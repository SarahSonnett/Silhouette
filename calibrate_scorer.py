"""Calibrate the solution scorer by injection-recovery, then train and evaluate it.

Two steps:

  python calibrate_scorer.py run   --n 3000 --workers 4 --hours 6
  python calibrate_scorer.py train

``run`` draws synthetic asteroids (irregular convex shapes, isotropic poles,
3–20 h periods, a scattering law that differs from the one the fit assumes)
observed with survey-like plans (1–6 apparitions, 12–60 points per curve,
0.5–4 % noise, phase angles to 28° or, for 15 % of bodies, to 60°), runs the
full scorer pipeline on each, and appends one row per candidate solution to
``results/scorer/injections.csv``. It resumes where it left off and stops
cleanly at ``--hours``.

``train`` fits the calibrated classifier with injection-grouped CV, holds out
20 % of injections for an honest test, compares against the baselines
(best-chi² indicator, the classical DAMIT-style uniqueness rule, raw likelihood
weights, the uncalibrated scorer), writes the model to
``silhouette/models/scorer_v1.pkl`` and figures/metrics to ``results/scorer/``.

Cores: ``--workers`` bounds the pool (default 4, leaving the rest of the
machine free); BLAS threads are pinned to one per worker.
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "scorer")
CSV = os.path.join(OUT, "injections.csv")


def cmd_run(args):
    from silhouette.calibration import run_injections
    os.makedirs(OUT, exist_ok=True)
    deadline = time.time() + args.hours * 3600.0 if args.hours else None
    print(f"[{time.strftime('%H:%M:%S')}] injections {args.start}..{args.start + args.n - 1} "
          f"on {args.workers} workers -> {CSV}", flush=True)
    n = run_injections(range(args.start, args.start + args.n), CSV,
                       n_workers=args.workers, deadline=deadline)
    print(f"[{time.strftime('%H:%M:%S')}] finished {n} new injections", flush=True)


def load_table(path=CSV):
    import pandas as pd
    df = pd.read_csv(path)
    return df


def cmd_train(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import log_loss, roc_auc_score

    from silhouette.calibration import (
        DEFAULT_MODEL_PATH, FEATURES, brier, classical_accept, fit_score_model,
        add_derived_features, reliability, upgrade_legacy_s2, yield_curve,
    )

    df = add_derived_features(upgrade_legacy_s2(load_table()))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["correct"])
    inj = df["inj"].astype(int).values
    uniq = np.unique(inj)
    rng = np.random.default_rng(args.seed)
    print(f"{len(df)} candidate rows from {uniq.size} injections")
    print(f"found-at-all rate (truth among candidates): "
          f"{df.groupby('inj')['correct'].max().mean():.3f}")

    # Nested, injection-grouped CV: each outer fold's model (with its own inner
    # CV for the isotonic map) predicts injections it never saw, so every
    # injection gets an honest out-of-sample probability and the evaluation
    # uses the whole set rather than one small holdout.
    from sklearn.model_selection import GroupKFold
    X_all, y_all = df[FEATURES].values, df["correct"].values
    p_oof = np.full(len(df), np.nan)
    for k, (tr_i, te_i) in enumerate(GroupKFold(n_splits=args.folds).split(X_all, y_all, inj)):
        m_k = fit_score_model(X_all[tr_i], y_all[tr_i], inj[tr_i], seed=args.seed)
        p_oof[te_i] = m_k.predict_from_matrix(X_all[te_i])
        if k == 0:
            model = m_k                       # used only for permutation importance
            imp_rows = te_i
    te = df.copy()
    y_te = y_all

    # ---- baselines on the same rows -----------------------------------------
    te["p_cal"] = p_oof
    te["p_rank0"] = (te["rank"] == 0).astype(float)
    base = {
        "calibrated scorer": te["p_cal"].values,
        "uncalibrated scorer": te["p_uncal"].values,
        "likelihood weight": te["like_weight"].values,
        "best-chi2 indicator": te["p_rank0"].values,
    }
    metrics = {}
    for name, p in base.items():
        pc = np.clip(p, 1e-4, 1 - 1e-4)
        metrics[name] = {
            "brier": brier(p, y_te),
            "log_loss": float(log_loss(y_te, pc, labels=[0, 1])),
            "auc": float(roc_auc_score(y_te, p)) if len(np.unique(y_te)) > 1 else float("nan"),
        }
    for k, v in metrics.items():
        print(f"  {k:22s} Brier {v['brier']:.4f}  logloss {v['log_loss']:.4f}  AUC {v['auc']:.3f}")

    # ---- yield at fixed false-solution rate (per object = per injection) ----
    groups = {i: g for i, g in te.groupby("inj")}
    n_obj = len(groups)
    top_p, top_ok, top_uncal, top_ok_uncal = [], [], [], []
    cl_acc = cl_ok = cl_acc_m = cl_ok_m = 0
    for i, g in groups.items():
        g = g.sort_values("p_cal", ascending=False)
        top_p.append(g["p_cal"].iloc[0])
        top_ok.append(g["correct"].iloc[0])
        gu = g.sort_values("p_uncal", ascending=False)
        top_uncal.append(gu["p_uncal"].iloc[0])
        top_ok_uncal.append(gu["correct"].iloc[0])
        rows = g.to_dict("records")
        acc = classical_accept(rows, allow_mirror=False)
        if acc is not None:
            cl_acc += 1
            cl_ok += int(acc["correct"])
        acc = classical_accept(rows, allow_mirror=True)
        if acc is not None:
            cl_acc_m += 1
            cl_ok_m += int(acc["correct"])
    top_p, top_ok = np.array(top_p), np.array(top_ok)
    thr, yl, fr = yield_curve(top_p, top_ok, n_obj)
    thr_u, yl_u, fr_u = yield_curve(np.array(top_uncal), np.array(top_ok_uncal), n_obj)
    classical = {
        "strict (no rival within 10%)": {
            "yield": cl_ok / n_obj, "false_rate": (cl_acc - cl_ok) / cl_acc if cl_acc else None,
            "accepted": cl_acc},
        "mirror allowed": {
            "yield": cl_ok_m / n_obj,
            "false_rate": (cl_acc_m - cl_ok_m) / cl_acc_m if cl_acc_m else None,
            "accepted": cl_acc_m},
    }
    print("classical rule:", json.dumps(classical, indent=1))

    def yield_at(fr_target, fr_arr, yl_arr):
        ok = np.where(np.nan_to_num(fr_arr, nan=1.0) <= fr_target)[0]
        return float(yl_arr[ok].max()) if ok.size else 0.0

    for target in (0.01, 0.02, 0.05, 0.10):
        print(f"  yield at false rate <= {target:.0%}: calibrated "
              f"{yield_at(target, fr, yl):.3f}, uncalibrated {yield_at(target, fr_u, yl_u):.3f}")

    # ---- head-to-head at the classical rule's own false rate, with bootstrap CIs
    per_obj = pd.DataFrame({"inj": list(groups.keys()), "p": top_p, "ok": top_ok})
    per_obj["n_app"] = [groups[i]["n_app"].iloc[0] for i in per_obj["inj"]]
    per_obj["found"] = [groups[i]["correct"].max() for i in per_obj["inj"]]
    cl_rows = {i: classical_accept(g.to_dict("records"), allow_mirror=False)
               for i, g in groups.items()}
    per_obj["cl_acc"] = [cl_rows[i] is not None for i in per_obj["inj"]]
    per_obj["cl_ok"] = [bool(cl_rows[i] is not None and cl_rows[i]["correct"])
                        for i in per_obj["inj"]]
    cl_fr = classical["strict (no rival within 10%)"]["false_rate"] or 0.0
    # threshold giving (at most) the classical false rate on the full test set
    thr_match = None
    for t_, f_ in zip(thr, fr):
        if np.isfinite(f_) and f_ <= cl_fr:
            thr_match = float(t_)
            break
    thr5 = next((float(t_) for t_, f_ in zip(thr, fr) if np.isfinite(f_) and f_ <= 0.05), 0.99)

    def boot_ci(stat, n_boot=500):
        vals = []
        for _ in range(n_boot):
            smp = per_obj.sample(len(per_obj), replace=True, random_state=int(rng.integers(1e9)))
            vals.append(stat(smp))
        return [float(np.nanpercentile(vals, 16)), float(np.nanpercentile(vals, 84))]

    def cal_yield(df_, t_):
        return float(((df_["p"] >= t_) & (df_["ok"] == 1)).mean())

    def cal_false(df_, t_):
        acc = df_["p"] >= t_
        return float((acc & (df_["ok"] == 0)).sum() / acc.sum()) if acc.sum() else np.nan

    head = {
        "classical_strict": {"yield": float(per_obj["cl_ok"].mean()),
                             "yield_ci68": boot_ci(lambda d: d["cl_ok"].mean()),
                             "false_rate": cl_fr},
        "calibrated_at_classical_false_rate": None if thr_match is None else {
            "threshold": thr_match, "yield": cal_yield(per_obj, thr_match),
            "yield_ci68": boot_ci(lambda d: cal_yield(d, thr_match)),
            "false_rate": cal_false(per_obj, thr_match)},
        "calibrated_at_5pct": {"threshold": thr5, "yield": cal_yield(per_obj, thr5),
                               "yield_ci68": boot_ci(lambda d: cal_yield(d, thr5)),
                               "false_rate": cal_false(per_obj, thr5),
                               "false_rate_ci68": boot_ci(lambda d: cal_false(d, thr5))},
    }
    rej = per_obj[~per_obj["cl_acc"]]
    head["classical_rejected"] = {
        "n": int(len(rej)),
        "mean_p_top": float(rej["p"].mean()) if len(rej) else None,
        "frac_top_correct": float(rej["ok"].mean()) if len(rej) else None,
        "n_p_top_gt_0.5": int((rej["p"] > 0.5).sum()),
        "frac_correct_when_p_gt_0.5": float(rej[rej["p"] > 0.5]["ok"].mean())
        if (rej["p"] > 0.5).any() else None,
    }
    by_app = []
    for napp, g in per_obj.groupby("n_app"):
        by_app.append({"n_app": int(napp), "n": int(len(g)),
                       "found_at_all": float(g["found"].mean()),
                       "classical_yield": float(g["cl_ok"].mean()),
                       "classical_accepted": int(g["cl_acc"].sum()),
                       "classical_wrong": int((g["cl_acc"] & ~g["cl_ok"]).sum()),
                       "calibrated_yield_at_5pct_thr": cal_yield(g, thr5),
                       "calibrated_accepted": int((g["p"] >= thr5).sum()),
                       "calibrated_wrong": int(((g["p"] >= thr5) & (g["ok"] == 0)).sum()),
                       "mean_p_top": float(g["p"].mean()),
                       "top_correct": float(g["ok"].mean())})
    print("head-to-head:", json.dumps(head, indent=1))
    print(f"{'n_app':>5s} {'N':>4s} {'found':>6s} {'classical':>10s} {'cl.wrong':>8s} "
          f"{'calib@5%':>9s} {'cal.wrong':>9s} {'<p_top>':>8s} {'top ok':>7s}")
    for r in by_app:
        print(f"{r['n_app']:5d} {r['n']:4d} {r['found_at_all']:6.2f} {r['classical_yield']:10.2f} "
              f"{r['classical_wrong']:8d} {r['calibrated_yield_at_5pct_thr']:9.2f} "
              f"{r['calibrated_wrong']:9d} {r['mean_p_top']:8.2f} {r['top_correct']:7.2f}")

    # ---- permutation importance on test ------------------------------------
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _Est(BaseEstimator, ClassifierMixin):
        def __init__(self, m=None):
            self.m = m
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            p = self.m.predict_from_matrix(X)
            return np.column_stack([1 - p, p])

        def predict(self, X):
            return (self.m.predict_from_matrix(X) >= 0.5).astype(int)

    pi = permutation_importance(_Est(model), X_all[imp_rows], y_all[imp_rows],
                                scoring="neg_brier_score", n_repeats=8,
                                random_state=args.seed, n_jobs=1)
    order = np.argsort(pi.importances_mean)[::-1]

    # ---- figures ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect calibration")
    for name, style in [("calibrated scorer", dict(color="C0", marker="o")),
                        ("uncalibrated scorer", dict(color="C1", marker="s")),
                        ("likelihood weight", dict(color="C2", marker="^"))]:
        mp, fy, n = reliability(base[name], y_te)
        ax.plot(mp, fy, label=f"{name} (Brier {metrics[name]['brier']:.3f})", **style)
    ax.set_xlabel("predicted P(correct)")
    ax.set_ylabel("observed fraction correct")
    ax.set_title("Reliability (out-of-sample)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.plot(np.nan_to_num(fr, nan=0) * 100, yl * 100, color="C0", label="calibrated scorer")
    ax.plot(np.nan_to_num(fr_u, nan=0) * 100, yl_u * 100, color="C1", label="uncalibrated scorer")
    for (name, v), mk in zip(classical.items(), ("X", "P")):
        if v["false_rate"] is not None:
            ax.plot(v["false_rate"] * 100, v["yield"] * 100, mk, ms=11, color="C3",
                    label=f"classical: {name}")
    ax.set_xlabel("false-solution rate among accepted (%)")
    ax.set_ylabel("objects correctly solved (%)")
    ax.set_title("Yield vs false-solution rate")
    ax.set_xlim(0, 40)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    k = order[:12][::-1]
    ax.barh([FEATURES[i] for i in k], pi.importances_mean[k], xerr=pi.importances_std[k],
            color="C0")
    ax.set_xlabel("permutation importance (ΔBrier)")
    ax.set_title("What the scorer relies on")
    fig.tight_layout()
    fig_path = os.path.join(OUT, "scorer_calibration.png")
    fig.savefig(fig_path, dpi=130)
    os.makedirs(os.path.join(HERE, "docs", "images"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "docs", "images", "scorer_calibration.png"), dpi=130)

    # P(correct) of the top candidate vs data richness -----------------------
    per = te.sort_values("p_cal", ascending=False).groupby("inj").head(1)
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, (col, lab) in enumerate([("n_app", "apparitions"), ("noise_frac", "fractional noise")]):
        a = ax2[j]
        if col == "n_app":
            xs = sorted(per[col].unique())
            ok = [per[per[col] == x]["correct"].mean() for x in xs]
            pp = [per[per[col] == x]["p_cal"].mean() for x in xs]
        else:
            bins = np.quantile(per[col], np.linspace(0, 1, 6))
            idx = np.clip(np.digitize(per[col], bins) - 1, 0, 4)
            xs = [per[col][idx == b].median() for b in range(5)]
            ok = [per["correct"][idx == b].mean() for b in range(5)]
            pp = [per["p_cal"][idx == b].mean() for b in range(5)]
        a.plot(xs, ok, "o-", label="top candidate actually correct")
        a.plot(xs, pp, "s--", label="mean predicted P(correct)")
        a.set_xlabel(lab)
        a.set_ylim(0, 1.02)
        a.set_ylabel("fraction / probability")
        a.legend(fontsize=8)
        if col == "noise_frac":
            a.set_xscale("log")
    fig2.suptitle("Out-of-sample: does the predicted probability track reality?")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "scorer_vs_data.png"), dpi=130)
    fig2.savefig(os.path.join(HERE, "docs", "images", "scorer_vs_data.png"), dpi=130)

    # ---- identifiability map: how often is a unique, correct answer reachable?
    per_obj["noise"] = [groups[i]["noise_frac"].iloc[0] for i in per_obj["inj"]]
    per_obj["lonc"] = [groups[i]["lon_coverage"].iloc[0] for i in per_obj["inj"]]
    nbins = [0.0, 0.01, 0.02, 1.0]
    nlab = ["<1%", "1-2%", ">2%"]
    per_obj["nbin"] = pd.cut(per_obj["noise"], nbins, labels=nlab)
    apps = sorted(per_obj["n_app"].unique())
    grids = {k: np.full((len(nlab), len(apps)), np.nan) for k in ("found", "ok", "p", "n")}
    for a_i, a in enumerate(apps):
        for n_i, nl in enumerate(nlab):
            g = per_obj[(per_obj["n_app"] == a) & (per_obj["nbin"] == nl)]
            if len(g):
                grids["found"][n_i, a_i] = g["found"].mean()
                grids["ok"][n_i, a_i] = g["ok"].mean()
                grids["p"][n_i, a_i] = g["p"].mean()
                grids["n"][n_i, a_i] = len(g)
    fig3, ax3 = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for a_, key, title in zip(ax3, ("found", "ok", "p"),
                              ("truth among the candidates", "top candidate correct",
                               "mean calibrated P(top)")):
        im = a_.imshow(grids[key], vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
        a_.set_xticks(range(len(apps)), [str(int(a)) for a in apps])
        a_.set_yticks(range(len(nlab)), nlab)
        a_.set_xlabel("apparitions")
        if key == "found":
            a_.set_ylabel("photometric noise")
        a_.set_title(title)
        for (yy, xx), v in np.ndenumerate(grids[key]):
            if np.isfinite(v):
                a_.text(xx, yy, f"{v:.2f}\nN={int(grids['n'][yy, xx])}", ha="center",
                        va="center", fontsize=7, color="w" if v < 0.6 else "k")
    fig3.colorbar(im, ax=ax3, shrink=0.85)
    fig3.suptitle("Identifiability map from injection-recovery (pole within 20° and true period)")
    fig3.savefig(os.path.join(OUT, "identifiability_map.png"), dpi=130, bbox_inches="tight")
    fig3.savefig(os.path.join(HERE, "docs", "images", "identifiability_map.png"), dpi=130,
                 bbox_inches="tight")

    # ---- ship a model trained on ALL injections (metrics above are the
    # out-of-sample nested-CV ones)
    final = fit_score_model(df[FEATURES].values, df["correct"].values,
                            df["inj"].astype(int).values, seed=args.seed)
    final.meta.update({"oof_metrics": metrics, "classical": classical,
                       "head_to_head": head, "n_eval_injections": n_obj, "eval": "nested grouped CV",
                       "settings": "calibration.FAST_INV + FAST_GRID, n_boot=10, tol 20 deg"})
    final.save(DEFAULT_MODEL_PATH)
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump({"metrics": metrics, "classical": classical, "head_to_head": head,
                   "by_n_app": by_app,
                   "yield_at_false_rate": {str(t): {"calibrated": yield_at(t, fr, yl),
                                                    "uncalibrated": yield_at(t, fr_u, yl_u)}
                                           for t in (0.01, 0.02, 0.05, 0.10)},
                   "importance": {FEATURES[i]: float(pi.importances_mean[i]) for i in order},
                   "n_rows": int(len(df)), "n_injections": int(uniq.size),
                   "found_at_all": float(df.groupby("inj")["correct"].max().mean())},
                  fh, indent=1)
    print(f"model -> {DEFAULT_MODEL_PATH}\nfigures -> {fig_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run injection-recovery trials")
    r.add_argument("--n", type=int, default=3000)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--hours", type=float, default=None, help="stop cleanly after this long")
    t = sub.add_parser("train", help="train + evaluate the calibrated classifier")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--folds", type=int, default=5, help="outer grouped-CV folds")
    args = ap.parse_args()
    {"run": cmd_run, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()

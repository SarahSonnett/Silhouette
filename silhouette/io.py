"""Flexible reader for tabular asteroid photometry.

Accepts whitespace- or comma-delimited tables with a one-line column header and
maps a wide range of common column names onto Silhouette's canonical fields.
This is deliberately permissive so the same reader handles SpinDoc-style
calibrated photometry files (``Frame Rhelio Delta alpha … MJD TmagCorr …``) and
files that additionally carry per-row ecliptic coordinates.

Canonical fields
----------------
time     : epoch (returned as MJD; JD inputs are auto-detected and converted)
mag      : apparent (calibrated) magnitude
merr     : magnitude uncertainty
rhelio   : heliocentric distance [au]
delta    : observer (geocentric) distance [au]
alpha    : solar phase angle [deg]
filt     : filter / band label (optional)
ecl_lon  : observer-centric ecliptic longitude of the target [deg] (optional)
ecl_lat  : observer-centric ecliptic latitude of the target [deg]  (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Canonical field -> set of accepted (lower-cased) header aliases.
_ALIASES: Dict[str, set] = {
    "time":    {"mjd", "jd", "time", "epoch", "date", "jdutc", "mjdutc"},
    "mag":     {"mag", "magnitude", "tmagcorr", "tmag", "redmag", "m", "mean_mag", "calmag"},
    "merr":    {"merr", "tmagfinalerr", "tmagerr", "magerr", "err", "error", "sigma", "dmag", "emag"},
    "rhelio":  {"rhelio", "rh", "r", "helio", "r_helio", "rsun", "sundist"},
    "delta":   {"delta", "geo", "d", "range", "obsdist", "obs_dist", "deldist"},
    "alpha":   {"alpha", "phase", "phaseangle", "phase_angle", "sunangle"},
    "filt":    {"filter", "filt", "band", "fltr"},
    "ecl_lon": {"ecl_lon", "eclon", "lambda", "lon", "elon", "ecliptic_lon",
                "obsecllon", "obs_ecl_lon", "eclipticlongitude"},
    "ecl_lat": {"ecl_lat", "eclat", "beta", "lat", "elat", "ecliptic_lat",
                "obsecllat", "obs_ecl_lat", "eclipticlatitude"},
}

_REQUIRED = ("time", "mag", "merr", "rhelio", "delta", "alpha")


@dataclass
class Photometry:
    """A loaded photometry table as parallel NumPy arrays."""

    time: np.ndarray            # MJD
    mag: np.ndarray
    merr: np.ndarray
    rhelio: np.ndarray
    delta: np.ndarray
    alpha: np.ndarray
    filt: Optional[np.ndarray] = None
    ecl_lon: Optional[np.ndarray] = None
    ecl_lat: Optional[np.ndarray] = None
    object_name: Optional[str] = None
    source: Optional[str] = None
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def has_ecliptic(self) -> bool:
        return self.ecl_lon is not None and self.ecl_lat is not None

    def __len__(self) -> int:
        return int(self.time.size)


def _normalise(name: str) -> str:
    return name.strip().lstrip("#").strip().lower().replace(" ", "").replace("-", "_")


def _resolve_columns(header: List[str], overrides: Optional[Dict[str, str]]) -> Dict[str, int]:
    """Map canonical fields to column indices from a header line."""
    norm = [_normalise(h) for h in header]
    colmap: Dict[str, int] = {}

    if overrides:
        for canon, colname in overrides.items():
            key = _normalise(colname)
            if key in norm:
                colmap[canon] = norm.index(key)
            else:
                raise KeyError(f"override column {colname!r} not found in header {header}")

    for canon, aliases in _ALIASES.items():
        if canon in colmap:
            continue
        for i, h in enumerate(norm):
            if h in aliases:
                colmap[canon] = i
                break
    return colmap


def read_photometry(
    infile: str,
    columns: Optional[Dict[str, str]] = None,
    delimiter: Optional[str] = None,
    object_name: Optional[str] = None,
) -> Photometry:
    """Read a tabular photometry file into a :class:`Photometry`.

    Parameters
    ----------
    infile : str
        Path to a delimited photometry table whose first row is a header.
    columns : dict, optional
        Explicit ``{canonical_field: header_name}`` overrides for any column
        whose name is not auto-recognised.
    delimiter : str, optional
        Field delimiter. ``None`` (default) splits on any whitespace; pass
        ``","`` for CSV.
    object_name : str, optional
        Target designation, stored for later ephemeris lookups.
    """
    with open(infile) as fh:
        first = fh.readline().rstrip("\n")
    header = first.split(delimiter) if delimiter else first.split()
    colmap = _resolve_columns(header, columns)

    missing = [f for f in _REQUIRED if f not in colmap]
    if missing:
        raise ValueError(
            f"Could not identify required column(s) {missing} in {infile!r}.\n"
            f"Header was: {header}\n"
            f"Pass explicit names via columns={{'time': 'MJD', ...}}."
        )

    raw = np.genfromtxt(infile, dtype=str, skip_header=1, delimiter=delimiter)
    if raw.ndim == 1:                       # single data row
        raw = raw[np.newaxis, :]

    def col_f(name: str) -> Optional[np.ndarray]:
        if name not in colmap:
            return None
        return raw[:, colmap[name]].astype(float)

    time = col_f("time")
    # Auto-detect JD vs MJD: JD epochs are ~2.4e6.
    if time is not None and np.nanmedian(time) > 2_400_000.0:
        time = time - 2_400_000.5

    filt = None
    if "filt" in colmap:
        filt = raw[:, colmap["filt"]].astype(str)

    return Photometry(
        time=time,
        mag=col_f("mag"),
        merr=col_f("merr"),
        rhelio=col_f("rhelio"),
        delta=col_f("delta"),
        alpha=col_f("alpha"),
        filt=filt,
        ecl_lon=col_f("ecl_lon"),
        ecl_lat=col_f("ecl_lat"),
        object_name=object_name,
        source=infile,
    )


__all__ = ["Photometry", "read_photometry"]

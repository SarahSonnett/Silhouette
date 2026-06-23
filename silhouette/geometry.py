"""Resolve the observer-centric ecliptic direction to each apparition.

The aspect angle the shape fit relies on needs the ecliptic longitude/latitude
of the target as seen from the observer at each apparition. Per the "either,
with fallback" design, Silhouette uses values carried in the photometry file
when present, and otherwise queries JPL Horizons (via ``astroquery``) at each
apparition's midpoint epoch.
"""

from __future__ import annotations

from typing import List, Optional

from .apparitions import Apparition


def _have_astroquery() -> bool:
    try:
        import astroquery.jplhorizons  # noqa: F401
        return True
    except Exception:
        return False


def fetch_horizons_ecliptic(
    target: str,
    mjd_epochs: List[float],
    location: str = "500@399",
) -> List[tuple]:
    """Return [(ecl_lon, ecl_lat), ...] from JPL Horizons for each MJD epoch.

    Parameters
    ----------
    target : str
        Object designation understood by Horizons (e.g. ``"433"`` or ``"Eros"``).
    mjd_epochs : list of float
        Apparition midpoint epochs (MJD).
    location : str
        Observer location code. Default ``"500@399"`` is the geocentre.
    """
    from astroquery.jplhorizons import Horizons

    out: List[tuple] = []
    for mjd in mjd_epochs:
        jd = mjd + 2_400_000.5
        obj = Horizons(id=target, location=location, epochs=jd)
        eph = obj.ephemerides()
        # ObsEclLon/ObsEclLat: observer-centric ecliptic coordinates of target.
        out.append((float(eph["ObsEclLon"][0]), float(eph["ObsEclLat"][0])))
    return out


def resolve_geometry(
    apparitions: List[Apparition],
    target: Optional[str] = None,
    prefer_file: bool = True,
    location: str = "500@399",
) -> List[Apparition]:
    """Ensure every apparition has ecliptic (lon, lat) filled in.

    Uses file-supplied coordinates when available (and ``prefer_file``), then
    falls back to Horizons for any apparition still missing geometry. Raises a
    clear error if geometry cannot be obtained.
    """
    need = [a for a in apparitions
            if not (prefer_file and a.ecl_lon is not None and a.ecl_lat is not None)]

    if need:
        if target is None:
            missing = [a for a in need if a.ecl_lon is None]
            if missing:
                raise ValueError(
                    "Some apparitions lack ecliptic coordinates and no Horizons "
                    "target was supplied. Either add ecl_lon/ecl_lat columns to "
                    "the photometry file or pass target=<designation>."
                )
        elif not _have_astroquery():
            raise ImportError(
                "astroquery is required to fetch geometry from JPL Horizons.\n"
                "Install it (`pip install astroquery`) or add ecl_lon/ecl_lat "
                "columns to the photometry file."
            )
        else:
            coords = fetch_horizons_ecliptic(
                target, [a.epoch_mid for a in need], location=location
            )
            for a, (lon, lat) in zip(need, coords):
                a.ecl_lon, a.ecl_lat = lon, lat
                a.geom_source = "horizons"

    return apparitions


__all__ = ["resolve_geometry", "fetch_horizons_ecliptic"]

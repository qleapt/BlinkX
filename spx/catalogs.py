"""Gaia DR3 cone search (Gaia archive, VizieR fallback) and WISE / 2MASS reference frames."""
from __future__ import annotations

import csv
import io
import json
import time

import numpy as np
import requests

from . import config


def _tap_csv(url: str, adql: str, timeout=(6, 40)) -> list[dict]:
    r = requests.get(url, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
                     timeout=timeout)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def gaia_cone(ra: float, dec: float, radius_deg: float) -> tuple[list[dict], str]:
    """Stars brighter than GAIA_MAX_G: dicts with ra, dec (epoch 2016.0), pmra, pmdec, parallax, g, id."""
    path = config.cache_dir() / f"gaia_{ra:.4f}_{dec:+.4f}_{radius_deg:.3f}.json"
    if path.exists():
        blob = json.loads(path.read_text())
        return blob["stars"], blob["source"]
    circle = f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.5f})"
    attempts = [
        ("Gaia archive", config.GAIA_TAP,
         f"SELECT TOP {config.GAIA_TOP} source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag "
         f"FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), {circle}) "
         f"AND phot_g_mean_mag < {config.GAIA_MAX_G} ORDER BY phot_g_mean_mag",
         ("source_id", "ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag")),
        ("VizieR", config.VIZIER_TAP,
         f'SELECT TOP {config.GAIA_TOP} Source, RA_ICRS, DE_ICRS, pmRA, pmDE, Plx, Gmag '
         f'FROM "I/355/gaiadr3" WHERE 1 = CONTAINS(POINT(\'ICRS\', RA_ICRS, DE_ICRS), {circle}) '
         f"AND Gmag < {config.GAIA_MAX_G} ORDER BY Gmag",
         ("Source", "RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Plx", "Gmag")),
    ]
    last = None
    for name, url, q, cols in attempts:
        try:
            rows = _tap_csv(url, q)
        except Exception as e:  # try the next service
            last = e
            continue
        stars = [dict(id=r[cols[0]], ra=_f(r[cols[1]]), dec=_f(r[cols[2]]), pmra=_f(r[cols[3]]),
                      pmdec=_f(r[cols[4]]), parallax=_f(r[cols[5]]), g=_f(r[cols[6]])) for r in rows]
        path.write_text(json.dumps({"t": time.time(), "source": name, "stars": stars}))
        return stars, name
    raise RuntimeError(f"Gaia catalogue unavailable ({last})")


def gaia_at(stars: list[dict], mjd: float):
    """RA, Dec arrays of the stars moved to `mjd` (linear proper motion)."""
    if not stars:
        return np.zeros(0), np.zeros(0)
    ra = np.array([s["ra"] for s in stars])
    dec = np.array([s["dec"] for s in stars])
    pmra = np.nan_to_num(np.array([s["pmra"] for s in stars]))
    pmdec = np.nan_to_num(np.array([s["pmdec"] for s in stars]))
    dt = (mjd - config.GAIA_REF_MJD) / 365.25
    dec2 = dec + pmdec * dt / 3.6e6
    ra2 = ra + pmra * dt / 3.6e6 / np.cos(np.radians(dec))
    return ra2, dec2


def nearest(stars: list[dict], ra: float, dec: float, mjd: float, max_arcsec: float = 6.0):
    """The Gaia star closest to (ra, dec) on date mjd (stars moved to that date), if within max_arcsec.
    Returns (star, separation in arcsec) or (None, None)."""
    r, d = gaia_at(stars, mjd)
    if not len(r):
        return None, None
    sep = np.hypot(((r - ra + 180) % 360 - 180) * np.cos(np.radians(dec)), d - dec) * 3600
    i = int(np.argmin(sep))
    return (stars[i], float(sep[i])) if sep[i] <= max_arcsec else (None, None)


def as_target(s: dict, name: str | None = None):
    """A Gaia star as a targets.Target (epoch 2016.0), so the motion check can predict its path."""
    from .targets import Target
    f = lambda v: 0.0 if v is None or not np.isfinite(v) else float(v)  # noqa: E731
    return Target(name or f"Gaia DR3 {s['id']}", float(s["ra"]), float(s["dec"]), config.GAIA_REF_MJD,
                  f(s.get("pmra")), f(s.get("pmdec")), f(s.get("parallax")), "Gaia DR3 (2016.0), matched to your mark")


# --- WISE 2010 and 2MASS reference frames ------------------------------------------------
def _sia(ra: float, dec: float, collection: str, band: str) -> list[dict]:
    path = config.cache_dir() / f"sia_{collection}_{band}_{ra:.4f}_{dec:+.4f}.json"
    if path.exists():
        return json.loads(path.read_text())
    # IRSA's SIA v2 service directly as CSV (faster than going through astroquery)
    r = requests.get(f"{config.IRSA}/SIA", timeout=60, params={
        "COLLECTION": collection, "POS": f"circle {ra:.6f} {dec:.6f} 0.008", "RESPONSEFORMAT": "CSV"})
    r.raise_for_status()
    rows = [dict(url=x["access_url"], mjd=float(x["t_min"]), band=x["energy_bandpassname"])
            for x in csv.DictReader(io.StringIO(r.text))
            if x.get("dataproduct_subtype") == "science" and x.get("energy_bandpassname") == band and x.get("t_min")]
    rows.sort(key=lambda r: r["mjd"])
    path.write_text(json.dumps(rows))
    return rows


def _cutout(url: str, ra: float, dec: float, size_arcmin: float):
    import gzip
    import hashlib
    from astropy.io import fits
    key = hashlib.sha1(f"{url}|{ra:.6f}|{dec:.6f}|{size_arcmin:.2f}".encode()).hexdigest()[:20]
    path = config.cache_dir() / f"sv_{key}.fits"
    if not path.exists():
        from .cutouts import _get_with_retry
        r = _get_with_retry(f"{url}?center={ra:.6f},{dec:.6f}&size={size_arcmin:.2f}arcmin", False)
        raw = r.content
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        path.write_bytes(raw)
    hd = fits.open(path)
    return hd[0].data.astype(float), hd[0].header


def survey_frames(ra: float, dec: float, sphx_band: str, size_arcmin: float) -> list[dict]:
    """Older reference images for one SPHEREx band: WISE 2010 (first visit, up to 8 frames)
    and 2MASS (~1998-2001). Each item: {survey, band, mjds, images: [(data, header)]}."""
    out = []
    wb = config.SURVEY_MATCH.get(sphx_band, {}).get("wise")
    if wb:
        rows = [r for r in _sia(ra, dec, "wise_allsky", wb) if r["mjd"] < config.WISE_CRYO_END_MJD]
        if rows:  # first visit only: frames within a few days of the first one
            visit = [r for r in rows if r["mjd"] - rows[0]["mjd"] < 5][:8]
            from concurrent.futures import ThreadPoolExecutor

            def one(r):
                try:
                    return r["mjd"], _cutout(r["url"], ra, dec, size_arcmin)
                except Exception:
                    return None
            with ThreadPoolExecutor(4) as ex:  # in parallel: these were 2-3 s each, one after another
                got = [g for g in ex.map(one, visit[:4]) if g]
            if got:
                out.append(dict(survey="WISE", band=wb, mjds=[m for m, _ in got], images=[i for _, i in got]))
    tb = config.SURVEY_MATCH.get(sphx_band, {}).get("2mass")
    if tb:
        rows = _sia(ra, dec, "twomass_allsky", tb)
        if rows:
            try:
                out.append(dict(survey="2MASS", band=tb, mjds=[rows[0]["mjd"]],
                                images=[_cutout(rows[0]["url"], ra, dec, size_arcmin)]))
            except Exception:
                pass
    return out


def gaia_hover(s: dict) -> str:
    plx = s["parallax"]
    dist = f" (~{1000 / plx:,.0f} pc)" if plx and plx > 0.2 else ""
    pm = "" if np.isnan(s["pmra"]) else f"<br>proper motion ({s['pmra']:+.1f}, {s['pmdec']:+.1f}) mas/yr"
    pl = "" if np.isnan(plx) else f"<br>parallax {plx:.2f} mas{dist}"
    return f"Gaia DR3 {s['id']}<br>G = {s['g']:.2f}{pm}{pl}"

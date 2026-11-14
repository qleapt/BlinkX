"""Gaia DR3 cone search (Gaia archive, VizieR fallback) and WISE / 2MASS reference frames."""
from __future__ import annotations

import csv
import io
import json
import time

import numpy as np
import requests

from . import config


def _tap_csv(url: str, adql: str, timeout: float = 40) -> list[dict]:
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


# --- WISE 2010 and 2MASS reference frames ------------------------------------------------
def _sia(ra: float, dec: float, collection: str, band: str) -> list[dict]:
    path = config.cache_dir() / f"sia_{collection}_{band}_{ra:.4f}_{dec:+.4f}.json"
    if path.exists():
        return json.loads(path.read_text())
    from astroquery.ipac.irsa import Irsa
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    t = Irsa.query_sia(pos=(SkyCoord(ra * u.deg, dec * u.deg), 0.5 * u.arcmin), collection=collection)
    rows = [dict(url=str(r["access_url"]), mjd=float(r["t_min"]), band=str(r["energy_bandpassname"]))
            for r in t if str(r["dataproduct_subtype"]) == "science" and str(r["energy_bandpassname"]) == band]
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
            imgs = []
            for r in visit:
                try:
                    imgs.append(_cutout(r["url"], ra, dec, size_arcmin))
                except Exception:
                    continue
            if imgs:
                out.append(dict(survey="WISE", band=wb, mjds=[r["mjd"] for r in visit][:len(imgs)], images=imgs))
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

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


def gaia_hover(s: dict) -> str:
    plx = s["parallax"]
    dist = f" (~{1000 / plx:,.0f} pc)" if plx and plx > 0.2 else ""
    pm = "" if np.isnan(s["pmra"]) else f"<br>proper motion ({s['pmra']:+.1f}, {s['pmdec']:+.1f}) mas/yr"
    pl = "" if np.isnan(plx) else f"<br>parallax {plx:.2f} mas{dist}"
    return f"Gaia DR3 {s['id']}<br>G = {s['g']:.2f}{pm}{pl}"

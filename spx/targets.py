"""Target lookup: coordinates, SIMBAD names, five presets, proper motion."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

J2000_MJD = 51544.5


@dataclass(frozen=True)
class Target:
    name: str
    ra: float            # deg, at ref_mjd
    dec: float           # deg, at ref_mjd
    ref_mjd: float = J2000_MJD
    pmra: float = 0.0    # mas/yr, includes cos(dec)
    pmdec: float = 0.0   # mas/yr
    parallax: float = 0.0  # mas
    source: str = ""

    @property
    def has_motion(self) -> bool:
        return abs(self.pmra) + abs(self.pmdec) > 50


# Reference values: Gaia DR3 (epoch 2016.0) where available, else SIMBAD/literature.
PRESETS = {
    "Barnard's Star": Target("Barnard's Star", 269.448502525, 4.739420051, 57388.5,
                             -801.551, 10362.394, 546.976, "Gaia DR3 (2016.0)"),
    "Epsilon Indi B": Target("Epsilon Indi B", 331.0438, -56.78269, J2000_MJD,
                             3966.7, -2536.2, 274.8, "SIMBAD (J2000)"),
    "WISE 0855-0714": Target("WISE 0855-0714", 133.795129, -7.245147, 55330.0,
                             -8118.0, 680.0, 439.0, "WISE All-Sky (2010.4)"),
    "RW Aurigae": Target("RW Aurigae", 76.956407, 30.401398, J2000_MJD,
                         3.0, -25.0, 6.4, "SIMBAD (J2000)"),
    "R Draconis": Target("R Draconis", 248.167433, 66.755117, J2000_MJD,
                         -5.0, 14.0, 2.5, "SIMBAD (J2000)"),
}
# Band shown first for each preset (where the object stands out best).
PRESET_BAND = {"Barnard's Star": "D4", "Epsilon Indi B": "D6", "WISE 0855-0714": "D6",
               "RW Aurigae": "D3", "R Draconis": "D6"}  # R Dra saturates beyond repair in D1-D3
ALIASES = {
    "barnard": "Barnard's Star", "barnards star": "Barnard's Star", "barnard's star": "Barnard's Star",
    "gj 699": "Barnard's Star", "epsilon indi b": "Epsilon Indi B", "eps ind b": "Epsilon Indi B",
    "epsilon indi ba": "Epsilon Indi B", "eps indi b": "Epsilon Indi B",
    "wise 0855": "WISE 0855-0714", "wise 0855-0714": "WISE 0855-0714", "wise 0855−0714": "WISE 0855-0714",
    "wise j085510.83-071442.5": "WISE 0855-0714", "rw aur": "RW Aurigae", "rw aurigae": "RW Aurigae",
    "r dra": "R Draconis", "r draconis": "R Draconis",
}


class TargetNotFound(ValueError):
    pass


def preset(text: str) -> Target | None:
    key = re.sub(r"\s+", " ", text.strip().lower().replace("’", "'"))
    name = ALIASES.get(key) or next((n for n in PRESETS if n.lower() == key), None)
    return PRESETS.get(name) if name else None


_DEC_NUM = r"[+\-−]?\d+(?:\.\d*)?|[+\-−]?\.\d+"


def parse_coords(text: str) -> Target | None:
    """Accept decimal degrees in many spellings ('269.45 4.74', '269.45, -4.74', '269.45° +4.74°',
    'RA=269.45 Dec=4.74', '269.45d 4.74d') or sexagesimal ('17 57 48.5 +04 41 36', '17h57m48.5s +4d41m36s')."""
    t = text.strip().lower().replace("−", "-").replace(",", " ").replace(";", " ")
    t = re.sub(r"(?<![a-z])(ra|dec|de|alpha|delta)(?![a-z])\s*[:=]?", " ", t)
    t = t.replace("=", " ").strip()
    if not t or re.search(r"[a-ce-gi-ln-rt-z]", t):  # letters other than d h m s: a name, not a position
        return None
    # decimal degrees: exactly two numbers, optionally followed by ° or d (no h/m/s/: anywhere)
    dec_tokens = re.fullmatch(rf"\s*({_DEC_NUM})\s*(?:°|deg|d)?\s+({_DEC_NUM})\s*(?:°|deg|d)?\s*", t)
    if dec_tokens:
        ra, dec = float(dec_tokens.group(1)), float(dec_tokens.group(2))
        if 0 <= ra < 360 and -90 <= dec <= 90:
            return Target(f"RA {ra:.5f}, Dec {dec:+.5f}", ra, dec, source="coordinates")
        return None
    try:
        if re.fullmatch(r"[\d\s:hmsd'\"+\-.°]+", t) and len(t.split()) >= 2:
            c = SkyCoord(t, unit=(u.hourangle, u.deg))
            return Target(f"RA {c.ra.deg:.5f}, Dec {c.dec.deg:+.5f}", c.ra.deg, c.dec.deg, source="coordinates")
    except (ValueError, u.UnitsError):
        return None
    return None


def looks_like_coords(text: str) -> bool:
    """True when the text is mostly numbers (the user meant a position, even if we could not read it)."""
    s = re.sub(r"(?i)(?<![a-z])(ra|dec)(?![a-z])", " ", text or "")
    if len(re.findall(r"\d+(?:\.\d*)?", s)) < 2:  # "M 42" is a name, "269.4 4.7" a position
        return False
    t = re.sub(r"[\s,;:=°'\"hmsdHMSD+\-.−]", "", s)
    return bool(t) and t.isdigit()


def _num(row, *names):
    for n in names:
        if n in row.colnames:
            v = row[n]
            try:
                if np.ma.is_masked(v):
                    continue
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def simbad_lookup(name: str) -> Target:
    """SIMBAD TAP (identifiers are matched case- and space-insensitively), with mirrors."""
    import csv
    import io
    import requests
    from . import config
    q = ("SELECT TOP 1 b.main_id, b.ra, b.dec, b.pmra, b.pmdec, b.plx_value FROM basic b "
         f"JOIN ident i ON i.oidref = b.oid WHERE i.id = '{name.strip()[:80].replace(chr(39), chr(39) * 2)}'")
    last = None
    for url in config.SIMBAD_TAPS:
        try:
            r = requests.get(url, params=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q),
                             timeout=(5, 20))  # short connect timeout: a dead mirror costs 5 s, not 20
            r.raise_for_status()
            rows = list(csv.DictReader(io.StringIO(r.text)))
        except Exception as e:  # try the next mirror
            last = e
            continue
        if not rows or not rows[0].get("ra"):
            raise TargetNotFound(name)
        row = rows[0]

        def f(k):
            try:
                return float(row.get(k) or 0.0)
            except ValueError:
                return 0.0
        return Target(name, f("ra"), f("dec"), J2000_MJD, f("pmra"), f("pmdec"), f("plx_value"),
                      f"SIMBAD: {row.get('main_id', name).strip()} (J2000)")
    raise RuntimeError(f"SIMBAD unreachable ({type(last).__name__})")


def resolve(text: str) -> Target:
    if not text or not text.strip():
        raise TargetNotFound("empty")
    hit = preset(text) or parse_coords(text)
    if hit:
        return hit
    if looks_like_coords(text):  # numbers we could not read: do not send them to SIMBAD
        raise TargetNotFound(text)
    return simbad_lookup(text.strip())


def position_at(t: Target, mjd: float) -> tuple[float, float]:
    """Linear proper-motion propagation (good to <0.1" for decades away from the poles)."""
    dt = (mjd - t.ref_mjd) / 365.25
    ddec = t.pmdec * dt / 3.6e6
    dec = t.dec + ddec
    cosd = np.maximum(np.cos(np.radians((t.dec + dec) / 2)), 1e-6)   # works for one date or an array of dates
    ra = (t.ra + t.pmra * dt / 3.6e6 / cosd) % 360
    return ra, dec


def predicted_offsets(t: Target, mjds, ref_mjd: float, with_parallax: bool = True):
    """Offsets (east, north) in arcsec of the target from its position at ref_mjd,
    including annual parallax as seen from Earth (SPHEREx is in Earth orbit)."""
    from astropy.coordinates import get_body_barycentric
    mjds = np.atleast_1d(np.asarray(mjds, float))
    ra0, dec0 = position_at(t, ref_mjd)
    out = []
    for m in mjds:
        ra, dec = position_at(t, m)
        de = (ra - ra0) * np.cos(np.radians(dec0)) * 3600
        dn = (dec - dec0) * 3600
        if with_parallax and t.parallax > 0:
            e = get_body_barycentric("earth", Time(m, format="mjd")).xyz.to_value(u.au)
            a, d = np.radians(ra), np.radians(dec)
            plx = t.parallax / 1000
            # standard parallax factors (barycentric Earth position X,Y,Z in au)
            de += plx * (e[0] * np.sin(a) - e[1] * np.cos(a))
            dn += plx * (e[0] * np.cos(a) * np.sin(d) + e[1] * np.sin(a) * np.sin(d) - e[2] * np.cos(d))
        out.append((de, dn))
    return np.array(out)


def mjd_to_date(mjd: float) -> str:
    return Time(mjd, format="mjd").to_datetime().strftime("%Y-%m-%d")

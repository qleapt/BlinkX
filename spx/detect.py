"""Optional helpers: find stars, and what changed between two dates.

The app never runs these unless the user switches them on (the user finds things; the tools measure
them). Everything works on the aligned, background-subtracted grid images (x = column, y = row,
row 0 = south). SPHEREx pixels are 6.15", so crowded stars blend and the faintest ones are missed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .photometry import centroid

OVERSAMPLE_NOISE = 2.0   # the display grid is 2x finer than the camera: neighbouring pixels share noise


def noise(img: np.ndarray) -> float:
    """Robust pixel noise (1.4826 x median absolute deviation)."""
    v = img[np.isfinite(img)]
    if v.size < 10:
        return float("nan")
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med))) or float(np.std(v))


def find_stars(img: np.ndarray, nsigma: float = 5.0, min_sep: int = 3, edge: int = 3) -> list[dict]:
    """Local brightness peaks more than nsigma above the noise, centred to a fraction of a pixel.
    Returns [{x, y, peak, snr}], brightest first."""
    z = np.where(np.isfinite(img), img, 0.0)
    sm = ndimage.gaussian_filter(z, 1.0)
    s = noise(sm)
    if not s > 0:
        return []
    peaks = (sm == ndimage.maximum_filter(sm, size=2 * min_sep + 1)) & (sm > nsigma * s)
    out = []
    ny, nx = img.shape
    for y, x in zip(*np.nonzero(peaks)):
        if x < edge or y < edge or x >= nx - edge or y >= ny - edge:
            continue
        cx, cy = centroid(z, float(x), float(y), r=3)
        if not (np.isfinite(cx) and np.hypot(cx - x, cy - y) < 2):
            cx, cy = float(x), float(y)
        out.append(dict(x=float(cx), y=float(cy), peak=float(z[y, x]), snr=float(sm[y, x] / s)))
    return sorted(out, key=lambda d: -d["snr"])


def aperture(img: np.ndarray, x: float, y: float, r: float = 2.5, r_in: float = 5, r_out: float = 8):
    """Light within r pixels minus the median of a ring around it. Returns (flux, n pixels)."""
    x0, x1 = int(max(0, np.floor(x - r_out))), int(min(img.shape[1], np.ceil(x + r_out) + 1))
    y0, y1 = int(max(0, np.floor(y - r_out))), int(min(img.shape[0], np.ceil(y + r_out) + 1))
    cut = img[y0:y1, x0:x1]
    yy, xx = np.indices(cut.shape)
    d = np.hypot(xx + x0 - x, yy + y0 - y)
    ok = np.isfinite(cut)
    inside, ring = (d <= r) & ok, (d >= r_in) & (d <= r_out) & ok
    if inside.sum() < 3 or ring.sum() < 5 or ((d <= r) & ~ok).any():
        return float("nan"), 0
    return float(np.sum(cut[inside] - np.median(cut[ring]))), int(inside.sum())


def _pa(dx: float, dy: float) -> float:
    """Direction in degrees east of north (east is -x on the grid)."""
    return float(np.degrees(np.arctan2(-dx, dy)) % 360)


COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


def changes(a: np.ndarray, b: np.ndarray, nsigma: float = 5.0, same_px: float = 1.5, move_px: float = 10.0,
            min_rel: float = 0.15) -> list[dict]:
    """What changed from photo a (older) to photo b (newer), strongest first.

    kind: 'moved' (x, y -> x2, y2), 'brightened' or 'faded'. score: how many noise levels the change is.
    A star seen at nearly the same spot on both dates is compared by brightness; one seen on one date
    only is paired with an unmatched star of similar brightness within move_px (moved), otherwise it
    counts as faded (only in a) or brightened (only in b)."""
    sa, sb = find_stars(a, nsigma), find_stars(b, nsigma)
    na, nb = noise(a), noise(b)
    out, used_a, used_b = [], set(), set()

    def err(n):
        return OVERSAMPLE_NOISE * np.sqrt(max(n, 1)) * np.hypot(na, nb)

    # Each photo averages a slightly different mix of wavelengths, so all stars can brighten or fade
    # together. `scale` is that shared change (median ratio of the stars seen on both dates) and
    # `spread` its star-to-star scatter; a star counts only if it changed more than the others.
    scale, spread = 1.0, 0.0

    def bright_change(x, y):
        fa, n = aperture(a, x, y)
        fb, _ = aperture(b, x, y)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return None
        fa_s = fa * scale
        e = np.hypot(err(n), spread * abs(fa_s))
        sig = (fb - fa_s) / e
        rel = abs(fb - fa_s) / max(abs(fa_s), abs(fb), 1e-12)
        if abs(sig) < nsigma or rel < min_rel:
            return None
        kind = "brightened" if fb > fa_s else "faded"
        pct = (fb / fa_s - 1) * 100 if fa_s > 0 else float("inf")
        extra = f", after allowing for all stars changing by {(scale - 1) * 100:+.0f}%" if abs(scale - 1) > 0.03 else ""
        return dict(kind=kind, x=x, y=y, x2=x, y2=y, score=float(abs(sig)), flux_a=fa, flux_b=fb,
                    text=(f"{'brighter' if fb > fa_s else 'fainter'} by {abs(pct):.0f}%" if np.isfinite(pct)
                          else "appeared") + f" ({abs(sig):.0f}σ{extra})")

    pairs = []
    if sa and sb:   # same star on both dates
        ta, tb = cKDTree([(s["x"], s["y"]) for s in sa]), cKDTree([(s["x"], s["y"]) for s in sb])
        for i, j in sorted((i, j) for i, js in enumerate(ta.query_ball_tree(tb, same_px)) for j in js):
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            pairs.append(((sa[i]["x"] + sb[j]["x"]) / 2, (sa[i]["y"] + sb[j]["y"]) / 2))
    logs = []
    for x, y in pairs:
        fa, n = aperture(a, x, y)
        fb, _ = aperture(b, x, y)
        if fa > 0 and fb > 0 and min(fa, fb) > 10 * err(n):   # well-measured stars only
            logs.append(np.log(fb / fa))
    if len(logs) >= 5:
        med = float(np.median(logs))
        scale, spread = float(np.exp(med)), float(1.4826 * np.median(np.abs(np.array(logs) - med)))
    for x, y in pairs:
        if c := bright_change(x, y):
            out.append(c)
    ua = [i for i in range(len(sa)) if i not in used_a]
    ub = [j for j in range(len(sb)) if j not in used_b]
    cand = []   # movers: an a-only star and a b-only star of similar brightness, close together
    for i in ua:
        fa, n = aperture(a, sa[i]["x"], sa[i]["y"])
        for j in ub:
            d = np.hypot(sb[j]["x"] - sa[i]["x"], sb[j]["y"] - sa[i]["y"])
            fb, _ = aperture(b, sb[j]["x"], sb[j]["y"])
            if d <= move_px and fa > 0 and fb > 0 and 0.5 < fb / (fa * scale) < 2:
                cand.append((d, i, j))
    for d, i, j in sorted(cand):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        dx, dy = sb[j]["x"] - sa[i]["x"], sb[j]["y"] - sa[i]["y"]
        pa = _pa(dx, dy)
        out.append(dict(kind="moved", x=sa[i]["x"], y=sa[i]["y"], x2=sb[j]["x"], y2=sb[j]["y"],
                        dist_px=float(d), pa_deg=pa, score=float(min(sa[i]["snr"], sb[j]["snr"])),
                        text=f"moved {d:.1f} px toward the {COMPASS[int(round(pa / 45)) % 8]}"))
    for i in ua:   # seen only on the older photo
        if i not in used_a and (c := bright_change(sa[i]["x"], sa[i]["y"])) and c["kind"] == "faded":
            out.append(c)
    for j in ub:   # seen only on the newer photo
        if j not in used_b and (c := bright_change(sb[j]["x"], sb[j]["y"])) and c["kind"] == "brightened":
            out.append(c)
    return sorted(out, key=lambda c: -c["score"])

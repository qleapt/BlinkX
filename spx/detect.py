"""Optional helpers (experimental): find stars, and what changed between two dates.

The app never runs these unless the user switches them on (the user finds things; the tools measure
them). They work on the aligned, background-subtracted grid images (x = column, y = row, row 0 =
south). They are tuned to be right rather than complete: a spot counts as a star only if

* it stands well above the noise after the large-scale background (nebulae, gradients) is removed,
* it has the shape of a star in this very photo (not a one-pixel spike, a wide blob or a streak),
* it is not a ring or wing leftover around a much brighter star, and
* when the single exposures behind a photo are given, it is seen in most of them (a cosmic-ray hit or a
  hot pixel is in one exposure only).

A change counts only if it is bigger than the noise, than the scatter between the single exposures,
than the brightening every star shares (each photo averages a different mix of wavelengths) and than
the star-to-star scatter of that, and if the single exposures agree. SPHEREx pixels are 6.15", so
crowded stars blend and the faintest ones are missed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .photometry import centroid

OVERSAMPLE_NOISE = 2.0   # the display grid is 2x finer than the camera: neighbouring pixels share noise
COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


def noise(img: np.ndarray) -> float:
    """Robust pixel noise (1.4826 x median absolute deviation)."""
    v = np.asarray(img)[np.isfinite(img)]
    if v.size < 10:
        return float("nan")
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med))) or float(np.std(v))


def background(img: np.ndarray, step: int = 4, box: int = 7) -> np.ndarray:
    """Large-scale background (nebulae, gradients): median over ~step*box pixels, smoothly interpolated."""
    small = ndimage.median_filter(img[::step, ::step], size=box, mode="nearest")
    big = ndimage.zoom(small, step, order=1)
    out = np.empty_like(img, dtype=float)
    out[...] = np.median(img)
    h, w = min(big.shape[0], img.shape[0]), min(big.shape[1], img.shape[1])
    out[:h, :w] = big[:h, :w]
    return out


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


def _shape(z: np.ndarray, x: int, y: int):
    """Concentration (light in 3x3 / light in 7x7), size (rms radius) and elongation of the spot at (x, y)."""
    cut = z[y - 3:y + 4, x - 3:x + 4]
    if cut.shape != (7, 7):
        return np.nan, np.nan, np.nan
    f7, f3 = cut.sum(), cut[2:5, 2:5].sum()
    w = np.clip(cut, 0, None) * np.exp(-(((np.indices((7, 7)) - 3) ** 2).sum(0)) / (2 * 2.5 ** 2))
    sw = w.sum()
    if f7 <= 0 or sw <= 0:
        return np.nan, np.nan, np.nan
    yy, xx = np.indices((7, 7)) - 3
    mx, my = (w * xx).sum() / sw, (w * yy).sum() / sw
    sxx, syy, sxy = (w * (xx - mx) ** 2).sum() / sw, (w * (yy - my) ** 2).sum() / sw, (w * (xx - mx) * (yy - my)).sum() / sw
    t = sxx + syy
    return float(f3 / f7), float(np.sqrt(t / 2)), float(np.hypot(sxx - syy, 2 * sxy) / t) if t > 0 else np.nan


def despike(z: np.ndarray, sigma: float, passes: int = 2) -> np.ndarray:
    """Replace sharp one-pixel outliers (hot pixels, cosmic-ray hits) by the median of their neighbours.
    A star's centre is never much brighter than its neighbours (the light is spread over several pixels)."""
    out = z.copy()
    for _ in range(passes):
        med = ndimage.median_filter(out, size=3, mode="nearest")
        bad = (out - med > 5 * sigma) & (out - med > 0.5 * np.abs(out))
        if not bad.any():
            break
        out[bad] = med[bad]
    return out


def _wing_radius(ratio: float, size: float) -> float:
    """How far from a star `ratio` times brighter its rings and wings can fake a fainter star (pixels)."""
    return 2.0 + 4.5 * max(size, 1.0) * np.log10(max(ratio, 1.0))


def _confirmed(singles, x: float, y: float, min_frac: float = 0.5) -> bool:
    """Seen in most single exposures, and not carried by one of them (cosmic ray, hot pixel)."""
    if not singles:
        return True
    fl, sig = [], []
    for s in singles:
        f, n = aperture(s, x, y)
        if np.isfinite(f):
            fl.append(f)
            sig.append(OVERSAMPLE_NOISE * np.sqrt(max(n, 1)) * noise(s))
    if len(fl) < 2:
        return True
    fl, sig = np.array(fl), np.array(sig)
    seen = int(np.sum(fl > 1.5 * sig))
    if seen < max(2, int(np.ceil(min_frac * len(fl)))):
        return False
    pos = fl[fl > 0]
    return not (len(fl) >= 3 and pos.size and pos.max() > 0.7 * pos.sum())


def find_stars(img: np.ndarray, nsigma: float = 5.0, min_sep: int = 3, edge: int = 4, singles=None) -> list[dict]:
    """Star-like spots more than nsigma above the noise, centred to a fraction of a pixel; brightest first.
    singles: the aligned single exposures behind img (optional), used to confirm each star.
    Returns [{x, y, peak, flux, snr}]."""
    img = np.asarray(img, float)
    ok = np.isfinite(img)
    if ok.sum() < 50:
        return []
    z = np.where(ok, img, np.median(img[ok]))
    z = z - background(z)
    z[~ok] = 0.0
    z = despike(z, noise(z[ok]))
    sm = ndimage.gaussian_filter(z, 1.2)
    s = noise(sm[ok])
    if not s > 0:
        return []
    near_bad = ndimage.binary_dilation(~ok, iterations=3)
    peaks = (sm == ndimage.maximum_filter(sm, size=2 * min_sep + 1)) & (sm > nsigma * s) & ~near_bad
    ny, nx = img.shape
    cands = []
    for y, x in zip(*np.nonzero(peaks)):
        if x < edge or y < edge or x >= nx - edge or y >= ny - edge:
            continue
        conc, size, ell = _shape(z, int(x), int(y))
        if not np.isfinite(conc):
            continue
        yy, xx = np.ogrid[-6:7, -6:7]
        ring = (np.hypot(yy, xx) >= 4.5) & (np.hypot(yy, xx) <= 5.5)
        cut = z[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
        drop = float(np.mean(cut[ring]) / z[y, x]) if cut.shape == (13, 13) and z[y, x] > 0 else 1.0
        cands.append(dict(x=int(x), y=int(y), snr=float(sm[y, x] / s), conc=conc, size=size, ell=ell, drop=drop,
                          flux=float(z[y - 3:y + 4, x - 3:x + 4].sum()), peak=float(z[y, x])))
    if not cands:
        return []
    # what a star looks like in this photo: from the clearly bright, round spots
    ref = [c for c in cands if c["snr"] > 25 and c["ell"] < 0.3]
    c0 = float(np.median([c["conc"] for c in ref])) if len(ref) >= 3 else 0.5
    s0 = float(np.median([c["size"] for c in ref])) if len(ref) >= 3 else 1.4
    keep = []
    for c in cands:
        if c["snr"] > 30 and c["ell"] < 0.3 and c["drop"] < 0.4:
            keep.append(c)   # bright, round and compact (light falls off fast): a star, even with a flat saturated top
            continue
        tol = 0.2 + 1.2 / c["snr"]
        if not (c0 * (1 - tol) <= c["conc"] <= min(0.9, c0 * (1 + tol))):
            continue   # one-pixel spike (too concentrated) or wide blob (too spread out)
        if not (s0 * (1 - tol) <= c["size"] <= s0 * (1 + tol)):
            continue
        if c["ell"] > 0.35 + 2.0 / c["snr"]:
            continue   # streak
        keep.append(c)
    keep.sort(key=lambda c: -c["flux"])
    out = []
    for i, c in enumerate(keep):   # rings and wings of much brighter stars
        if any(b["flux"] > 8 * c["flux"] and np.hypot(b["x"] - c["x"], b["y"] - c["y"]) < _wing_radius(b["flux"] / c["flux"], s0)
               for b in keep[:i]):
            continue
        cx, cy = centroid(z, float(c["x"]), float(c["y"]), r=3)
        if not (np.isfinite(cx) and np.hypot(cx - c["x"], cy - c["y"]) < 1.5):
            cx, cy = float(c["x"]), float(c["y"])
        if not _confirmed(singles, cx, cy):
            continue
        out.append(dict(x=float(cx), y=float(cy), peak=c["peak"], flux=c["flux"], snr=c["snr"]))
    return sorted(out, key=lambda d: -d["snr"])


def _pa(dx: float, dy: float) -> float:
    """Direction in degrees east of north (east is -x on the grid)."""
    return float(np.degrees(np.arctan2(-dx, dy)) % 360)


def _measure(img, singles, x, y):
    """Brightness at (x, y): (mean, error, per-exposure values). The error is the larger of the pixel-noise
    error and the scatter between the single exposures."""
    f, n = aperture(img, x, y)
    if not np.isfinite(f):
        return np.nan, np.nan, []
    e = OVERSAMPLE_NOISE * np.sqrt(max(n, 1)) * noise(img)
    per = [v for v in (aperture(s, x, y)[0] for s in (singles or [])) if np.isfinite(v)]
    if len(per) >= 2:
        return float(np.mean(per)), float(max(e, np.std(per, ddof=1) / np.sqrt(len(per)))), per
    return f, float(e), []


def changes(a: np.ndarray, b: np.ndarray, nsigma: float = 5.0, same_px: float = 1.5, move_px: float = 10.0,
            min_rel: float = 0.3, singles_a=None, singles_b=None) -> list[dict]:
    """What changed from photo a (older) to photo b (newer), strongest first.

    kind: 'moved' (x, y -> x2, y2), 'brightened' or 'faded'; score: how many uncertainties the change is.
    singles_a / singles_b: the aligned single exposures behind a and b (optional, makes it much surer)."""
    sa = find_stars(a, nsigma, singles=singles_a)
    sb = find_stars(b, nsigma, singles=singles_b)
    allstars = [(s["x"], s["y"], s["flux"]) for s in sa + sb]
    s0 = 1.4

    def near_brighter(x, y, flux):
        return any(f > 8 * flux and np.hypot(x - bx, y - by) < _wing_radius(f / max(flux, 1e-12), s0)
                   for bx, by, f in allstars)

    used_a, used_b, pairs = set(), set(), []
    if sa and sb:   # the same star on both dates
        ta, tb = cKDTree([(s["x"], s["y"]) for s in sa]), cKDTree([(s["x"], s["y"]) for s in sb])
        for i, j in sorted((i, j) for i, js in enumerate(ta.query_ball_tree(tb, same_px)) for j in js):
            if i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                pairs.append(((sa[i]["x"] + sb[j]["x"]) / 2, (sa[i]["y"] + sb[j]["y"]) / 2))
    # the change every star shares, and its star-to-star scatter (in log brightness)
    logs = []
    for x, y in pairs:
        fa, ea, _ = _measure(a, singles_a, x, y)
        fb, eb, _ = _measure(b, singles_b, x, y)
        if fa > 10 * ea and fb > 10 * eb:
            logs.append(np.log(fb / fa))
    scale, spread = 1.0, 0.10
    if len(logs) >= 5:
        med = float(np.median(logs))
        scale, spread = float(np.exp(med)), max(0.03, float(1.4826 * np.median(np.abs(np.array(logs) - med))))

    def brightness_change(x, y):
        fa, ea, pa_ = _measure(a, singles_a, x, y)
        fb, eb, pb_ = _measure(b, singles_b, x, y)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return None
        ref = scale * fa
        if fb > 0 and ref > 0:   # scatter between stars multiplies brightness: compare the ratio (log scale)
            sig = np.log(fb / ref) / np.sqrt(spread ** 2 + (eb / fb) ** 2 + (ea / fa) ** 2)
        else:
            sig = (fb - ref) / np.sqrt(eb ** 2 + (scale * ea) ** 2 + (spread * max(abs(fb), abs(ref))) ** 2)
        rel = abs(fb - ref) / max(abs(ref), abs(fb), 1e-12)
        if abs(sig) < nsigma or rel < min_rel or near_brighter(x, y, max(fa, fb)):
            return None
        mid = (fb + ref) / 2
        if pa_ and pb_:   # the single exposures must agree, not one odd exposure
            up = fb > ref
            agree_b = np.mean([(v > mid) == up for v in pb_])
            agree_a = np.mean([(scale * v < mid) == up for v in pa_])
            if min(agree_a, agree_b) < 0.75:
                return None
        pct = (fb / ref - 1) * 100 if ref > 0 else float("inf")
        extra = f", after allowing for all stars changing by {(scale - 1) * 100:+.0f}%" if abs(scale - 1) > 0.03 else ""
        return dict(kind="brightened" if fb > ref else "faded", x=x, y=y, x2=x, y2=y, score=float(abs(sig)),
                    flux_a=fa, flux_b=fb,
                    text=(f"{'brighter' if fb > ref else 'fainter'} by {abs(pct):.0f}%" if np.isfinite(pct)
                          else "appeared") + f" ({abs(sig):.0f}σ{extra})")

    out = [c for c in (brightness_change(x, y) for x, y in pairs) if c]
    ua = [i for i in range(len(sa)) if i not in used_a]
    ub = [j for j in range(len(sb)) if j not in used_b]
    cand = []   # movers: an older-only star and a newer-only star of similar brightness, close together
    for i in ua:
        fa, _, _ = _measure(a, singles_a, sa[i]["x"], sa[i]["y"])
        left, _, _ = _measure(b, singles_b, sa[i]["x"], sa[i]["y"])      # what stayed at the old spot
        for j in ub:
            d = np.hypot(sb[j]["x"] - sa[i]["x"], sb[j]["y"] - sa[i]["y"])
            if not same_px <= d <= move_px:
                continue
            fb, _, _ = _measure(b, singles_b, sb[j]["x"], sb[j]["y"])
            before, _, _ = _measure(a, singles_a, sb[j]["x"], sb[j]["y"])  # what was at the new spot before
            if (fa > 0 and fb > 0 and 0.5 < fb / (scale * fa) < 2 and left < 0.6 * fb and before < 0.6 * fa
                    and not near_brighter(sa[i]["x"], sa[i]["y"], fa)):
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
        if i not in used_a and (c := brightness_change(sa[i]["x"], sa[i]["y"])) and c["kind"] == "faded":
            out.append(c)
    for j in ub:   # seen only on the newer photo
        if j not in used_b and (c := brightness_change(sb[j]["x"], sb[j]["y"])) and c["kind"] == "brightened":
            out.append(c)
    return sorted(out, key=lambda c: -c["score"])

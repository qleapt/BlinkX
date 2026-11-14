"""Accuracy bench for the optional helpers (spx/detect.py) on synthetic SPHEREx-like epochs.

Each epoch = 4 single exposures (noise, stars) averaged into a stack, plus the things that fooled the
first version: single hot pixels and cosmic-ray hits (one exposure only), an extended nebula blob,
ring leftovers around a very bright star, an all-stars brightness shift and star-to-star colour
scatter between epochs. The helpers must be right at least 99% of the time (precision) and still find
the clear, real things.
"""
import numpy as np

from spx import detect

N, PSF, NOISE1 = 101, 1.5, 0.08   # grid size, star width (grid px), single-exposure noise


def _gauss(img, x, y, amp, s=PSF):
    yy, xx = np.indices(img.shape)
    img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))


def _field(seed, n_stars=30):
    rng = np.random.default_rng(seed)
    xy = []
    while len(xy) < n_stars:   # stars at least 6 px apart and 8 px from the edge
        x, y = rng.uniform(8, N - 9, 2)
        if all(np.hypot(x - a, y - b) > 6 for a, b, _ in xy) and np.hypot(x - 50, y - 50) > 18:
            xy.append((x, y, float(np.exp(rng.uniform(np.log(0.25), np.log(6.0))))))
    return xy


def _epoch(stars, seed, scale=1.0, bright=True, artifacts=True):
    """(stack, singles) for one epoch."""
    rng = np.random.default_rng(seed)
    singles = []
    for k in range(4):
        im = rng.normal(0, NOISE1, (N, N))
        for x, y, a in stars:
            _gauss(im, x, y, a * scale)
        if bright:   # a saturated-looking star with a ring of leftovers around it
            _gauss(im, 50, 50, 80.0, 1.6)
            yy, xx = np.indices(im.shape)
            r = np.hypot(xx - 50, yy - 50)
            im += 0.9 * np.exp(-((r - 7) ** 2) / 2) * (1 + 0.6 * np.cos(5 * np.arctan2(yy - 50, xx - 50)))
        if artifacts:
            hx, hy = rng.integers(5, N - 5, 2)
            im[hy, hx] += 4.0                            # hot pixel / cosmic ray: one pixel, one exposure
            cx, cy = rng.integers(10, N - 14, 2)
            im[cy, cx:cx + 4] += 2.5                      # short cosmic-ray streak, one exposure
            _gauss(im, 20 + k * 0.1, 80, 0.35, 7.0)       # extended nebula blob (in every exposure)
        singles.append(im)
    return np.mean(singles, axis=0), singles


def _real(found, stars, tol=1.6):
    return [f for f in found if min(np.hypot(f["x"] - x, f["y"] - y) for x, y, _ in stars + [(50, 50, 80)]) < tol]


def test_star_detection_is_at_least_99_percent_right():
    n_found = n_false = n_bright = n_bright_found = 0
    for seed in range(12):
        stars = _field(seed)
        img, singles = _epoch(stars, 100 + seed)
        found = detect.find_stars(img, nsigma=5, singles=singles)
        good = _real(found, stars)
        n_found += len(found)
        n_false += len(found) - len(good)
        for x, y, a in stars:
            if a > 1.0:   # clearly visible stars must be found
                n_bright += 1
                n_bright_found += any(np.hypot(f["x"] - x, f["y"] - y) < 1.6 for f in found)
    assert n_found > 200
    assert n_false / n_found <= 0.01, f"{n_false} false of {n_found}"
    assert n_bright_found / n_bright >= 0.95, f"found {n_bright_found} of {n_bright} clear stars"


def test_detection_without_single_exposures_still_rejects_spikes_blobs_and_rings():
    n_found = n_false = 0
    for seed in range(12):
        stars = _field(seed)
        img, _ = _epoch(stars, 200 + seed)
        found = detect.find_stars(img, nsigma=5)
        n_found += len(found)
        n_false += len(found) - len(_real(found, stars))
    assert n_false / n_found <= 0.01, f"{n_false} false of {n_found}"


def test_change_finder_is_at_least_99_percent_right():
    """Everything brightens 25% (other wavelengths) and each star's colour scatters by ±10%; one star moved,
    one flared x2.5, one vanished. Only those three may be reported."""
    n_rep = n_false = n_true_found = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        stars = _field(seed)
        mover, flare, gone = 0, 1, 2
        stars[mover] = (stars[mover][0], stars[mover][1], 3.0)
        stars[flare] = (stars[flare][0], stars[flare][1], 1.5)
        stars[gone] = (stars[gone][0], stars[gone][1], 2.0)
        b_stars = [(x, y, a * float(rng.normal(1, 0.10))) for x, y, a in stars]
        mx, my, ma = stars[mover]
        b_stars[mover] = (mx + 3.0, my + 1.0, ma)
        b_stars[flare] = (stars[flare][0], stars[flare][1], stars[flare][2] * 2.5)
        del b_stars[gone]
        a, sa = _epoch(stars, 300 + seed)
        b, sb = _epoch(b_stars, 400 + seed, scale=1.25)
        ch = detect.changes(a, b, nsigma=5, singles_a=sa, singles_b=sb)
        want = {"moved": (mx, my), "brightened": stars[flare][:2], "faded": stars[gone][:2]}
        for c in ch:
            n_rep += 1
            w = want.get(c["kind"])
            if w is None or np.hypot(c["x"] - w[0], c["y"] - w[1]) > 1.6:
                n_false += 1
        for kind, (x, y) in want.items():
            n_true_found += any(c["kind"] == kind and np.hypot(c["x"] - x, c["y"] - y) < 1.6 for c in ch)
    assert n_rep > 0
    assert n_false / n_rep <= 0.01, f"{n_false} false of {n_rep} reported"
    assert n_true_found >= 0.9 * 3 * 8, f"found {n_true_found} of {3 * 8} real changes"


def test_saturated_bright_star_is_still_found():
    """A very bright star with a flat, clipped centre (saturated) is a star, not a blob."""
    stars = _field(3)
    img, singles = _epoch(stars, 77, bright=False)
    for im in [img] + singles:
        sat = np.zeros_like(im)
        _gauss(sat, 50, 50, 300.0, 1.7)
        im += np.minimum(sat, 25.0)          # clipped at 25: a flat top 4-5 px wide
    found = detect.find_stars(img, nsigma=5, singles=singles)
    assert any(np.hypot(f["x"] - 50, f["y"] - 50) < 1.0 for f in found)

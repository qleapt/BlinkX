"""Background removal, reprojection to one north-up grid (2x oversampled), stacks, add/subtract."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy.wcs import WCS
from reproject import reproject_interp

from . import config
from .cutouts import Stamp
from .targets import mjd_to_date


def grid(ra: float, dec: float, fov_arcmin: float, oversample: int = config.OVERSAMPLE):
    """North-up, east-left TAN grid centred on (ra, dec)."""
    scale = config.PIXEL_SCALE_ARCSEC / oversample / 3600
    n = int(np.ceil(fov_arcmin / 60 / scale))
    n += (n + 1) % 2  # odd: target on the central pixel
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cdelt = [-scale, scale]
    return w, (n, n)


@dataclass
class Aligned:
    image: np.ndarray      # background-subtracted MJy/sr on the grid, NaN = no data
    weight: np.ndarray     # 1/variance, 0 where bad
    stamp: Stamp
    bad_near_target: list = field(default_factory=list)
    flagged: np.ndarray | None = None  # bad-flagged pixels on the grid


def clean(st: Stamp, bad_names=None):
    """Zodi and local background removed; dead pixels NaN.
    Returns (image, variance, suspect mask)."""
    dead = st.bad_mask(config.DEAD_FLAGS) | ~np.isfinite(st.image) | ~(st.variance > 0)
    suspect = st.bad_mask(bad_names) & ~dead
    img = (st.image - st.zodi).astype(np.float64)
    img[dead] = np.nan
    good = img[np.isfinite(img) & ~suspect]
    if good.size > 10:
        med = np.median(good)
        for _ in range(3):  # sigma-clipped median of the local sky
            mad = np.median(np.abs(good - med)) * 1.4826 or 1.0
            keep = np.abs(good - med) < 3 * mad
            if keep.sum() < 10:
                break
            med = np.median(good[keep])
        img -= med
    var = st.variance.astype(np.float64)
    var[dead] = np.nan
    return img, var, suspect


def flags_near(st: Stamp, ra: float, dec: float, radius_px: float = 2.5) -> list[str]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    x, y = st.wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    yy, xx = np.indices(st.flags.shape)
    near = (xx - x) ** 2 + (yy - y) ** 2 <= radius_px ** 2
    bits = int(np.bitwise_or.reduce(st.flags[near].astype(np.int64))) if near.any() else 0
    return [n for n, b in st.flag_bits.items() if bits & int(b) and n in config.WARN_FLAGS]


def align(st: Stamp, out_wcs: WCS, shape, bad_names=None) -> Aligned:
    img, var, suspect = clean(st, bad_names)
    w_in = st.wcs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, fp = reproject_interp((img, w_in), out_wcs, shape_out=shape, order="bilinear")
        v, _ = reproject_interp((var, w_in), out_wcs, shape_out=shape, order="bilinear")
        s, _ = reproject_interp((suspect.astype(float), w_in), out_wcs, shape_out=shape, order="bilinear")
    a[fp == 0] = np.nan
    wt = np.where(np.isfinite(a) & np.isfinite(v) & (v > 0), 1.0 / np.where(v > 0, v, 1), 0.0)
    flagged = np.nan_to_num(s) > 0
    wt = np.where(flagged, wt * config.SUSPECT_WEIGHT, wt)
    return Aligned(a, wt, st, flagged=flagged)


def fill_holes(img: np.ndarray, passes: int = 3) -> np.ndarray:
    """Fill small NaN holes (saturated cores) with the max of their neighbours, for display."""
    from scipy.ndimage import maximum_filter, convolve
    out = img.copy()
    for _ in range(passes):
        nan = np.isnan(out)
        if not nan.any():
            break
        fin = np.isfinite(out)
        nb = convolve(fin.astype(int), np.ones((3, 3), int), mode="constant")
        mx = maximum_filter(np.where(fin, out, -np.inf), size=3)
        fill = nan & (nb >= 3) & np.isfinite(mx)
        out[fill] = mx[fill]
    return out


def align_array(img: np.ndarray, w_in: WCS, out_wcs: WCS, shape) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, fp = reproject_interp((img, w_in), out_wcs, shape_out=shape, order="bilinear")
    a[fp == 0] = np.nan
    return a


@dataclass
class Frame:
    image: np.ndarray
    label: str
    band: str
    mjd_start: float
    mjd_end: float
    n: int
    wl_min: float = float("nan")
    wl_max: float = float("nan")
    kind: str = "stack"        # stack | single | survey | diff | sum
    flags: list = field(default_factory=list)
    stamps: list = field(default_factory=list)
    flag_mask: np.ndarray | None = None  # pixels where the shown value comes from flagged data

    @property
    def mjd_mid(self) -> float:
        return 0.5 * (self.mjd_start + self.mjd_end)


def weighted_mean(images, weights) -> np.ndarray:
    num = np.zeros_like(images[0], dtype=np.float64)
    den = np.zeros_like(num)
    for im, w in zip(images, weights):
        ok = np.isfinite(im) & (w > 0)
        num[ok] += im[ok] * w[ok]
        den[ok] += w[ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def date_range(a: float, b: float) -> str:
    da, db = mjd_to_date(a), mjd_to_date(b)
    return da if da == db else f"{da} – {db}"


def stack_frame(items: list[Aligned], label_prefix: str = "") -> Frame:
    img = fill_holes(weighted_mean([a.image for a in items], [a.weight for a in items]))
    mjds = [a.stamp.mjd for a in items]
    wls = [a.stamp.wavelength for a in items if np.isfinite(a.stamp.wavelength)]
    band = items[0].stamp.band
    flags = sorted({f for a in items for f in a.bad_near_target})
    wl = f" · {min(wls):.2f}–{max(wls):.2f} µm" if wls else ""
    label = f"{label_prefix}{date_range(min(mjds), max(mjds))} · {band} · {len(items)} exp{wl}"
    fm = None
    if all(a.flagged is not None for a in items):
        # flagged in every exposure that has data there -> the stack shows flagged data
        fm = np.all([a.flagged | ~np.isfinite(a.image) for a in items], axis=0) & \
            np.any([np.isfinite(a.image) for a in items], axis=0)
    return Frame(img, label, band, min(mjds), max(mjds), len(items),
                 min(wls) if wls else np.nan, max(wls) if wls else np.nan, "stack", flags,
                 [a.stamp for a in items], fm)


def _or_masks(frames):
    ms = [f.flag_mask for f in frames if f.flag_mask is not None]
    return np.any(ms, axis=0) if ms else None


def single_frame(a: Aligned, i: int) -> Frame:
    st = a.stamp
    wl = f" · {st.wavelength:.3f} µm" if np.isfinite(st.wavelength) else ""
    return Frame(fill_holes(a.image), f"#{i} {mjd_to_date(st.mjd)} · {st.band}{wl}", st.band, st.mjd, st.mjd_end, 1,
                 st.wavelength, st.wavelength, "single", a.bad_near_target, [st], a.flagged)


def subtract(b: Frame, a: Frame) -> Frame:
    return Frame(b.image - a.image, f"{b.label.split(' · ')[0]}  minus  {a.label.split(' · ')[0]} · {a.band}",
                 a.band, a.mjd_start, b.mjd_end, a.n + b.n, kind="diff", flag_mask=_or_masks([a, b]),
                 flags=sorted(set(a.flags) | set(b.flags)))


def add(frames: list[Frame]) -> Frame:
    w = [np.isfinite(f.image).astype(float) * f.n for f in frames]
    img = weighted_mean([f.image for f in frames], w)
    return Frame(img, f"All epochs added · {frames[0].band} · {sum(f.n for f in frames)} exp",
                 frames[0].band, min(f.mjd_start for f in frames), max(f.mjd_end for f in frames),
                 sum(f.n for f in frames), kind="sum", flag_mask=_or_masks(frames),
                 flags=sorted({x for f in frames for x in f.flags}))

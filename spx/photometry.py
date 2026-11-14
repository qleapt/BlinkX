"""Quick-look aperture photometry (spectra, light curves) and centroids. Not calibrated."""
from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_area

from . import config
from .cutouts import Stamp

R_AP, R_IN, R_OUT = 2.0, 4.0, 7.0  # pixels
SUB = 5                             # sub-pixel sampling of the aperture edge


def aperture_weights(shape, x0: float, y0: float, r: float) -> np.ndarray:
    """Fraction of each pixel inside a circle (sub-sampled)."""
    yy, xx = np.indices(shape, dtype=float)
    off = (np.arange(SUB) + 0.5) / SUB - 0.5
    w = np.zeros(shape)
    for dy in off:
        for dx in off:
            w += ((xx + dx - x0) ** 2 + (yy + dy - y0) ** 2) <= r * r
    return w / SUB ** 2


def measure(st: Stamp, ra: float, dec: float) -> dict:
    """Flux in mJy of the source at (ra, dec), with error from VARIANCE."""
    w = st.wcs
    x, y = (float(v) for v in w.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg)))
    img = st.image.astype(float) - st.zodi
    var = st.variance.astype(float)
    dead = st.bad_mask(config.DEAD_FLAGS) | ~np.isfinite(img) | ~(var > 0)
    bad = st.bad_mask()
    ap = aperture_weights(img.shape, x, y, R_AP)
    yy, xx = np.indices(img.shape)
    r2 = (xx - x) ** 2 + (yy - y) ** 2
    ann = (r2 >= R_IN ** 2) & (r2 <= R_OUT ** 2) & ~dead & ~bad
    omega = proj_plane_pixel_area(w) * (np.pi / 180) ** 2  # sr per pixel
    out = dict(mjd=st.mjd, band=st.band, wavelength=st.wavelength, bandwidth=st.bandwidth, uri=st.uri,
               flux=np.nan, err=np.nan, flags=[], x=x, y=y)
    inside = ap > 0
    if ann.sum() < 8:  # bright star: the sky ring is flagged too; use its finite pixels anyway
        ann = (r2 >= R_IN ** 2) & (r2 <= R_OUT ** 2) & ~dead
    if inside.any() and ap[inside & ~dead].sum() < 0.6 * ap.sum():
        out["flags"] = ["SATURATED CORE (no usable pixels)"]
        # brightness index from the PSF wings (ring 3-5 px), sky from 7-10 px: relative units only
        wing = (r2 >= 9) & (r2 <= 25) & ~dead
        far = (r2 >= 49) & (r2 <= 100) & ~dead
        if wing.sum() >= 10 and far.sum() >= 10:
            out["wing"] = float(np.mean(img[wing]) - np.median(img[far]))
        return out
    if not inside.any() or ann.sum() < 8:
        return out
    sky = np.median(img[ann])
    sky_var = (1.2533 * np.std(img[ann])) ** 2 / ann.sum()
    use = inside & ~dead
    a = ap[use]
    flux = np.sum((img[use] - sky) * a) * ap.sum() / a.sum()  # scale up for lost dead pixels
    err2 = np.sum(var[use] * a ** 2) + a.sum() ** 2 * sky_var
    bits = int(np.bitwise_or.reduce(st.flags[inside].astype(np.int64)))
    out.update(flux=flux * omega * 1e9, err=np.sqrt(err2) * omega * 1e9,  # MJy -> mJy
               flags=sorted(n for n, b in st.flag_bits.items() if bits & int(b) and n in config.BAD_FLAGS))
    return out


def fit_motion(years, de, dn, err):
    """Weighted straight-line fit of offsets (arcsec) against time (years).
    Returns (pm_east, pm_north, err_east, err_north) in arcsec/yr, or None."""
    years, de, dn, err = (np.asarray(v, float) for v in (years, de, dn, err))
    ok = np.isfinite(de) & np.isfinite(dn) & np.isfinite(err) & (err > 0)
    if ok.sum() < 2 or np.ptp(years[ok]) < 0.1:
        return None
    w = 1 / err[ok] ** 2
    A = np.vstack([np.ones(ok.sum()), years[ok] - years[ok].mean()]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    pe = cov @ A.T @ (w * de[ok])
    pn = cov @ A.T @ (w * dn[ok])
    s = np.sqrt(cov[1, 1])
    if ok.sum() > 2:  # inflate by the reduced chi-square when the scatter is larger than the errors
        chi = (np.sum(w * (de[ok] - A @ pe) ** 2) + np.sum(w * (dn[ok] - A @ pn) ** 2)) / (2 * (ok.sum() - 2))
        s *= np.sqrt(max(1.0, chi))
    return float(pe[1]), float(pn[1]), float(s), float(s)


def centroid(img: np.ndarray, x0: float, y0: float, r: float = 4.0, iters: int = 3):
    """Background-subtracted, thresholded first-moment centroid near (x0, y0) (pixels)."""
    yy, xx = np.indices(img.shape)
    x, y = x0, y0
    for _ in range(iters):
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        ring = np.isfinite(img) & (d2 > (2 * r) ** 2) & (d2 < (3 * r) ** 2)
        sky = np.nanmedian(img[ring]) if ring.sum() > 5 else 0.0
        m = np.isfinite(img) & (d2 <= r * r)
        if m.sum() < 4:
            return np.nan, np.nan
        v = np.where(m, img - sky, 0.0)
        v = np.where(v > 0.2 * v.max(), v, 0.0)
        if v.sum() <= 0:
            return np.nan, np.nan
        x, y = (v * xx).sum() / v.sum(), (v * yy).sum() / v.sum()
    return float(x), float(y)

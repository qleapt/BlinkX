import io

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from spx import archive, cutouts, stack, targets
from spx.cutouts import Stamp


# --- targets -------------------------------------------------------------------
def test_barnard_proper_motion_j2000_to_2026():
    b = targets.PRESETS["Barnard's Star"]
    t2000 = targets.Target("b", 269.452075, 4.693391, targets.J2000_MJD, b.pmra, b.pmdec)
    ra, dec = targets.position_at(t2000, 61041.0)  # 2026.0
    assert abs((dec - t2000.dec) * 3600 - 269) < 3


def test_parse_coords_and_presets():
    assert targets.parse_coords("269.45, 4.74").ra == pytest.approx(269.45)
    c = targets.parse_coords("17 57 48.5 +04 41 36")
    assert c.ra == pytest.approx(269.452, abs=1e-3)
    assert targets.preset("barnard's star").name == "Barnard's Star"
    assert targets.preset("WISE 0855") is not None
    assert targets.parse_coords("Barnard's Star") is None


# --- archive -------------------------------------------------------------------
def _wise0855_mjds():
    return (list(np.linspace(60800, 60809, 8)) + list(np.linspace(60985, 60999, 20))
            + [61015, 61018] + list(np.linspace(61150, 61174, 30)))


def test_epoch_grouping_wise0855_pattern():
    ep = archive.group_epochs(_wise0855_mjds())
    assert len(set(ep)) == 3


def test_dense_field_is_split_into_bins():
    ep = archive.group_epochs(np.arange(60800, 61300, 2.0))
    assert len(set(ep)) >= 8


def test_thinning_respects_caps():
    exps = [archive.Exposure("D6", m, m + .01, f"x/{i}.fits", "l2b-v27-2026-197")
            for i, m in enumerate(np.arange(60800, 61300, 0.5))]
    q = archive.select(exps, "D6", quick=True)
    assert sum(len(v) for v in q.values()) <= 24
    full = archive.select(exps, "D6", quick=False, cap=50)
    assert sum(len(v) for v in full.values()) <= 50


# --- cutouts -------------------------------------------------------------------
def _mef() -> bytes:
    h = fits.Header()
    h["MSKN0001"], h["MSKM0001"] = "OVERFLOW", 2
    hl = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((49, 49), np.float32), name="IMAGE"),
        fits.ImageHDU(np.zeros((49, 49), np.int32), header=h, name="FLAGS"),
        fits.ImageHDU(np.ones((49, 49), np.float32), name="VARIANCE"),
        fits.ImageHDU(np.zeros((49, 49), np.float32), name="ZODI"),
        fits.BinTableHDU.from_columns([fits.Column("PSF", "4000E", array=np.zeros((200, 4000)))], name="EPSF"),
        fits.BinTableHDU.from_columns([fits.Column("X", "E", array=np.arange(3.0))], name="WCS-WAVE"),
    ])
    b = io.BytesIO()
    hl.writeto(b)
    return b.getvalue()


def test_stream_parser_stops_before_epsf():
    raw = _mef()
    p = cutouts.HeadParser()
    used = 0
    for i in range(0, len(raw), 1000):
        used = i + 1000
        if p.feed(raw[i:i + 1000]):
            break
    assert p.done and used < len(raw) / 4
    hd = fits.open(io.BytesIO(p.prefix))
    assert [h.name for h in hd] == ["PRIMARY", "IMAGE", "FLAGS", "VARIANCE", "ZODI"]
    assert cutouts.flag_bits_from_header(hd["FLAGS"].header) == {"OVERFLOW": 2}


class _Resp:
    def __init__(self, code, data=b""):
        self.status_code, self.data = code, data

    def iter_content(self, n):
        for i in range(0, len(self.data), n):
            yield self.data[i:i + n]

    def raise_for_status(self):
        pass

    def close(self):
        pass


def test_503_is_retried():
    raw = _mef()
    calls = []

    def get(url, stream, timeout):
        calls.append(1)
        return _Resp(503) if len(calls) < 3 else _Resp(200, raw)

    out = cutouts.fetch_prefix("http://x", get=get, sleep=lambda s: None)
    assert len(calls) == 3 and out.startswith(b"SIMPLE")


# --- stack ---------------------------------------------------------------------
def _stamp(rot_deg, ra0=150.0, dec0=10.0, src=(150.0, 10.0), n=49, bad=None):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [25, 25]
    s = 6.15 / 3600
    t = np.radians(rot_deg)
    w.wcs.cd = [[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]]
    x0, y0 = w.world_to_pixel_values(*src)
    yy, xx = np.indices((n, n))
    img = 100 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.3 ** 2)) + 1.0
    flags = np.zeros((n, n), np.int32)
    if bad is not None:
        img[bad] = 1e6
        flags[bad] = 2
    h = w.to_header()
    return Stamp("D6", 61000, 61000.01, "u", "v", img.astype(np.float32), flags,
                 np.ones((n, n), np.float32), np.zeros((n, n), np.float32), h.tostring(), {"OVERFLOW": 2})


def _centroid(img):
    im = np.nan_to_num(img)
    im = np.where(im > im.max() * 0.2, im, 0)
    yy, xx = np.indices(im.shape)
    return (im * xx).sum() / im.sum(), (im * yy).sum() / im.sum()


def test_rotated_exposures_align():
    w, shape = stack.grid(150.0, 10.0, 3)
    a = stack.align(_stamp(0), w, shape)
    b = stack.align(_stamp(37, ra0=150.003, dec0=9.998), w, shape)
    ca, cb = _centroid(a.image), _centroid(b.image)
    assert np.hypot(ca[0] - cb[0], ca[1] - cb[1]) < 0.2 * stack.config.OVERSAMPLE
    c = (shape[1] - 1) / 2
    assert np.hypot(ca[0] - c, ca[1] - c) < 0.4


def test_aperture_photometry_recovers_flux():
    from spx import photometry
    st = _stamp(0)  # Gaussian sigma 1.3 px, peak 100 MJy/sr, sky 1
    m = photometry.measure(st, 150.0, 10.0)
    omega = (6.15 / 206264.806) ** 2
    total = 100 * 2 * np.pi * 1.3 ** 2 * omega * 1e9
    enclosed = 1 - np.exp(-2.0 ** 2 / (2 * 1.3 ** 2))  # r = 2 px aperture
    assert m["flux"] == pytest.approx(total * enclosed, rel=0.05)
    assert m["err"] > 0 and m["flags"] == []


def test_centroid_subpixel():
    from spx import photometry
    yy, xx = np.indices((40, 40))
    img = 50 * np.exp(-((xx - 20.3) ** 2 + (yy - 18.7) ** 2) / (2 * 2.0 ** 2)) + np.random.default_rng(1).normal(0, .1, (40, 40))
    x, y = photometry.centroid(img, 19, 19)
    assert abs(x - 20.3) < 0.1 and abs(y - 18.7) < 0.1


def test_gaia_propagation_and_subtract_pair():
    from spx import catalogs
    s = [dict(id=1, ra=269.448502525, dec=4.739420051, pmra=-801.551, pmdec=10362.394, parallax=547, g=8.2)]
    r, d = catalogs.gaia_at(s, 57388.5 + 365.25)
    assert (d[0] - 4.739420051) * 3600 == pytest.approx(10.362, abs=0.01)
    a = stack.Frame(np.zeros((5, 5)), "Epoch 1: x", "D6", 0, 1, 1)
    b = stack.Frame(np.zeros((5, 5)), "Epoch 2: y", "D6", 2, 3, 1)
    a.image[1, 2], b.image[3, 2] = 10, 10
    dif = stack.subtract(b, a).image
    assert dif[3, 2] == 10 and dif[1, 2] == -10


def test_motion_fit_recovers_proper_motion():
    from spx import photometry
    yrs = np.array([-0.4, 0.0, 0.5, -16.0])
    de, dn = -0.8 * yrs + 0.05, 10.36 * yrs - 0.1
    pe, pn, ee, en = photometry.fit_motion(yrs, de, dn, [0.5, 0.5, 0.5, 0.6])
    assert pe == pytest.approx(-0.8, abs=0.01) and pn == pytest.approx(10.36, abs=0.01) and ee > 0
    assert photometry.fit_motion([0.0], [0], [0], [1]) is None


def test_masked_pixels_are_ignored():
    w, shape = stack.grid(150.0, 10.0, 3)
    good = stack.align(_stamp(0), w, shape)
    bad = stack.align(_stamp(0, bad=(slice(5, 10), slice(5, 10))), w, shape)
    s = stack.weighted_mean([good.image, bad.image], [good.weight, bad.weight])
    assert np.nanmax(s) < 200

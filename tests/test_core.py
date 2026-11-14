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


def test_parse_coords_many_spellings():
    for text in ["269.452 4.693", "269.452, 4.693", "269.452° 4.693°", "RA=269.452 Dec=4.693",
                 "ra 269.452 dec 4.693", "269.452d +4.693d"]:
        t = targets.parse_coords(text)
        assert t is not None and t.ra == pytest.approx(269.452) and t.dec == pytest.approx(4.693), text
    assert targets.parse_coords("17h57m48.5s +4d41m36s").ra == pytest.approx(269.452, abs=1e-3)
    assert targets.parse_coords("RA=10 Dec=-4.5").dec == pytest.approx(-4.5)
    for name in ["M 42", "HD 209458", "Andromeda", "2MASS J0523-1403"]:
        assert targets.parse_coords(name) is None and not targets.looks_like_coords(name), name
    assert targets.looks_like_coords("400 10")  # out of range: a bad position, not a SIMBAD name


def test_race_remembers_the_faster_source():
    """When the IRSA copy wins a race, the next download asks both at once (no hedge wait)."""
    import time as _t
    from spx import cutouts

    route = cutouts.Route()

    def slow_s3():
        _t.sleep(0.6)
        return "s3"

    def fast_irsa():
        return "irsa"
    assert cutouts.race(slow_s3, fast_irsa, hedge_after=0.2, route=route) == "irsa"
    assert route.prefer_irsa
    t0 = _t.time()
    assert cutouts.race(slow_s3, fast_irsa, hedge_after=0.5, route=route) == "irsa"
    assert _t.time() - t0 < 0.3  # did not wait for the hedge
    # a fast AWS answer resets the preference
    route2 = cutouts.Route()
    assert cutouts.race(lambda: "s3", fast_irsa, hedge_after=0.5, route=route2) == "s3"
    assert not route2.prefer_irsa


# --- viewer --------------------------------------------------------------------
def _decode_png16(b64, lo, s, a):
    """Python twin of the viewer's JavaScript decoder."""
    import base64
    from PIL import Image
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")).astype(int)[::-1]
    q = rgb[..., 0] * 256 + rgb[..., 1]
    v = lo + s * np.sinh((q - 1) / 65534 * a)
    return np.where(q == 0, np.nan, v), rgb[..., 2] > 127


def test_viewer_encoding_keeps_faint_and_bright_values():
    from spx import viewer
    rng = np.random.default_rng(1)
    img = rng.normal(0, 0.05, (41, 41))
    img[20, 20] = 400.0            # a bright star
    img[3, 3] = np.nan             # no data
    flags = np.zeros_like(img, bool)
    flags[10, 12] = True
    enc = viewer.encode_frames([img], [flags])
    v, f = _decode_png16(enc["frames"][0], enc["lo"], enc["s"], enc["a"])
    assert np.isnan(v[3, 3])
    ok = np.isfinite(img)
    faint = ok & (np.abs(img) < 0.3)
    assert np.max(np.abs(v[faint] - img[faint])) < 0.01   # well below the noise (0.05)
    assert abs(v[20, 20] - 400) / 400 < 0.01
    assert f[10, 12] and f.sum() == 1


def test_viewer_payload_has_overlays_per_frame():
    from spx import viewer
    from types import SimpleNamespace
    fr = [SimpleNamespace(image=np.zeros((5, 5)), flag_mask=None, label=f"Epoch {i}", mjd_mid=61000.0 + 180 * i,
                          kind="stack") for i in range(3)]
    data = viewer.payload(fr, target=[dict(x=2, y=2, label="t")] * 3, scale=3.075, center=(2, 2),
                          radec=(10.0, 20.0))
    assert len(data["frames"]) == 3 and data["frames"][1]["date"] == "2026-05-20"
    assert data["frames"][0]["target"]["x"] == 2


# --- helpers: star detection and change finding ---------------------------------
def _sky(stars, n=81, noise=0.05, seed=3):
    """Synthetic background-subtracted frame: gaussian stars (x, y, peak) on noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((n, n))
    img = rng.normal(0, noise, (n, n))
    for x, y, peak in stars:
        img += peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.6 ** 2))
    return img


def test_detect_finds_injected_stars_and_no_noise_peaks():
    from spx import detect
    stars = [(20.3, 30.7, 2.0), (60.0, 15.2, 1.0), (45.6, 62.1, 0.6)]
    found = detect.find_stars(_sky(stars), nsigma=5)
    assert len(found) == 3
    for x, y, _ in stars:
        assert min(np.hypot(f["x"] - x, f["y"] - y) for f in found) < 0.3
    assert detect.find_stars(_sky([]), nsigma=5) == []


def test_changes_classifies_moved_brightened_faded():
    from spx import detect
    still, mover, flare, gone = (15.0, 15.0, 1.5), (40.0, 40.0, 1.5), (65.0, 20.0, 0.8), (20.0, 65.0, 1.2)
    a = _sky([still, mover, flare, gone], seed=1)
    b = _sky([still, (43.0, 41.0, 1.5), (65.0, 20.0, 2.0)], seed=2)
    ch = detect.changes(a, b, nsigma=5)
    kinds = {c["kind"]: c for c in ch}
    assert set(kinds) == {"moved", "brightened", "faded"}, ch
    m = kinds["moved"]
    assert abs(m["x"] - 40) < 0.5 and abs(m["x2"] - 43) < 0.5 and abs(m["dist_px"] - np.hypot(3, 1)) < 0.4
    assert abs(kinds["brightened"]["x"] - 65) < 0.5 and abs(kinds["faded"]["y"] - 65) < 0.5
    assert ch == sorted(ch, key=lambda c: -c["score"])  # ranked, strongest first


def test_changes_ignore_a_shift_shared_by_all_stars():
    """Each epoch averages a different mix of wavelengths, so all stars can look ~30% brighter at once.
    Only the star that changed more than the others counts."""
    from spx import detect
    pos = [(12 + 13 * (i % 5), 12 + 15 * (i // 5), 0.6 + 0.15 * i) for i in range(15)]
    a = _sky(pos, seed=4)
    b_stars = [(x, y, p * 1.3) for x, y, p in pos]
    b_stars[7] = (pos[7][0], pos[7][1], pos[7][2] * 3.0)   # the real flare
    ch = detect.changes(a, _sky(b_stars, seed=5), nsigma=5)
    assert len(ch) == 1 and ch[0]["kind"] == "brightened", ch
    assert abs(ch[0]["x"] - pos[7][0]) < 0.5 and abs(ch[0]["y"] - pos[7][1]) < 0.5


# --- Check my find ----------------------------------------------------------------
def test_nearest_gaia_star_uses_the_mark_date():
    """A fast star sits 20" from its 2016 position by the mark date: it must still be found there."""
    from spx import catalogs, config
    fast = dict(id=1, ra=150.0, dec=10.0, pmra=0.0, pmdec=2000.0, parallax=100.0, g=12.0)   # 2"/yr north
    slow = dict(id=2, ra=150.0, dec=10.0 + 18 / 3600, pmra=0.0, pmdec=0.0, parallax=1.0, g=12.0)
    mjd = config.GAIA_REF_MJD + 10 * 365.25          # 10 years later: the fast star is 20" north
    star, sep = catalogs.nearest([fast, slow], 150.0, 10.0 + 20 / 3600, mjd, max_arcsec=6.0)
    assert star["id"] == 1 and sep < 0.5
    assert catalogs.nearest([slow], 151.0, 10.0, mjd, max_arcsec=6.0) == (None, None)
    t = catalogs.as_target(star)
    assert t.ref_mjd == config.GAIA_REF_MJD and t.pmdec == 2000.0 and t.has_motion


def test_fetch_many_keeps_reporting_while_waiting(monkeypatch):
    """First photos of different detectors download side by side, and the progress callback keeps ticking
    while nothing has arrived yet (so the loading screen never looks frozen)."""
    import time as _t
    from types import SimpleNamespace
    from spx import cutouts

    exps = [SimpleNamespace(detector=d, mjd=60000.0 + d, uri=f"u{d}") for d in (1, 2, 3)]
    monkeypatch.setattr(cutouts, "has_wave_table", lambda e: False)
    monkeypatch.setattr(cutouts, "is_cached", lambda *a: False)

    def slow(e, *a):
        _t.sleep(0.8)
        return SimpleNamespace(mjd=e.mjd, uri=e.uri)
    monkeypatch.setattr(cutouts, "get_stamp", slow)
    calls = []
    t0 = _t.time()
    stamps, fails = cutouts.fetch_many(exps, 0, 0, 1, progress=lambda d, n: calls.append((round(_t.time() - t0, 1), d, n)))
    assert len(stamps) == 3 and fails == 0
    assert _t.time() - t0 < 1.6                       # parallel, not 3 x 0.8 s
    assert any(d == 0 for _, d, _ in calls)           # ticked before the first photo arrived

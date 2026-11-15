"""Fast loading: the AWS reader's round trips, shared downloads, background loading for /RTP, hedged searches,
and the alignment shortcut (all must give exactly the same results as before)."""
import io
import threading
import time
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from spx import archive, config, cutouts, stack, targets, teleport, warp
from spx.archive import Exposure
from spx.cutouts import Stamp

NY, NX = 60, 50
EXP = Exposure("D4", 61000.0, 61000.01, "ibe/data/spherex/test/level2/x/l2b-v27-2026-001/4/file.fits",
               "l2b-v27-2026-001")


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 10.0]
    w.wcs.crpix = [25.5, 30.5]
    w.wcs.cdelt = [-6.15 / 3600, 6.15 / 3600]
    return w


def _spherex_like_file(epsf_rows=10):
    """A small file laid out like a SPHEREx level-2 image: IMAGE, FLAGS (RICE_1, one tile per row), VARIANCE,
    ZODI, then EPSF and WCS-WAVE. Returns (bytes, the four planes)."""
    rng = np.random.default_rng(3)
    planes = dict(image=rng.normal(5, 2, (NY, NX)).astype(np.float32),
                  flags=rng.integers(0, 1 << 20, (NY, NX)).astype(np.int32),
                  variance=rng.uniform(0.5, 2, (NY, NX)).astype(np.float32),
                  zodi=rng.uniform(1, 1.1, (NY, NX)).astype(np.float32))
    img = fits.ImageHDU(planes["image"], header=_wcs().to_header(), name="IMAGE")
    flags = fits.CompImageHDU(planes["flags"], compression_type="RICE_1", tile_shape=(1, NX), name="FLAGS")
    flags.header["MSKN1"], flags.header["MSKM1"] = "OVERFLOW", 2
    epsf = fits.ImageHDU(np.ones((epsf_rows, 9, 9), np.float32), name="EPSF")
    wave = fits.BinTableHDU.from_columns([fits.Column("X", "E", array=np.zeros(3))], name="WCS-WAVE")
    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), img, flags, fits.ImageHDU(planes["variance"], name="VARIANCE"),
                  fits.ImageHDU(planes["zodi"], name="ZODI"), epsf, wave]).writeto(buf)
    return buf.getvalue(), planes


@pytest.fixture
def s3(monkeypatch, tmp_path):
    """The AWS copy served from memory: fetch_s3_stamp reads `blob`; every request is counted."""
    monkeypatch.setenv("SPHEREX_BLINK_CACHE", str(tmp_path))
    monkeypatch.setattr(cutouts, "_LAYOUTS", {})
    monkeypatch.setattr(cutouts, "_LAYOUTS_READ", [False])
    state = {"blob": b"", "calls": []}

    def fake(url, start, stop, size=None, lane=None):
        state["calls"].append((start, stop))
        if size is not None:
            size.append(len(state["blob"]))
        return state["blob"][start:stop]
    monkeypatch.setattr(cutouts, "_range", fake)
    return state


def _expected(planes, ra=150.0, dec=10.0, size=1.0):
    x, y = _wcs().world_to_pixel_values(ra, dec)
    half = int(np.ceil(size * 60 / config.PIXEL_SCALE_ARCSEC / 2))
    x0, x1 = max(0, int(round(float(x))) - half), min(NX, int(round(float(x))) + half + 1)
    y0, y1 = max(0, int(round(float(y))) - half), min(NY, int(round(float(y))) + half + 1)
    return {k: v[y0:y1, x0:x1] for k, v in planes.items()}


def _same(st, exp_planes):
    return all(np.array_equal(getattr(st, k), v) for k, v in exp_planes.items())


def test_aws_read_learns_the_layout_once_then_needs_far_fewer_requests(s3):
    s3["blob"], planes = _spherex_like_file()
    first = cutouts.fetch_s3_stamp(EXP, 150.0, 10.0, 1.0)
    assert _same(first, _expected(planes)) and first.flag_bits == {"OVERFLOW": 2}
    n_first = len(s3["calls"])
    s3["calls"].clear()
    second = cutouts.fetch_s3_stamp(EXP, 150.0, 10.0, 1.0)
    assert _same(second, _expected(planes))
    # header; IMAGE, VARIANCE, ZODI, FLAGS header+table, two header-end checks; FLAGS rows
    assert len(s3["calls"]) == 8 < n_first
    assert (config.cache_dir() / "s3_layout.json").exists()     # remembered for the next session


def test_aws_read_is_still_right_when_a_file_ends_differently(s3):
    s3["blob"], _ = _spherex_like_file(epsf_rows=10)
    cutouts.fetch_s3_stamp(EXP, 150.0, 10.0, 1.0)                 # learns where VARIANCE / ZODI sit
    s3["blob"], planes = _spherex_like_file(epsf_rows=37)         # same layout, a longer tail
    st = cutouts.fetch_s3_stamp(EXP, 150.0, 10.0, 1.0)
    assert _same(st, _expected(planes))
    s3["calls"].clear()
    assert _same(cutouts.fetch_s3_stamp(EXP, 150.0, 10.0, 1.0), _expected(planes))
    assert len(s3["calls"]) == 8                                  # the new tail was learned


def test_aws_read_relearns_at_once_when_the_headers_changed(s3):
    s3["blob"], planes = _spherex_like_file()
    key = cutouts._layout_key(EXP)
    wrong = (2880 * 7, 2880, 2880, 123456)                        # learned from a differently built file
    cutouts._LAYOUTS[key] = wrong
    cutouts._LAYOUTS_READ[0] = True
    assert _same(cutouts.fetch_s3_stamp(EXP, 150.0, 10.0, 1.0), _expected(planes))   # no IRSA needed
    assert cutouts._LAYOUTS[key] != wrong


def _stamp(n=41, rot=23.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 10.0]
    w.wcs.crpix = [21, 21]
    s, t = 6.15 / 3600, np.radians(rot)
    w.wcs.cd = [[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]]
    rng = np.random.default_rng(5)
    img = rng.normal(10, 3, (n, n)).astype(np.float32)
    flags = np.zeros((n, n), np.int32)
    flags[5:8, 9:12] = 2
    img[20, 20] = np.nan
    return Stamp("D4", 61000, 61000.01, "u", "v", img, flags, rng.uniform(0.5, 2, (n, n)).astype(np.float32),
                 np.full((n, n), 1.0, np.float32), w.to_header().tostring(), {"OVERFLOW": 2})


def test_align_with_one_transform_equals_three_separate_reprojections():
    st = _stamp()
    w, shape = stack.grid(150.0, 10.0, 3)
    got = stack.align(st, w, shape)
    img, var, suspect = stack.clean(st)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, fp = reproject_interp((img, st.wcs), w, shape_out=shape, order="bilinear")
        v, _ = reproject_interp((var, st.wcs), w, shape_out=shape, order="bilinear")
        s, _ = reproject_interp((suspect.astype(float), st.wcs), w, shape_out=shape, order="bilinear")
    a[fp == 0] = np.nan
    wt = np.where(np.isfinite(a) & np.isfinite(v) & (v > 0), 1.0 / np.where(v > 0, v, 1), 0.0)
    flagged = np.nan_to_num(s) > 0
    wt = np.where(flagged, wt * config.SUSPECT_WEIGHT, wt)
    assert np.array_equal(got.image, a, equal_nan=True) and np.array_equal(got.weight, wt)
    assert np.array_equal(got.flagged, flagged)


def test_a_stamp_builds_its_wcs_once():
    st = _stamp()
    assert st.wcs is st.wcs


def test_the_same_photo_asked_twice_at_once_is_downloaded_once(monkeypatch, tmp_path):
    monkeypatch.setenv("SPHEREX_BLINK_CACHE", str(tmp_path))
    calls = []

    def slow(exp, ra, dec, size, quiet=False):
        calls.append(quiet)
        time.sleep(0.3)
        return _stamp()
    monkeypatch.setattr(cutouts, "_download", slow)
    out = []
    ts = [threading.Thread(target=lambda q=q: out.append(cutouts.get_stamp(EXP, 1.0, 2.0, 7.0, quiet=q)))
          for q in (True, False)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1 and len(out) == 2
    assert np.array_equal(out[0].image, out[1].image, equal_nan=True)


def test_background_loading_waits_for_the_page_the_user_waits_for_and_can_be_called_off(monkeypatch):
    got = []
    monkeypatch.setattr(cutouts, "has_wave_table", lambda e: True)
    monkeypatch.setattr(cutouts, "get_stamp", lambda e, *a, **k: got.append(e.uri) or Stamp(
        "D4", e.mjd, e.mjd, e.uri, "v", *(np.zeros((2, 2), np.float32),) * 4, ""))
    exps = [Exposure("D4", 61000.0 + i, 61000.0 + i, f"u{i}", "v") for i in range(4)]
    with cutouts._FG_IDLE:
        cutouts._FG_BUSY[0] += 1                                  # a foreground load is running
    res = []
    t = threading.Thread(target=lambda: res.append(cutouts.fetch_many(exps, 0, 0, 1, background=True)))
    t.start()
    time.sleep(0.4)
    assert got == []                                              # nothing taken from the user's load
    with cutouts._FG_IDLE:
        cutouts._FG_BUSY[0] -= 1
        cutouts._FG_IDLE.notify_all()
    t.join(5)
    assert sorted(got) == ["u0", "u1", "u2", "u3"] and len(res[0][0]) == 4
    got.clear()
    stamps, missing = cutouts.fetch_many(exps, 0, 0, 1, background=True, stop=lambda: True)
    assert got == [] and stamps == [] and missing == 4


def test_rtp_takes_the_spot_loaded_ahead_and_keeps_the_queue_full(monkeypatch):
    loaded = []
    monkeypatch.setattr(teleport, "_load", lambda spot, want: loaded.append((spot.text, want)))
    q = []
    want = teleport.settings("blink", "D4", 5, False, False)
    teleport.prepare(q, want, 2)
    for s in q:
        s.job.result(5)
    assert len(q) == 2 and [t for t, _ in loaded] == [s.text for s in q]
    first = q[0]
    assert teleport.take(q) == first.text and first.want == ()    # used: its background work stops
    assert targets.parse_coords(first.text) is not None
    teleport.prepare(q, want, 2)
    assert len(q) == 2
    for s in q:
        s.job.result(5)
    loaded.clear()
    other = teleport.settings("blink", "D6", 5, False, False)    # another band: the same spots, loaded again
    teleport.prepare(q, other, 2)
    for s in q:
        s.job.result(5)
    assert sorted(loaded) == sorted((s.text, other) for s in q)
    assert targets.parse_coords(teleport.take([])) is not None   # nothing waiting: a fresh random spot


def test_a_spot_that_could_not_load_is_tried_again_later_unless_it_was_used(monkeypatch):
    monkeypatch.setattr(teleport, "RETRY_AFTER", 0.05)
    calls = []

    def flaky(spot, want, stop):
        calls.append(spot.text)
        if len(calls) < 3:
            raise RuntimeError("archive unreachable")
    monkeypatch.setattr(teleport, "_fill", flaky)
    q = []
    teleport.prepare(q, teleport.settings("blink", "D4", 5, False, False), 1)
    t0 = time.time()
    while len(calls) < 3 and time.time() - t0 < 5:
        time.sleep(0.02)
    assert len(calls) == 3                                   # failed twice, loaded on the third try
    calls.clear()
    monkeypatch.setattr(teleport, "_fill", lambda spot, want, stop: calls.append(1) or 1 / 0)
    q2 = []
    teleport.prepare(q2, teleport.settings("blink", "D4", 5, False, False), 1)
    q2[0].job.result(5)
    teleport.take(q2)                                        # used: no more tries for it
    time.sleep(0.3)
    assert calls == [1]


def test_rtp_settings_follow_what_each_view_loads():
    assert teleport.settings("spectrum", "D1", 5, True, True) == ("spectrum", "D1", 5, False, False)
    assert teleport.settings("blink", "D4", 10, True, True) == ("view", "D4", 10, True, True)
    assert teleport.settings("motion", "D4", 10, True, True) == ("view", "D4", 10, False, True)
    assert teleport.settings("movie", "D4", 10, True, True) == ("view", "D4", 10, True, False)


def test_a_slow_search_answer_is_asked_again_and_the_first_good_one_wins():
    n = {"calls": 0}

    def fn():
        n["calls"] += 1
        if n["calls"] == 1:
            time.sleep(1.0)
            return "slow"
        return "fast"
    t0 = time.time()
    assert archive.hedged(fn, after=0.1, tries=3) == "fast" and time.time() - t0 < 0.8


def test_a_failed_search_is_retried_and_all_failures_raise():
    n = {"calls": 0}

    def flaky():
        n["calls"] += 1
        if n["calls"] < 3:
            raise ConnectionError("reset")
        return "ok"
    assert archive.hedged(flaky, after=5, tries=3) == "ok" and n["calls"] == 3
    with pytest.raises(ConnectionError):
        archive.hedged(lambda: (_ for _ in ()).throw(ConnectionError("down")), after=5, tries=2)


def test_two_searches_for_one_position_at_once_share_one_request(monkeypatch, tmp_path):
    monkeypatch.setenv("SPHEREX_BLINK_CACHE", str(tmp_path))
    calls = []

    def slow(ra, dec, use_cache):
        calls.append((ra, dec))
        time.sleep(0.3)
        return [EXP]
    monkeypatch.setattr(archive, "_search", slow)
    out = []
    ts = [threading.Thread(target=lambda: out.append(archive.search(12.3456, -7.6543))) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1 and out == [[EXP]] * 3


def test_loading_screen_fade_does_not_hold_the_page():
    assert warp.HANDOVER < 0.2
    assert "__CSS__" not in warp.SCRIPT and ".spx-warp{position:fixed" in warp.SCRIPT and "spx-ghost" in warp.SCRIPT


def test_loading_screen_script_survives_streamlits_html_sanitizer():
    """st.html drops a whole script whose text contains '<' followed by a letter, '/' or '!' (DOMPurify's
    mXSS check), and the star field would then never start."""
    import re
    body = warp.SCRIPT.removeprefix("<script>").removesuffix("</script>")
    assert warp.SCRIPT.startswith("<script>") and warp.SCRIPT.endswith("</script>")
    assert not re.search(r"<[/\w!]", body)

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


def _fresh_rtp(monkeypatch, tmp_path):
    monkeypatch.setenv("SPHEREX_BLINK_CACHE", str(tmp_path))
    for name, v in (("_WANT", {}), ("_QUEUED", set()), ("_DONE", set()), ("_BUSY", {1: 0, 2: 0})):
        monkeypatch.setattr(teleport, name, v)


def _drain():
    teleport._JOBS.submit(lambda: None).result(10)   # every job queued before this one is done


def test_rtp_keeps_a_stock_on_disk_and_takes_the_readiest_spot(monkeypatch, tmp_path):
    _fresh_rtp(monkeypatch, tmp_path)
    steps = []
    exps = [Exposure("D4", 61000.0 + 200 * k + i, 61000.0 + 200 * k + i, f"e{k}-{i}", "v")
            for k in range(2) for i in range(3)]
    monkeypatch.setattr(teleport.archive, "search", lambda ra, dec: exps)
    monkeypatch.setattr(teleport.archive, "saved", lambda ra, dec: exps)
    monkeypatch.setattr(teleport.archive, "select", lambda es, b, quick=True, **k: {0: es[:3], 1: es[3:]})
    monkeypatch.setattr(teleport.archive, "complete_future", lambda ra, dec: None)
    have = set()
    monkeypatch.setattr(cutouts, "is_cached", lambda e, *a: e.uri in have)

    def fetch(todo, *a, **k):
        steps.append(len(todo))
        return [], len(todo)
    monkeypatch.setattr(cutouts, "fetch_many", fetch)
    want = teleport.settings("blink", "D4", 5, False, False)
    teleport.prepare(want, 3)
    _drain()
    spots = teleport.stock()
    assert len(spots) == 3 and all(targets.parse_coords(s) for s in spots)   # saved: it survives a restart
    assert steps == [2, 2, 2, 6]       # one photo per date of every spot first, then all of the next one
    assert all(teleport.level(s, want) == 1 for s in spots)
    have.update({"e0-0", "e1-0"})                                    # one per date at hand, for the second spot
    ra1 = teleport._where(spots[1])[1]
    monkeypatch.setattr(cutouts, "is_cached", lambda e, ra, *a: e.uri in have and ra == ra1)
    assert teleport.level(spots[1], want) == 2
    assert teleport.take(want) == spots[1]                           # the readiest one, not just the oldest
    assert teleport.stock() == [spots[0], spots[2]]
    steps.clear()
    teleport.prepare(want, 3)                                        # topped up again; the others are not redone
    _drain()
    assert len(teleport.stock()) == 3 and steps == [2]                # only the new spot's first step
    steps.clear()
    teleport._WANT.clear()                                           # a restart: the stock is still there
    assert teleport.take(want) in spots
    teleport._WANT.clear()
    teleport._save([])
    assert targets.parse_coords(teleport.take(want)) is not None   # nothing waiting: a fresh random spot


def test_a_spot_that_could_not_load_is_tried_again_later_unless_it_was_used(monkeypatch, tmp_path):
    _fresh_rtp(monkeypatch, tmp_path)
    monkeypatch.setattr(teleport, "RETRY_AFTER", 0.05)
    calls = []

    def flaky(text, want, stop):
        calls.append(text)
        if len(calls) < 3:
            raise RuntimeError("archive unreachable")
    monkeypatch.setattr(teleport, "_first", flaky)
    monkeypatch.setattr(teleport, "_rest", lambda *a: None)
    want = teleport.settings("blink", "D4", 5, False, False)
    teleport.prepare(want, 1)
    t0 = time.time()
    while len(calls) < 3 and time.time() - t0 < 5:
        time.sleep(0.02)
    assert len(calls) == 3                                   # failed twice, loaded on the third try
    _fresh_rtp(monkeypatch, tmp_path)
    calls.clear()
    monkeypatch.setattr(teleport, "_first", lambda text, want, stop: calls.append(1) or 1 / 0)
    teleport.prepare(want, 1)
    _drain()
    teleport.take(want)                                      # used: no more tries for it
    time.sleep(0.3)
    assert calls == [1]


def test_the_search_uses_an_old_list_when_the_archive_is_slow(monkeypatch, tmp_path):
    monkeypatch.setenv("SPHEREX_BLINK_CACHE", str(tmp_path))
    monkeypatch.setattr(config, "SEARCH_STALE_WAIT", 0.2)
    import json as _json
    old = [Exposure("D1", 61000.0, 61000.0, "old", "v")]
    (tmp_path / "sia_10.0000_+20.0000.json").write_text(
        _json.dumps({"t": time.time() - 40 * 3600, "rows": [dict(e.__dict__) for e in old]}))
    gate = threading.Event()
    monkeypatch.setattr(archive, "_search", lambda ra, dec, use_cache: gate.wait(5) or [])
    t0 = time.time()
    assert archive.search(10.0, 20.0) == old and time.time() - t0 < 1   # NASA slow: the saved list at once
    gate.set()
    monkeypatch.setattr(archive, "_search", lambda ra, dec, use_cache: [Exposure("D1", 1.0, 1.0, "new", "v")])
    assert archive.search(10.0, 20.0)[0].uri == "new"                # NASA answering: the fresh list


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


def test_the_loading_screens_count_down_and_the_motion_bar_shows_real_progress():
    import re as _re
    html = warp.card("Flying in", "", 2, 0.5, "3 of 8 photos", left=47.2, lid="L1")
    m = _re.search(r'class="spx-eta" data-lid="L1" data-left="47.2"[^>]*>48 s left<', html)
    assert m and 'data-fs="{} s left"' in html                   # the browser counts it down from there
    assert 'data-left=""' in warp.card("x", "", 1, left=None, lid="L")   # no honest estimate: says so instead
    assert "data-lid" not in warp.card("x", "", 1)               # no estimate yet: no countdown
    bar = warp.bar_card("Measuring", "", 4, 0.8, "5 of 9 photos measured", left=2.0, lid="L2")
    assert "spx-bar" in bar and 'aria-valuenow="80"' in bar and "width:80.0%" in bar and "2 s left" in bar
    assert bar.count('class="done"') == 4 and bar.count('class="now"') == 1
    assert abs(sum(warp.MOTION_WEIGHTS) - 1) < 1e-9 and len(warp.MOTION_WEIGHTS) == len(warp.MOTION_STAGES)
    assert warp.eta_text(125) == "2 min left" and warp.eta_text(0) == "almost done"
    body = warp.SCRIPT.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    assert "spx-eta" in body and not _re.search(r"<[/\w!]", body)   # survives st.html's sanitizer

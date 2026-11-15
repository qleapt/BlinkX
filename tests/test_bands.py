"""Band changes at once and nothing over ~2 s: the WCS built from its own cards, the order photos are fetched in
(at hand, one per date, the rest), the page shown at the soft deadline, the other bands loaded ahead, and the
scripts that hand them to the browser."""
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from spx import config, cutouts, prefetch, stack, viewer
from spx.archive import Exposure
from spx.cutouts import Stamp


def _sip_header() -> str:
    """An IMAGE-like header: a TAN-SIP WCS among many cards a WCS never reads (as in SPHEREx files)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    w.wcs.crval = [269.45, 4.69]
    w.wcs.crpix = [1020.5, 1020.5]
    s, t = 6.15 / 3600, np.radians(31.0)
    w.wcs.cd = [[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]]
    w.wcs.dateobs, w.wcs.radesys = "2026-01-01T00:00:00", "ICRS"
    h = w.to_header(relax=True)
    for name in ("A", "B", "AP", "BP"):
        h[f"{name}_ORDER"] = 3
        for i in range(4):
            for j in range(4 - i):
                h[f"{name}_{i}_{j}"] = 1e-9 * (i + 1) * (j + 2) * (1 if name in ("A", "AP") else -1)
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 2040, 2040
    for k in range(40):
        h[f"KSN{k}"] = k * 0.5                  # instrument housekeeping
        h.add_history(f"processing step {k}")
    return h.tostring()


def test_wcs_from_its_own_cards_is_identical_to_the_whole_header():
    text = _sip_header()
    full = cutouts.image_wcs(fits.Header.fromstring(text))
    fast = cutouts.build_wcs(text)
    assert full.wcs.compare(fast.wcs) and full.to_header_string(relax=True) == fast.to_header_string(relax=True)
    for nm in ("a", "b", "ap", "bp"):
        assert np.array_equal(getattr(full.sip, nm), getattr(fast.sip, nm))
    px = np.random.default_rng(2).uniform(-100, 2100, (2, 400))
    w1, w2 = np.array(full.pixel_to_world_values(*px)), np.array(fast.pixel_to_world_values(*px))
    assert np.array_equal(w1, w2)
    assert np.array_equal(np.array(full.world_to_pixel_values(*w1)), np.array(fast.world_to_pixel_values(*w1)))


def test_a_header_wcs_is_built_once_and_handed_out_as_copies(monkeypatch):
    built = []
    real = cutouts.build_wcs
    monkeypatch.setattr(cutouts, "build_wcs", lambda t: built.append(1) or real(t))
    monkeypatch.setattr(cutouts, "_WCS_MEMO", type(cutouts._WCS_MEMO)())
    text = _sip_header()
    a, b = cutouts.header_wcs(text), cutouts.header_wcs(text)
    assert len(built) == 1 and a is not b                         # never shared between threads
    assert np.array_equal(a.pixel_to_world_values(10, 20), b.pixel_to_world_values(10, 20))


def _stamp_for(e):
    z = np.zeros((2, 2), np.float32)
    return Stamp(e.band, e.mjd, e.mjd, e.uri, "v", z, z.astype(np.int32), z, z, "")


def _exps(n_epochs=3, per=3):
    return [Exposure("D4", 61000.0 + 200 * k + i, 61000.0 + 200 * k + i, f"e{k}-{i}", "v")
            for k in range(n_epochs) for i in range(per)]


def test_photos_at_hand_then_one_per_date_come_first_and_the_page_shows_at_the_deadline(monkeypatch):
    exps = _exps()
    epoch = {e.uri: e.uri.split("-")[0] for e in exps}
    at_hand = {"e1-2"}                                            # e.g. loaded ahead in the background
    lead = {"e0-0", "e2-0"}                                       # one per date that has nothing at hand
    started, lock = {}, threading.Lock()

    def get(e, *a, **k):
        with lock:
            started[e.uri] = time.time()
        time.sleep(0 if e.uri in at_hand else 0.1 if e.uri in lead else 1.5)
        return _stamp_for(e)
    monkeypatch.setattr(cutouts, "get_stamp", get)
    monkeypatch.setattr(cutouts, "is_cached", lambda e, *a: e.uri in at_hand)
    monkeypatch.setattr(cutouts, "has_wave_table", lambda e: True)
    stats, t0 = {}, time.time()
    got, fails = cutouts.fetch_many(exps, 0, 0, 1, deadline=0.3, stats=stats, first=lead,
                                    enough=lambda ss: {epoch[s.uri] for s in ss} == {"e0", "e1", "e2"})
    took = time.time() - t0
    assert {s.uri for s in got} == at_hand | lead and fails == 0 and stats["skipped"] == len(exps) - 3
    assert 0.25 < took < 1.0                                      # shown at the deadline, not after the rest
    later = [t for u, t in started.items() if u not in at_hand | lead]
    assert min(later) >= max(started[u] for u in lead) - 0.01      # the rest only after one photo per date
    assert cutouts._FG_BUSY[0] > 0                                # still arriving: the background keeps waiting
    deadline = time.time() + 5
    while cutouts._FG_BUSY[0] and time.time() < deadline:
        time.sleep(0.05)
    assert cutouts._FG_BUSY[0] == 0                               # ... until the last one is in


def test_everything_arriving_before_the_deadline_is_shown_in_one_go(monkeypatch):
    exps = _exps(2, 2)
    monkeypatch.setattr(cutouts, "get_stamp", lambda e, *a, **k: time.sleep(0.05) or _stamp_for(e))
    monkeypatch.setattr(cutouts, "is_cached", lambda e, *a: False)
    monkeypatch.setattr(cutouts, "has_wave_table", lambda e: True)
    stats = {}
    got, _ = cutouts.fetch_many(exps, 0, 0, 1, deadline=1.5, stats=stats, first={"e0-0", "e1-0"},
                                enough=lambda ss: True)
    assert len(got) == 4 and stats["skipped"] == 0


def test_bands_load_ahead_nearest_first():
    assert prefetch.order("D4") == ["D3", "D5", "D2", "D6", "D1"]
    assert prefetch.order("D1") == ["D2", "D3", "D4", "D5", "D6"]
    assert prefetch.order("D4", include=True)[0] == "D4"


class _Tgt:
    name = "T"


def test_other_bands_are_lined_up_ahead_and_handed_to_the_page(monkeypatch):
    calls = []
    monkeypatch.setattr(prefetch.archive, "select", lambda exps, b, quick=True, **k:
                        {} if k else {0: [e for e in exps if e.band == b]})
    monkeypatch.setattr(cutouts, "fetch_many", lambda todo, *a, **k: (calls.append(todo[0].band) or
                                                                     [_stamp_for(e) for e in todo], 0))
    monkeypatch.setattr(stack, "epoch_frames", lambda stamps, groups, tgt, w, s: ({}, [stamps[0].band]))
    monkeypatch.setattr(prefetch, "_READY", type(prefetch._READY)())
    exps = [Exposure(b, 61000.0, 61000.0, f"u-{b}", "v") for b in config.BANDS]
    prefetch.start(_Tgt(), 10.0, 20.0, 5, "D3", exps)
    prefetch._JOBS.submit(lambda: None).result(10)                # the job queued before this one is done
    assert calls == ["D2", "D4", "D1", "D5", "D6"]                # nearest band first
    for b in ("D1", "D2", "D4", "D5", "D6"):
        assert prefetch.ready("T", 10.0, 20.0, 5, b, exps)[1] == [b]
    assert prefetch.ready("T", 10.0, 20.0, 5, "D3", exps) is None  # the page's own band is the page's job
    calls.clear()
    prefetch.start(_Tgt(), 10.0, 20.0, 5, "D3", exps)             # the same view again: nothing is done twice
    prefetch._JOBS.submit(lambda: None).result(10)
    assert calls == []


def test_another_spot_calls_the_loading_ahead_off(monkeypatch):
    gate, calls = threading.Event(), []

    def fetch(todo, *a, stop=None, **k):
        calls.append(todo[0].band)
        gate.wait(5)
        return [], len(todo)
    monkeypatch.setattr(prefetch.archive, "select", lambda exps, b, quick=True, **k:
                        {} if k else {0: [e for e in exps if e.band == b]})
    monkeypatch.setattr(cutouts, "fetch_many", fetch)
    exps = [Exposure(b, 61000.0, 61000.0, f"w-{b}", "v") for b in config.BANDS]
    prefetch.start(_Tgt(), 1.0, 2.0, 5, "D1", exps)
    time.sleep(0.2)
    prefetch.start(_Tgt(), 3.0, 4.0, 5, "D1", [])                # the user went elsewhere
    gate.set()
    prefetch._JOBS.submit(lambda: None).result(10)
    assert calls == ["D2"]                                        # the first spot stopped after its current band


def _script_bodies(html):
    return re.findall(r"<script>(.*?)</script>", html, re.S)


def test_band_scripts_survive_streamlits_html_sanitizer():
    """st.html drops a whole script whose text contains '<' followed by a letter, '/' or '!' (see warp.SCRIPT)."""
    rec = {"data": {"title": "x</script><b>bold</b>", "frames": []}, "store_key": "k", "settings": {"a": 1}}
    for html in (viewer.BAND_SCRIPT, viewer.alt_html("view<1>", "T|D2|5|blink", rec)):
        bodies = _script_bodies(html)
        assert len(bodies) == 1 and not re.search(r"<[/\w!]", bodies[0])
    js = _script_bodies(viewer.alt_html("v", "T|D2|5|blink", rec))[0]
    m = json.loads(re.search(r"const m = (\{.*?\}); let A", js, re.S).group(1))
    assert m == {"k": "v", "t": "T|D2|5|blink", "r": rec}         # the browser gets exactly the record


def test_the_viewer_can_show_a_band_handed_over_and_undo_it():
    page = (Path(viewer.__file__).parent / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "window.spxShowBand" in page and "window.spxRestore" in page and "__spxAlts" in page
    assert ".st-key-band input" in viewer.BAND_SCRIPT and "spxShowBand" in viewer.BAND_SCRIPT

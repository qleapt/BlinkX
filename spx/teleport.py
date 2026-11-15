"""/RTP (random teleport) without the wait: random spots are picked ahead of time and loaded quietly in the
background, so a jump lands at once, even on a slow connection or while NASA's archive is slow to answer.

A spot is still a uniformly random position on the sphere (targets.random_position), only chosen earlier. The spots
ready to jump to are kept on disk (rtp_stock.json in the cache folder), so they survive a restart and the stock is
there from the first page on. Each spot is loaded in two steps, so several spots get ready quickly:

1. the archive's list of photos and one photo per date (enough for the page to show it at once);
2. for the next spot to be used only: the rest of its photos, lined up, and the extras (Gaia, older surveys).

/RTP takes the readiest spot (take()). The background work pauses while a load the user is waiting for runs
(cutouts.fetch_many(background=True)), uses a few connections of its own, and a spot's work stops as soon as it is
used or the view settings change. Whatever it loaded lands in the usual caches.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from . import archive, catalogs, config, cutouts, stack, targets

_JOBS = ThreadPoolExecutor(1, thread_name_prefix="spx-rtp")   # one job at a time, in order
_READY: OrderedDict = OrderedDict()   # stack.frames_key -> (by_uri, frames) lined up in the background
_READY_LOCK = threading.Lock()
_KEEP = 3   # lined-up spots kept in memory (each is a few MB)
RETRY_AFTER = 60.0   # s: a spot whose loading failed (e.g. the archive could not be reached) is tried again
RETRIES = 10

_LOCK = threading.RLock()
_WANT: dict = {}      # spot text -> the settings it is being loaded for; a change (or the spot being used) stops it
_QUEUED: set = set()  # (text, want, step) waiting or running
_DONE: set = set()    # (text, want, 2) finished
_BUSY = {1: 0, 2: 0}  # jobs waiting or running per step


def settings(mode: str, band: str, fov, gaia: bool, survey: bool) -> tuple:
    """What a spot is loaded for: exactly what the app loads in this view (the extras only where it shows them)."""
    return ("spectrum" if mode == "spectrum" else "view", band, fov,
            bool(gaia and mode in ("blink", "sub", "movie")), bool(survey and mode in ("blink", "sub", "motion")))


def _path():
    return config.cache_dir() / "rtp_stock.json"


def stock() -> list:
    """The spots ready to jump to (their archive list is saved), oldest first."""
    try:
        spots = json.loads(_path().read_text()).get("spots", [])
    except Exception:
        return []
    return [s for s in spots if isinstance(s, str) and targets.parse_coords(s) is not None]


def _save(spots: list) -> None:
    try:
        _path().write_text(json.dumps({"spots": spots}))
    except OSError:
        pass


def _where(text: str):
    t = targets.parse_coords(text)
    ra0, dec0 = targets.position_at(t, config.REFERENCE_MJD)   # exactly as the app computes the field centre
    return t, ra0, dec0


def plan(exps, want: tuple):
    """(photos, uri -> date or band) exactly as the page loads them for these settings."""
    kind, band = want[0], want[1]
    todo, group_of = [], {}
    if kind == "spectrum":
        for b in config.BANDS:
            for g in archive.select(exps, b, quick=True, cap=config.SPEC_CAP_PER_BAND,
                                    quick_total=config.SPEC_QUICK_PER_BAND).values():
                for e in g:
                    todo.append(e)
                    group_of[e.uri] = b
    else:
        for k, g in archive.select(exps, band, quick=True).items():
            for e in g:
                todo.append(e)
                group_of[e.uri] = k + 1
    return todo, group_of


def leads(todo, group_of) -> list:
    """One photo per date (per band for the spectrum): what the page needs to show the spot."""
    first = {}
    for e in todo:
        first.setdefault(group_of[e.uri], e)
    return list(first.values())


def level(text: str, want: tuple) -> int:
    """How ready a spot is for these settings: 2 = its list and one photo per date are on this computer (the jump
    shows at once); 1 = its list; 0 = nothing yet. Looks at the disk only."""
    try:
        _, ra0, dec0 = _where(text)
        exps = archive.saved(round(ra0, 4), round(dec0, 4))
        if not exps:
            return 0
        todo, group_of = plan(exps, want)
        first = leads(todo, group_of)
        size = want[2] * 1.45
        return 2 if first and all(cutouts.is_cached(e, ra0, dec0, size) for e in first) else 1
    except Exception:
        return 0


def _candidates() -> list:
    spots = stock()
    return spots + [s for s in _WANT if s not in spots]


def take(want: tuple) -> str:
    """The text for /RTP: the readiest spot loaded ahead (the oldest of equals), or a fresh random one when none is
    waiting. The spot leaves the stock and its background work stops; the page loads it, sharing what is done."""
    with _LOCK:
        cands = _candidates()
        if cands:
            best = max(cands, key=lambda s: (level(s, want), -cands.index(s)))
            spots = stock()
            if best in spots:
                spots.remove(best)
                _save(spots)
            _WANT.pop(best, None)
            return best
    return targets.teleport_text(*targets.random_position())


def prepare(want: tuple, ahead: int = config.TELEPORT_AHEAD) -> None:
    """Keep `ahead` spots ready for the settings `want` (from settings()), loading in the background: first the list
    and one photo per date of every spot, then everything for the next one to be used. Returns at once."""
    with _LOCK:
        cands = _candidates()
        while len(cands) < ahead:
            cands.append(targets.teleport_text(*targets.random_position()))
        for s in cands:
            if _WANT.get(s) != want:   # new, or the view changed: load it (again) for these settings
                _WANT[s] = want
                _submit(s, want, 1)
        _submit(cands[0], want, 2)


def busy(step: int = 1) -> bool:
    """True while spots still wait for step 1 (list and one photo per date), or for any step when step=2."""
    with _LOCK:
        return bool(_BUSY[1] or (step >= 2 and _BUSY[2]))


def wait_idle(step: int, stop, poll: float = 0.5) -> None:
    """Wait (in a background thread) while busy(step), or until stop() is True: the other bands of the view wait
    for the next /RTP spots this way (see prefetch.py)."""
    while not stop() and busy(step):
        time.sleep(poll)


def aligned(key):
    """The photos lined up in the background for this view (handed over once), or None."""
    with _READY_LOCK:
        return _READY.pop(key, None)


def _submit(text: str, want: tuple, step: int, tries: int = 0) -> None:
    k = (text, want, step)
    with _LOCK:
        if k in _QUEUED or k in _DONE:
            return
        _QUEUED.add(k)
        _BUSY[step] += 1
    _JOBS.submit(_job, text, want, step, tries)


def _job(text: str, want: tuple, step: int, tries: int) -> None:
    stop = lambda: _WANT.get(text) != want   # noqa: E731  (the spot was used, or the settings changed)
    ok = False
    try:
        if not stop():
            (_first if step == 1 else _rest)(text, want, stop)
        ok = True
    except Exception:
        # e.g. NASA's archive could not be reached: try again a little later, so the spot is ready once it is back.
        if tries < RETRIES and not stop():
            def again():
                if not stop():
                    _submit(text, want, step, tries + 1)
            timer = threading.Timer(RETRY_AFTER, again)
            timer.daemon = True
            timer.start()
    finally:
        with _LOCK:
            _QUEUED.discard((text, want, step))
            _BUSY[step] -= 1
            if ok and step == 2 and not stop():
                _DONE.add((text, want, step))


def _first(text: str, want: tuple, stop) -> None:
    """Step 1: the archive's list (then the spot is in the stock) and one photo per date."""
    _, ra0, dec0 = _where(text)
    exps = archive.search(round(ra0, 4), round(dec0, 4))
    if stop():
        return
    with _LOCK:
        spots = stock()
        if not exps:   # no photos here yet: forget it, another spot is picked next time
            _WANT.pop(text, None)
            if text in spots:
                spots.remove(text)
                _save(spots)
            return
        if text not in spots and _WANT.get(text) == want:
            spots.append(text)
            _save(spots)
    todo, group_of = plan(exps, want)
    cutouts.fetch_many(leads(todo, group_of), ra0, dec0, want[2] * 1.45, background=True, stop=stop)


def _rest(text: str, want: tuple, stop) -> None:
    """Step 2: every photo of the view, lined up, and the extras the view shows."""
    kind, band, fov, gaia, survey = want
    t, ra0, dec0 = _where(text)
    ra4, dec4 = round(ra0, 4), round(dec0, 4)
    exps = archive.search(ra4, dec4)
    if not exps or stop():
        return
    size = fov * 1.45
    todo, group_of = plan(exps, want)
    stamps, _ = cutouts.fetch_many(todo, ra0, dec0, size, background=True, stop=stop)
    if kind == "view" and len(stamps) == len(todo) and not stop():   # all there: line them up as well
        groups = {}
        for e in todo:
            groups.setdefault(group_of[e.uri], []).append(e)
        wgrid, shape = stack.grid(ra0, dec0, fov)
        ready = stack.epoch_frames(stamps, groups, t, wgrid, shape)
        with _READY_LOCK:
            _READY[stack.frames_key(t.name, ra0, dec0, fov, band, groups, stamps)] = ready
            while len(_READY) > _KEEP:
                _READY.popitem(last=False)
    if stop():
        return
    if gaia:
        _quietly(catalogs.gaia_cone, ra4, dec4, round(fov / 60 * 0.75, 3))
    if survey:
        _quietly(catalogs.survey_frames, round(ra0, 5), round(dec0, 5), band, size)
    # A dense field's complete list (TAP, a little later) can pick other photos than the fast list: load again
    # then, from the cache plus what is new.
    tap = archive.complete_future(ra4, dec4)
    if tap is not None:
        def again(f):
            try:
                full = f.result()
            except Exception:
                return
            if full and {e.uri for e in full} != {e.uri for e in exps} and not stop():
                with _LOCK:
                    _DONE.discard((text, want, 2))
                _submit(text, want, 2)
        tap.add_done_callback(again)


def _quietly(fn, *args):
    try:
        fn(*args)
    except Exception:
        pass

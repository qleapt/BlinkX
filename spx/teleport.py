"""/RTP (random teleport) without the wait: the next random spots are picked ahead of time and loaded quietly in
the background (archive search, photos, lined up), so a jump lands at once.

A spot is still a uniformly random position on the sphere (targets.random_position), only chosen earlier. The
background work pauses while a load the user is waiting for runs (cutouts.fetch_many(background=True)), uses a
few connections of its own, and stops as soon as its spot is used or the view settings change. Whatever it
loaded lands in the usual caches, so a spot that was only half ready still loads faster.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import archive, catalogs, config, cutouts, stack, targets

_JOBS = ThreadPoolExecutor(1, thread_name_prefix="spx-rtp")   # one spot at a time
_READY: OrderedDict = OrderedDict()   # stack.frames_key -> (by_uri, frames) lined up in the background
_READY_LOCK = threading.Lock()
_KEEP = 3   # lined-up spots kept in memory (at most TELEPORT_AHEAD can be waiting; each is a few MB)
RETRY_AFTER = 60.0   # s: a spot whose loading failed (e.g. the archive could not be reached) is tried again
RETRIES = 10


@dataclass
class Spot:
    text: str           # what /RTP writes into the search box: "RA Dec" in degrees
    want: tuple = ()    # the settings it is loaded for (see settings()); () once it has been used
    job: object = None  # Future of the background work


def settings(mode: str, band: str, fov, gaia: bool, survey: bool) -> tuple:
    """What a spot is loaded for: exactly what the app loads in this view (the extras only where it shows them)."""
    return ("spectrum" if mode == "spectrum" else "view", band, fov,
            bool(gaia and mode in ("blink", "sub", "movie")), bool(survey and mode in ("blink", "sub", "motion")))


def take(queue: list) -> str:
    """The text for /RTP: the oldest spot loaded ahead, or a fresh random one when none is waiting."""
    if queue:
        spot = queue.pop(0)
        spot.want = ()   # used: its background work stops; the page loads it, sharing whatever is done
        return spot.text
    return targets.teleport_text(*targets.random_position())


def prepare(queue: list, want: tuple, ahead: int) -> None:
    """Keep `ahead` spots waiting, each loading in the background for the settings `want` (from settings())."""
    for spot in queue:
        if spot.want != want:   # the view changed: load the same spot again for the new settings
            spot.want = want
            spot.job = _JOBS.submit(_load, spot, want)
    while len(queue) < ahead:
        spot = Spot(targets.teleport_text(*targets.random_position()), want)
        spot.job = _JOBS.submit(_load, spot, want)
        queue.append(spot)


def aligned(key):
    """The photos lined up in the background for this view (handed over once), or None."""
    with _READY_LOCK:
        return _READY.pop(key, None)


def _load(spot: Spot, want: tuple, tries: int = 0) -> None:
    stop = lambda: spot.want != want   # noqa: E731  (the spot was used, or the settings changed)
    try:
        if not stop():
            _fill(spot, want, stop)
    except Exception:
        # e.g. NASA's archive could not be reached: try again a little later, so the spot is ready once it is
        # back. A spot that never loads ahead is simply loaded when it is used, as before.
        if tries < RETRIES and not stop():
            timer = threading.Timer(RETRY_AFTER, lambda: stop() or _JOBS.submit(_load, spot, want, tries + 1))
            timer.daemon = True
            timer.start()


def _quietly(fn, *args):
    try:
        fn(*args)
    except Exception:
        pass


def _fill(spot: Spot, want: tuple, stop) -> None:
    kind, band, fov, gaia, survey = want
    t = targets.parse_coords(spot.text)
    ra0, dec0 = targets.position_at(t, config.REFERENCE_MJD)   # exactly as the app computes the field centre
    ra4, dec4 = round(ra0, 4), round(dec0, 4)
    exps = archive.search(ra4, dec4)
    if not exps or stop():
        return
    size = fov * 1.45
    if kind == "spectrum":
        todo = [e for b in config.BANDS
                for g in archive.select(exps, b, quick=True, cap=config.SPEC_CAP_PER_BAND,
                                        quick_total=config.SPEC_QUICK_PER_BAND).values() for e in g]
        cutouts.fetch_many(todo, ra0, dec0, size, background=True, stop=stop)
    else:
        groups = {k + 1: v for k, v in archive.select(exps, band, quick=True).items()}
        todo = [e for g in groups.values() for e in g]
        stamps, _ = cutouts.fetch_many(todo, ra0, dec0, size, background=True, stop=stop)
        if len(stamps) == len(todo) and not stop():   # all there: line them up as well
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
                _JOBS.submit(_load, spot, want)
        tap.add_done_callback(again)

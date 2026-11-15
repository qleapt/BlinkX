"""Band changes without the wait: once a view is on screen, the other bands of the same spot and field are loaded
quietly in the background (photos, lined up), nearest band first, then the all-band spectrum's photos, so choosing
another band, or the spectrum, shows it at once.

Like the /RTP spots loaded ahead (teleport.py) this uses the background lane: it pauses while a load you are waiting
for runs, and it stops as soon as you move to another spot or field. Everything it loads lands in the usual caches,
so a band that was only half ready still loads faster.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from . import archive, catalogs, config, cutouts, stack

_JOBS = ThreadPoolExecutor(1, thread_name_prefix="spx-bands")
_READY: OrderedDict = OrderedDict()   # stack.frames_key -> (by_uri, frames), lined up in the background
_LOCK = threading.Lock()
_KEEP = 8          # lined-up bands kept in memory (one spot's six, and a little more)
_VIEW = [None]     # the spot and field being loaded ahead; another one calls the old work off


def view_key(name: str, ra0: float, dec0: float, fov, exps=(), survey: bool = False) -> tuple:
    """What the loading ahead is for. A longer photo list (the complete one replacing the fast one) or the older
    surveys switched on start it again; whatever is done already is not done twice."""
    return name, round(ra0, 6), round(dec0, 6), fov, len(exps), bool(survey)


def order(band: str, include: bool = False) -> list:
    """The bands to load ahead: nearest to the one shown first (it too when include, e.g. from the spectrum)."""
    bands = list(config.BANDS)
    i = bands.index(band) if band in bands else 0
    return sorted((b for b in bands if include or b != band), key=lambda b: (abs(bands.index(b) - i), bands.index(b)))


def start(tgt, ra0: float, dec0: float, fov, band: str, exps, survey: bool = False, include: bool = False) -> None:
    """Load the other bands of this view in the background (returns at once). The same view again does nothing; another
    spot or field calls the earlier work off."""
    view = view_key(tgt.name, ra0, dec0, fov, exps, survey)
    with _LOCK:
        if _VIEW[0] == view:
            return
        _VIEW[0] = view
    _JOBS.submit(_run, view, tgt, ra0, dec0, fov, order(band, include), list(exps), survey)


def stop() -> None:
    """Call the loading ahead off (nothing is being looked at, e.g. the name was not found)."""
    with _LOCK:
        _VIEW[0] = None


def groups_for(exps, band: str) -> dict:
    """The quick look's epochs for a band, numbered as the page numbers them."""
    return {k + 1: v for k, v in archive.select(exps, band, quick=True).items()}


def ready(name: str, ra0: float, dec0: float, fov, band: str, exps):
    """The quick look of `band` lined up in the background, or None (not yet, or not all its photos arrived)."""
    groups = groups_for(exps, band)
    todo = [e for g in groups.values() for e in g]
    return aligned(stack.frames_key(name, ra0, dec0, fov, band, groups, todo)) if todo else None


def aligned(key):
    """The photos lined up in the background for this key (stack.frames_key), or None. Kept for switching back."""
    with _LOCK:
        got = _READY.get(key)
        if got is not None:
            _READY.move_to_end(key)
        return got


def _keep(key, value) -> None:
    with _LOCK:
        _READY[key] = value
        _READY.move_to_end(key)
        while len(_READY) > _KEEP:
            _READY.popitem(last=False)


def _run(view, tgt, ra0, dec0, fov, bands, exps, survey) -> None:
    stale = lambda: _VIEW[0] != view   # noqa: E731  (another spot or field is being looked at now)
    size = fov * 1.45
    wgrid, shape = stack.grid(ra0, dec0, fov)
    for band in bands:   # the quick look of each band, lined up
        if stale():
            return
        groups = groups_for(exps, band)
        todo = [e for g in groups.values() for e in g]
        if not todo:
            continue
        key = stack.frames_key(tgt.name, ra0, dec0, fov, band, groups, todo)
        if aligned(key) is None:
            try:
                stamps, _ = cutouts.fetch_many(todo, ra0, dec0, size, background=True, stop=stale)
                if len(stamps) == len(todo) and not stale():
                    _keep(key, stack.epoch_frames(stamps, groups, tgt, wgrid, shape))
            except Exception:
                pass
        if survey and not stale():
            try:
                catalogs.survey_frames(round(ra0, 5), round(dec0, 5), band, size)
            except Exception:
                pass
    # then the photos of the all-band spectrum (they land in the cache; the spectrum measures them when opened)
    spec = [e for b in config.BANDS
            for g in archive.select(exps, b, quick=True, cap=config.SPEC_CAP_PER_BAND,
                                    quick_total=config.SPEC_QUICK_PER_BAND).values() for e in g]
    spec = [e for e in spec if not cutouts.is_cached(e, ra0, dec0, size)]
    if spec and not stale():
        try:
            cutouts.fetch_many(spec, ra0, dec0, size, background=True, stop=stale)
        except Exception:
            pass

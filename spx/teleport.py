"""/RTP (random teleport) without the wait: random spots are picked ahead of time and loaded quietly in the
background, so a jump lands at once, even on a slow connection or while NASA's archive is slow to answer.

A spot is a uniformly random position on the sphere (targets.random_position), only chosen earlier. config.TELEPORT_STOCK
spots are kept ready, a new set each time the app starts (the stock on disk, rtp_stock.json, is started afresh), and
every jump uses one up and starts loading a new one. The last run's spots that were never jumped to are kept aside as
spares: while the new set is still loading (just after a start, or while NASA's search is slow), a spare that is ready
is used instead of making the jump wait for the archive. Each spot is loaded in steps, so many get ready quickly:

1. the archive's list of photos and one photo per date (enough for the page to show it at once);
2. for the next config.TELEPORT_FULL spots: the rest of its photos, lined up, the first photos of every other band
   (a band change after the jump is quick too) and the extras (Gaia, older surveys).

While NASA's search services are down (archive.down()) a new random spot cannot be searched; spots searched on this
computer before (their list is saved; their photos come from AWS, which stays up) stand in, so /RTP keeps working.
Spots used recently (rtp_seen.json) are not picked again while others are left. Spots are compared by position
(4 decimals, as the saved lists are named), so a spot rebuilt from a saved list is never the one just used.

/RTP takes the readiest spot (take()). The background work pauses while a load the user is waiting for runs
(cutouts.fetch_many(background=True)), uses a few connections of its own, and a spot's work stops as soon as it is
used or the view settings change. Whatever it loaded lands in the usual caches.
"""
from __future__ import annotations

import json
import random
import re
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
SEEN_KEEP = 400      # spots remembered as used (not picked again while others are left)

_LOCK = threading.RLock()
_WANT: dict = {}      # spot text -> the settings it is being loaded for; a change (or the spot being used) stops it
_QUEUED: set = set()  # (text, want, step) waiting or running
_DONE: set = set()    # (text, want, step) finished
_PRIO: set = set()    # spots among the next TELEPORT_FULL: busy() counts only their work
_BUSY = {1: 0, 2: 0}  # priority jobs waiting or running per step
_FRESH = [False]      # the stock was started afresh in this run of the app
_SPARE: list = []     # the last run's spots never jumped to: used only while nothing of the new set is ready


def settings(mode: str, band: str, fov, gaia: bool, survey: bool) -> tuple:
    """What a spot is loaded for: exactly what the app loads in this view (the extras only where it shows them)."""
    return ("spectrum" if mode == "spectrum" else "view", band, fov,
            bool(gaia and mode in ("blink", "sub", "movie")), bool(survey and mode in ("blink", "sub", "motion")))


def _path():
    return config.cache_dir() / "rtp_stock.json"


def _seen_path():
    return config.cache_dir() / "rtp_seen.json"


def _read(path) -> list:
    try:
        return json.loads(path.read_text()).get("spots", [])
    except Exception:
        return []


def stock() -> list:
    """The spots ready to jump to (their archive list is saved), oldest first."""
    return [s for s in _read(_path()) if isinstance(s, str) and targets.parse_coords(s) is not None]


def _save(spots: list) -> None:
    try:
        _path().write_text(json.dumps({"spots": spots}))
    except OSError:
        pass


def seen() -> list:
    """Spots used lately (jumped to, or of an earlier run's stock), the longest ago first."""
    return [s for s in _read(_seen_path()) if isinstance(s, str)]


def _key(text: str):
    """A spot's position as the saved lists name it (4 decimals): the same spot whatever its text precision."""
    t = targets.parse_coords(text)
    return None if t is None else (f"{t.ra:.4f}", f"{t.dec:+.4f}")


def _keys(texts) -> set:
    return {_key(s) for s in texts}


def _remember(text: str) -> None:
    k = _key(text)
    spots = [s for s in seen() if _key(s) != k] + [text]
    try:
        _seen_path().write_text(json.dumps({"spots": spots[-SEEN_KEEP:]}))
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


def _searched(text: str) -> bool:
    """True when the spot's photo list is saved on this computer (it loads without NASA's search)."""
    try:
        _, ra0, dec0 = _where(text)
        return archive.known(round(ra0, 4), round(dec0, 4))
    except Exception:
        return False


_SAVED_NAME = re.compile(r"(?:sia|tap)_(\d+\.\d{4})_([+-]\d+\.\d{4})\.json$")


def _saved_spots() -> list:
    """Spots searched on this computer before (not the challenge targets): they need no search."""
    skip = set()
    for name in targets.PRESETS:
        ra, dec = targets.position_at(targets.preset(name), config.REFERENCE_MJD)
        skip.add((f"{ra:.4f}", f"{dec:+.4f}"))
    out = set()
    for p in config.cache_dir().glob("*_*_*.json"):
        m = _SAVED_NAME.match(p.name)
        if m and (m.group(1), m.group(2)) not in skip:
            out.add(targets.teleport_text(float(m.group(1)), float(m.group(2))))
    return sorted(out)


def _pick(taken) -> str | None:
    """A new spot: a random position on the sphere, or, while NASA's search is down, a spot searched before (not
    used recently, while others are left). None when there is nothing to pick. taken: texts (any precision)."""
    if not archive.down():
        return targets.teleport_text(*targets.random_position())
    taken = _keys(taken)
    pool = [s for s in _saved_spots() if _key(s) not in taken and archive.saved(*map(float, s.split()))]
    if not pool:
        return None
    recent = _keys(seen())
    fresh = [s for s in pool if _key(s) not in recent]
    if fresh:
        return random.choice(fresh)
    order = {_key(s): i for i, s in enumerate(seen())}   # every one used lately: the longest ago first
    return min(pool, key=lambda s: order.get(_key(s), -1))


def _start_afresh() -> None:
    """Once per run of the app: a new set of spots (those of the last run count as used)."""
    if _FRESH[0]:
        return
    _FRESH[0] = True
    used = _keys(seen())
    _SPARE[:] = [s for s in stock() if _key(s) not in used]   # never jumped to: still new to the user
    _save([])


def _candidates() -> list:
    """The spots being kept ready, without any used already (also dropped from the stock on disk)."""
    used = _keys(seen())
    spots = stock()
    keep = [s for s in spots if _key(s) not in used]
    if len(keep) != len(spots):
        _save(keep)
    for s in [s for s in _WANT if _key(s) in used]:
        _WANT.pop(s, None)
    return keep + [s for s in _WANT if s not in keep]


def _spare(want: tuple, better_than: int):
    """A spare (see the module notes) that shows at once (level 2), when nothing of the new set is that ready."""
    if better_than >= 2:
        return None
    used = _keys(seen())
    for s in list(_SPARE):
        if _key(s) in used:
            _SPARE.remove(s)
        elif level(s, want) == 2:
            _SPARE.remove(s)
            return s
    return None


def take(want: tuple) -> str:
    """The text for /RTP: the readiest spot loaded ahead (the oldest of equals), or a new one when none is waiting.
    The spot leaves the stock (it is not picked again soon) and its background work stops; the page loads it,
    sharing what is done. The next prepare() starts loading a new spot in its place."""
    with _LOCK:
        _start_afresh()
        used = _keys(seen())
        cands = [s for s in _candidates() if _key(s) not in used]
        if archive.down():   # a spot that still needs a search cannot load now
            cands = [s for s in cands if _searched(s)] or cands
        best = max(cands, key=lambda s: (level(s, want), -cands.index(s))) if cands else None
        spare = _spare(want, level(best, want) if best else -1)
        if spare is not None:   # the new set is not ready yet: a ready spare instead of a wait for the archive
            best = spare
        elif best is not None:
            spots = stock()
            if best in spots:
                spots.remove(best)
                _save(spots)
            _WANT.pop(best, None)
            _PRIO.discard(best)
        else:
            best = _pick(set(seen())) or targets.teleport_text(*targets.random_position())
        _remember(best)
        return best


def prepare(want: tuple, ahead: int = config.TELEPORT_STOCK) -> None:
    """Keep `ahead` spots ready for the settings `want` (from settings()), loading in the background: the next
    config.TELEPORT_FULL completely, first, then the list and one photo per date of every other. Returns at once."""
    with _LOCK:
        _start_afresh()
        used = set(seen())
        used_k = _keys(used)
        cands = [s for s in _candidates() if _key(s) not in used_k]
        if archive.down():   # spots searched before take their turn first (the others need the search)
            cands = [s for s in cands if _searched(s)] + [s for s in cands if not _searched(s)]
        while len(cands) < ahead:
            s = _pick(set(cands) | used | set(_SPARE))
            if s is None or _key(s) in _keys(cands):
                break
            cands.append(s)
        head = cands[:config.TELEPORT_FULL]
        _PRIO.clear()
        _PRIO.update(head)
        for s in cands:
            if _WANT.get(s) != want:   # new, or the view changed: load it (again) for these settings
                _WANT[s] = want
            _submit(s, want, 1)
            if s in head:
                _submit(s, want, 2)


def busy(step: int = 1) -> bool:
    """True while the next spots (config.TELEPORT_FULL) still wait for step 1 (list and one photo per date), or for
    any step when step=2. The rest of the stock does not hold anything back."""
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
        prio = text in _PRIO
        if prio:
            _BUSY[step] += 1
    _JOBS.submit(_job, text, want, step, tries, prio)


def _job(text: str, want: tuple, step: int, tries: int, prio: bool) -> None:
    stop = lambda: _WANT.get(text) != want   # noqa: E731  (the spot was used, or the settings changed)
    ok = False
    try:
        if not stop():
            (_first if step == 1 else _rest)(text, want, stop)
        ok = True
    except archive.ArchiveBusy:
        if _searched(text):
            _again(text, want, step, tries, stop)
        else:   # NASA's search is down and this spot was never searched: a spot searched before takes its place
            with _LOCK:
                _WANT.pop(text, None)
            threading.Thread(target=prepare, args=(want,), daemon=True).start()
    except Exception:
        _again(text, want, step, tries, stop)
    finally:
        with _LOCK:
            _QUEUED.discard((text, want, step))
            if prio:
                _BUSY[step] -= 1
            if ok and not stop():   # done for these settings: the next prepare() does not do it again
                _DONE.add((text, want, step))


def _again(text, want, step, tries, stop) -> None:
    """Try again a little later (e.g. the connection dropped), so the spot is ready once it is back."""
    if tries < RETRIES and not stop():
        def again():
            if not stop():
                _submit(text, want, step, tries + 1)
        timer = threading.Timer(RETRY_AFTER, again)
        timer.daemon = True
        timer.start()


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
            _remember(text)
            return
        if text not in spots and _WANT.get(text) == want:
            spots.append(text)
            _save(spots)
    todo, group_of = plan(exps, want)
    cutouts.fetch_many(leads(todo, group_of), ra0, dec0, want[2] * 1.45, background=True, stop=stop)


def _rest(text: str, want: tuple, stop) -> None:
    """Step 2: every photo of the view, lined up, the first photos of the other bands, and the extras."""
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
    if kind == "view" and not stop():   # every other band's first photos: a band change after the jump is quick too
        other = []
        for b in config.BANDS:
            if b != band:
                o_todo, o_group = plan(exps, (kind, b) + tuple(want[2:]))
                other += leads(o_todo, o_group)
        cutouts.fetch_many(other, ra0, dec0, size, background=True, stop=stop)
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

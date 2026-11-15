"""Find SPHEREx exposures covering a position (IRSA SIA, TAP), group them into epochs, thin."""
from __future__ import annotations

import csv
import io
import json
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, asdict

import numpy as np
import requests

from . import config


@dataclass(frozen=True)
class Exposure:
    band: str        # D1..D6
    mjd: float       # start of exposure
    mjd_end: float
    uri: str         # path below IRSA root
    version: str     # processing version, e.g. l2b-v27-2026-197

    @property
    def detector(self) -> int:
        return int(self.band[1])

    @property
    def url(self) -> str:
        return f"{config.IRSA}/{self.uri}"


class ArchiveBusy(RuntimeError):
    pass


def _adql(ra: float, dec: float) -> str:
    return (
        "SELECT p.energy_bandpassname, p.time_bounds_lower, p.time_bounds_upper, a.uri "
        f"FROM {config.ARTIFACT_TABLE} a JOIN {config.PLANE_TABLE} p ON a.planeid = p.planeid "
        f"WHERE 1 = CONTAINS(POINT('ICRS', {ra:.6f}, {dec:.6f}), p.poly) "
        "ORDER BY p.time_bounds_lower"
    )


def parse_rows(text: str) -> list[Exposure]:
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        uri = r["uri"].strip()
        if not uri.endswith(".fits"):
            continue
        band = r["energy_bandpassname"].strip().replace("SPHEREx-", "")
        m = re.search(r"(l2b(?:_[a-z]+)?-v\d+-\d{4}-\d+)", uri)  # also l2b_retry-v27-...
        lo = float(r["time_bounds_lower"])
        hi = float(r.get("time_bounds_upper") or lo)
        out.append(Exposure(band, lo, hi, uri, m.group(1) if m else ""))
    return out


# One keep-alive session for IRSA: a search after the first skips the connection set-up (TLS, ~0.5-1 s).
_IRSA = requests.Session()
_IRSA.mount("https://", requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=16))
_COPIES = ThreadPoolExecutor(12, thread_name_prefix="spx-sia")      # copies of one request (see hedged)
_COLLECTIONS = ThreadPoolExecutor(8, thread_name_prefix="spx-col")  # one task per release collection


def hedged(fn, after: float, tries: int, pool=None):
    """fn() with insurance against a slow or failed answer: when no answer came within `after` seconds, or one
    failed, another copy starts (at most `tries` in all). The first good answer wins; slower copies are left to
    finish on their own. Raises the last error when every copy failed."""
    pool = pool or _COPIES
    pending, started, last = set(), 0, None
    while True:
        if started < tries:
            pending.add(pool.submit(fn))
            started += 1
        done, pending = wait(pending, timeout=after if started < tries else None, return_when=FIRST_COMPLETED)
        for f in done:
            if f.exception() is None:
                return f.result()
            last = f.exception()
        if not pending and started >= tries:
            raise last
        if done and not pending:   # the only copy failed at once (a hiccup): breathe before the next one
            time.sleep(0.5)


def _sia_rows(col: str, ra: float, dec: float):
    r = _IRSA.get(config.SIA_URL, timeout=(6, 60), params={
        "COLLECTION": col, "POS": f"circle {ra:.6f} {dec:.6f} 0.0001", "RESPONSEFORMAT": "CSV",
        "MAXREC": 50000})
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))
    return rows, len(rows) >= 50000 or "OVERFLOW" in r.text[-500:]


def search_sia(ra: float, dec: float) -> list[Exposure]:
    """Same exposures through IRSA's SIA image search, all release collections in parallel. A collection that
    answers slowly or fails is asked again (staggered copies), so one stuck request does not hold the page."""
    futs = [_COLLECTIONS.submit(hedged, lambda c=col: _sia_rows(c, ra, dec), config.SEARCH_HEDGE_AFTER,
                                config.SEARCH_TRIES) for col in config.SIA_COLLECTIONS]
    answers = [f.result() for f in futs]
    if any(cut for _, cut in answers):
        raise RuntimeError("SIA result truncated")
    tables = [rows for rows, _ in answers]
    out = {}
    for rows in tables:
        for x in rows:
            url = x.get("access_url", "")
            if x.get("dataproduct_subtype", "science") != "science" or not url.endswith(".fits"):
                continue
            uri = url.split(config.IRSA + "/", 1)[-1].lstrip("/")
            m = re.search(r"(l2b(?:_[a-z]+)?-v\d+-\d{4}-\d+)", uri)  # also l2b_retry-v27-...
            lo = float(x["t_min"])
            out[uri] = Exposure(x["energy_bandpassname"].replace("SPHEREx-", ""), lo,
                                float(x.get("t_max") or lo), uri, m.group(1) if m else "")
    return sorted(out.values(), key=lambda e: e.mjd)


def _from_cache(path, hours: float):
    """Exposures saved by an earlier search, or None when there is none or it is too old."""
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    if time.time() - blob["t"] < hours * 3600:
        return [Exposure(**e) for e in blob["rows"]]
    return None


def _tap_path(ra, dec):
    return config.cache_dir() / f"tap_{ra:.4f}_{dec:+.4f}.json"


def _sia_path(ra, dec):
    return config.cache_dir() / f"sia_{ra:.4f}_{dec:+.4f}.json"


def is_cached(ra: float, dec: float) -> bool:
    """True when a search for this position was saved recently (it is then answered without the network).
    Only looks at the files' age, so it costs nothing on every page update."""
    now = time.time()
    for path, hours in ((_tap_path(ra, dec), config.TAP_CACHE_HOURS), (_sia_path(ra, dec), config.SIA_CACHE_HOURS)):
        try:
            if now - path.stat().st_mtime < hours * 3600:
                return True
        except OSError:
            pass
    return False


def saved(ra: float, dec: float) -> list[Exposure] | None:
    """The list saved by an earlier search of this position, however old (the complete one when there is one), or
    None. No network."""
    for path in (_tap_path(ra, dec), _sia_path(ra, dec)):
        try:
            rows = _from_cache(path, float("inf"))
        except Exception:
            rows = None
        if rows:
            return rows
    return None


_REFRESH = ThreadPoolExecutor(4, thread_name_prefix="spx-refresh")


def search(ra: float, dec: float, use_cache: bool = True) -> list[Exposure]:
    """All exposures whose footprint contains (ra, dec). Cached on disk for 12 h. Two searches for the same
    position at the same time (a /RTP spot being loaded ahead, then used) share one request. A position searched
    longer ago waits config.SEARCH_STALE_WAIT for a fresh list and otherwise uses the saved one (new photos of the
    week then appear on the next visit): NASA's archive being slow or down never holds a known spot."""
    if use_cache:
        rows = _from_cache(_tap_path(ra, dec), config.TAP_CACHE_HOURS)
        if rows is not None:
            return rows
        old = saved(ra, dec)
        if old:
            fut = _REFRESH.submit(_fresh, ra, dec, use_cache)
            try:
                return fut.result(timeout=config.SEARCH_STALE_WAIT) or old
            except Exception:   # still asking (it carries on and is saved), or the archive failed
                return old
    return _fresh(ra, dec, use_cache)


def _fresh(ra: float, dec: float, use_cache: bool) -> list[Exposure]:
    key = (f"{ra:.4f}", f"{dec:+.4f}", use_cache)
    with _PENDING_LOCK:
        fut = _SEARCHING.get(key)
        mine = fut is None
        if mine:
            fut = _SEARCHING[key] = Future()
    if not mine:
        return list(fut.result())
    try:
        rows = _search(ra, dec, use_cache)
        fut.set_result(rows)
        return list(rows)
    except BaseException as e:
        fut.set_exception(e)
        raise
    finally:
        with _PENDING_LOCK:
            _SEARCHING.pop(key, None)


def _search(ra: float, dec: float, use_cache: bool) -> list[Exposure]:
    # Fast path: SIA answers in 1-3 s but can return only part of the list in very dense fields
    # (R Draconis: 539 of 2,493), so the complete TAP search runs in the background and replaces it.
    sia_path = _sia_path(ra, dec)
    rows = _from_cache(sia_path, config.SIA_CACHE_HOURS) if use_cache else None
    if rows is None:
        try:
            rows = search_sia(ra, dec)
            sia_path.write_text(json.dumps({"t": time.time(), "rows": [asdict(e) for e in rows]}))
        except Exception:
            rows = None
    if rows:
        _complete_later(ra, dec)
        return rows
    if rows is None:   # SIA failed even after its retries: TAP, with SIA tried again alongside
        return _tap_or_sia(ra, dec)
    return search_tap(ra, dec)   # SIA found nothing here: make sure with the complete search


_BG = ThreadPoolExecutor(2)
_PENDING: set = set()
_PENDING_LOCK = threading.Lock()
_SEARCHING: dict = {}   # position -> Future of a search in progress
_TAPS: dict = {}        # position -> Future of the complete TAP list (started once per position)


def _quiet_tap(ra, dec):
    try:
        return search_tap(ra, dec)
    except Exception:
        return None


def _complete_later(ra: float, dec: float):
    """Start the complete TAP search in the background, once per position; its Future (result None on failure)."""
    key = (round(ra, 4), round(dec, 4))
    with _PENDING_LOCK:
        if key not in _PENDING:
            _PENDING.add(key)
            _TAPS[key] = _BG.submit(_quiet_tap, ra, dec)
        return _TAPS.get(key)


def _tap_or_sia(ra: float, dec: float) -> list[Exposure]:
    """SIA failed: the complete TAP search (as patient as before) while SIA is asked again every few seconds.
    The first list with photos wins; a list from SIA leaves TAP running to replace it later, as usual."""
    tap = _TAP_POOL.submit(search_tap, ra, dec)
    with _PENDING_LOCK:
        key = (round(ra, 4), round(dec, 4))
        _PENDING.add(key)
        _TAPS[key] = tap
    while True:
        try:
            return tap.result(timeout=8)
        except FutureTimeout:
            pass
        try:
            rows = search_sia(ra, dec)
        except Exception:
            continue
        if rows:
            _sia_path(ra, dec).write_text(json.dumps({"t": time.time(), "rows": [asdict(e) for e in rows]}))
            return rows


_TAP_POOL = ThreadPoolExecutor(2, thread_name_prefix="spx-tap")


def complete_future(ra: float, dec: float):
    """The background search for this position's complete (TAP) list: a Future whose result is the list (None,
    or an error, when it failed), or None when no such search was started in this session."""
    with _PENDING_LOCK:
        return _TAPS.get((round(ra, 4), round(dec, 4)))


def is_complete(ra: float, dec: float) -> bool:
    """True once the full TAP list for this position is cached."""
    return (config.cache_dir() / f"tap_{ra:.4f}_{dec:+.4f}.json").exists()


def search_tap(ra: float, dec: float) -> list[Exposure]:
    path = config.cache_dir() / f"tap_{ra:.4f}_{dec:+.4f}.json"
    last = None
    for attempt in range(4):
        try:
            r = _IRSA.get(config.TAP_URL, params={"QUERY": _adql(ra, dec), "FORMAT": "CSV"},
                          timeout=90)
            if r.status_code in (502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(2 + 3 * attempt)
                continue
            r.raise_for_status()
            rows = parse_rows(r.text)
            path.write_text(json.dumps({"t": time.time(), "rows": [asdict(e) for e in rows]}))
            return rows
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 + 3 * attempt)
    raise ArchiveBusy(last or "archive did not answer")


def group_epochs(mjds, gap: float = config.EPOCH_GAP_DAYS,
                 max_span: float = config.EPOCH_MAX_SPAN_DAYS) -> np.ndarray:
    """Epoch index per MJD. New epoch after a gap > `gap` days or when a run is longer
    than `max_span` days (dense fields near the ecliptic poles)."""
    mjds = np.asarray(mjds, float)
    order = np.argsort(mjds)
    idx = np.zeros(len(mjds), int)
    e, start, prev = 0, None, None
    for i in order:
        m = mjds[i]
        if prev is not None and (m - prev > gap or m - start > max_span):
            e += 1
            start = m
        if start is None:
            start = m
        idx[i] = e
        prev = m
    return idx


def evenly(items: list, n: int) -> list:
    """Pick n items spread evenly over the (time-sorted) list."""
    if len(items) <= n:
        return list(items)
    pos = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(pos))]


def select(exps: list[Exposure], band: str, quick: bool = True,
           cap: int = config.LOAD_ALL_CAP,
           quick_total: int = config.QUICK_MAX_TOTAL) -> dict[int, list[Exposure]]:
    """Exposures of one band grouped by epoch (epochs defined on all bands), thinned.
    Quick look: <= QUICK_PER_GROUP per epoch and <= QUICK_MAX_TOTAL overall.
    Load all: <= cap overall, spread over epochs in proportion."""
    if not exps:
        return {}
    ep = group_epochs([e.mjd for e in exps])
    groups: dict[int, list[Exposure]] = {}
    for e, k in zip(exps, ep):
        if e.band == band:
            groups.setdefault(int(k), []).append(e)
    for k in groups:
        groups[k].sort(key=lambda x: x.mjd)
    ng = max(1, len(groups))
    if quick:
        per = max(1, min(config.QUICK_PER_GROUP, quick_total // ng))
    else:
        per = max(1, cap // ng)
    return {k: evenly(v, per) for k, v in sorted(groups.items())}


def renumber(groups: dict[int, list]) -> dict[int, list]:
    return {i + 1: v for i, (_, v) in enumerate(sorted(groups.items()))}


def data_through(exps: list[Exposure]) -> tuple[float | None, list[str]]:
    if not exps:
        return None, []
    return max(e.mjd for e in exps), sorted({e.version.split("-")[1] for e in exps if e.version})

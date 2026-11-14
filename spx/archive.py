"""Find SPHEREx exposures covering a position (IRSA TAP), group them into epochs, thin."""
from __future__ import annotations

import csv
import io
import json
import re
import time
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
        m = re.search(r"(l2b-v\d+-\d{4}-\d+)", uri)
        lo = float(r["time_bounds_lower"])
        hi = float(r.get("time_bounds_upper") or lo)
        out.append(Exposure(band, lo, hi, uri, m.group(1) if m else ""))
    return out


def search(ra: float, dec: float, use_cache: bool = True) -> list[Exposure]:
    """All exposures whose footprint contains (ra, dec). Cached on disk for 12 h."""
    path = config.cache_dir() / f"tap_{ra:.4f}_{dec:+.4f}.json"
    if use_cache and path.exists():
        blob = json.loads(path.read_text())
        if time.time() - blob["t"] < config.TAP_CACHE_HOURS * 3600:
            return [Exposure(**e) for e in blob["rows"]]
    last = None
    for attempt in range(4):
        try:
            r = requests.get(config.TAP_URL, params={"QUERY": _adql(ra, dec), "FORMAT": "CSV"},
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

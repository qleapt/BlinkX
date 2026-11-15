"""How long the loading will take: learned on this computer, then corrected by the real speed.

Before the first photo arrives the estimate uses what earlier visits measured (seconds per photo that had
to be downloaded, kept in the cache folder as timing.json = {"per_photo": seconds}); photos already saved on
this computer take almost no time. Once photos arrive, the real speed of this download takes over step by step.
Steps without a count (the archive search) use how long they usually took: timing.json also holds
{"steps": {"search": seconds}}.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PER_PHOTO = 0.6   # s per photo to download, before anything was learned (parallel downloads)
CACHED_SECONDS = 0.03     # s per photo already saved on this computer
ALIGN_SECONDS = 0.04      # s per photo to line it up and average
DEFAULT_STEP = {"search": 4.0, "resolve": 1.5, "fetch": 6.0, "motion": 0.8}   # s, before anything was learned


class Estimator:
    def __init__(self, path: Path | str | None = None):
        if path is None:
            from . import config
            path = config.cache_dir() / "timing.json"
        self.path = Path(path)
        self.per_photo, self.known, self.steps = DEFAULT_PER_PHOTO, False, {}
        try:
            blob = json.loads(self.path.read_text())
        except Exception:
            blob = {}
        try:
            v = float(blob.get("per_photo", 0))
            if 0 < v < 120 and blob.get("known", True):
                self.per_photo, self.known = v, True
            self.steps = {k: float(x) for k, x in (blob.get("steps") or {}).items() if 0 < float(x) < 600}
        except (TypeError, ValueError, AttributeError):
            pass
        self.prior = 0.0

    def start(self, new: int, cached: int) -> float:
        """Seconds the download should take, before it starts."""
        self.prior = new * self.per_photo + cached * CACHED_SECONDS
        return self.prior

    def update(self, done: int, total: int, elapsed: float) -> float | None:
        """Seconds left, from the expectation and (more and more) from the speed seen so far; None while nothing
        has arrived although the expected time is clearly over (no false promise)."""
        if done >= total:
            return 0.0
        if done < 3 and elapsed > 1.2 * self.prior + 2:
            return None
        left = max(0.0, self.prior - elapsed)
        if done >= 3 and elapsed > 0.5:
            seen = (total - done) * elapsed / done
            w = min(1.0, done / max(5.0, 0.2 * total))
            left = w * seen + (1 - w) * left
        return left

    def learn(self, new: int, seconds: float) -> None:
        """Remember how long the photos that had to be downloaded took (a running average)."""
        if new < 3 or seconds <= 0:
            return
        v = seconds / new
        self.per_photo = 0.5 * self.per_photo + 0.5 * v if self.known else v
        self.known = True
        self._save()

    def step_left(self, kind: str, elapsed: float) -> float | None:
        """Seconds a step (e.g. 'search') usually still needs, or None once it takes clearly longer than usual."""
        usual = self.steps.get(kind, DEFAULT_STEP.get(kind, 5.0))
        if elapsed > 1.3 * usual + 1:
            return None
        return max(0.0, usual - elapsed)

    def usual(self, kind: str) -> float:
        """How long a step usually takes on this computer (learned), in seconds."""
        return self.steps.get(kind, DEFAULT_STEP.get(kind, 5.0))

    def learn_step(self, kind: str, seconds: float) -> None:
        if seconds <= 0:
            return
        old = self.steps.get(kind)
        self.steps[kind] = seconds if old is None else 0.5 * old + 0.5 * seconds
        self._save()

    def _save(self):
        try:
            self.path.write_text(json.dumps({"per_photo": self.per_photo, "known": self.known, "steps": self.steps}))
        except OSError:
            pass


def words(seconds: float | None) -> str:
    """'about 25 s left', 'about 2 min left', 'almost done', or 'taking longer than usual' (None)."""
    if seconds is None:
        return "taking longer than usual"
    if seconds < 5:
        return "almost done"
    if seconds < 90:
        s = int(round(seconds)) if seconds < 20 else 5 * int(round(seconds / 5))
        return f"about {s} s left"
    return f"about {int(round(seconds / 60))} min left"

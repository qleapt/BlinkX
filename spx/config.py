"""All archive endpoints, tables, bands, flags and limits live here.

This is the only file to edit when IRSA changes an endpoint or a new
SPHEREx release changes the table names.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

# --- Archive -------------------------------------------------------------------
IRSA = "https://irsa.ipac.caltech.edu"
TAP_URL = f"{IRSA}/TAP/sync"
# Fast exposure search: IRSA's SIA v2 image search, one call per release collection, in parallel
# (~1-3 s instead of 6-15 s for TAP). Add a new release's collection name here. TAP is the fallback.
SIA_URL = f"{IRSA}/SIA"
SIA_COLLECTIONS = ["spherex_dr1", "spherex_qr3"]
ARTIFACT_TABLE ="spherex.artifact"
PLANE_TABLE = "spherex.plane"
# SIMBAD name lookup: main site first, then the official mirrors (the main site was down on 2026-11-14)
SIMBAD_TAPS = [
    "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
    "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync",
    "https://simbad.idia.ac.za/simbad/sim-tap/sync",
]
GAIA_TAP ="https://gea.esac.esa.int/tap-server/tap/sync"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
GAIA_MAX_G = 19.0
GAIA_TOP = 800
GAIA_REF_MJD = 57388.5  # Gaia DR3 epoch 2016.0
# 'Check my find': Gaia positions are for 2016; the fastest star (Barnard's, 10.4"/yr) has moved ~2'
# since, so search this far around a mark, then match within FIND_MATCH_ARCSEC on the mark's date.
FIND_CONE_DEG = 0.05
FIND_MATCH_ARCSEC = 6.0
# Older surveys shown before the SPHEREx frames (IRSA SIA collections; cutouts come back gzipped)
SURVEY_MATCH = {
    "D1": {"2mass": "J"}, "D2": {"2mass": "J"}, "D3": {"2mass": "K"},
    "D4": {"wise": "W1"}, "D5": {"wise": "W2"}, "D6": {"wise": "W2"},
}
WISE_CRYO_END_MJD = 55414.0  # 2010-08-06, end of the 4-band cryogenic survey
SPECTROPHOTOMETRY_TOOL = "https://irsa.ipac.caltech.edu/applications/spherex/tool-spectrophotometry"

# --- Instrument ----------------------------------------------------------------
# band: (min um, max um, resolving power)
BANDS = {
    "D1": (0.75, 1.09, 39),
    "D2": (1.10, 1.62, 41),
    "D3": (1.63, 2.41, 41),
    "D4": (2.42, 3.82, 35),
    "D5": (3.83, 4.41, 112),
    "D6": (4.42, 5.00, 128),
}
PIXEL_SCALE_ARCSEC = 6.15
OVERSAMPLE = 2  # display grid is 2x finer than the detector

# Flag names that make a pixel unusable. SOURCE (a known star) is NOT bad.
BAD_FLAGS = [
    "TRANSIENT", "OVERFLOW", "SUR_ERROR", "NONFUNC", "MISSING_DATA", "HOT", "COLD",
    "NONLINEAR", "PERSIST", "OUTLIER", "STREAK", "BLOOM", "SNOWBALL",
]
# Always removed. The other BAD_FLAGS only get SUSPECT_WEIGHT, so a clean exposure wins,
# but a saturated star flagged in every exposure is still shown (with a warning).
DEAD_FLAGS = ["NONFUNC", "MISSING_DATA", "HOT", "COLD"]
SUSPECT_WEIGHT = 1e-6
# Flags worth reporting near the target (saturation-like): shown, not hidden.
WARN_FLAGS = ["OVERFLOW", "NONLINEAR", "PERSIST", "BLOOM", "TRANSIENT", "OUTLIER"]

# --- Limits --------------------------------------------------------------------
# Fast path: the same SPHEREx files on NASA's public AWS bucket, read with HTTP byte ranges
# (only the rows we need; no throttling). IRSA cutouts remain the fallback.
S3_BASE = "https://nasa-irsa-spherex.s3.us-east-1.amazonaws.com"
USE_S3 = True
S3_CONCURRENT = 24
MAX_CONCURRENT = 4         # 8 parallel gave HTTP 503
HEDGE_AFTER = 5.0          # s: also ask IRSA when the AWS read is this slow (distant/slow links)
SEARCH_HEDGE_AFTER = 2.5   # s: ask the SIA search again when an answer is this slow (it usually takes 1-3 s,
                           # but now and then 7-30 s: a second request then usually answers first)
SEARCH_TRIES = 3           # SIA requests per collection (staggered) before falling back to TAP
SEARCH_RETRY_SIA = 6.0     # s: while the complete search (TAP) runs, a failed quick search (SIA) is asked again
ARCHIVE_DOWN_FOR = 60.0    # s: after NASA's search services failed, new searches try once quickly (a few seconds) instead
                           # of retrying for half a minute; spots searched before never wait for them (saved list)
# /RTP: random spots are picked ahead of time and loaded quietly in the background, so a jump lands at once (see
# teleport.py). Background loading pauses while a load you are waiting for runs, and uses few connections.
TELEPORT_STOCK = 20        # /RTP spots kept ready: a new set each time the app starts, one new spot loaded per jump
TELEPORT_FULL = 3          # of those, the next ones loaded completely (all photos lined up, every band's first photos)
BG_STAMPS = 3              # photos loaded side by side in the background
BG_S3_CONCURRENT = 6       # AWS requests at once for background loading (the page you look at gets the rest)
FETCH_SOFT_DEADLINE = 1.5  # s: then show what has arrived if every epoch has an image (one photo per epoch is
                           # loaded first); the rest keep arriving and the page fills them in by itself
PROBE_TIMEOUT = 45.0       # s: the one quick search while NASA's search counts as down (an outage fails at once)
LOADER_AFTER = 1.0         # s: the loading screen only appears for a wait: NASA's archive asked, or a page this slow
WAIT_EXTRAS = 0.5          # s: the photos never wait longer for the Gaia list or the older surveys (added on arrival)
REFRESH_EVERY = 1.0        # s: how often the page checks whether the photos still on their way have arrived
RETRIES = 12
REQUEST_TIMEOUT = 60        # seconds
EPOCH_GAP_DAYS = 45         # a gap longer than this starts a new epoch
EPOCH_MAX_SPAN_DAYS = 60    # dense fields: split long runs into bins
QUICK_PER_GROUP = 4         # quick look: exposures per band x epoch
QUICK_MAX_TOTAL = 24
LOAD_ALL_CAP = 80           # hard cap per band per "Load all"
MOVIE_CAP = 60              # every-exposure movie
SPEC_QUICK_PER_BAND = 12    # spectrum quick look: exposures per band (all six bands)
SPEC_CAP_PER_BAND = 40      # spectrum "Load all"
FOV_CHOICES = (3, 5, 10, 15)  # arcmin
TAP_CACHE_HOURS = 12
SIA_CACHE_HOURS = 12        # fast exposure list; the complete TAP list replaces it when ready
REFERENCE_MJD = 61041.0     # 2026-01-01: cutouts are centred on the target at this date
CACHE_VERSION = 1           # bump to invalidate processed stamps
_SEED_LOCK = threading.Lock()
_SEEDED: set[Path] = set()


def _seed_cache(p: Path) -> None:
    """Populate a fresh cache with bundled index JSONs so challenge targets and /RTP work if IRSA is down."""
    if p in _SEEDED or "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        return
    with _SEED_LOCK:
        if p in _SEEDED:
            return
        seed = Path(__file__).resolve().parent / "data" / "seed_cache.zip"
        if seed.exists():
            try:
                with zipfile.ZipFile(seed) as z:
                    for name in z.namelist():
                        dst = p / Path(name).name
                        if not dst.exists():
                            dst.write_bytes(z.read(name))
            except OSError:
                pass
        _SEEDED.add(p)


def cache_dir() -> Path:
    """Cutout cache: $SPHEREX_BLINK_CACHE, else %LOCALAPPDATA%/spherex-blink/cache, else temp."""
    env = os.environ.get("SPHEREX_BLINK_CACHE")
    if env:
        p = Path(env)
    elif os.environ.get("LOCALAPPDATA"):
        p = Path(os.environ["LOCALAPPDATA"]) / "spherex-blink" / "cache"
    else:
        p = Path(tempfile.gettempdir()) / "spherex-blink-cache"
    p.mkdir(parents=True, exist_ok=True)
    _seed_cache(p)
    return p
        p = Path(tempfile.gettempdir()) / "spherex-blink-cache"
    p.mkdir(parents=True, exist_ok=True)
    return p

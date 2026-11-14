"""All archive endpoints, tables, bands, flags and limits live here.

This is the only file to edit when IRSA changes an endpoint or a new
SPHEREx release changes the table names.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# --- Archive -------------------------------------------------------------------
IRSA = "https://irsa.ipac.caltech.edu"
TAP_URL = f"{IRSA}/TAP/sync"
ARTIFACT_TABLE = "spherex.artifact"
PLANE_TABLE = "spherex.plane"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
GAIA_MAX_G = 19.0
GAIA_TOP = 800
GAIA_REF_MJD = 57388.5  # Gaia DR3 epoch 2016.0
SPECTROPHOTOMETRY_TOOL ="https://irsa.ipac.caltech.edu/applications/spherex/tool-spectrophotometry"

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
MAX_CONCURRENT = 3          # 8 parallel gave HTTP 503; 4 still gets "busy" often
RETRIES = 12
REQUEST_TIMEOUT = 60        # seconds
EPOCH_GAP_DAYS = 45         # a gap longer than this starts a new epoch
EPOCH_MAX_SPAN_DAYS = 60    # dense fields: split long runs into bins
QUICK_PER_GROUP = 4         # quick look: exposures per band x epoch
QUICK_MAX_TOTAL = 24
LOAD_ALL_CAP = 80           # hard cap per band per "Load all"
MOVIE_CAP = 60              # every-exposure movie
FOV_CHOICES = (3, 5, 10, 15)  # arcmin
TAP_CACHE_HOURS = 12
REFERENCE_MJD = 61041.0     # 2026-01-01: cutouts are centred on the target at this date
CACHE_VERSION = 1           # bump to invalidate processed stamps


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
    return p

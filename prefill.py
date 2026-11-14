"""Fill the local cache for the five challenge targets so a demo loads instantly.

Usage:  python prefill.py            (default 5' field)
        python prefill.py 10         (also fill the 10' field)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from spx import archive, catalogs, config, cutouts, photometry, targets


def prefill(fov: float = 5.0):
    size = fov * 1.45
    for name, t in targets.PRESETS.items():
        t0 = time.time()
        ra, dec = targets.position_at(t, config.REFERENCE_MJD)
        exps = archive.search(round(ra, 4), round(dec, 4))
        band = targets.PRESET_BAND[name]
        blink = [e for g in archive.select(exps, band).values() for e in g]
        spec = [e for b in config.BANDS
                for g in archive.select(exps, b, quick_total=config.SPEC_QUICK_PER_BAND).values() for e in g]
        todo = list({e.uri: e for e in blink + spec}.values())
        stamps, fails = cutouts.fetch_many(todo, ra, dec, size)
        try:
            stars, src = catalogs.gaia_cone(round(ra, 4), round(dec, 4), round(fov / 60 * 0.75, 3))
        except Exception as e:  # optional overlay
            stars, src = [], f"Gaia failed: {e}"
        try:
            old = catalogs.survey_frames(round(ra, 5), round(dec, 5), band, size)
        except Exception as e:
            old = [f"survey failed: {e}"]
        rows = [photometry.measure(s, *targets.position_at(t, s.mjd)) for s in stamps if s.band == band]
        flux = np.nanmedian([r["flux"] for r in rows]) if rows else np.nan
        flagged = sum(bool(r["flags"]) for r in rows)
        epochs = len(set(archive.group_epochs([e.mjd for e in exps])))
        print(f"{name}: {len(exps)} exposures, {epochs} epochs; fetched {len(stamps)}/{len(todo)} "
              f"(failed {fails}); {band} median flux {flux:.3g} mJy, {flagged}/{len(rows)} flagged; "
              f"Gaia {len(stars)} ({src}); older frames "
              f"{[o if isinstance(o, str) else o['survey'] + ' ' + o['band'] for o in old]}; {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    for f in (sys.argv[1:] or ["5"]):
        prefill(float(f))

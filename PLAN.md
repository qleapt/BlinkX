# blinkX — design and build plan

NASA Space Apps Challenge 2026 · SPHEREx challenge · built 14–15 Nov 2026

**Status: complete.** All milestones below (M0–M6) are done, plus the extras listed in section 7. This file records
how the tool was planned, what was measured on the archive before writing code, and why the design looks the way it
does. For using the tool, see [README.md](README.md) and [TUTORIAL.md](TUTORIAL.md).

## 1. What we are building

A public web tool (Streamlit, Python). A user types a sky position or an object name. The tool fetches small SPHEREx
image cutouts on demand and plays them like a short video, so moving or varying objects stand out.

Scope agreed by the team on 14 Nov:

- Run locally, keep it ready for Streamlit Community Cloud (one `requirements.txt`, no secrets).
- Core: blink animation, add / subtract, band picker, field-of-view picker, URL parameters, flags, Gaia overlay.
- Extras, in this order: spectra and light curves, WISE 2010 / 2MASS frames, motion check against Gaia.
- Not in scope: automatic mover detection, user accounts, bulk downloads.

## 2. Verified facts about the archive (measured 14 Nov 2026)

### Finding exposures (no bulk download)

- Endpoint: `https://irsa.ipac.caltech.edu/TAP/sync` with `QUERY=<ADQL>` and `FORMAT=CSV`. Takes 5–12 s per position.
- Query that works:

```sql
SELECT p.energy_bandpassname, p.time_bounds_lower, a.uri
FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid
WHERE 1 = CONTAINS(POINT('ICRS', 133.795129, -7.245147), p.poly)
ORDER BY p.time_bounds_lower
```

- Example row: `SPHEREx-D6,60800.700028,ibe/data/spherex/dr1/level2/2025W19_1B/l2b-v27-2026-197/6/level2_2025W19_1B_0116_1D6_spx_l2b-v27-2026-197.fits`
- Other useful `spherex.plane` columns: `obsid`, `productid`, `time_bounds_upper`, `time_exposure`,
  `energy_bounds_lower/upper` (metres), `metrics_maglimit`, `quality_flag`, `provenance_version`.
- These tables hold both current releases (DR1 and QR3), so new weekly data appears without code changes. Newest
  exposure seen: MJD 61315 (2 Oct 2026).
- Python: `from astroquery.ipac.irsa import Irsa` then `Irsa.query_tap(adql)`. The module `astroquery.irsa` named in
  the original guide no longer exists.

### Coverage of the five challenge targets

| Target | Exposures | Per band D1…D6 | Dates | Epochs |
|---|---|---|---|---|
| WISE 0855−0714 | 258 | 36/48/45/36/48/45 | 5 May 2025 – 14 May 2026 | May 2025 (thin: 3–8 per band), Nov 2025 (plus a few in Dec), Apr–May 2026 |
| Barnard's Star | 358 | 60/61/59/59/61/58 | 15 Aug 2025 – 22 Sep 2026 | about 3 |
| Epsilon Indi Ba/Bb | 465 | 48/93/99/45/84/96 | 24 Apr 2025 – 2 Oct 2026 | about 4 |
| RW Aurigae | 335 | 56/55/56/56/56/56 | 1 Sep 2025 – 16 Sep 2026 | about 3 |
| R Draconis | 2,493 | 584/89/574/584/89/573 | 24 Apr 2025 – 27 Sep 2026 | nearly continuous (close to the north ecliptic pole) |

Bands D1/D4, D2/D5 and D3/D6 share a pointing, so their counts match.

### Cutouts

- URL: `https://irsa.ipac.caltech.edu/<uri>?center=<ra>,<dec>&size=<degrees>` (units such as `size=5arcmin` are accepted).
- A 5′ cutout is 49×49 px and the response is 3,985,920 bytes. A 15′ cutout is 4,285,440 bytes. Size barely depends
  on field of view because a 3.86 MB PSF table is always included. There is no parameter to drop it, the `Range`
  header is ignored, and there is no gzip.
- HDU order: primary, `IMAGE` (float32, MJy/sr), `FLAGS` (int32), `VARIANCE` (float32), `ZODI` (float32, modelled
  zodiacal light, not subtracted from IMAGE), `EPSF` (binary table, 8752 bytes × 441 rows), `WCS-WAVE` (wavelength
  lookup table).
- For a 5′ cutout the first five HDUs end at byte **112,320**. Everything we need for images is in the first 3% of
  the response.
- **Fast path:** stream the response, parse HDU by HDU (2880-byte blocks, header until `END`, data size from
  `BITPIX`/`NAXISn`), stop when the `EPSF` header appears, close the connection. The first 300 kB arrive in about
  2 s; the full 4 MB takes about 10 s.
- **Throttle:** 8 parallel requests gave four immediate HTTP 503. Use at most 4 concurrent requests and retry 503
  with backoff and jitter.
- Wavelength at a pixel (from the IRSA tutorial) needs the `WCS-WAVE` HDU, which comes after `EPSF`:

```python
spatial_wcs = WCS(header)                      # header of the IMAGE HDU
x, y = spatial_wcs.world_to_pixel(coord)
spectral_wcs = WCS(header, fobj=hdul, key="W"); spectral_wcs.sip = None
wavelength, bandwidth = spectral_wcs.pixel_to_world(x, y)   # micrometres
```

  Solution used: one `WCS-WAVE` table per detector is shipped in `spx/data/wcswave_D1…D6_v27.fits` and attached to
  the truncated cutouts, so the wavelength is known without downloading the PSF table.
- Flag names are written in each file's header (`MSKN…` keywords): TRANSIENT, OVERFLOW, SUR_ERROR, NONFUNC, DICHROIC,
  MISSING_DATA, HOT, COLD, FULLSAMPLE, PHANMISS, NONLINEAR, PERSIST, OUTLIER, CROSSTALK, SOURCE, GHOST, GHOST_FPA,
  GHOST_EXT, STREAK, BLOOM, SNOWBALL, HALO. `SOURCE` marks a known source and must not be treated as bad.

### Instrument

| Band | Wavelength (µm) | Resolving power |
|---|---|---|
| D1 | 0.75–1.09 | 39 |
| D2 | 1.10–1.62 | 41 |
| D3 | 1.63–2.41 | 41 |
| D4 | 2.42–3.82 | 35 |
| D5 | 3.83–4.41 | 112 |
| D6 | 4.42–5.00 | 128 |

Pixel scale 6.15″. Each exposure of the same sky position is at a different wavelength inside the band (linear
variable filter).

## 3. Corrections to the original data guide

1. Use `astroquery.ipac.irsa`, not `astroquery.irsa`.
2. Cutouts are plain URLs on files found by the TAP query. There is no astroquery cutout call.
3. "Subtract the older image from the newer one" only works per band on aligned stacks. Single exposures differ in
   wavelength and in pixel grid.
4. Catalog coordinates are old. Barnard's Star has moved about 4.5′ since J2000, so it is outside a 5′ box centred on
   the SIMBAD position. Always move the target to the observation date using proper motion.
5. Motion between SPHEREx epochs is smaller than one 6.15″ pixel: Barnard's Star about 5.2″, WISE 0855 about 4.1″,
   Epsilon Indi B about 2.4″ per six months. So we need difference images, a 2× oversampled display, centroid
   tracking, and WISE 2010 frames (WISE 0855 has moved about 130″, 21 pixels, since 2010).
6. Gaia has no entry for WISE 0855 (too faint and red). Fall back to SIMBAD or preset values for proper motion and
   parallax.
7. A larger field of view costs almost no extra transfer. The number of exposures is what costs time.

## 4. Design

Approach chosen: Streamlit + IRSA cutout URLs with the partial-stream fast path + disk cache. Considered and set
aside: direct byte-range reads from the full files or the AWS mirror (more moving parts for a similar gain), and a
JavaScript front end with a FastAPI back end (better embedding, about twice the work).

```
name or RA/Dec ─► targets.resolve ─► position at the observation date (proper motion applied)
       │
archive.search (TAP) ─► exposure table (band, MJD, URI) ─► epochs (gap > 45 days)
       │
cutouts.fetch (≤4 parallel, partial stream, disk cache) ─► Stamp(image, flags, variance, zodi, WCS, wavelength)
       │
stack.align (zodi and local background removed, reprojected to one north-up grid, 2× oversampled)
       ├─► stack per band × epoch ─► Blink | Add | Subtract ─► sky viewer, GIF export
       ├─► photometry per exposure ─► Spectrum (flux vs wavelength) | Light curve (flux vs time)
       └─► centroid per epoch ─► Motion check against the predicted track
overlays: Gaia DR3 moved to each frame's date, flag mask, predicted position, scale bar, N/E
```

Rules:

- **Quick look first.** Load at most 4 exposures per band × epoch nearest the band centre (about 12 cutouts), show
  the blink, then "Load all" fills in. Dense fields (R Dra) are thinned per time bin with a hard cap per load.
- **Cache.** Processed stamps as compressed `.npz` in `cache/`, keyed by a hash of URI, centre and size. TAP results
  cached for 12 hours. A new processing version has a new URI, so it refreshes by itself.
- **Bad pixels** are masked using a configurable list of flag names (default: TRANSIENT, OVERFLOW, SUR_ERROR,
  NONFUNC, MISSING_DATA, HOT, COLD, NONLINEAR, PERSIST, OUTLIER, STREAK, BLOOM, SNOWBALL).
- **Stacks** are inverse-variance weighted means with flagged pixels excluded.
- **Two kinds of motion.** Stars move between epochs: blink the stacks. Asteroids and comets move within hours: the
  "every exposure" movie shows single exposures ordered by time or by wavelength.
- **Subtract** is (epoch B − epoch A) in one band with a diverging colour map and symmetric limits.
- **Photometry.** Circular aperture r = 2 px, background annulus 4–7 px, error from `VARIANCE`, MJy/sr × pixel solid
  angle → mJy. Points with bad flags inside the aperture are drawn hollow. Labelled "quick look, not calibrated",
  with a link to the IRSA Spectrophotometry Tool.
- **URL parameters.** `?target=…` or `?ra=…&dec=…`, plus `fov` (arcmin), `band` (D1–D6), `mode`. Read with
  `st.query_params`, validated and clamped, written back by the widgets. Embedding: add `&embed=true` inside an iframe.
- **Easy to update.** URIs come from the archive. `spx/config.py` is the only place with endpoints, table names, the
  band table, the bad-flag list and limits.

## 5. Files

See the *Code* section of [README.md](README.md) for the current list. Main pieces:

```
app.py              Streamlit user interface
prefill.py          fill the cache for the five challenge targets
spx/config.py       endpoints, tables, bands, bad flags, limits (the only file to edit for a new release)
spx/targets.py      coordinate parsing, SIMBAD lookup, five presets, proper-motion propagation
spx/archive.py      TAP exposure search, epoch grouping, thinning
spx/cutouts.py      throttled partial-stream fetch, FITS parsing, wavelength, disk cache
spx/stack.py        background removal, reprojection, stacks, add / subtract
spx/photometry.py   aperture photometry, spectra, light curves, centroids
spx/catalogs.py     Gaia cone search and propagation, WISE / 2MASS frames
spx/render.py       stretch, colour maps, charts, GIF export
spx/viewer/         browser sky viewer (frontend/index.html)
spx/data/           WCS-WAVE wavelength tables, one per detector
tests/              pytest on synthetic data
```

## 6. Milestones

Every milestone ended with tests green, the app running, and a `git commit`. Each stop point is a complete, demoable
tool.

| Milestone | Content | Status |
|---|---|---|
| M0 — Setup | Python 3.12 virtual environment, `requirements.txt`, `.gitignore`, project skeleton | ✅ |
| M1 — Core blink viewer | config, target lookup with proper motion, TAP search and epochs, partial-stream cutouts, aligned stacks, blink animation, URL parameters. **Stop point 1: the core challenge objective is met.** | ✅ |
| M2 — Add / subtract, flags, Gaia | Blink / Add / Subtract / Every exposure, flag overlay, exposures table, Gaia DR3 overlay with VizieR fallback, GIF export | ✅ |
| M3 — Spectra and light curves | Flux against wavelength for all six bands coloured by epoch; flux against time per band | ✅ |
| M4 — WISE 2010 and 2MASS frames | AllWISE W1/W2 and 2MASS J/H/Ks cutouts reprojected to the same grid, shown first in the blink (J→D2, H and Ks→D3, W1→D4, W2→D6) | ✅ |
| M5 — Motion check | Centroid per epoch, predicted track from proper motion and parallax, measured against predicted motion with a plain verdict | ✅ |
| M6 — Polish and release | Five-target run, first-run help, friendly errors, cache prefill, README, tutorial, MIT license, GitHub upload | ✅ |

## 7. Added after the plan

- **Sky viewer** in the browser: brightness, contrast and stretch change instantly, zoom and pan, real-date timeline,
  Compare (blink two dates, side by side, swipe, difference).
- **Check my find** and the motion engine (`spx/motion.py`): joint fit of all marks, model choice
  (still / drift / drift + parallax / curved), leave-one-out tests, classification and path prediction; a second
  object can be compared with the first.
- **/RTP**: random sky spots kept ready on disk and loaded in the background.
- **Prefetch** of the other bands, a full-window loading screen and a time-left estimate.
- Interface translations (`spx/i18n.py`).

## 8. Acceptance on the five challenge targets

| Target | What we expect to show |
|---|---|
| Barnard's Star | Inside the frame when chosen by name. Shift of about one pixel per epoch, a clear pair in Subtract, motion check close to Gaia. |
| Epsilon Indi B | The pair (unresolved) visible; small shift per epoch in the direction of Epsilon Indi A's motion; A itself visible at 10′ or more. |
| WISE 0855 | Detected in the D6 stack (it is faint, W2 ≈ 13.9), absent in D1–D3, large jump against WISE 2010. |
| RW Aurigae | Spectrum and light curve per epoch with error bars; any change reported. |
| R Draconis | Dense light curve from hundreds of thinned exposures. Saturation and non-linearity flags are shown, not hidden. |

These are quick-look results from a hackathon tool, not calibrated measurements.

## 9. Risks and how they were handled

| Risk | Level | What we did |
|---|---|---|
| Archive slow or busy (503, 4 parallel) | High | Partial-stream fast path, quick look first, disk cache, pre-filled cache for the five targets |
| Motion is sub-pixel per epoch | High | 2× oversampling, Subtract mode, WISE 2010 frames, motion check |
| Reusing the wavelength table across files does not hold | Medium | Checked against the files' own tables; tables shipped per detector in `spx/data/` |
| Wavelength mix differs between epochs (thin May 2025 epoch) | Medium | Wavelength coverage of each stack shown; warning when stacks are not comparable |
| Bright stars saturate (R Dra, Barnard's Star, Epsilon Indi A) | Medium | Flags shown on frames and on spectrum points |
| Dense fields (R Dra: 2,493 exposures) | Medium | Thinning and a hard cap per load |
| SIMBAD position epoch unclear for some objects (WISE 0855) | Medium | Presets for the five targets; catalog and predicted positions both shown |
| Gaia archive or SIMBAD slow or down | Low | VizieR fallback for Gaia, mirror fallback for SIMBAD, overlays optional |

## 10. Use of AI

An AI coding assistant was used to write code from this plan; the team set the scope, made the decisions and tested
the tool. Details in the *Use of AI* section of [README.md](README.md).

## 11. Sources

- SPHEREx data access: https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-access
- SPHEREx data products: https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-products
- Cutout tutorial (TAP query, wavelength code): https://caltech-ipac.github.io/irsa-tutorials/spherex-cutouts
- Intro tutorial (SIA query, flags): https://caltech-ipac.github.io/irsa-tutorials/spherex-intro
- Cutout parameters: https://irsa.ipac.caltech.edu/ibe/cutouts.html
- SPHEREx cutout capabilities: https://irsa.ipac.caltech.edu/data/SPHEREx/docs/cutout_tool.html
- SPHEREx data tools and dates: https://spherex.caltech.edu/page/data-tools
- Ultracool dwarfs with SPHEREx (detection limits): https://arxiv.org/abs/2604.22012
- Space Apps submission guide: https://www.spaceappschallenge.org/resources/project-submission-guide/

# SPHEREx Blink Viewer — implementation plan

NASA Space Apps 2026 · written Sat 2026-11-14 12:40 (UTC+3) · **must be finished Sun 2026-11-15 13:00 local**

Status: waiting for team approval. No code exists yet.

## 1. What we are building

A public web tool (Streamlit, Python). A user types a sky position or an object name. The tool fetches small SPHEREx image cutouts on demand and plays them like a short video, so moving or varying objects stand out.

Decisions made by the team on 14 Nov:

- Claude writes the code. The team tests and steers.
- Run locally first. Upload to GitHub together at the end. Keep it ready for Streamlit Community Cloud (one `requirements.txt`, no secrets).
- Core: blink animation, add / subtract, band picker, field-of-view picker, URL parameters, flags, Gaia overlay.
- Extras, in this order: spectra and light curves, WISE 2010 / 2MASS frames, motion check against Gaia.
- Not in scope: automatic mover detection, user accounts, bulk downloads.
- The token budget is limited (Pro plan, extra usage off). Work lean (section 8).

## 2. Verified facts about the archive (measured 2026-11-14 from this PC)

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
- Other useful `spherex.plane` columns: `obsid`, `productid`, `time_bounds_upper`, `time_exposure`, `energy_bounds_lower/upper` (metres), `metrics_maglimit`, `quality_flag`, `provenance_version`.
- These tables hold both current releases (DR1 and QR3), so new weekly data appears without code changes. Newest exposure seen: MJD 61315 (2 Oct 2026).
- Python: `from astroquery.ipac.irsa import Irsa` then `Irsa.query_tap(adql)`. The module `astroquery.irsa` named in the original guide no longer exists.

### Coverage of the five challenge targets

| Target | Exposures | Per band D1…D6 | Dates | Epochs |
|---|---|---|---|---|
| WISE 0855−0714 | 258 | 36/48/45/36/48/45 | 5 May 2025 – 14 May 2026 | May 2025 (thin: 3–8 per band), Nov 2025 (plus a few in Dec), Apr–May 2026 |
| Barnard's Star | 358 | 60/61/59/59/61/58 | 15 Aug 2025 – 22 Sep 2026 | about 3 (confirm) |
| Epsilon Indi Ba/Bb | 465 | 48/93/99/45/84/96 | 24 Apr 2025 – 2 Oct 2026 | about 4 (confirm) |
| RW Aurigae | 335 | 56/55/56/56/56/56 | 1 Sep 2025 – 16 Sep 2026 | about 3 (confirm) |
| R Draconis | 2,493 | 584/89/574/584/89/573 | 24 Apr 2025 – 27 Sep 2026 | nearly continuous (close to the north ecliptic pole) |

Bands D1/D4, D2/D5 and D3/D6 share a pointing, so their counts match.

### Cutouts

- URL: `https://irsa.ipac.caltech.edu/<uri>?center=<ra>,<dec>&size=<degrees>` (units such as `size=5arcmin` are accepted).
- A 5′ cutout is 49×49 px and the response is 3,985,920 bytes. A 15′ cutout is 4,285,440 bytes. Size barely depends on field of view because a 3.86 MB PSF table is always included. There is no parameter to drop it, the `Range` header is ignored, and there is no gzip.
- HDU order: primary, `IMAGE` (float32, MJy/sr), `FLAGS` (int32), `VARIANCE` (float32), `ZODI` (float32, modelled zodiacal light, not subtracted from IMAGE), `EPSF` (binary table, 8752 bytes × 441 rows), `WCS-WAVE` (wavelength lookup table).
- For a 5′ cutout the first five HDUs end at byte **112,320**. Everything we need for images is in the first 3% of the response.
- **Fast path:** stream the response, parse HDU by HDU (2880-byte blocks, header until `END`, data size from `BITPIX`/`NAXISn`), stop when the `EPSF` header appears, close the connection. The first 300 kB arrived in 2 s; the full 4 MB takes about 10 s (0.4 MB/s per stream from here).
- **Throttle:** 8 parallel requests gave four immediate HTTP 503. Use at most 4 concurrent requests and retry 503 with backoff and jitter.
- Wavelength at a pixel (from the IRSA tutorial) needs the `WCS-WAVE` HDU, which comes after `EPSF`:

```python
spatial_wcs = WCS(header)                      # header of the IMAGE HDU
x, y = spatial_wcs.world_to_pixel(coord)
spectral_wcs = WCS(header, fobj=hdul, key="W"); spectral_wcs.sip = None
wavelength, bandwidth = spectral_wcs.pixel_to_world(x, y)   # micrometres
```

  Plan: download one full cutout per detector once, cache its `WCS-WAVE` HDU, and attach it to truncated files. Check first that this gives the same wavelength as the file's own table (section 10). If not, fall back to full downloads for spectra only.
- Flag names are written in each file's header (`MSKN…` keywords). Names seen in the tutorial: TRANSIENT, OVERFLOW, SUR_ERROR, NONFUNC, DICHROIC, MISSING_DATA, HOT, COLD, FULLSAMPLE, PHANMISS, NONLINEAR, PERSIST, OUTLIER, CROSSTALK, SOURCE, GHOST, GHOST_FPA, GHOST_EXT, STREAK, BLOOM, SNOWBALL, HALO. `SOURCE` marks a known source and must not be treated as bad.

### Instrument

| Band | Wavelength (µm) | Resolving power |
|---|---|---|
| D1 | 0.75–1.09 | 39 |
| D2 | 1.10–1.62 | 41 |
| D3 | 1.63–2.41 | 41 |
| D4 | 2.42–3.82 | 35 |
| D5 | 3.83–4.41 | 112 |
| D6 | 4.42–5.00 | 128 |

Pixel scale 6.15″. Each exposure of the same sky position is at a different wavelength inside the band (linear variable filter).

### This PC

Windows 11, UTC+3. `git`, `node`, `winget` present. **No Python, pip, uv or conda.** No GitHub CLI.

## 3. Corrections to the original data guide

1. Use `astroquery.ipac.irsa`, not `astroquery.irsa`.
2. Cutouts are plain URLs on files found by the TAP query. There is no astroquery cutout call.
3. "Subtract the older image from the newer one" only works per band on aligned stacks. Single exposures differ in wavelength and in pixel grid.
4. Catalog coordinates are old. Barnard's Star has moved about 4.5′ since J2000, so it is outside a 5′ box centred on the SIMBAD position. Always move the target to the observation date using proper motion.
5. Motion between SPHEREx epochs is smaller than one 6.15″ pixel: Barnard's Star about 5.2″, WISE 0855 about 4.1″, Epsilon Indi B about 2.4″ per six months. So we need difference images, a 2× oversampled display, centroid tracking, and WISE 2010 frames (WISE 0855 has moved about 130″, 21 pixels, since 2010).
6. Gaia has no entry for WISE 0855 (too faint and red). Fall back to SIMBAD or preset values for proper motion and parallax.
7. A larger field of view costs almost no extra transfer here. The number of exposures is what costs time.

## 4. Design

Approach chosen: Streamlit + IRSA cutout URLs with the partial-stream fast path + disk cache. Considered and set aside: direct byte-range reads from the full files or the AWS mirror (more moving parts for a similar gain), and a JavaScript front end with a FastAPI back end (better embedding, about twice the work).

```
name or RA/Dec ─► targets.resolve ─► position at the observation date (proper motion applied)
       │
archive.search (TAP) ─► exposure table (band, MJD, URI) ─► epochs (gap > 45 days)
       │
cutouts.fetch (≤4 parallel, partial stream, disk cache) ─► Stamp(image, flags, variance, zodi, WCS, wavelength)
       │
stack.align (zodi and local background removed, reprojected to one north-up grid, 2× oversampled)
       ├─► stack per band × epoch ─► Blink | Add | Subtract ─► Plotly animation, GIF export
       ├─► photometry per exposure ─► Spectrum (flux vs wavelength) | Light curve (flux vs time)
       └─► centroid per epoch ─► Motion check against the predicted track
overlays: Gaia DR3 moved to each frame's date, flag mask, predicted position, scale bar, N/E
```

Rules:

- **Quick look first.** Load at most 4 exposures per band × epoch nearest the band centre (about 12 cutouts), show the blink, then "Load all" fills in. Dense fields (R Dra) are thinned per time bin with a hard cap per load.
- **Cache.** Processed stamps as compressed `.npz` in `cache/`, keyed by a hash of URI, centre and size. TAP results cached for 12 hours. A new processing version has a new URI, so it refreshes by itself.
- **Bad pixels** are masked using a configurable list of flag names (default: TRANSIENT, OVERFLOW, SUR_ERROR, NONFUNC, MISSING_DATA, HOT, COLD, NONLINEAR, PERSIST, OUTLIER, STREAK, BLOOM, SNOWBALL).
- **Stacks** are inverse-variance weighted means with flagged pixels excluded.
- **Two kinds of motion.** Stars move between epochs: blink the stacks. Asteroids and comets move within hours: the "every exposure" movie shows single exposures ordered by time or by wavelength.
- **Subtract** is (epoch B − epoch A) in one band with a diverging colour map and symmetric limits.
- **Photometry.** Circular aperture r = 2 px, background annulus 4–7 px, error from `VARIANCE`, MJy/sr × pixel solid angle → mJy. Points with bad flags inside the aperture are drawn hollow. Label it "quick look, not calibrated" and link the IRSA Spectrophotometry Tool.
- **URL parameters.** `?target=…` or `?ra=…&dec=…`, plus `fov` (arcmin), `band` (D1–D6), `mode` (blink, add, sub, movie, spectrum). Read with `st.query_params`, validated and clamped, written back by the widgets. Embedding: add `&embed=true` inside an iframe.
- **Easy to update.** URIs come from the archive. `spx/config.py` is the only place with endpoints, table names, the band table, the bad-flag list and limits. The footer shows "data through <date>" and the processing versions seen.

## 5. Files

```
app.py              Streamlit UI: sidebar inputs, tabs Blink | Spectrum | Motion | Exposures
spx/config.py       endpoints, tables, bands, bad flags, limits (the only file to edit for a new release)
spx/targets.py      coordinate parsing, SIMBAD lookup, five presets, proper-motion propagation
spx/archive.py      TAP exposure search, epoch grouping, thinning
spx/cutouts.py      throttled partial-stream fetch, FITS parsing, wavelength, disk cache
spx/stack.py        background removal, reprojection, stacks, add / subtract
spx/photometry.py   aperture photometry, spectra, light curves, centroids
spx/catalogs.py     Gaia cone search and propagation, WISE / 2MASS frames
spx/render.py       stretch, colour maps, Plotly figures, GIF export
tests/              pytest on synthetic data, plus opt-in live checks (-m live)
requirements.txt  README.md  .gitignore  .streamlit/config.toml  .claude/launch.json
```

About 1,500–1,800 lines of Python.

## 6. Milestones

Every milestone ends with tests green, the app running, and a `git commit`. Each stop point is a complete, demoable tool.

### M0 — Setup (needs team OK for the installs)

- `winget install --id astral-sh.uv -e`, then `uv venv --python 3.12` and `uv pip install -r requirements.txt` (streamlit, astropy, astroquery, reproject, numpy, scipy, pillow, plotly, requests, pytest). Everything stays inside the project `.venv`.
- `git init`, `.gitignore` (`.venv/`, `cache/`, `__pycache__/`), skeleton, `.claude/launch.json` running `uv run streamlit run app.py --server.port 8501 --server.headless true`.
- Check: `uv run python -c "import astropy, astroquery, reproject, streamlit, plotly"`.

### M1 — Core blink viewer

1. `config.py`.
2. `targets.py`. Test: Barnard's Star moved from J2000 to 2026.0 shifts +269″ ± 3″ in Dec.
3. `archive.py`. Tests: epoch grouping on the WISE 0855 date pattern gives 3 epochs; thinning respects the cap.
4. `cutouts.py`. Tests: the stream reader stops before `EPSF` on an in-memory MEF; a mocked 503 is retried. Live check: one WISE 0855 D6 cutout is 49×49 with wavelength inside 4.42–5.00 µm.
5. `stack.py`. Tests: two synthetic exposures with rotated WCS align to better than 0.2 px; masked pixels are ignored.
6. `render.py` and `app.py`: sidebar (target box, five presets, field of view 3/5/10/15′, band, stretch), Plotly animation of epoch stacks with play/pause, slider and hover value, frame label with date range, band and number of exposures, progress bar while loading, URL parameters.

You will see: Barnard's Star and WISE 0855 blinking across three epochs at `http://localhost:8501/?target=Barnard's+Star`.

**Stop point 1: the core challenge objective is met.**

### M2 — Add / subtract, flags, exposures table, Gaia

- Modes: Blink, Add, Subtract (pick epochs A and B), Every exposure (by time or by wavelength).
- Flag overlay toggle. Exposures table: date, band, wavelength, flags in the aperture, link to the file.
- Gaia DR3 cone search with `astroquery.gaia`, falling back to VizieR `I/355/gaiadr3`. Markers moved to each frame's date. Hover card with G magnitude, proper motion, parallax and distance.
- GIF export.

You will see: a positive/negative pair at Barnard's Star in Subtract mode, and Gaia markers that follow the stars.

**Stop point 2.**

### M3 — Spectra and light curves

- `photometry.py`. Spectrum tab: flux against wavelength for all six bands, coloured by epoch (loads the remaining bands with a progress bar). Light curve: flux against time per band.
- Load the `dataviz` skill before writing chart code.

You will see: WISE 0855 bright only near 4.5–5 µm; RW Aur and R Dra compared between epochs.

**Stop point 3.**

### M4 — WISE 2010 and 2MASS frames

- IRSA image search (`Irsa.query_sia`) for AllWISE W1/W2 and 2MASS J/H/Ks tiles, same `?center&size` cutout, reprojected to our grid. Band matching: J→D2, H and Ks→D3, W1→D4, W2→D6.
- These frames come first in the blink, labelled with their year.

You will see: WISE 0855 jumping about 130″ between 2010 and 2026.

**Stop point 4.**

### M5 — Motion check

- Centroid per epoch stack and per WISE frame. Predicted track from proper motion and parallax (astropy, Earth position). Plot of offsets against time. Card: measured against predicted motion, with a plain verdict.

**Stop point 5.**

### M6 — Polish and release (start no later than Sun 10:30)

- Run all five targets end to end (section 7) and fix what breaks.
- First-run help text ("what am I looking at?") and friendly errors: name not found, no coverage, archive busy.
- Fill the cache for the five targets so the demo is instant.
- README: what it is, how to run, URL parameters, embed snippet, data sources, how to update for a new release, AI-use note.
- Together: create the GitHub repository and push. Optional: deploy on Streamlit Community Cloud.

## 7. Acceptance on the five challenge targets

| Target | What we expect to show |
|---|---|
| Barnard's Star | Inside the frame when chosen by name. Shift of about one pixel per epoch, a clear pair in Subtract, motion check close to Gaia. |
| Epsilon Indi B | The pair (unresolved) visible; small shift per epoch in the direction of Epsilon Indi A's motion; A itself visible at 10′ or more. |
| WISE 0855 | Detected in the D6 stack (it is faint, W2 ≈ 13.9), absent in D1–D3, large jump against WISE 2010. |
| RW Aurigae | Spectrum and light curve per epoch with error bars; any change reported. |
| R Draconis | Dense light curve from hundreds of thinned exposures. Saturation and non-linearity flags are likely and must be shown, not hidden. |

These are expectations, not promises. If something is not detectable in the data, the README says so.

## 8. Working rules (token budget)

- One session builds. No sub-agents and no multi-agent workflows.
- Write each module in one go. Batch shell commands. Do not re-read a file that was just written. Keep command output short.
- About 15 tests, only for the maths and the stream parser. Live archive checks are opt-in.
- Check the app through page text. One screenshot per milestone at most.
- Load a skill only when its step starts: `dataviz` (charts), `superpowers:systematic-debugging` (when stuck), `superpowers:verification-before-completion` (before saying "done"), `ecc:documentation-lookup` (API details), `run` (launching the app).
- If a milestone runs over, cut scope and finish the stop point. Order is value order.

## 9. Risks

| Risk | Level | What we do |
|---|---|---|
| Token budget runs out before the deadline | High | Lean rules above; value-ordered stop points; fresh session at lower effort |
| Archive slow or busy (503, 4 parallel) | High | Partial-stream fast path, quick look first, disk cache, pre-filled cache for the five targets |
| Motion is sub-pixel per epoch | High | 2× oversampling, Subtract mode, WISE 2010 frames, motion check |
| Reusing the wavelength table across files does not hold | Medium | Verify early; fall back to full downloads for spectra only |
| Wavelength mix differs between epochs (thin May 2025 epoch) | Medium | Show the wavelength coverage of each stack; warn when stacks are not comparable |
| Bright stars saturate (R Dra, Barnard's Star, Epsilon Indi A) | Medium | Show flags on frames and on spectrum points |
| Dense fields (R Dra: 2,493 exposures) | Medium | Thinning and a hard cap per load |
| SIMBAD position epoch unclear for some objects (WISE 0855) | Medium | Presets for the five targets; show catalog and predicted positions |
| Gaia archive slow | Low | VizieR fallback, overlay is optional |

## 10. To verify while building

- Wavelength from a cached `WCS-WAVE` equals the file's own, on at least 3 exposures per detector.
- Which flag bits mean "bad" (Explanatory Supplement, Table 16).
- Current `astroquery.simbad` field names for proper motion and parallax.
- SIA collection names for AllWISE and 2MASS, and that their files accept `?center&size`.
- Streamlit details: `st.query_params`, Plotly selection events, embed mode.
- Reference epoch of the WISE 0855 position (the designation suggests about 2010).
- Approximate preset values to confirm from SIMBAD: Barnard's Star PM (−0.80, +10.36) ″/yr, parallax 0.547″; Epsilon Indi B PM (+3.96, −2.54) ″/yr, parallax 0.275″; WISE 0855 PM (−8.12, +0.68) ″/yr, parallax 0.439″.

## 11. Schedule (local time, UTC+3)

Usage limits reset in 5-hour windows; the weekly limit resets after the deadline.

| When | Work |
|---|---|
| Sat 12:45–14:30 | M0, start M1 |
| Sat 14:30–19:30 | finish M1, M2, start M3 |
| Sat 19:30–00:30 | M3, M4, M5 |
| Sun morning | M6: five-target run, polish, README. Feature freeze 10:30 |
| Sun 11:30–12:30 | GitHub upload together |
| Sun 13:00 | finished |

Space Apps submissions close Sun 23:59 local. Check the official Project Submission Guide for the demo format and the AI-use disclosure.

## 12. Sources

- SPHEREx data access: https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-access
- SPHEREx data products: https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-products
- Cutout tutorial (TAP query, wavelength code): https://caltech-ipac.github.io/irsa-tutorials/spherex-cutouts
- Intro tutorial (SIA query, flags): https://caltech-ipac.github.io/irsa-tutorials/spherex-intro
- Cutout parameters: https://irsa.ipac.caltech.edu/ibe/cutouts.html
- SPHEREx cutout capabilities: https://irsa.ipac.caltech.edu/data/SPHEREx/docs/cutout_tool.html
- SPHEREx data tools and dates: https://spherex.caltech.edu/page/data-tools
- Ultracool dwarfs with SPHEREx (detection limits): https://arxiv.org/abs/2604.22012
- Space Apps submission guide: https://www.spaceappschallenge.org/resources/project-submission-guide/

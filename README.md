# SPHEREx Blink Viewer

**NASA Space Apps Challenge 2026 · SPHEREx challenge.** Type an object name or a sky position and watch
SPHEREx images of that patch of sky play as a short movie. Stars that move (like Barnard's Star) or change
brightness (like R Draconis) stand out, just like the "blink comparators" astronomers used to find Pluto.

**New here? Start with the [tutorial](TUTORIAL.md):** five short lessons with real targets, no astronomy needed.

The tool fetches small image cutouts on demand from the NASA/IPAC Infrared Science Archive (IRSA). Nothing is
bulk-downloaded, and new SPHEREx data appear automatically as the archive is updated each week.

## What you can do

| View | What it shows |
|---|---|
| **Blink epochs** | One averaged image per survey pass (epoch), aligned north-up, played as an animation. Older **2MASS (~2000)** and **WISE (2010)** images come first, so fast movers jump visibly. |
| **Subtract (B − A)** | Newer minus older epoch. A moving star leaves a light/dark pair; a variable star turns light or dark. |
| **Every exposure** | Single exposures ordered by time or by wavelength (for asteroids and comets, which move within hours). |
| **Spectrum & light curve** | Brightness in all six bands (0.75–5 µm) against wavelength, coloured by epoch, and against time. A plain-language "what changed?" summary. |
| **Motion check** | Where the target's light is centred in each frame, against the track predicted from its catalogue (Gaia) proper motion and parallax. Then the motion engine fits all positions together (still / drift / drift + parallax / curved), estimates what kind of object it is with the evidence listed, and predicts the path for the coming years with an uncertainty cone. |

All image views draw in one **sky viewer** in the browser: brightness, contrast and stretch change instantly
(the page receives the pixel values, not pictures), zoom with the wheel, drag to move, a timeline with the real
dates, and **Compare**: blink two dates, side by side with a linked crosshair, a swipe divider, or the difference.
Overlays: the predicted position of the target on each date (orange), Gaia DR3 stars moved to each date (blue,
hover for brightness, motion and distance; off until you tick them) and flagged pixels (pink, e.g. saturation).

**You find things; the tools measure them.** Nothing is pointed out by default. Mark what you see (the mark stays exactly
where you click, even on dark sky; *Snap to star centre* is optional): the viewer gives positions, distances, directions and speed between dates, a card with that
spot on every date and its brightness, and real sizes from a distance. **Check my find** analyses *all* the marks
of one object together (`spx/motion.py`): each is measured on its own photo, one path is fitted through all of them,
every pair is compared and every mark is tested against the others (leave-one-out), Gaia is checked at each mark's
date, and the object is classified (standing still, moving star, asteroid / comet, or not one object) with the
reasons. The path is predicted years ahead (hours for an asteroid), on the sky and against time, with a table.
Mark a **second object** on the same photos (for example a star next to an asteroid on two exposures 14 minutes
apart) and put its marks under *object B*: both get their own check, and the app compares them: their distance on
each photo, how fast B moves relative to A and in which direction, whether they drift apart or close in, whether
they move together, and how far apart they will be. Optional **helpers**, off until switched on: star detection, and
*Find what changed* (moved / brightened / faded, strongest first, allowing for all stars brightening together).
Each view can be downloaded (GIF, CSV).

## Run it

Needs Python 3.10+ (tested with 3.12).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
.venv/bin/python -m streamlit run app.py         # opens http://localhost:8501
```

Optional, before a demo: `python prefill.py` downloads everything the five challenge targets need, so they load
instantly. Tests: `python -m pytest -q` (offline, synthetic data).

The cache lives in `%LOCALAPPDATA%\spherex-blink\cache` on Windows, else in the system temp folder; set
`SPHEREX_BLINK_CACHE` to choose another folder.

The app is ready for [Streamlit Community Cloud](https://streamlit.io/cloud): one `requirements.txt`, no secrets.

## Link to a view (URL parameters)

| Parameter | Meaning | Example |
|---|---|---|
| `target` | object name (SIMBAD) or preset | `?target=Barnard's Star` |
| `ra`, `dec` | position in degrees (instead of `target`) | `?ra=269.452&dec=4.693` |
| `fov` | field of view in arcmin: 3, 5, 10, 15 | `&fov=10` |
| `band` | SPHEREx band D1 … D6 | `&band=D6` |
| `mode` | `blink`, `sub`, `movie`, `spectrum`, `motion` | `&mode=sub` |
| `gaia`, `old` | `0` skips loading Gaia stars / older surveys | `&old=0` |
| `lang` | page language: `en` (English) or `tr` (Türkçe); also in the sidebar | `&lang=tr` |

The page writes its current settings back to the address bar, so you can copy and share any view.

Embed in another page:

```html
<iframe src="https://YOUR-APP.streamlit.app/?target=WISE+0855&band=D6&fov=10&embed=true"
        width="100%" height="900" style="border:0"></iframe>
```

## How it works

1. **Find the images.** IRSA's SIA image search lists the SPHEREx exposures covering the target in 1–3 s
   (one call per release, in parallel). Because SIA can return only part of the list in very dense fields, the
   complete ADQL query to IRSA's TAP service (`spherex.plane` joined with `spherex.artifact`) is asked at the
   same time and replaces it as soon as it arrives; when one of the two services is failing, the other one's
   answer is used, and the loading screen says so after 8 s. The position is first moved to 2026 with the
   proper motion: Barnard's Star has moved 4.5′ since 2000.
2. **Group into epochs.** A gap of more than 45 days starts a new epoch; dense fields near the ecliptic poles
   (R Draconis) are split into 60-day bins. A quick look loads up to 4 exposures per epoch; "Load all" adds more.
3. **Fetch only the pixels we need.** The same files are mirrored on NASA's public AWS bucket
   (`nasa-irsa-spherex`). The app reads the FITS headers and then only the rows around the target, with HTTP
   byte-range requests, up to 24 at a time (the compressed FLAGS layer is decoded row by row). That gives exactly
   the same pixels as an IRSA cutout (checked) without the archive's "server busy" throttling. If AWS fails, it
   falls back to IRSA cutouts, streaming each one and hanging up once the image layers have arrived. The
   wavelength table is identical for all exposures of a detector (checked); the six tables ship with the app in
   `spx/data`. Gaia stars and the older WISE / 2MASS images download in the background at the same time.
   Everything is cached on disk. A target never seen before loads in about 7–10 s (≈ 65 s before these changes);
   the all-band spectrum in about 13 s. Each photo takes three round trips to AWS (the places of the VARIANCE
   and ZODI layers follow from the file's length, learned once per detector), and the connections open while the
   archive search runs. For `/RTP` twenty random spots are kept ready on disk (`rtp_stock.json`, a new set each
   time the app starts): first the archive's list and one photo per date of each, then everything for the next
   three, before the other bands' far end and the spectrum. A jump takes the readiest spot and lands at once,
   even on a slow connection; until the new set is ready, a ready spot from the last run that was never jumped
   to stands in, a spot just visited never comes back, and a spot without photos jumps on to another. That
   background work pauses while a load you are waiting for runs. A spot whose dates all have a photo opens with
   no loading screen and fills in the rest quietly; "Load all" waits once and shows every photo together. A spot searched more than 12 h ago waits at most 2 s for NASA's archive and otherwise uses
   its saved list, so a slow or failing archive never holds a known spot. Once a view is on screen, the other five bands of the same spot are loaded and
   lined up the same way (nearest band first, then the spectrum's photos) and handed to the browser, so a band
   chosen in the sidebar appears in the same frame. A spot never seen before is shown as soon as every date
   has one photo (photos already on disk first, then one per date): the rest are filled in by themselves, and
   Gaia and the older surveys never hold the page for more than half a second. The loading screen only appears
   when NASA's archive is actually asked; it counts the time left down every second (learned on this computer,
   corrected by the speed seen). The Motion check shows one progress bar for the whole check instead: download,
   measuring the positions, fitting the path.
4. **Align and stack.** Zodiacal light and the local sky are removed, each exposure is reprojected onto one
   north-up grid at 2× the detector resolution (3.1″ pixels), and the exposures of an epoch are averaged with
   inverse-variance weights. Dead pixels are removed; flagged pixels (saturation, cosmic rays…) get a tiny weight,
   so a clean exposure always wins but a saturated star is still shown, with a warning.
5. **Measure.** Aperture photometry (radius 2 pixels, sky annulus 4–7 pixels, errors from the variance layer,
   MJy/sr × pixel solid angle → mJy) and centroids for the motion check. These are quick-look numbers, not
   calibrated; for science use the [IRSA SPHEREx Spectrophotometry Tool](https://irsa.ipac.caltech.edu/applications/spherex/tool-spectrophotometry).

## The five challenge targets

What the tool shows with the data available on 14 Nov 2026 (quick look, 5′ field unless noted). Click a preset
in the sidebar to reproduce.

| Target | Result |
|---|---|
| **Barnard's Star** | 358 exposures, 3 epochs. The star moves about 5″ north per epoch (less than one SPHEREx pixel) and is centred within ~1″ of its Gaia prediction each time. **Subtract** shows a clear light/dark pair. Motion check: 11.0 ± 5.3″/yr against Gaia's 10.39″/yr (agrees; the star is saturated, which limits the precision). With a 10′ field the 2010 WISE frame shows it ~2.8′ further south. |
| **Epsilon Indi B** | 458 exposures, 4 epochs; the Ba/Bb pair is seen as one source (~21 mJy in D6). Motion check including the April 2010 WISE frame: **4.71 ± 0.04″/yr, the catalogue says 4.71″/yr**. Epsilon Indi A is ~6.7′ away: choose the 15′ field to see both. |
| **WISE 0855−0714** | Detected in D6 at its predicted position (~0.5 mJy, matching its WISE W2 = 13.9 mag). A Gaia star (G = 16.3) is 4–11″ away on the SPHEREx dates; it dominates D1–D3, and the tool says so. In the May 2010 WISE frame the brown dwarf is detected at 25σ, **127″ east** of where it is now: a big jump in the 10′ blink. |
| **RW Aurigae** | 335 exposures, 3 epochs; spectrum per epoch with error bars, and a change report per band. The star is bright enough that its pixels are flagged in every exposure (shown as hollow points), so the numbers are indicative: D5/D6 come out ~50% fainter in 2026 than in 2025. |
| **R Draconis** | 2,493 exposures, split into 9 time bins of ≤ 60 days. The centre of this very bright Mira variable is saturated in most exposures (60 of 68 in the quick look); the tool reports this instead of plotting wrong numbers, and shows a brightness index from the star's wings, which follows its variation. |

These are quick-look results from a hackathon tool, not calibrated measurements.

## Updating for a new SPHEREx release

Nothing to do for new weekly data: the archive query always returns the latest files, and a new processing
version has new file names, so the cache refreshes by itself. If IRSA renames tables or endpoints, edit
`spx/config.py`, the only file that holds endpoints, table names, the band table, flag lists and limits.

## Code

```
app.py              Streamlit user interface
prefill.py          fill the cache for the five challenge targets
spx/config.py       endpoints, tables, bands, flags, limits
spx/targets.py      name / coordinate lookup, presets, proper motion and parallax
spx/archive.py      TAP search, epoch grouping, thinning
spx/cutouts.py      throttled partial-stream download, FITS parsing, wavelength, cache
spx/stack.py        background removal, reprojection, stacks, subtract, survey frames
spx/photometry.py   aperture photometry, centroids, motion fit
spx/catalogs.py     Gaia DR3 (Gaia archive, VizieR fallback), WISE 2010 and 2MASS frames
spx/render.py       stretches, charts (spectrum, light curve, motion, miss distance), GIF export
spx/viewer/         the sky viewer: pixel values packed into PNGs, two-way component (frontend/index.html)
spx/detect.py       optional helpers: star detection, what changed between two dates
spx/motion.py       motion engine: joint fit of N positions, model choice, leave-one-out, classification, prediction
spx/teleport.py     /RTP: a stock of random spots kept ready on disk, loaded quietly in the background
spx/prefetch.py     the other bands of the view loaded and lined up ahead, so a band change is instant
spx/warp.py         full-window loading screen (canvas star field)
spx/eta.py          time-left estimate while loading, learned on this computer
tests/              pytest on synthetic data
```

## Data and credits

- SPHEREx Level 2 images: NASA/IPAC Infrared Science Archive (IRSA), Caltech/IPAC, NASA.
- Gaia DR3: European Space Agency, Gaia mission, via the ESA Gaia archive or CDS VizieR (I/355).
- WISE All-Sky single-exposure images and 2MASS images: IRSA.
- Target names: SIMBAD, CDS Strasbourg (the app falls back to the official CfA and IDIA mirrors when the main
  site is down, as it was on 14 Nov 2026).

## License

MIT, see [LICENSE](LICENSE). The SPHEREx, WISE, 2MASS and Gaia data the app shows are not part of this
repository: they are fetched from their archives and keep their own terms (see *Data and credits*).

## Use of AI

This tool was written with Claude (Anthropic) as a coding assistant, working from a plan the team approved.
The team chose the scope and tested the tool; the archive measurements behind the design (cutout layout,
throttling, wavelength tables) were checked live against IRSA during the build.

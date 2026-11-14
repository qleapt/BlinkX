# SPHEREx Blink Viewer

**NASA Space Apps Challenge 2026 · SPHEREx challenge.** Type an object name or a sky position and watch
SPHEREx images of that patch of sky play as a short movie. Stars that move (like Barnard's Star) or change
brightness (like R Draconis) stand out, just like the "blink comparators" astronomers used to find Pluto.

The tool fetches small image cutouts on demand from the NASA/IPAC Infrared Science Archive (IRSA). Nothing is
bulk-downloaded, and new SPHEREx data appear automatically as the archive is updated each week.

## What you can do

| View | What it shows |
|---|---|
| **Blink epochs** | One averaged image per survey pass (epoch), aligned north-up, played as an animation. Older **2MASS (~2000)** and **WISE (2010)** images come first, so fast movers jump visibly. |
| **Add** | All epochs averaged: the deepest image. |
| **Subtract (B − A)** | Newer minus older epoch. A moving star leaves a light/dark pair; a variable star turns light or dark. |
| **Every exposure** | Single exposures ordered by time or by wavelength (for asteroids and comets, which move within hours). |
| **Spectrum & light curve** | Brightness in all six bands (0.75–5 µm) against wavelength, coloured by epoch, and against time. A plain-language "what changed?" summary. |
| **Motion check** | Where the target's light is centred in each frame, against the track predicted from its catalogue (Gaia) proper motion and parallax, with a verdict. |

Overlays: the predicted position of the target on each date (orange), Gaia DR3 stars moved to each date (cyan,
hover for brightness, motion and distance), and pixels the pipeline flagged (pink, e.g. saturation).
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
| `mode` | `blink`, `add`, `sub`, `movie`, `spectrum`, `motion` | `&mode=sub` |
| `gaia`, `old` | `0` hides Gaia stars / older surveys | `&old=0` |

The page writes its current settings back to the address bar, so you can copy and share any view.

Embed in another page:

```html
<iframe src="https://YOUR-APP.streamlit.app/?target=WISE+0855&band=D6&fov=10&embed=true"
        width="100%" height="900" style="border:0"></iframe>
```

## How it works

1. **Find the images.** An ADQL query to IRSA's TAP service (`spherex.plane` joined with `spherex.artifact`) lists
   every SPHEREx exposure whose footprint contains the target. The position is first moved to 2026 with the
   proper motion: Barnard's Star has moved 4.5′ since 2000.
2. **Group into epochs.** A gap of more than 45 days starts a new epoch; dense fields near the ecliptic poles
   (R Draconis) are split into 60-day bins. A quick look loads up to 4 exposures per epoch; "Load all" adds more.
3. **Fetch cutouts fast.** A cutout is ~4 MB, but 97% is a PSF table we don't need. The app streams the response,
   reads the FITS headers block by block and hangs up once the image, flags, variance and zodiacal-light layers
   have arrived (~100 kB). The wavelength table is identical for all exposures of a detector (checked), so it is
   downloaded once per detector. Requests are limited to 3 at a time and "server busy" answers are retried after
   the delay the archive asks for. Everything is cached on disk.
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
spx/stack.py        background removal, reprojection, stacks, add / subtract, survey frames
spx/photometry.py   aperture photometry, centroids, motion fit
spx/catalogs.py     Gaia DR3 (Gaia archive, VizieR fallback), WISE 2010 and 2MASS frames
spx/render.py       stretches, Plotly animation and charts, GIF export
tests/              pytest on synthetic data
```

## Data and credits

- SPHEREx Level 2 images: NASA/IPAC Infrared Science Archive (IRSA), Caltech/IPAC, NASA.
- Gaia DR3: European Space Agency, Gaia mission, via the ESA Gaia archive or CDS VizieR (I/355).
- WISE All-Sky single-exposure images and 2MASS images: IRSA.
- Target names: SIMBAD, CDS Strasbourg (the app falls back to the official CfA and IDIA mirrors when the main
  site is down, as it was on 14 Nov 2026).

## Use of AI

This tool was written with Claude (Anthropic) as a coding assistant, working from a plan the team approved.
The team chose the scope and tested the tool; the archive measurements behind the design (cutout layout,
throttling, wavelength tables) were checked live against IRSA during the build.

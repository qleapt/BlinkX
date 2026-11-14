# SPHEREx Blink Viewer: tutorial

A hands-on guide for first-time users: students, judges, amateur astronomers. In about 20 minutes you will catch
a star moving across the sky, watch the coldest known brown dwarf jump, measure a star's speed and read an
infrared spectrum, all with real NASA data.

No astronomy background needed. Words in *italics* are explained in the [glossary](#glossary) at the end.

---

## 0. The idea in one minute

NASA's **SPHEREx** space telescope photographs the whole sky every six months, in 102 infrared colours. Each
time it passes over the same spot, it takes a new picture. Put those pictures one after the other, like frames of
a flip-book, and anything that **moved** or **changed brightness** jumps out, while everything else stays still.

Astronomers call this *blinking*. In 1930 Clyde Tombaugh found Pluto this way, flipping between two glass
photographs by hand. This tool does it for you in the browser, for any spot in the sky.

---

## 1. Start the app

If someone already shared a link with you, just open it and skip to step 2.

To run it on your own computer (Python 3.10 or newer):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py         # Windows: .venv\Scripts\python -m streamlit run app.py
```

Your browser opens at **http://localhost:8501**.

> Tip for demos: run `python prefill.py` once beforehand. It downloads everything the five example targets need,
> so they appear instantly instead of after 30–60 seconds.

---

## 2. A tour of the screen

**Sidebar (left)**

| Control | What it does |
|---|---|
| **Object name or RA Dec** | Type a star name (`Vega`, `M42`, `Proxima Cen`) or a position in degrees (`269.452 4.693`). |
| **Challenge targets** | Five ready-made examples, one click each. |
| **Field of view** | How much sky you see: 3, 5, 10 or 15 *arcminutes*. Bigger shows more neighbours. |
| **Band** | Which slice of infrared light, D1 (0.75 µm, near red) to D6 (5 µm, warm-object light). |
| **Load the Gaia star list** | Off at first. Fetches ESA's Gaia stars; tick *Gaia stars* in the viewer to see them. |
| **Older surveys** | Off at first. Adds 2MASS (~2000) and WISE (2010) pictures before the SPHEREx ones. |

**Top of the page: the view buttons** Blink · Subtract · Every exposure · Spectrum · Motion check. Under them,
*How to use* explains the view you are in, *What the bands mean* lists D1–D6 with their wavelength and photon
energy, and *Brightness scales* shows what linear, sqrt, log and asinh do and when to use each.

**The sky viewer** (all picture views)

- The **picture**: north is up, east is left (that's how the sky looks from below). The white bar is 1 arcminute.
- The **orange ring**: where your target should be on that date, from its catalogue motion.
- **Play**, **Prev / Next** and the **timeline** under the picture (real dates; long gaps are squeezed, marked ≈).
  **Speed** and **Cross-fade** set how it plays.
- **Dark / Bright** and the stretch menu change brightness and contrast instantly. Scroll to zoom, drag to move,
  **Fit** to see everything again.
- **Compare:** *Blink A ↔ B* flips two dates you pick, *Side by side* shows them next to each other with a linked
  crosshair, *Swipe* puts a divider you drag across, *Difference* subtracts them (red = brighter now, blue = was
  brighter before).
- Tick boxes: *Target ring*, *Gaia stars*, *Flagged pixels* (pink: pixels the NASA pipeline marked as unreliable,
  often a saturated star), *My marks*.
- The **title** of each frame: which *epoch*, dates, band, number of exposures averaged, and wavelength range.
  A warning means the pipeline flagged pixels at the target, usually because it is too bright.
- The line under the picture reads the position and brightness of the pixel under the mouse.

---

## 3. Lesson 1: catch a moving star (Barnard's Star)

Barnard's Star is the fastest-moving star in our sky. It is a small red dwarf just 6 light-years away.

1. Click **Barnard's Star** in the sidebar. The first load takes up to a minute while the tool fetches the
   pictures from NASA's archive. The status line shows progress.
2. Press **Play**. Watch the bright star in the orange ring: it creeps **upward (north)** from frame to frame,
   while the other stars stay put.
3. That shift is only about **5 arcseconds per six months**, less than one SPHEREx pixel (6.15″). You can still see
   it because the tool averages several exposures per epoch and draws the picture at twice the detector resolution.
4. Choose **Subtract** at the top (or *Compare → Difference* in the viewer). This shows the newest epoch minus the
   oldest. Stars that did not move cancel to white. Barnard's Star leaves a **red spot** (where it is now) next to a
   **blue spot** (where it was).
5. Tick **Flagged pixels** in the viewer. The star's centre turns pink: it is so bright that it *saturates* the detector.
   The tool still shows it, but warns you.

**Try:** switch on *Include older photos* in the sidebar, set the field of view to **10′** and View back to **Blink**. The first frame is now a WISE picture from
2010, and the star is almost 3 arcminutes further south in it: 16 years of motion.

---

## 4. Lesson 2: the coldest brown dwarf jumps (WISE 0855−0714)

WISE 0855 is the coldest known brown dwarf, around −20 °C, like a winter day on Earth. It is so cold that it gives off
almost no visible or near-infrared light, only "warm" infrared around 4–5 µm.

1. Click **WISE 0855-0714**. The band switches to **D6** (4.4–5.0 µm) automatically.
2. Set the field of view to **10′** and press **Play**.
3. The first frame is **WISE, May 2010**. The orange circle (and the faint dot inside it) sits about **127″ east
   (left)** of the picture's centre. Then come the SPHEREx epochs (2025–2026): circle and brown dwarf have jumped
   to the centre. Choose *Compare → Blink A ↔ B* to flip between the first and the last frame.
4. Now change the band to **D1** (0.75–1.1 µm). At the predicted position you will see a different, ordinary star.
   WISE 0855 itself is invisible at these short wavelengths: a direct sign of how cold it is.

**Why it moves so fast on the sky:** it is only 7.4 light-years away, so its real motion through space shows up
as more than 8 arcseconds per year.

---

## 5. Lesson 3: measure a star's speed (Epsilon Indi B)

Epsilon Indi B is a pair of brown dwarfs orbiting the star Epsilon Indi A, 12 light-years away. SPHEREx sees the
pair as one dot.

1. Click **Epsilon Indi B**, then **Motion check** at the top.
2. The chart is a map of the sky around the target (east to the left). The **grey line** is the path the catalogue
   predicts: a steady drift (*proper motion*) plus a small yearly loop (*parallax*, caused by Earth going around
   the Sun).
3. The **coloured dots** are where the tool measured the centre of the dot in each SPHEREx epoch. The **diamond** is
   the 2010 WISE picture.
4. Read the cards below the chart: SPHEREx measures **4.71″ per year**, and the catalogue says **4.71″ per year**.
   The verdict: *Agrees*.

5. Below, **What SPHEREx's own measurements say**: the same positions fitted on their own, the kind of object with
   the evidence for and against, and **Where it goes next**: the path continued for the coming years (move *Predict
   ahead*), with a widening band for the uncertainty and the catalogue's own prediction as a dotted line.

**Try:** the same view for Barnard's Star. The agreement is looser (± 5″/yr) because that star is saturated,
which makes its centre harder to pin down. Honest error bars are part of the result.

---

## 6. Lesson 4: an infrared spectrum and a light curve (RW Aurigae, R Draconis)

Every SPHEREx exposure of a given spot sees it through a slightly different colour (the camera's filter changes
colour across the field). Put many exposures together and you get a **spectrum**: brightness against wavelength.

1. Click **RW Aurigae** (a young star still surrounded by its dusty disk, known for sudden dimming events), then
   **Spectrum** at the top. The tool fetches pictures in **all six bands**; this takes a minute or two
   the first time.
2. **Top chart: spectrum.** Each dot is one exposure; colours are epochs. Hover for date, wavelength and error.
   **Dates to show** lets you compare two dates directly.
   **Hollow dots** had flagged pixels in the measurement (here the star is bright, so all are flagged: treat the
   numbers as indicative).
3. **Second chart: light curve** for the band chosen in the sidebar: brightness against time.
4. **What changed?** A plain-language summary per band: how much brighter or fainter the latest epoch is, and
   whether that is bigger than the scatter (σ).
5. **Table of measurements** → **Download CSV** to analyse the numbers yourself.
6. Want more points? Click **Load more exposures**.

Now click **R Draconis**, a giant pulsating star (a *Mira variable*) that swells and shrinks over about 8 months.
It is so bright that the centre of its image is saturated in most exposures. The tool tells you how many could
not be measured instead of plotting wrong numbers, and adds a **brightness index from the star's outer glow**,
which still rises and falls with the star. It shows the trend, not a calibrated brightness.

---

## 7. Lesson 5: explore on your own

- **Any object:** type a name SIMBAD knows (`Vega`, `M 42`, `Proxima Cen`, `Betelgeuse`, `RR Lyr`) or coordinates.
- **Fast movers in our Solar System (asteroids, comets):** they move within hours, not months. Choose
  **Every exposure** and order **by time**. You see single exposures instead of epoch averages.
- **Colour changes:** **Every exposure** ordered **by wavelength** flips through the colours of one spot.

Ideas: the closest star (Proxima Centauri) moves almost 4″ per year. Bright nebulae look different at
1 µm and at 4 µm. Find a star that changed brightness between epochs.

### Mark, measure and check your find (works everywhere, made for typed positions)

A position you type (e.g. `201.365 -43.019`) has no catalogued star to follow, so there is no predicted
path to check. The app does not point anything out: you look, then the tools measure. In the viewer:

- **Mark:** click anything to drop a numbered marker. With **Snap to star centre** on, the
  marker jumps to the middle of the nearest star. The panel lists each marker's RA and Dec.
- **Distances between marks:** the separation (″) and direction between consecutive markers. Mark the
  *same* star on two dates (switch photos with ) and it also shows the **speed in ″ per year**.
  Under ~3″ (half a pixel) is within the measuring error.
- **Measure (line):** click the start, then the end. The line's length on the sky is shown on the image.
- **Real size:** type the object's distance (parsecs or light-years) to turn angles into AU and km
  (size in AU = angle in ″ × distance in pc).
- **Same spot on every date:** a row of small pictures of your last mark on each date, with a fixed crosshair and
  the brightness there (sky removed). Click one to show that date.
- With no markers the panel says *Nothing marked yet*. **Undo** / **Clear** remove marks;
  **Download CSV** saves them. Marks stay on your computer only.
- **Check my find** (under the viewer): choose the marks you made of *one* object (all are kept at first) and press
  **Analyse the motion**. Three or more dates are best. The app measures each mark on its own photo, fits one path
  through all of them, compares every pair, tests each mark against the others (a mark on the wrong star stands
  out), looks for a Gaia star at every mark on its date, and says what the object probably is (standing still,
  moving star, asteroid / comet, or not one object) with every reason listed. **Where it goes next** continues the
  path for years (or hours, for an asteroid) with its uncertainty. *Also measure it on the other photos* (off at
  first) adds the photos you did not mark, at the spot your marks predict. **Spectrum of the first mark** measures
  its colours.
- **Helpers** (under the viewer, off until you switch them on): *Detect stars* draws green crosses on the stars
  it finds; *Find what changed* lists what moved, brightened or faded between two dates, strongest first, labelled
  C1, C2… on the picture. Check each one by eye.

---

## 8. Share and save

- **Share a view:** the address bar always holds your current settings. Copy it and send it. Example:
  `http://localhost:8501/?target=WISE+0855&band=D6&fov=10&mode=blink`
  (`mode` can be `blink`, `sub`, `movie`, `spectrum`, `motion`).
- **Animated GIF:** **Download GIF** under the picture, for slides and social media.
- **Data:** the **Photo list** tab lists every picture used, with a link to the original NASA file. The spectrum
  view exports a CSV.
- **Embed in a website:** see the iframe snippet in the [README](README.md#link-to-a-view-url-parameters).

---

## 9. Reading the pictures carefully

| What you see | What it means |
|---|---|
| A warning in the frame title, pink pixels, hollow dots | The pipeline flagged those pixels, usually saturation from a bright star. Shown, never hidden. |
| A yellow warning about a nearby Gaia star | Another star sits within ~15″ of your target. SPHEREx pixels are 6″, so its light mixes in. |
| Grey "survey" frames look sharper or different | 2MASS and WISE are other telescopes with other colours and pixel sizes. Use them for **position**, not brightness. |
| Numbers say "quick look, not calibrated" | Fine for spotting changes; for research use the IRSA SPHEREx Spectrophotometry Tool. |
| Shifts smaller than a pixel | Real: averaging many pixels pins down a star's centre far better than one pixel. Check the error bars. |

---

## 10. Troubleshooting

| Problem | What to do |
|---|---|
| First load takes 30–60 s | Normal on a slow or distant connection. The app shows the picture once every epoch has a photo; the rest keep downloading and are added automatically when they arrive. The second time it is instant (cached). |
| Coordinates are not recognised | Use two numbers in degrees, RA (0–360) then Dec (−90 to +90): `269.452 4.693`, `269.452°, +4.693°` and `RA=269.452 Dec=4.693` all work. The line under the box shows how your text was read. |
| "The NASA/IPAC archive is busy" | Wait a minute and reload. |
| "Could not find …" | Check the spelling, try another name (e.g. `HD 209100`), or type coordinates in degrees. |
| "Name lookup failed" | The name service is down; coordinates always work. |
| "No SPHEREx images cover … yet" | SPHEREx is still filling in the sky; new data arrive weekly. |
| The target is outside the picture | Use a larger field of view; very fast stars move far between 2000 and today. |
| A frame looks empty or grey | Older surveys may not cover every band; try another band. |

---

## Glossary

- **Arcminute (′), arcsecond (″):** sky angles. The full Moon is about 30′ wide; 1′ = 60″.
- **Band:** a range of infrared wavelengths. SPHEREx has six (D1–D6) covering 0.75–5 µm.
- **Blink comparator:** flipping between pictures of the same sky to spot changes.
- **Brown dwarf:** an object between a planet and a star, too small to burn hydrogen.
- **Epoch:** one pass of SPHEREx over a spot. Passes come about every six months.
- **Exposure:** one single picture. An epoch average combines several.
- **Flagged pixel:** a pixel the NASA pipeline marked as unreliable (saturated, hit by a cosmic ray, …).
- **Gaia DR3:** the European Space Agency's catalogue of positions and motions for over a billion stars.
- **Light curve:** brightness of an object against time.
- **mJy (millijansky), MJy/sr:** units of brightness used in infrared astronomy.
- **Parallax:** the small yearly back-and-forth of a nearby star caused by Earth's orbit.
- **Proper motion:** how fast a star drifts across the sky, in arcseconds per year.
- **Saturation:** a star so bright that it overflows the detector pixels, like an overexposed photo.
- **Spectrum:** brightness against wavelength; it tells what an object is made of and how hot it is.
- **Stretch:** how brightness values map to grey levels on screen (asinh shows faint and bright things together).
- **µm (micrometre):** a millionth of a metre. Visible light is 0.4–0.7 µm; SPHEREx sees 0.75–5 µm.

---

*Data: NASA/IPAC Infrared Science Archive (SPHEREx, WISE, 2MASS), ESA Gaia, CDS SIMBAD.
Built for NASA Space Apps Challenge 2026.*

"""SPHEREx Blink Viewer: type a sky position or object name, watch SPHEREx images of it blink."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from astropy.time import Time

from spx import archive, catalogs, config, cutouts, detect, i18n, photometry, render, stack, targets, viewer, warp

st.set_page_config(page_title="SPHEREx Blink Viewer", layout="wide")
# English / Turkish: everything Streamlit shows goes through i18n.tr() when Turkish is chosen
i18n.install(lambda: st.session_state.get("lang", "en"))

PRESET_BAND = targets.PRESET_BAND
MODES = {"blink": "Blink epochs", "add": "Add", "sub": "Subtract", "movie": "Every exposure",
         "spectrum": "Spectrum & light curve", "motion": "Motion check"}
MODE_LABEL = {"blink": "Blink", "add": "Add", "sub": "Subtract", "movie": "Every exposure",
              "spectrum": "Spectrum", "motion": "Motion check"}
MODE_HELP = {"blink": "flip between dates: movers jump", "add": "stack all dates: see faint things",
             "sub": "newer − older: see only what changed", "movie": "one frame per single photo",
             "spectrum": "brightness at each infrared colour", "motion": "does it move as predicted?"}
PRESET_HELP = {"Barnard's Star": "Fastest-moving star in the sky (10″ a year)",
               "Epsilon Indi B": "Pair of cold brown dwarfs, 12 light-years away",
               "WISE 0855-0714": "Coldest known brown dwarf (about as cold as a freezer); fast mover",
               "RW Aurigae": "Young star that dims and flares unpredictably",
               "R Draconis": "Pulsating giant star (brightens and fades over ~8 months)"}
BAND_LABEL = {b: f"{b} · {lo:.2f}–{hi:.2f} µm" for b, (lo, hi, _) in config.BANDS.items()}

qp = st.query_params
ss = st.session_state


def _init():
    if ss.get("_init"):
        return
    ss._init = True
    if "ra" in qp and "dec" in qp:
        ss.target_text = f"{qp['ra']} {qp['dec']}"
    else:
        ss.target_text = qp.get("target", "Barnard's Star")[:100]
    try:
        f = float(qp.get("fov", 5))
    except ValueError:
        f = 5
    ss.fov = min(config.FOV_CHOICES, key=lambda c: abs(c - f))
    p = targets.preset(ss.target_text)
    b = str(qp.get("band", "")).upper()
    ss.band = b if b in config.BANDS else PRESET_BAND.get(p.name if p else "", "D3")
    m = str(qp.get("mode", "blink")).lower()
    ss.mode = m if m in MODES else "blink"
    ss.stretch = "asinh"
    ss.speed = 700
    ss.gaia = qp.get("gaia", "1") != "0"
    ss.survey = qp.get("old", "1") != "0"
    ss.showflags = False
    lang = str(qp.get("lang", "en")).lower()
    ss.lang = lang if lang in i18n.LANGS else "en"


def _pick(name):
    ss.target_text = name
    ss.band = PRESET_BAND.get(name, ss.band)


_init()

def _input_hint(text):
    """One line under the search box saying how the text will be read."""
    if targets.preset(text):
        st.caption("A challenge target (built-in catalogue values)")
    elif (t := targets.parse_coords(text)) is not None:
        st.caption(f"Read as a sky position: RA {t.ra:.5f}°, Dec {t.dec:+.5f}°")
    elif targets.looks_like_coords(text):
        st.caption("Warning: this looks like numbers but is not a valid position. Use two numbers in degrees: "
                   "RA from 0 to 360, then Dec from −90 to +90, e.g. `269.452 4.693`.")
    elif text.strip():
        st.caption("Will be looked up by name in the SIMBAD star catalogue")


GLOSSARY = """
- **RA, Dec** (right ascension, declination): a position on the sky, like longitude and latitude on Earth,
  in degrees (°). RA runs 0–360, Dec from −90 (south pole) to +90 (north pole).
- **″ arcsecond, ′ arcminute**: tiny angles. 60″ = 1′, 60′ = 1°. The full Moon is about 30′ (1,800″) wide.
- **Pixel**: one dot of the camera. A SPHEREx pixel covers 6.15″ of sky.
- **Band / µm**: SPHEREx sees infrared light, invisible to our eyes, split into six bands D1–D6.
  µm (micrometre) = a millionth of a metre; visible light is 0.4–0.7 µm.
- **Epoch**: one pass of the survey over this spot. SPHEREx comes back about every six months.
- **Proper motion**: a star's own slow drift across the sky, in ″ per year (″/yr).
- **Parallax**: a small yearly back-and-forth wobble of nearby stars, because we watch from a moving Earth.
- **MJy/sr, mJy**: units of brightness (megajansky per steradian for the sky, millijansky for a star).
- **Flagged pixels**: pixels the NASA pipeline marked as unreliable (overexposed, hit by a cosmic ray…).
- **σ (sigma)**: the measurement uncertainty. A change bigger than 3σ is very likely real.
- **Gaia, 2MASS, WISE, SIMBAD**: a star-mapping satellite (ESA), two older infrared sky surveys
  (~2000 and 2010) and a big catalogue of named objects.
"""

# --- sidebar ------------------------------------------------------------------------
i18n.GLOSSARY_EN = GLOSSARY

with st.sidebar:
    st.segmented_control("Language / Dil", list(i18n.LANGS), key="lang", required=True, format_func=i18n.LANGS.get,
                         help="English or Turkish / İngilizce veya Türkçe")
    st.header("SPHEREx Blink")
    st.markdown("**① What to look at**")
    st.text_input("Object name or sky position (RA Dec in degrees)", key="target_text",
                  placeholder="Barnard's Star  ·  M 42  ·  269.452 4.693",
                  help="Type the name of a star, galaxy or nebula (looked up in the SIMBAD catalogue), or a "
                       "sky position as two numbers in degrees: RA (0–360) then Dec (−90 to +90), for "
                       "example `269.452 4.693` or `269.452°, +4.693°`. Hours-minutes-seconds also work: "
                       "`17 57 48.5 +04 41 36`. Press Enter or Go.")
    st.button("Go", type="primary", use_container_width=True, help="Load this object or position")
    _input_hint(ss.target_text)
    st.caption("…or pick a challenge target:")
    cols = st.columns(2)
    for i, name in enumerate(targets.PRESETS):
        cols[i % 2].button(name, on_click=_pick, args=(name,), use_container_width=True,
                           help=PRESET_HELP.get(name))
    st.markdown("**② Colour and size**")
    st.selectbox("Band (which infrared colour)", list(config.BANDS), key="band", format_func=BAND_LABEL.get,
                 help="SPHEREx splits infrared light into six bands: D1 has the shortest wavelengths "
                      "(just beyond red light), D6 the longest (sees cool dust and cold brown dwarfs). "
                      "µm = micrometre.")
    st.radio("Field of view (how much sky)", config.FOV_CHOICES, key="fov", horizontal=True,
             format_func=lambda f: f"{f}′",
             help="Width of the picture in arcminutes (′). 60′ = 1°; the full Moon is about 30′ wide. "
                  "Bigger fields take a little longer to load.")
    st.markdown("**③ Extras**")
    st.toggle("Load the Gaia star list", key="gaia",
              help="Fetches the stars catalogued by ESA's Gaia satellite. They stay hidden until you tick "
                   "*Gaia stars* in the viewer; then each is drawn as a blue circle where it should be on that "
                   "date. Hover a circle for its details.")
    st.toggle("Include older photos (2MASS ~2000, WISE 2010)", key="survey",
              help="Adds images from earlier infrared surveys before the SPHEREx frames, so motion over 15–25 "
                   "years becomes visible. Other telescopes: J → D1/D2, Ks → D3, W1 → D4, W2 → D5/D6.")
    st.caption("Brightness, contrast, speed and flagged pixels are set in the viewer itself.")
    with st.expander("Words used here"):
        st.markdown(GLOSSARY)
    st.html(warp.SCRIPT, unsafe_allow_javascript=True)  # loading animation, installed once per page


# write the state back to the URL so it can be shared
_t = targets.parse_coords(ss.target_text)
if _t:
    qp.pop("target", None)
    qp["ra"], qp["dec"] = f"{_t.ra:.5f}", f"{_t.dec:.5f}"
else:
    for _k in ("ra", "dec"):
        qp.pop(_k, None)
    qp["target"] = ss.target_text
qp["fov"], qp["band"], qp["mode"], qp["lang"] = str(ss.fov), ss.band, ss.mode, ss.lang


# --- data -------------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def resolve(text):
    return targets.resolve(text)


def search(ra, dec):  # cached on disk by archive.search (fast list first, complete list soon after)
    return archive.search(ra, dec)


class Loader:
    """The star-field loading screen (spx/warp.py) with the current step, in the middle of the page."""

    def __init__(self):
        self.box, self.t0 = st.empty(), time.time()
        self.title, self.detail, self.stage, self.frac, self.ptext = "", "", 0, None, ""

    def _draw(self):
        T = lambda s: i18n.tr(s, ss.lang)  # noqa: E731  (the card is HTML, so it is translated here)
        self.box.markdown(warp.card(T(self.title), T(self.detail), self.stage, self.frac, T(self.ptext),
                                    stages=[T(s) for s in warp.STAGES]), unsafe_allow_html=True)

    def step(self, title, detail="", stage=0):
        if stage != self.stage:
            self.frac, self.ptext = None, ""
        self.title, self.detail, self.stage = title, detail, stage
        self._draw()

    def progress(self, frac, text):
        if (min(1.0, frac), text) != (self.frac, self.ptext):
            self.frac, self.ptext = min(1.0, frac), text
            self._draw()

    def note(self, text):
        """A line under the steps without a progress bar (e.g. how long we have been waiting)."""
        if text != self.ptext:
            self.ptext = text
            self._draw()

    def done(self, arrive=True):
        secs = time.time() - self.t0
        if arrive and secs > 0.8:  # long enough to have been seen: slow down and fade out before the photos
            self.stage = 4
            self._draw()
            time.sleep(0.45)
        self.box.empty()
        return secs


st.title("SPHEREx Blink Viewer")
st.caption("SPHEREx photographs the whole sky every six months in 102 infrared colours. "
           "Flip between its photos of one spot: whatever moves or changes brightness stands out.")
st.segmented_control("View", list(MODES), key="mode", required=True, format_func=MODE_LABEL.get,
                     label_visibility="collapsed",
                     help="How to look at the photos:\n\n" + "\n".join(f"- **{MODE_LABEL[m]}**: {MODE_HELP[m]}"
                                                                       for m in MODES))
st.caption(f"**{MODE_LABEL[ss.mode]}**: {MODE_HELP[ss.mode]}")
with st.expander("How to use this page (30 seconds)"):
    st.markdown(
        "1. **Choose a target** in the sidebar: a challenge target, any name (e.g. *M 42*), or a sky position "
        "in degrees (e.g. `269.452 4.693`). Press **Go**.\n"
        "2. **Press Play** under the image. The photos are months apart: a star that *jumps* is moving, one "
        "that *flickers* changes brightness.\n"
        "3. **Switch the view** with the buttons above: *Subtract* shows only what changed, *Spectrum* shows "
        "colour, *Motion check* compares the measured path with the predicted one.\n"
        "4. **Mark** (in the viewer's toolbar): click to mark objects, draw lines to measure sizes and "
        "distances, and mark the same object on two dates to get its speed. *Compare* puts two dates side "
        "by side, under a swipe divider, or subtracts them.\n\n"
        "Hover over the small question-mark icons for short explanations; *Words used here* in the sidebar explains "
        "the astronomy terms.")
load = Loader()
load.step("Finding your target…", "Reading the position you typed, or looking the name up in the SIMBAD catalogue", 0)
try:
    tgt = resolve(ss.target_text)
except targets.TargetNotFound:
    load.done(arrive=False)
    if targets.looks_like_coords(ss.target_text):
        st.error(f"**{ss.target_text}** is not a valid sky position. Use two numbers in degrees: RA from 0 to 360, "
                 "then Dec from −90 to +90, for example `269.452 4.693` or `269.452, -4.693`.")
    else:
        st.error(f"Could not find **{ss.target_text}** in the SIMBAD catalogue. Check the spelling, try another "
                 "name (e.g. *M 42*, *HD 209458*), or type a position like `269.452 4.693`.")
    st.stop()
except Exception as e:  # SIMBAD down
    load.done(arrive=False)
    st.error(f"Name lookup failed ({e}). Coordinates like `269.452 4.693` always work.")
    st.stop()

ra0, dec0 = targets.position_at(tgt, config.REFERENCE_MJD)
band = ss.band
# Side requests that only need the position start now and run while SPHEREx images load.
_bg = ThreadPoolExecutor(2)
fut_gaia = (_bg.submit(catalogs.gaia_cone, round(ra0, 4), round(dec0, 4), round(ss.fov / 60 * 0.75, 3))
            if ss.gaia and ss.mode in ("blink", "add", "sub", "movie") else None)
fut_survey = (_bg.submit(catalogs.survey_frames, round(ra0, 5), round(dec0, 5), band, ss.fov * 1.45)
              if ss.survey and ss.mode in ("blink", "sub", "motion") else None)
_bg.shutdown(wait=False)

def waiting(t0, what=""):
    """Plain words for the loading card: how long we have waited, and why when it is long."""
    el = time.time() - t0
    return (what + f"{el:.0f} s" + (" · NASA's servers or the connection are slow right now: it keeps trying, and "
                                    "whatever arrives is saved, so the next visit is fast" if el > 12 else ""))


load.step("Asking NASA's archive which photos cover this spot…",
          "SPHEREx has photographed every part of the sky several times since 2025", 1)
try:
    _t_search, _ex = time.time(), ThreadPoolExecutor(1)
    _fs = _ex.submit(search, round(ra0, 4), round(dec0, 4))
    _ex.shutdown(wait=False)
    while True:  # keep the loading card alive while the archive answers
        try:
            exps = _fs.result(timeout=0.5)
            break
        except FutureTimeout:
            load.note(waiting(_t_search))
except archive.ArchiveBusy:
    load.done(arrive=False)
    st.warning("The NASA/IPAC archive is busy right now. Please try again in a minute.")
    st.stop()
if not exps:
    load.done(arrive=False)
    st.info(f"SPHEREx has no images covering {tgt.name} yet. New data arrive weekly.")
    st.stop()


def fetch(todo, label, group_of=None):
    """Download the cutouts. With group_of (uri -> epoch), stop waiting after a while once every epoch has
    at least one photo: on a slow connection the rest keep downloading into the cache for the next view."""
    load.step(f"Flying in: downloading {len(todo)} SPHEREx photos ({label})…",
              "Reading only the pixels around your target from NASA's archive", 2)
    stats, enough = {}, None
    if group_of:
        need = set(group_of.values())

        def enough(got):
            return need <= {group_of.get(x.uri) for x in got}
    t_fetch = time.time()
    out, nfail = cutouts.fetch_many(todo, ra0, dec0, ss.fov * 1.45,
                                    lambda d, n: load.progress(d / n, waiting(t_fetch, f"{d} of {n} photos · ")),
                                    deadline=config.FETCH_SOFT_DEADLINE if group_of else None,
                                    enough=enough, stats=stats)
    if not out:
        load.done(arrive=False)
        st.warning("No photos could be downloaded (the archive or your connection is busy). "
                   "Please try again in a minute.")
        st.stop()
    ss["_skipped"] = stats.get("skipped", 0)
    load.progress(1.0, f"{len(out)} photos" + (f" ({nfail} failed)" if nfail else ""))
    load.step("Lining the photos up…", "Turning every photo north-up on one grid and averaging each epoch", 3)
    return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _gaia_near(ra, dec):
    return catalogs.gaia_cone(ra, dec, config.FIND_CONE_DEG)[0]


def _drop_find():
    ss.pop("find", None)


def resolve_find():
    """The object of 'Check my find' in this field: (Target, plain-English note), or (None, "")."""
    f = ss.get("find")
    if not f or f.get("field") != tgt.name:
        return None, ""
    m = f["marks"][0] if f.get("marks") else f["mark"]
    try:
        star, sep = catalogs.nearest(_gaia_near(round(m["ra"], 5), round(m["dec"], 5)),
                                     m["ra"], m["dec"], m["mjd"], config.FIND_MATCH_ARCSEC)
        trouble = ""
    except Exception:
        star, sep = None, None
        trouble = "The Gaia catalogue could not be reached right now. "
    if star:
        return catalogs.as_target(star, name=f"Your find (Gaia DR3 {star['id']})"), (
            f"Your mark ({m.get('date', '')}) at RA {m['ra']:.5f}°, Dec {m['dec']:+.5f}° matches Gaia DR3 "
            f"{star['id']} (G = {star['g']:.1f}), {sep:.1f}″ away on that date.")
    from spx.targets import Target
    return Target("Your find", m["ra"], m["dec"], m["mjd"], source="your mark (no catalogue match)"), (
        trouble + f"No Gaia star within {config.FIND_MATCH_ARCSEC:g}″ of your mark (RA {m['ra']:.5f}°, "
        f"Dec {m['dec']:+.5f}°), so there is no prediction: it is treated as standing still at the spot you "
        "marked, and what SPHEREx measures is its own. Very cold or very faint objects are often missing from Gaia.")


if ss.mode == "spectrum":
    spec_t, find_note = resolve_find()
    spec_t = spec_t or tgt
    spec_key = f"specall|{tgt.name}"
    spec_all = ss.get(spec_key, False)
    todo, epoch_of = [], {}
    for b in config.BANDS:
        for k, g in archive.select(exps, b, quick=not spec_all, cap=config.SPEC_CAP_PER_BAND,
                                   quick_total=config.SPEC_QUICK_PER_BAND).items():
            for e in g:
                todo.append(e)
                epoch_of[e.uri] = k + 1
    nep = max(epoch_of.values())
    fold = nep > len(render.EPOCH_COLORS)  # dense fields: merge time bins so colours never repeat
    stamps = fetch(todo, "all-band")
    rows = []
    for s in stamps:
        m = photometry.measure(s, *targets.position_at(spec_t, s.mjd))
        k = epoch_of[s.uri]
        m.update(date=targets.mjd_to_date(s.mjd), datetime=Time(s.mjd, format="mjd").iso,
                 epoch=1 + (k - 1) * len(render.EPOCH_COLORS) // nep if fold else k)
        rows.append(m)
    by_epoch = {}
    for r in sorted(rows, key=lambda r: (r["epoch"], r["wavelength"])):
        by_epoch.setdefault(r["epoch"], []).append(r)
    load_secs = load.done()
    st.subheader(f"{spec_t.name}: SPHEREx spectrum, 0.75–5.0 µm")
    st.caption(f"{len(stamps)} images loaded in {load_secs:.1f} s. " + find_note)
    if find_note:
        st.button(f"← Back to {tgt.name}", on_click=_drop_find, help="Forget this find and measure the target again.")
    close = {}
    try:  # warn when a Gaia neighbour falls inside the 12" aperture (SPHEREx pixels are 6.15")
        nb, _ = catalogs.gaia_cone(*(round(v, 4) for v in targets.position_at(spec_t, config.REFERENCE_MJD)), 0.02)
        mids = [min(r["mjd"] for r in rows), max(r["mjd"] for r in rows)]
        pr0, pd0 = targets.position_at(spec_t, mids[0])
        gr0, gd0 = catalogs.gaia_at(nb, mids[0])
        sep0 = np.hypot((gr0 - pr0) * np.cos(np.radians(pd0)), gd0 - pd0) * 3600
        own = [s["g"] for s, d in zip(nb, sep0) if d < 2.0]
        g_limit = (min(own) + 3) if own else 99  # only neighbours within 3 mag of the target matter
        for m in mids:
            pr, pd_ = targets.position_at(spec_t, m)
            gr, gd = catalogs.gaia_at(nb, m)
            sep = np.hypot((gr - pr) * np.cos(np.radians(pd_)), gd - pd_) * 3600
            for s, d in zip(nb, sep):
                if 2.0 < d < 15 and s["g"] < g_limit:  # < 2" is the target's own Gaia entry
                    close.setdefault(s["id"], (s["g"], []))[1].append(d)
        for g, ds in close.values():
            dist = f"{min(ds):.0f}″" if round(min(ds)) == round(max(ds)) else f"{min(ds):.0f}–{max(ds):.0f}″"
            st.warning(f"A Gaia star (G = {g:.1f}) lies {dist} from the target on these dates. "
                       "SPHEREx pixels are 6.15″, so its light blends into the measurement — especially at short "
                       "wavelengths where a faint, cold target is dark.")
    except Exception:
        pass
    s1, s2 = st.columns([3, 1], vertical_alignment="bottom")
    ep_label = {k: f"Epoch {k}: {v[0]['date']}" for k, v in by_epoch.items() if v}
    show_eps = s1.multiselect("Dates to show", list(ep_label), default=list(ep_label), format_func=ep_label.get,
                              help="Pick two dates to compare their spectra directly; remove one to hide it.")
    logy = s2.toggle("Logarithmic flux axis", value=False,
                     help="Spreads out faint and bright values; useful when the spectrum spans a big range.")
    st.plotly_chart(render.spectrum_fig({k: v for k, v in by_epoch.items() if k in show_eps}, logy),
                    use_container_width=True)
    st.caption("Each point is one exposure; SPHEREx sees each spot at a slightly different wavelength every "
               "time, so many exposures together make a spectrum. Colours are survey epochs"
               + (" (time bins merged)" if fold else "") + ". Hollow points have flagged pixels in the aperture "
               "(often saturation): treat them with care. Quick look, not calibrated; for science use the "
               f"[IRSA SPHEREx Spectrophotometry Tool]({config.SPECTROPHOTOMETRY_TOOL}).")
    nbad = sum(not np.isfinite(r["flux"]) for r in rows)
    if nbad:
        sat = sum(bool(r["flags"]) and r["flags"][0].startswith("SATURATED") for r in rows)
        bands_sat = sorted({r["band"] for r in rows if r["flags"] and r["flags"][0].startswith("SATURATED")})
        st.info(f"{nbad} of {len(rows)} exposures could not be measured"
                + (f"; in {sat} of them the target is so bright that the pixels at its centre are saturated "
                   f"(bands {', '.join(bands_sat)}). They are listed in the table, not plotted." if sat else "."))
    st.subheader(f"Light curve in {band} (choose the band in the sidebar)")
    st.plotly_chart(render.lightcurve_fig(by_epoch, band, logy), use_container_width=True)
    wing_rows = {k: [dict(r, flux=r["wing"], err=0.0, flags=[]) for r in v
                     if r["band"] == band and np.isfinite(r.get("wing", np.nan))] for k, v in by_epoch.items()}
    if any(wing_rows.values()):
        st.markdown(f"**Saturated exposures: brightness index from the star's wings ({band})**")
        st.plotly_chart(render.lightcurve_fig(wing_rows, band, False,
                                              ytitle="Wing brightness (MJy/sr, relative)"),
                        use_container_width=True)
        st.caption("When the centre of a very bright star is saturated, the light 3–5 pixels from the centre still "
                   "rises and falls with the star's brightness. This index shows the trend only; it is not a flux, "
                   "and wavelength differences inside the band add scatter.")
    # plain-language change report per band: median clean flux, first vs last epoch
    notes = []
    for b in config.BANDS:
        per = {k: [r for r in v if r["band"] == b and not r["flags"] and np.isfinite(r["flux"])]
               for k, v in by_epoch.items()}
        per = {k: v for k, v in per.items() if len(v) >= 2}
        flagged_note = ""
        if len(per) < 2:  # bright stars: every point is flagged; use them, and say so
            per = {k: [r for r in v if r["band"] == b and np.isfinite(r["flux"])] for k, v in by_epoch.items()}
            per = {k: v for k, v in per.items() if len(v) >= 2}
            flagged_note = " (from flagged points)"
        if len(per) >= 2:
            k1, k2 = min(per), max(per)
            f1, f2 = (np.median([r["flux"] for r in per[k]]) for k in (k1, k2))
            e = np.hypot(*(1.25 * np.std([r["flux"] for r in per[k]]) / np.sqrt(len(per[k])) for k in (k1, k2)))
            sig = abs(f2 - f1) / e if e > 0 else 0
            word = "brighter" if f2 > f1 else "fainter"
            notes.append(f"**{b}**: epoch {k2} is {abs(f2 / f1 - 1) * 100:.0f}% {word} than epoch {k1}"
                         + (" — a real change is likely" if sig > 3 else " — within the scatter")
                         + f" ({sig:.1f}σ){flagged_note}")
    if notes:
        st.markdown("**What changed?** (median of unflagged points; different wavelengths inside a band add "
                    "scatter)\n\n- " + "\n- ".join(notes)
                    + ("\n\nNote: the neighbouring star above moves relative to the target, so part of any change "
                       "can come from it entering or leaving the aperture." if close else ""))
    with st.expander("Table of measurements"):
        df = pd.DataFrame([{k: (", ".join(v) if k == "flags" else v) for k, v in r.items()
                            if k not in ("x", "y", "datetime")} for r in rows]).sort_values(["band", "mjd"])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), f"spherex_{tgt.name.replace(' ', '_')}.csv")
    if not spec_all:
        st.button(f"Load more exposures (up to {config.SPEC_CAP_PER_BAND} per band)",
                  on_click=lambda: ss.__setitem__(spec_key, True))
    st.stop()

load_key = f"all|{tgt.name}|{band}"
load_all = ss.get(load_key, False)
groups = {k + 1: v for k, v in archive.select(exps, band, quick=not load_all).items()}
n_band = sum(e.band == band for e in exps)
todo = [e for g in groups.values() for e in g]
# cutouts are 1.45x the field so rotated frames cover the square grid
stamps = fetch(todo, band, {e.uri: k for k, g in groups.items() for e in g})

wgrid, shape = stack.grid(ra0, dec0, ss.fov)
arcsec_px = config.PIXEL_SCALE_ARCSEC / config.OVERSAMPLE


@st.cache_resource(max_entries=4, show_spinner=False)
def aligned_frames(key, _stamps, _groups, _tgt):
    """Line every photo up on the grid and average each epoch. Kept in memory, so changing a display
    control or the view never lines the photos up again. key: target, field, band, photos."""
    by_uri = {}
    for s in _stamps:
        a = stack.align(s, wgrid, shape)
        a.bad_near_target = stack.flags_near(s, *targets.position_at(_tgt, s.mjd))
        by_uri[s.uri] = a
    frames = []
    for k, g in _groups.items():
        items = [by_uri[e.uri] for e in g if e.uri in by_uri]
        if items:
            frames.append(stack.stack_frame(items, f"Epoch {k}: "))
    return by_uri, frames


by_uri, frames = aligned_frames(
    (tgt.name, round(ra0, 6), round(dec0, 6), ss.fov, band,
     tuple((k, tuple(sorted(e.uri for e in g))) for k, g in groups.items()), tuple(sorted(s.uri for s in stamps))),
    stamps, groups, tgt)
frames = list(frames)
sphx_frames = list(frames)


@st.cache_resource(max_entries=4, show_spinner=False)
def old_survey_frames(key, _items, ref):
    """WISE / 2MASS frames on the same grid (kept in memory like aligned_frames)."""
    return sorted([f for f in (stack.survey_frame(it, wgrid, shape, ref, band) for it in _items) if f is not None],
                  key=lambda f: f.mjd_start)


survey_pending = False
if fut_survey is not None and frames:
    try:
        if not fut_survey.done():
            load.step("Adding the older WISE and 2MASS photos…", "From 2010 and around 2000, to compare with today", 3)
        try:  # never hold the SPHEREx movie back for long: the download continues and is cached
            items = fut_survey.result(timeout=3)
        except FutureTimeout:
            items, survey_pending = [], True
        ref = float(np.nanmax([np.nanpercentile(f.image, 99.5) for f in sphx_frames]))
        skey = (round(ra0, 6), round(dec0, 6), ss.fov, band,
                tuple((it["survey"], it["band"], tuple(it["mjds"]), len(it["images"])) for it in items))
        frames = list(old_survey_frames(skey, items, ref)) + frames
    except Exception as e:
        st.caption(f"Older survey images unavailable ({type(e).__name__}).")


def target_at(mjd, t=None):
    """Where the target (or t) is on that date, in grid pixels (a ring), or the typed position (a cross)."""
    if t is not None:
        x, y = wgrid.world_to_pixel_values(*targets.position_at(t, mjd))
        return dict(x=float(x), y=float(y), label=f"{t.name}: predicted position on {targets.mjd_to_date(mjd)}")
    if tgt.source == "coordinates":  # nothing to track: just mark the position that was asked for
        x, y = wgrid.world_to_pixel_values(tgt.ra, tgt.dec)
        return dict(x=float(x), y=float(y), cross=True,
                    label=f"RA {tgt.ra:.5f}, Dec {tgt.dec:+.5f} (the position you typed)")
    x, y = wgrid.world_to_pixel_values(*targets.position_at(tgt, mjd))
    return dict(x=float(x), y=float(y), label=f"{tgt.name}: predicted position on {targets.mjd_to_date(mjd)}")


gaia_stars, gaia_note, gaia_pending = [], "", False
if fut_gaia is not None:
    try:  # the photos are ready: wait only briefly for the star list (it is cached once it arrives)
        gaia_stars, src = fut_gaia.result(timeout=8)
        gaia_note = f"{len(gaia_stars)} Gaia DR3 stars (G < {config.GAIA_MAX_G:g}, via {src})"
    except FutureTimeout:
        gaia_pending = True
        gaia_note = "Gaia star list still loading"
    except Exception:
        gaia_note = "Gaia catalogue unavailable right now."
gaia_text = [catalogs.gaia_hover(s) for s in gaia_stars]


def gaia_on(mjd):
    """Gaia stars moved to that date, in grid pixels, with their hover text."""
    if not gaia_stars:
        return []
    r, d = catalogs.gaia_at(gaia_stars, mjd)
    x, y = wgrid.world_to_pixel_values(r, d)
    return [dict(x=round(float(a), 2), y=round(float(b), 2), text=t)
            for a, b, t in zip(np.atleast_1d(x), np.atleast_1d(y), gaia_text)
            if -2 < a < shape[1] + 1 and -2 < b < shape[0] + 1]


COORD_NOTE = ("You typed a sky position, so nothing is followed here (the orange cross marks the spot). "
              "Look for anything that moves or changes yourself, then **Mark** it on two dates to measure it.")


def sky_viewer(shown_frames, diverging=False, key_extra="", stars=None, changes=None, follow=None):
    """The sky viewer (spx/viewer) on the given frames. Returns the user's marks (list of dicts).
    stars / changes: results of the optional helpers (None while they are switched off)."""
    cx, cy = (float(v) - 1 for v in wgrid.wcs.crpix)
    data = viewer.payload(shown_frames, target=[target_at(f.mjd_mid, follow) for f in shown_frames],
                          gaia=[gaia_on(f.mjd_mid) for f in shown_frames], scale=arcsec_px, center=(cx, cy),
                          radec=(ra0, dec0), diverging=diverging, mode=ss.mode,
                          title=f"{tgt.name}|{band}|{ss.fov}|{key_extra}", stars=stars, changes=changes)
    store = f"{tgt.name}|{ss.fov}|{band}|{key_extra}"
    first_sphx = len(shown_frames) - len(sphx_frames) if len(shown_frames) == len(frames) else 0
    res = viewer.show(data, key=f"viewer|{ss.mode}", store_key=store,
                      settings=dict(stretch=ss.stretch, speed=ss.speed, gaia=False, flags=ss.showflags, lang=ss.lang,
                                    a=first_sphx))
    if isinstance(res, dict) and res.get("key") == "spx-measure:" + store:
        ss["_marks"] = dict(res, store=store)
    got = ss.get("_marks") or {}
    return got.get("marks", []) if got.get("store") == store else []


def pending_notes():
    """Say so when part of the data is still arriving over a slow connection; redraw when it has."""
    msgs = []
    if ss.get("_skipped"):
        msgs.append(f"{ss['_skipped']} more SPHEREx photos are still downloading (slow connection); each "
                    "epoch already has at least one")
    if survey_pending:
        msgs.append("the older WISE (2010) / 2MASS (~2000) photos are still downloading")
    if gaia_pending:
        msgs.append("the Gaia star list is still loading")
    if not msgs:
        return
    have = {s.uri for s in stamps}
    missing = [e for e in todo if e.uri not in have] if ss.get("_skipped") else []
    akey = f"_auto|{tgt.name}|{band}|{ss.fov}"
    t_first = ss.setdefault(akey + "|t", time.time())

    @st.fragment(run_every=3)
    def watch():  # checks every 3 s; redraws the whole page once everything still missing has arrived
        arrived = ((not survey_pending or fut_survey is None or fut_survey.done())
                   and (not gaia_pending or fut_gaia is None or fut_gaia.done())
                   and all(cutouts.is_cached(e, ra0, dec0, ss.fov * 1.45) for e in missing))
        if arrived and ss.get(akey, 0) < 5 and time.time() - t_first < 600:
            ss[akey] = ss.get(akey, 0) + 1
            st.rerun()
        st.info("" + "; ".join(msgs) + ". They are added here automatically as soon as they arrive.")
    watch()


def _go_pair(m1, m2):
    ss["find"] = dict(marks=[m1, m2], field=tgt.name)
    ss["mode"] = "motion"


def _go_spectrum(mark):
    ss["find"] = dict(mark=mark, field=tgt.name)
    ss["mode"] = "spectrum"


def check_my_find(marks, key):
    """'Check my find', only on request: compare two of the user's marks, each on its own photo."""
    if not marks:
        return
    fmt = lambda i: f"#{i + 1} · {marks[i].get('date', '')} · RA {marks[i]['ra']:.5f}°, Dec {marks[i]['dec']:+.5f}°"  # noqa
    with st.container(border=True):
        st.markdown("**Check my find**")
        if len(marks) < 2:
            st.caption("Mark the same object on two different photos (two dates) to measure how it moved between "
                       "them. Only those two photos are compared.")
            st.button("Spectrum of mark #1", on_click=_go_spectrum, args=(marks[0],), key=f"find_sp|{key}",
                      help="Measures the marked object's brightness at every infrared colour SPHEREx sees.")
            return
        c1, c2 = st.columns(2)
        i = c1.selectbox("First mark (earlier photo)", range(len(marks)), index=len(marks) - 2, format_func=fmt,
                         key=f"find_a|{key}", help="The mark on the first photo.")
        j = c2.selectbox("Second mark (later photo)", range(len(marks)), index=len(marks) - 1, format_func=fmt,
                         key=f"find_b|{key}", help="The same object, marked on the second photo.")
        i, j = min(i, len(marks) - 1), min(j, len(marks) - 1)
        b1, b2 = st.columns(2)
        same = abs(marks[i]["mjd"] - marks[j]["mjd"]) < 0.01
        b1.button("Compare these two marks", type="primary", on_click=_go_pair, args=(marks[i], marks[j]),
                  key=f"find_go|{key}", use_container_width=True, disabled=i == j or same,
                  help="Measures how far and how fast the object moved between the two photos you marked it on. "
                       "Only those two photos are used, so fast movers (asteroids, meteors) are not mixed up "
                       "with other photos. If the object is a Gaia star, its predicted move is shown too.")
        b2.button("Spectrum of the first mark", on_click=_go_spectrum, args=(marks[i],), key=f"find_sp|{key}",
                  use_container_width=True,
                  help="Measures the marked object's brightness at every infrared colour SPHEREx sees.")
        if same:
            st.caption("These two marks are on the same photo: pick marks made on two different photos.")


def motion_check(t, note=""):
    """Measured positions of target t in every frame against its predicted path (plot, metrics, table)."""
    r_ref, d_ref = targets.position_at(t, config.REFERENCE_MJD)

    def offset(px, py):  # arcsec east / north of t's catalogue position on 2026-01-01
        r, d = wgrid.pixel_to_world_values(px, py)
        de = ((r - r_ref + 180) % 360 - 180) * np.cos(np.radians(d_ref)) * 3600
        return float(de), float((d - d_ref) * 3600)

    st.subheader(f"Motion check: {t.name} in {band}")
    if note:
        st.caption(note)
    pts = []
    for f in frames:
        r, d = targets.position_at(t, f.mjd_mid)
        par = targets.predicted_offsets(t, [f.mjd_mid], config.REFERENCE_MJD)[0] - \
            targets.predicted_offsets(t, [f.mjd_mid], config.REFERENCE_MJD, with_parallax=False)[0]
        gx, gy = wgrid.world_to_pixel_values(r, d)
        gx, gy = float(gx) - par[0] / arcsec_px, float(gy) + par[1] / arcsec_px  # east is -x
        cx, cy = photometry.centroid(f.image, gx, gy, r=4)
        if not np.isfinite(cx):
            continue
        de, dn = offset(cx, cy)
        err = 0.6 * (1 if f.kind == "survey" else 2)
        if f.kind == "stack" and len(f.stamps) >= 3:  # scatter of single-exposure centroids
            singles = [photometry.centroid(by_uri[s.uri].image, cx, cy, r=4) for s in f.stamps if s.uri in by_uri]
            singles = np.array([offset(*c) for c in singles if np.isfinite(c[0])])
            if len(singles) >= 3:
                err = max(0.3, float(np.hypot(*singles.std(axis=0)) / np.sqrt(len(singles))))
        pe, pn = targets.predicted_offsets(t, [f.mjd_mid], config.REFERENCE_MJD)[0]
        k = int(f.label.split(":")[0].split()[-1]) if f.label.startswith("Epoch") else 0
        pts.append(dict(label=f.label.split(" · ")[0] if k == 0 else f"Epoch {k}", epoch=k, mjd=f.mjd_mid,
                        date=targets.mjd_to_date(f.mjd_mid), de=de, dn=dn, err=err,
                        off=float(np.hypot(de - pe, dn - pn)), par_e=par[0], par_n=par[1], pred_e=pe, pred_n=pn))
    if len(pts) < 2:
        st.info("The object needs to be visible in at least two photos. Try a band where it is bright, or a "
                "larger field of view.")
        return
    m_lo, m_hi = min(p["mjd"] for p in pts) - 60, max(p["mjd"] for p in pts) + 60
    grid_mjd = np.linspace(m_lo, m_hi, int(min(2000, max(200, (m_hi - m_lo) / 3))))
    tr = targets.predicted_offsets(t, grid_mjd, config.REFERENCE_MJD)
    st.plotly_chart(render.motion_fig(tr[:, 0], tr[:, 1], pts), use_container_width=True)
    pending_notes()
    st.plotly_chart(render.miss_fig(pts), use_container_width=True)
    yrs = [(p["mjd"] - config.REFERENCE_MJD) / 365.25 for p in pts]
    fit = photometry.fit_motion(yrs, [p["de"] - p["par_e"] for p in pts], [p["dn"] - p["par_n"] for p in pts],
                                [p["err"] for p in pts])
    cat_e, cat_n = t.pmra / 1000, t.pmdec / 1000
    has_cat = bool(t.pmra or t.pmdec or t.parallax)
    if fit:
        pe, pn, ee, _ = fit
        mu, cat_mu = np.hypot(pe, pn), np.hypot(cat_e, cat_n)
        diff = np.hypot(pe - cat_e, pn - cat_n)
        c1, c2, c3 = st.columns(3)
        c1.metric("Measured motion", f"{mu:.2f} ± {ee:.2f} ″/yr",
                  help="How fast SPHEREx sees the object drift across the sky, in arcseconds per year: the slope "
                       "of a straight line fitted through the measured positions after removing the parallax "
                       "wobble. ± is the uncertainty (1σ).")
        if has_cat:
            c2.metric("Catalogue motion", f"{cat_mu:.2f} ″/yr",
                      help=f"What the catalogue says ({t.source}): the speed the grey line is drawn with.")
            c3.metric("Difference", f"{diff:.2f} ″/yr", f"{diff / ee:.1f} σ", delta_color="off",
                      help="Measured minus catalogue. σ (sigma) counts how many uncertainties apart they are: "
                           "below about 3σ they agree.")
        if t.has_motion:
            verdict = (f"**Agrees** with the catalogue: SPHEREx sees {t.name} move {mu:.2f}″ per year, "
                       f"the catalogue says {cat_mu:.2f}″." if diff < 3 * ee + 0.1 * cat_mu else
                       f"**Differs** from the catalogue by {diff:.2f}″/yr ({diff / ee:.1f}σ). Saturation "
                       "(an overexposed star), a blended neighbour or too few epochs can cause this; check the blink.")
        else:
            verdict = (f"**Moves**: {mu:.2f} ± {ee:.2f}″/yr, more than 3× the uncertainty." if mu > 3 * ee else
                       f"No motion detected: less than about {3 * ee:.2f}″/yr (3σ).")
        st.markdown(verdict)
    st.caption("**Reading the plot:** each coloured point is where the object's light is centred in one photo "
               "(east to the left, as on the sky), relative to its catalogue position on 2026-01-01 (the 0, 0 "
               "point). The grey line is where it *should* be. Error bars show the measuring uncertainty. "
               "Saturated (overexposed) stars and blended neighbours make centres less reliable.")

    plx = t.parallax / 1000
    with st.expander("How the prediction (grey line) and the measurement are calculated", expanded=False):
        st.markdown(
            f"**The prediction** starts from the catalogue ({t.source}) and adds two effects:\n\n"
            f"1. **Proper motion** (the star's own drift): **{cat_e:+.3f}″/yr** east and **{cat_n:+.3f}″/yr** north. "
            "After *t* years the star has drifted *t* × that. *t* is counted from 2026-01-01.\n"
            f"2. **Parallax** (the yearly wobble from watching on a moving Earth): **π = {plx:.4f}″**. Earth's "
            "position around the Sun on each date, *X, Y, Z* in AU (Earth–Sun distances), comes from the planetary "
            "ephemeris in astropy. With the star's RA α and Dec δ:\n\n"
            "   - east shift = π · (X sin α − Y cos α)\n"
            "   - north shift = π · (X cos α sin δ + Y sin α sin δ − Z cos δ)\n"
            "3. **Predicted position** = drift + wobble. Repeating this for many dates draws the grey line; for a "
            "nearby star it is a slanted line with loops (the loops are the parallax).\n\n"
            "**The measurement:** in each epoch photo the app looks within 4 display pixels (12″) of the predicted "
            "spot and takes the *centroid*, the brightness-weighted average position of the light, which can be "
            "pinned down to a fraction of a pixel. The error bar is the spread between the single photos of that "
            "epoch divided by √(number of photos). Finally the parallax part is subtracted from every point and "
            "a straight line is fitted through position versus time (weighted least squares): its slope is the "
            "**measured motion** above.\n\n"
            "**Step by step for this object** (all values in arcseconds, ″):")
        df = pd.DataFrame([{
            "Photo": p["label"], "Date": p["date"],
            "Years from 2026-01-01": round((p["mjd"] - config.REFERENCE_MJD) / 365.25, 3),
            "Drift east": round(p["pred_e"] - p["par_e"], 2), "Drift north": round(p["pred_n"] - p["par_n"], 2),
            "Wobble east": round(p["par_e"], 2), "Wobble north": round(p["par_n"], 2),
            "Predicted east": round(p["pred_e"], 2), "Predicted north": round(p["pred_n"], 2),
            "Measured east": round(p["de"], 2), "Measured north": round(p["dn"], 2),
            "± error": round(p["err"], 2), "Miss (measured − predicted)": round(p["off"], 2)} for p in pts])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption("Drift = proper motion × years. Wobble = parallax shift. Predicted = drift + wobble. "
                   "Miss = distance between the measured and predicted points.")
        st.download_button("Download this table (CSV)", df.to_csv(index=False),
                           f"motion_{t.name.replace(' ', '_')}_{band}.csv", help="Save the step-by-step numbers.")
    with st.expander("Watch it move: the photos with the predicted position (orange ring) on each date"):
        sky_viewer(frames, key_extra=f"motion|{t.name}", follow=t)


def photo_of(mjd):
    """The photo a mark was made on (epoch stack, older survey, or a single exposure), by its date."""
    for f in frames:
        if abs(f.mjd_mid - mjd) < 0.01:
            return f
    for i, a in enumerate(by_uri.values()):
        if abs(0.5 * (a.stamp.mjd + a.stamp.mjd_end) - mjd) < 0.01:
            return stack.single_frame(a, i + 1)
    return None


def where_on(f, m):
    """Centre of the marked object on its own photo (light-weighted), and its uncertainty in arcsec."""
    cx, cy = photometry.centroid(f.image, float(m["x"]), float(m["y"]), r=3)
    if not np.isfinite(cx) or np.hypot(cx - m["x"], cy - m["y"]) > 3:
        cx, cy = float(m["x"]), float(m["y"])
    err = 0.6 if f.kind == "survey" else 1.2
    if f.kind == "stack" and len(f.stamps) >= 3:  # scatter of the single exposures of that photo
        cs = [photometry.centroid(by_uri[s.uri].image, cx, cy, r=3) for s in f.stamps if s.uri in by_uri]
        cs = np.array([c for c in cs if np.isfinite(c[0])])
        if len(cs) >= 3:
            err = max(0.3, float(np.hypot(*cs.std(axis=0)) / np.sqrt(len(cs)) * arcsec_px))
    r, d = wgrid.pixel_to_world_values(cx, cy)
    return float(r), float(d), err, (cx, cy)


def pair_check(m1, m2):
    """Motion of one object between the two photos the user marked it on. No other photo is used, so a fast
    mover (asteroid, meteor) is never confused with what other photos show."""
    if m2["mjd"] < m1["mjd"]:
        m1, m2 = m2, m1
    fa, fb = photo_of(m1["mjd"]), photo_of(m2["mjd"])
    st.subheader(f"Motion between your two marks ({band})")
    if fa is None or fb is None:
        st.warning("One of the photos you marked is not in this view any more (another band or field?). "
                   "Go back, and mark the object again on two photos.")
        return
    ra1, de1, e1, p1 = where_on(fa, m1)
    ra2, de2, e2, p2 = where_on(fb, m2)
    de_ = ((ra2 - ra1 + 180) % 360 - 180) * np.cos(np.radians((de1 + de2) / 2)) * 3600
    dn_ = (de2 - de1) * 3600
    dist, err = float(np.hypot(de_, dn_)), float(np.hypot(e1, e2))
    days = fb.mjd_mid - fa.mjd_mid
    pa = float(np.degrees(np.arctan2(de_, dn_)) % 360)
    way = detect.COMPASS[int(round(pa / 45)) % 8]
    st.caption(f"Photo 1: {fa.label}  \nPhoto 2: {fb.label}  \nOnly these two photos are used. "
               f"Time between them: {days:.2f} days ({days / 365.25:.3f} years).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Moved", f"{dist:.1f} ± {err:.1f}″", help="Distance on the sky between the object's centre on photo 1 "
              "and on photo 2. ± is the measuring uncertainty (1σ) of the two centres together.")
    c2.metric("Direction", f"{way} ({pa:.0f}°)", help="Direction of the move, in degrees from north towards east.")
    speed = (f"{dist / (days / 365.25):.2f}″ per year" if days > 30 else
             f"{dist / days:.2f}″ per day" if days > 1 else f"{dist / (days * 24):.2f}″ per hour")
    c3.metric("Speed", speed if days > 0 else "–", help="Distance divided by the time between the two photos.")
    if dist > 3 * err:
        st.markdown(f"**It moved**: {dist:.1f}″ toward the {way}, more than 3× the uncertainty ({3 * err:.1f}″).")
    else:
        st.markdown(f"**No clear motion** between these two photos: the shift ({dist:.1f}″) is within 3× the "
                    f"uncertainty ({3 * err:.1f}″). One SPHEREx pixel is 6.15″.")
    t, note = resolve_find()
    pred = None
    if t is not None and (t.pmra or t.pmdec or t.parallax):
        o = targets.predicted_offsets(t, [fa.mjd_mid, fb.mjd_mid], config.REFERENCE_MJD)
        pred = (float(o[1][0] - o[0][0]), float(o[1][1] - o[0][1]))
        miss = float(np.hypot(de_ - pred[0], dn_ - pred[1]))
        st.markdown(("**Agrees** with Gaia" if miss < 3 * err else "**Differs** from Gaia") +
                    f": Gaia predicts a move of {np.hypot(*pred):.1f}″ between these two dates; the measured move is "
                    f"{miss:.1f}″ away from that (uncertainty {err:.1f}″).")
    st.caption(note)
    st.plotly_chart(render.pair_fig(de_, dn_, e1, e2, pred), use_container_width=True)
    st.dataframe(pd.DataFrame([
        {"Mark": "photo 1", "Photo": fa.label, "RA (°)": round(ra1, 6), "Dec (°)": round(de1, 6), "± (″)": round(e1, 2)},
        {"Mark": "photo 2", "Photo": fb.label, "RA (°)": round(ra2, 6), "Dec (°)": round(de2, 6), "± (″)": round(e2, 2)}]),
        hide_index=True, use_container_width=True)
    st.caption("The centres are measured again on each photo (light-weighted centre near your mark), so a mark a "
               "little off the object does not matter.")
    with st.expander("The two photos (blink them with Play)"):
        sky_viewer([fa, fb], key_extra="pair", follow=t)


if ss.mode == "motion":
    load_secs = load.done()
    f_ = ss.get("find") or {}
    if f_.get("field") == tgt.name and f_.get("marks"):
        st.button(f"Back to {tgt.name}" if tgt.source != "coordinates" else "Back", on_click=_drop_find,
                  help="Forget these marks and show the usual motion check.")
        pair_check(*f_["marks"][:2])
        st.stop()
    if tgt.source == "coordinates":
        st.subheader(f"Motion check at {tgt.name} ({band})")
        st.info("A motion check compares where a **known object** is predicted to be with where SPHEREx sees it. "
                "You typed a sky position, so nothing is followed here.\n\n"
                "**Mark it yourself:** choose *Mark*, click an object on the first date, switch to a later "
                "date with Next or the timeline, and click the *same* object again. The panel shows how far it moved "
                "and its speed in ″ per year. Then **Check my find** below compares it with the Gaia catalogue.")
        check_my_find(sky_viewer(frames, key_extra="motion"), "motion")
        pending_notes()
        st.caption(f"{len(stamps)} photos loaded in {load_secs:.1f} s. Stars move very little: anything below "
                   "about 3″ (half a SPHEREx pixel) between dates is within the measuring error.")
        st.stop()
    motion_check(tgt, f"{len(stamps)} photos loaded in {load_secs:.1f} s")
    st.stop()

load_secs = load.done()
if tgt.source == "coordinates":
    st.caption(COORD_NOTE)
tab_view, tab_exp = st.tabs(["Sky viewer", "Photo list"])
with tab_view:
    mode = ss.mode
    diverging = False
    if mode == "blink" or (mode in ("add", "sub") and len(frames) < 2 and mode != "add"):
        shown = frames
        if mode == "sub":
            st.info("Subtract needs photos from at least two dates (epochs) in this band.")
    elif mode == "add":
        ep_names = [f.label.split(" · ")[0] for f in sphx_frames]
        pick = st.multiselect("Dates to add together", range(len(sphx_frames)), default=list(range(len(sphx_frames))),
                              format_func=lambda i: ep_names[i],
                              help="Adding more dates lowers the noise, so fainter things show up. Leave out a date "
                                   "to see what it brings, or to remove one with a problem.")
        use = [sphx_frames[i] for i in pick] or sphx_frames
        shown = [stack.add(use)]
        if len(use) > 1:
            # random noise of one date, from the difference of two dates (the stars cancel there)
            sig1 = detect.noise(use[-1].image - use[0].image) / np.sqrt(2)
            gain = float(np.sqrt(sum(f.n for f in use) / np.median([f.n for f in use])))
            st.caption(f"**Depth gained:** {len(use)} dates ({sum(f.n for f in use)} photos) added → the random noise "
                       f"is {gain:.1f}× lower than on one date (about {sig1 / gain:.3g} instead of {sig1:.3g} MJy/sr), "
                       f"so stars about {2.5 * np.log10(gain):.1f} magnitudes fainter can show up (1 magnitude ≈ "
                       "2.5× fainter). In crowded fields faint stars also blend together, which adding cannot undo.")
    elif mode == "sub":
        labels = [f.label.split(":")[0] for f in frames]
        c1, c2 = st.columns(2)
        ia = c1.selectbox("Older photo (A)", range(len(frames)), index=len(frames) - len(sphx_frames),
                          format_func=lambda i: labels[i], help="The 'before' picture.")
        ib = c2.selectbox("Newer photo (B)", range(len(frames)), index=len(frames) - 1,
                          format_func=lambda i: labels[i], help="The 'after' picture. The view shows B − A.")
        diff = stack.subtract(frames[ib], frames[ia])
        cut = st.slider("Hide changes smaller than (× noise)", 0.0, 6.0, 0.0, 0.5,
                        help="Sets every pixel whose change is smaller than this many noise levels (σ) to 'no "
                             "change' (white), so only clear changes stay coloured. 0 shows everything.")
        if cut > 0:
            import dataclasses
            nz = detect.noise(diff.image)
            diff = dataclasses.replace(diff, image=np.where(np.abs(diff.image) < cut * nz, 0.0, diff.image))
        shown = [diff]
        diverging = True
        with st.expander("How to read the colours"):
            st.markdown("**Red** = brighter now (in B) · **white / grey** = no change (steady stars cancel out) · "
                        "**blue** = was brighter before (in A).\n\nA **moving star** leaves a blue + red pair — blue "
                        "where it *was*, red where it *is now*. A star that **changed brightness** leaves a single red "
                        "(brightened) or blue (faded) spot. Small rings around very bright stars are leftovers of "
                        "imperfect cancelling, not real changes. The stronger the colour, the bigger the change.")
    else:  # movie
        order = st.radio("Order", ["time", "wavelength"], horizontal=True,
                         help="time: oldest to newest. wavelength: from the bluest to the reddest infrared "
                              "colour inside the band.")
        al = sorted(by_uri.values(), key=lambda a: a.stamp.mjd if order == "time" else a.stamp.wavelength)
        nflag = sum(bool(a.bad_near_target) for a in al)
        if nflag == len(al):
            st.caption(f"All {nflag} photos have warnings at the target (it is probably overexposed in this band), "
                       "so none are hidden. Try another band.")
        elif nflag and st.toggle(f"Hide photos with warnings at the target ({nflag} of {len(al)})", value=False,
                               help="Leaves out single photos whose pixels at the target were marked unreliable by the "
                                    "NASA pipeline (overexposed, cosmic-ray hits…)."):
            al = [a for a in al if not a.bad_near_target] or al
        al = archive.evenly(al, config.MOVIE_CAP)
        shown = [stack.single_frame(a, i + 1) for i, a in enumerate(al)]

    # Optional helpers, off until switched on (the user finds things; the tools measure them). Their
    # switches sit below the viewer; their values are known here already, so the viewer can draw results.
    ss.setdefault("h_sigma", 5.0)
    labels = [f.label.split(" · ")[0] for f in frames]
    h_stars, h_change = ss.get("h_stars", False), ss.get("h_change", False) and mode in ("blink", "sub")
    if mode == "sub" and len(frames) >= 2:
        pair = (ia, ib)
    else:
        for k_, d_ in (("h_a", len(frames) - len(sphx_frames)), ("h_b", len(frames) - 1)):
            if not 0 <= ss.get(k_, -1) < len(frames):
                ss[k_] = max(0, d_)
        pair = (ss.h_a, ss.h_b)
    def singles_of(f):  # the aligned single exposures behind an epoch photo: each star must be in most of them
        return [by_uri[x.uri].image for x in f.stamps if x.uri in by_uri] if f.kind == "stack" else None
    found = [detect.find_stars(f.image, ss.h_sigma, singles=singles_of(f)) for f in shown] if h_stars else None
    changed = []
    if h_change and len(frames) >= 2 and pair[0] != pair[1]:
        changed = detect.changes(frames[pair[0]].image, frames[pair[1]].image, ss.h_sigma,
                                 singles_a=singles_of(frames[pair[0]]), singles_b=singles_of(frames[pair[1]]))[:40]
        yrs = abs(frames[pair[1]].mjd_mid - frames[pair[0]].mjd_mid) / 365.25
        for i, c in enumerate(changed):
            c["label"] = f"C{i + 1}"
            if c["kind"] == "moved":  # in arcseconds, and per year, like the marking tool
                arc = c["dist_px"] * arcsec_px
                c["text"] = (f"moved {arc:.1f}″ toward the {detect.COMPASS[int(round(c['pa_deg'] / 45)) % 8]}"
                             + (f" ({arc / yrs:.1f}″ per year)" if yrs > 0.05 else ""))
    my_marks = sky_viewer(shown, diverging, key_extra=mode, stars=found, changes=changed)
    pending_notes()
    check_my_find(my_marks, mode)

    with st.expander("Helpers — off until you switch them on", expanded=bool(h_stars or h_change)):
        st.caption("The app does not point things out by itself: look first, then let a helper check what you saw.")
        h1, h2, h3 = st.columns(3)
        h1.toggle("Detect stars", key="h_stars",
                  help="Finds spots well above the noise on each photo and draws small green crosses on them. "
                       "Faint or crowded stars can be missed: one SPHEREx pixel is 6.15″ wide.")
        if mode in ("blink", "sub"):
            h2.toggle("Find what changed", key="h_change",
                      help="Compares two dates and lists stars that moved, brightened or faded, strongest first. "
                           "Labels C1, C2… appear on the image.")
        h3.slider("Detection threshold (σ)", 3.0, 10.0, key="h_sigma", step=0.5,
                  help="How far above the noise a spot must be. Higher = fewer but surer detections. "
                       "σ (sigma) is the noise level of the photo.")
        if h_stars and found:
            st.caption(f"Detected stars: {', '.join(f'{len(s)}' for s in found)} (per photo, above {ss.h_sigma:g}σ).")
        if h_change:
            if mode == "blink":
                p1, p2 = st.columns(2)
                p1.selectbox("Compare from (older)", range(len(frames)), key="h_a", format_func=lambda i: labels[i],
                             help="The 'before' photo for Find what changed.")
                p2.selectbox("…to (newer)", range(len(frames)), key="h_b", format_func=lambda i: labels[i],
                             help="The 'after' photo.")
            if any(frames[i].kind == "survey" for i in pair):
                st.caption("Note: one photo is from an older survey (another telescope, scaled for display): only "
                           "*moved* results mean something; brightness changes between telescopes do not.")
            if changed:
                rows = []
                for c in changed:
                    r1, d1 = wgrid.pixel_to_world_values(c["x"], c["y"])
                    rows.append({"label": c["label"], "what": c["kind"], "details": c["text"],
                                 "RA (°)": round(float(r1), 5), "Dec (°)": round(float(d1), 5),
                                 "strength (σ)": round(c["score"], 1)})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                             column_config={"strength (σ)": st.column_config.NumberColumn(
                                 help="How many noise levels the change is; above ~5 it is very likely real")})
                st.caption(f"{labels[pair[0]]} → {labels[pair[1]]}. Check each one by eye: rings around very bright "
                           "stars and stars at the edge can be false alarms.")
            elif pair[0] == pair[1]:
                st.info("Pick two different photos to compare.")
            else:
                st.info(f"No change above {ss.h_sigma:g}σ between {labels[pair[0]]} and {labels[pair[1]]}.")
        st.caption("*Note: the helpers are experimental. They can miss things or report false alarms, so always check "
                   "their results by eye.*")

    c1, c2, c3 = st.columns([2, 2, 3])
    if not load_all and n_band > len(todo):
        c1.button(f"Load all {band} photos ({min(n_band, config.LOAD_ALL_CAP)} of {n_band})",
                  on_click=lambda: ss.__setitem__(load_key, True),
                  help="The quick look averages a few photos per date. Loading all gives a cleaner, deeper image "
                       "but takes longer.")
    _gif_args = (shown, ss.stretch, diverging, ss.speed)  # built only when the button is pressed
    c2.download_button("Download GIF", lambda: render.to_gif(*_gif_args),
                       file_name=f"spherex_{tgt.name.replace(' ', '_')}_{band}_{mode}.gif", mime="image/gif",
                       help="Save the animation as a GIF to share.")
    with c3.expander("What am I looking at?"):
        st.markdown(
            "Each frame is a SPHEREx photo of the same patch of sky, taken months apart and averaged per "
            "*epoch* (one pass of the survey). North is up, east is left; the white bar is 1 arcminute (1′). "
            + ("The small orange cross marks the position you typed. "
               if tgt.source == "coordinates" else
               "The orange ring is where the target should be on that date, from its catalogue proper motion. ")
            + "Blue circles are Gaia stars. A star that moves (like Barnard's Star) shifts against the "
            "background; a variable star changes brightness. *Compare* in the viewer puts two dates side by "
            "side, under a swipe divider or subtracted. Motion between epochs is often smaller than one SPHEREx "
            "pixel (6.15″), so the image is shown 2× finer than the camera's pixels. Your marks stay in your "
            "browser (remembered on this computer).")

with tab_exp:
    rows = [dict(date=targets.mjd_to_date(a.stamp.mjd), mjd=round(a.stamp.mjd, 4), band=a.stamp.band,
                 wavelength_um=round(a.stamp.wavelength, 4), flags_at_target=", ".join(a.bad_near_target),
                 file=a.stamp.uri and f"{config.IRSA}/{a.stamp.uri}") for a in by_uri.values()]
    st.dataframe(pd.DataFrame(rows).sort_values("mjd"), hide_index=True, use_container_width=True,
                 column_config={
                     "date": st.column_config.TextColumn("date", help="Day the photo was taken"),
                     "mjd": st.column_config.NumberColumn("MJD", help="Modified Julian Date: astronomers' day count"),
                     "wavelength_um": st.column_config.NumberColumn("wavelength (µm)",
                                                                    help="Infrared colour at the target in this photo"),
                     "flags_at_target": st.column_config.TextColumn("warnings at target",
                                                                    help="Pipeline flags on the pixels at the target "
                                                                         "(e.g. overexposed)"),
                     "file": st.column_config.LinkColumn("file", display_text="FITS",
                                                         help="The original data file at NASA/IPAC")})
    st.caption(f"{len(exps)} SPHEREx photos cover this position in all bands; {n_band} in {band}.")

last, versions = archive.data_through(exps)
where = (f"Position RA {ra0:.5f}, Dec {dec0:+.5f}. " if tgt.source == "coordinates" else
         f"{tgt.name}: RA {ra0:.5f}, Dec {dec0:+.5f} on 2026-01-01 ({tgt.source}). ")
st.caption(where +
           f"SPHEREx data through {targets.mjd_to_date(last)} · processing {', '.join(versions)} · "
           f"{gaia_note} · loaded in {load_secs:.1f} s · "
           "Data: NASA/IPAC Infrared Science Archive (IRSA, via its AWS copy), ESA Gaia.")

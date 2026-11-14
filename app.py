"""SPHEREx Blink Viewer: type a sky position or object name, watch SPHEREx images of it blink."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from astropy.time import Time

from spx import archive, catalogs, config, cutouts, photometry, render, stack, targets, viewer, warp

st.set_page_config(page_title="SPHEREx Blink Viewer", page_icon="🔭", layout="wide")

PRESET_BAND = targets.PRESET_BAND
MODES = {"blink": "Blink epochs", "add": "Add", "sub": "Subtract", "movie": "Every exposure",
         "spectrum": "Spectrum & light curve", "motion": "Motion check"}
MODE_LABEL = {"blink": "▶ Blink", "add": "➕ Add", "sub": "➖ Subtract", "movie": "🎞 Every exposure",
              "spectrum": "🌈 Spectrum", "motion": "🧭 Motion check"}
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


def _pick(name):
    ss.target_text = name
    ss.band = PRESET_BAND.get(name, ss.band)


_init()

def _input_hint(text):
    """One line under the search box saying how the text will be read."""
    if targets.preset(text):
        st.caption("⭐ A challenge target (built-in catalogue values)")
    elif (t := targets.parse_coords(text)) is not None:
        st.caption(f"📍 Read as a sky position: RA {t.ra:.5f}°, Dec {t.dec:+.5f}°")
    elif targets.looks_like_coords(text):
        st.caption("⚠️ This looks like numbers but is not a valid position. Use two numbers in degrees: "
                   "RA from 0 to 360, then Dec from −90 to +90, e.g. `269.452 4.693`.")
    elif text.strip():
        st.caption("🔎 Will be looked up by name in the SIMBAD star catalogue")


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
with st.sidebar:
    st.header("🔭 SPHEREx Blink")
    st.markdown("**① What to look at**")
    st.text_input("Object name or sky position (RA Dec in degrees)", key="target_text",
                  placeholder="Barnard's Star  ·  M 42  ·  269.452 4.693",
                  help="Type the name of a star, galaxy or nebula (looked up in the SIMBAD catalogue), or a "
                       "sky position as two numbers in degrees: RA (0–360) then Dec (−90 to +90), for "
                       "example `269.452 4.693` or `269.452°, +4.693°`. Hours-minutes-seconds also work: "
                       "`17 57 48.5 +04 41 36`. Press Enter or 🔍 Go.")
    st.button("🔍 Go", type="primary", use_container_width=True, help="Load this object or position")
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
    with st.expander("📖 Words used here"):
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
qp["fov"], qp["band"], qp["mode"] = str(ss.fov), ss.band, ss.mode


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
        self.box.markdown(warp.card(self.title, self.detail, self.stage, self.frac, self.ptext),
                          unsafe_allow_html=True)

    def step(self, title, detail="", stage=0):
        if stage != self.stage:
            self.frac, self.ptext = None, ""
        self.title, self.detail, self.stage = title, detail, stage
        self._draw()

    def progress(self, frac, text):
        self.frac, self.ptext = min(1.0, frac), text
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
with st.expander("❓ How to use this page (30 seconds)"):
    st.markdown(
        "1. **Choose a target** in the sidebar: a challenge target, any name (e.g. *M 42*), or a sky position "
        "in degrees (e.g. `269.452 4.693`). Press **🔍 Go**.\n"
        "2. **Press ▶ Play** under the image. The photos are months apart: a star that *jumps* is moving, one "
        "that *flickers* changes brightness.\n"
        "3. **Switch the view** with the buttons above: *Subtract* shows only what changed, *Spectrum* shows "
        "colour, *Motion check* compares the measured path with the predicted one.\n"
        "4. **📍 Mark** (in the viewer's toolbar): click to mark objects, draw lines to measure sizes and "
        "distances, and mark the same object on two dates to get its speed. *Compare* puts two dates side "
        "by side, under a swipe divider, or subtracts them.\n\n"
        "Hover over the small ❔ icons for short explanations; *📖 Words used here* in the sidebar explains "
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

load.step("Asking NASA's archive which photos cover this spot…",
          "SPHEREx has photographed every part of the sky several times since 2025", 1)
try:
    exps = search(round(ra0, 4), round(dec0, 4))
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
    out, nfail = cutouts.fetch_many(todo, ra0, dec0, ss.fov * 1.45,
                                    lambda d, n: load.progress(d / n, f"{d} of {n} photos"),
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


if ss.mode == "spectrum":
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
        m = photometry.measure(s, *targets.position_at(tgt, s.mjd))
        k = epoch_of[s.uri]
        m.update(date=targets.mjd_to_date(s.mjd), datetime=Time(s.mjd, format="mjd").iso,
                 epoch=1 + (k - 1) * len(render.EPOCH_COLORS) // nep if fold else k)
        rows.append(m)
    by_epoch = {}
    for r in sorted(rows, key=lambda r: (r["epoch"], r["wavelength"])):
        by_epoch.setdefault(r["epoch"], []).append(r)
    load_secs = load.done()
    st.subheader(f"{tgt.name}: SPHEREx spectrum, 0.75–5.0 µm")
    st.caption(f"{len(stamps)} images loaded in {load_secs:.1f} s")
    close = {}
    try:  # warn when a Gaia neighbour falls inside the 12" aperture (SPHEREx pixels are 6.15")
        nb, _ = catalogs.gaia_cone(round(ra0, 4), round(dec0, 4), 0.02)
        mids = [min(r["mjd"] for r in rows), max(r["mjd"] for r in rows)]
        pr0, pd0 = targets.position_at(tgt, mids[0])
        gr0, gd0 = catalogs.gaia_at(nb, mids[0])
        sep0 = np.hypot((gr0 - pr0) * np.cos(np.radians(pd0)), gd0 - pd0) * 3600
        own = [s["g"] for s, d in zip(nb, sep0) if d < 2.0]
        g_limit = (min(own) + 3) if own else 99  # only neighbours within 3 mag of the target matter
        for m in mids:
            pr, pd_ = targets.position_at(tgt, m)
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
    logy = st.toggle("Logarithmic flux axis", value=False)
    st.plotly_chart(render.spectrum_fig(by_epoch, logy), use_container_width=True)
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
                    + ("\n\n⚠ The neighbouring star above moves relative to the target, so part of any change "
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


def target_at(mjd):
    """Where the target is on that date, in grid pixels (a ring), or the typed position (a cross)."""
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


def sky_offset(px, py):
    r, d = wgrid.pixel_to_world_values(px, py)
    de = ((r - ra0 + 180) % 360 - 180) * np.cos(np.radians(dec0)) * 3600
    return float(de), float((d - dec0) * 3600)


COORD_NOTE = ("📍 You typed a sky position, so nothing is followed here (the orange cross marks the spot). "
              "Look for anything that moves or changes yourself, then **📍 Mark** it on two dates to measure it.")


def sky_viewer(shown_frames, diverging=False, key_extra=""):
    """The sky viewer (spx/viewer) on the given frames. Returns the user's marks (list of dicts)."""
    cx, cy = (float(v) - 1 for v in wgrid.wcs.crpix)
    data = viewer.payload(shown_frames, target=[target_at(f.mjd_mid) for f in shown_frames],
                          gaia=[gaia_on(f.mjd_mid) for f in shown_frames], scale=arcsec_px, center=(cx, cy),
                          radec=(ra0, dec0), diverging=diverging, mode=ss.mode,
                          title=f"{tgt.name}|{band}|{ss.fov}|{key_extra}")
    store = f"{tgt.name}|{ss.fov}|{band}|{key_extra}"
    first_sphx = len(shown_frames) - len(sphx_frames) if len(shown_frames) == len(frames) else 0
    res = viewer.show(data, key=f"viewer|{ss.mode}", store_key=store,
                      settings=dict(stretch=ss.stretch, speed=ss.speed, gaia=False, flags=ss.showflags,
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
        st.info("⏳ " + "; ".join(msgs) + ". They are added here automatically as soon as they arrive.")
    watch()


if ss.mode == "motion":
    load_secs = load.done()
    if tgt.source == "coordinates":
        st.subheader(f"Motion check at {tgt.name} ({band})")
        st.info("A motion check compares where a **known object** is predicted to be with where SPHEREx sees it. "
                "You typed a sky position, so there is no catalogued object (and no prediction) here.\n\n"
                "**Do it by hand instead:** choose 📍 *Mark*, click a star on the first date, switch to a later "
                "date with ▶ or the timeline, and click the *same* star again. The panel shows how far it moved "
                "and its speed in ″ per year. If you mark nothing, there is no marked object to measure.")
        sky_viewer(frames, key_extra="motion")
        pending_notes()
        st.caption(f"{len(stamps)} photos loaded in {load_secs:.1f} s. Stars move very little: anything below "
                   "about 3″ (half a SPHEREx pixel) between dates is within the measuring error.")
        st.stop()
    st.subheader(f"Motion check: {tgt.name} in {band}")
    st.caption(f"{len(stamps)} photos loaded in {load_secs:.1f} s")
    pts = []
    for f in frames:
        r, d = targets.position_at(tgt, f.mjd_mid)
        par = targets.predicted_offsets(tgt, [f.mjd_mid], config.REFERENCE_MJD)[0] - \
            targets.predicted_offsets(tgt, [f.mjd_mid], config.REFERENCE_MJD, with_parallax=False)[0]
        gx, gy = wgrid.world_to_pixel_values(r, d)
        gx, gy = float(gx) - par[0] / arcsec_px, float(gy) + par[1] / arcsec_px  # east is -x
        cx, cy = photometry.centroid(f.image, gx, gy, r=4)
        if not np.isfinite(cx):
            continue
        de, dn = sky_offset(cx, cy)
        err = 0.6 * (1 if f.kind == "survey" else 2)
        if f.kind == "stack" and len(f.stamps) >= 3:  # scatter of single-exposure centroids
            singles = [photometry.centroid(by_uri[s.uri].image, cx, cy, r=4) for s in f.stamps if s.uri in by_uri]
            singles = np.array([sky_offset(*c) for c in singles if np.isfinite(c[0])])
            if len(singles) >= 3:
                err = max(0.3, float(np.hypot(*singles.std(axis=0)) / np.sqrt(len(singles))))
        pe, pn = targets.predicted_offsets(tgt, [f.mjd_mid], config.REFERENCE_MJD)[0]
        k = int(f.label.split(":")[0].split()[-1]) if f.label.startswith("Epoch") else 0
        pts.append(dict(label=f.label.split(" · ")[0] if k == 0 else f"Epoch {k}", epoch=k, mjd=f.mjd_mid,
                        date=targets.mjd_to_date(f.mjd_mid), de=de, dn=dn, err=err,
                        off=float(np.hypot(de - pe, dn - pn)), par_e=par[0], par_n=par[1], pred_e=pe, pred_n=pn))
    if len(pts) < 2:
        st.info("The target needs to be visible in at least two photos. Try a band where it is bright, or a "
                "larger field of view.")
        st.stop()
    m_lo, m_hi = min(p["mjd"] for p in pts) - 60, max(p["mjd"] for p in pts) + 60
    grid_mjd = np.linspace(m_lo, m_hi, int(min(2000, max(200, (m_hi - m_lo) / 3))))
    tr = targets.predicted_offsets(tgt, grid_mjd, config.REFERENCE_MJD)
    st.plotly_chart(render.motion_fig(tr[:, 0], tr[:, 1], pts), use_container_width=True)
    pending_notes()
    yrs = [(p["mjd"] - config.REFERENCE_MJD) / 365.25 for p in pts]
    fit = photometry.fit_motion(yrs, [p["de"] - p["par_e"] for p in pts], [p["dn"] - p["par_n"] for p in pts],
                                [p["err"] for p in pts])
    cat_e, cat_n = tgt.pmra / 1000, tgt.pmdec / 1000
    if fit:
        pe, pn, ee, _ = fit
        mu, cat_mu = np.hypot(pe, pn), np.hypot(cat_e, cat_n)
        diff = np.hypot(pe - cat_e, pn - cat_n)
        c1, c2, c3 = st.columns(3)
        c1.metric("Measured motion", f"{mu:.2f} ± {ee:.2f} ″/yr",
                  help="How fast SPHEREx sees the target drift across the sky, in arcseconds per year: the slope "
                       "of a straight line fitted through the measured positions after removing the parallax "
                       "wobble. ± is the uncertainty (1σ).")
        c2.metric("Catalogue motion", f"{cat_mu:.2f} ″/yr",
                  help=f"What the catalogue says ({tgt.source}): the speed the grey line is drawn with.")
        c3.metric("Difference", f"{diff:.2f} ″/yr", f"{diff / ee:.1f} σ", delta_color="off",
                  help="Measured minus catalogue. σ (sigma) counts how many uncertainties apart they are: "
                       "below about 3σ they agree.")
        if tgt.has_motion:
            verdict = (f"✅ **Agrees** with the catalogue: SPHEREx sees {tgt.name} move {mu:.2f}″ per year, "
                       f"the catalogue says {cat_mu:.2f}″." if diff < 3 * ee + 0.1 * cat_mu else
                       f"⚠ **Differs** from the catalogue by {diff:.2f}″/yr ({diff / ee:.1f}σ). Saturation "
                       "(an overexposed star), a blended neighbour or too few epochs can cause this; check the blink.")
        else:
            verdict = (f"🚀 **Moves**: {mu:.2f} ± {ee:.2f}″/yr, more than 3× the uncertainty." if mu > 3 * ee else
                       f"No motion detected: less than about {3 * ee:.2f}″/yr (3σ).")
        st.markdown(verdict)
    st.caption("**Reading the plot:** each coloured point is where the target's light is centred in one photo "
               "(east to the left, as on the sky), relative to its catalogue position on 2026-01-01 (the 0, 0 "
               "point). The grey line is where it *should* be. Error bars show the measuring uncertainty. "
               "Saturated (overexposed) stars and blended neighbours make centres less reliable.")

    plx = tgt.parallax / 1000
    with st.expander("🧮 How the prediction (grey line) and the measurement are calculated", expanded=False):
        st.markdown(
            f"**The prediction** starts from the catalogue ({tgt.source}) and adds two effects:\n\n"
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
            "**Step by step for this target** (all values in arcseconds, ″):")
        st.dataframe(pd.DataFrame([{
            "Photo": p["label"], "Date": p["date"],
            "Years from 2026-01-01": round((p["mjd"] - config.REFERENCE_MJD) / 365.25, 3),
            "Drift east": round(p["pred_e"] - p["par_e"], 2), "Drift north": round(p["pred_n"] - p["par_n"], 2),
            "Wobble east": round(p["par_e"], 2), "Wobble north": round(p["par_n"], 2),
            "Predicted east": round(p["pred_e"], 2), "Predicted north": round(p["pred_n"], 2),
            "Measured east": round(p["de"], 2), "Measured north": round(p["dn"], 2),
            "± error": round(p["err"], 2), "Miss (measured − predicted)": round(p["off"], 2)} for p in pts]),
            hide_index=True, use_container_width=True)
        st.caption("Drift = proper motion × years. Wobble = parallax shift. Predicted = drift + wobble. "
                   "Miss = distance between the measured and predicted points.")
    st.stop()

load_secs = load.done()
if tgt.source == "coordinates":
    st.caption(COORD_NOTE)
tab_view, tab_exp = st.tabs(["🔭 Sky viewer", "📋 Photo list"])
with tab_view:
    mode = ss.mode
    diverging = False
    if mode == "blink" or (mode in ("add", "sub") and len(frames) < 2 and mode != "add"):
        shown = frames
        if mode == "sub":
            st.info("Subtract needs photos from at least two dates (epochs) in this band.")
    elif mode == "add":
        shown = [stack.add(sphx_frames)]
    elif mode == "sub":
        labels = [f.label.split(":")[0] for f in frames]
        c1, c2 = st.columns(2)
        ia = c1.selectbox("Older photo (A)", range(len(frames)), index=len(frames) - len(sphx_frames),
                          format_func=lambda i: labels[i], help="The 'before' picture.")
        ib = c2.selectbox("Newer photo (B)", range(len(frames)), index=len(frames) - 1,
                          format_func=lambda i: labels[i], help="The 'after' picture. The view shows B − A.")
        shown = [stack.subtract(frames[ib], frames[ia])]
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
        al = archive.evenly(al, config.MOVIE_CAP)
        shown = [stack.single_frame(a, i + 1) for i, a in enumerate(al)]

    sky_viewer(shown, diverging, key_extra=mode)
    pending_notes()

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

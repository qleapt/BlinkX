"""SPHEREx Blink Viewer: type a sky position or object name, watch SPHEREx images of it blink."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from spx import archive, catalogs, config, cutouts, render, stack, targets

st.set_page_config(page_title="SPHEREx Blink Viewer", page_icon="🔭", layout="wide")

PRESET_BAND = {"Barnard's Star": "D4", "Epsilon Indi B": "D6", "WISE 0855-0714": "D6",
               "RW Aurigae": "D3", "R Draconis": "D3"}
MODES = {"blink": "Blink epochs", "add": "Add (all epochs)", "sub": "Subtract (B − A)", "movie": "Every exposure"}
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
    ss.showflags = False


def _pick(name):
    ss.target_text = name
    ss.band = PRESET_BAND.get(name, ss.band)


_init()

# --- sidebar ------------------------------------------------------------------------
with st.sidebar:
    st.header("🔭 SPHEREx Blink")
    st.text_input("Object name or RA Dec (degrees)", key="target_text",
                  help="e.g. Barnard's Star, M 42, or 269.452 4.693")
    st.caption("Challenge targets")
    cols = st.columns(2)
    for i, name in enumerate(targets.PRESETS):
        cols[i % 2].button(name, on_click=_pick, args=(name,), use_container_width=True)
    st.radio("Field of view (arcmin)", config.FOV_CHOICES, key="fov", horizontal=True)
    st.selectbox("Band", list(config.BANDS), key="band", format_func=BAND_LABEL.get)
    st.radio("View", list(MODES), key="mode", format_func=MODES.get)
    st.toggle("Gaia DR3 stars (moved to each date)", key="gaia")
    st.toggle("Show flagged pixels", key="showflags",
              help="Pink: the shown value comes from pixels the pipeline flagged (saturation, cosmic rays…).")
    with st.expander("Display"):
        st.selectbox("Stretch", render.STRETCHES, key="stretch")
        st.slider("Frame time (ms)", 200, 2000, key="speed", step=100)

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


@st.cache_data(ttl=config.TAP_CACHE_HOURS * 3600, show_spinner=False)
def search(ra, dec):
    return archive.search(ra, dec)


st.title("SPHEREx Blink Viewer")
try:
    with st.spinner("Looking up the target…"):
        tgt = resolve(ss.target_text)
except targets.TargetNotFound:
    st.error(f"Could not find **{ss.target_text}**. Try another name (SIMBAD) or coordinates like `269.452 4.693`.")
    st.stop()
except Exception as e:  # SIMBAD down
    st.error(f"Name lookup failed ({e}). Coordinates like `269.452 4.693` always work.")
    st.stop()

ra0, dec0 = targets.position_at(tgt, config.REFERENCE_MJD)
try:
    with st.spinner("Asking the NASA/IPAC archive which SPHEREx images cover this spot (5–15 s)…"):
        exps = search(round(ra0, 4), round(dec0, 4))
except archive.ArchiveBusy:
    st.warning("The NASA/IPAC archive is busy right now. Please try again in a minute.")
    st.stop()
if not exps:
    st.info(f"SPHEREx has no images covering {tgt.name} yet. New data arrive weekly.")
    st.stop()

band = ss.band
load_key = f"all|{tgt.name}|{band}"
load_all = ss.get(load_key, False)
groups = archive.renumber(archive.select(exps, band, quick=not load_all))
n_band = sum(e.band == band for e in exps)
todo = [e for g in groups.values() for e in g]
size = ss.fov * 1.45  # margin so the rotated cutouts cover the square grid

with st.status(f"Fetching {len(todo)} {band} cutouts…", expanded=False) as status:
    bar = st.progress(0.0)
    stamps, fails = cutouts.fetch_many(todo, ra0, dec0, size,
                                       lambda d, n: bar.progress(d / n, f"{d}/{n} cutouts"))
    status.update(label=f"{len(stamps)} {band} cutouts ready" + (f" ({fails} failed)" if fails else ""),
                  state="complete")
if not stamps:
    st.warning("No cutouts could be downloaded (archive busy?). Please try again in a minute.")
    st.stop()

wgrid, shape = stack.grid(ra0, dec0, ss.fov)
arcsec_px = config.PIXEL_SCALE_ARCSEC / config.OVERSAMPLE
by_uri = {}
for s in stamps:
    a = stack.align(s, wgrid, shape)
    a.bad_near_target = stack.flags_near(s, *targets.position_at(tgt, s.mjd))
    by_uri[s.uri] = a

frames = []
for k, g in groups.items():
    items = [by_uri[e.uri] for e in g if e.uri in by_uri]
    if items:
        frames.append(stack.stack_frame(items, f"Epoch {k}: "))


def target_mark(mjd):
    r, d = targets.position_at(tgt, mjd)
    x, y = wgrid.world_to_pixel_values(r, d)
    return dict(x=[float(x)], y=[float(y)], name="predicted position", symbol="circle-open",
                color="#ffb000", size=22, text=[f"{tgt.name} predicted on {targets.mjd_to_date(mjd)}"])


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def gaia(ra, dec, r):
    return catalogs.gaia_cone(ra, dec, r)


gaia_stars, gaia_note = [], ""
if ss.gaia:
    try:
        with st.spinner("Loading Gaia DR3 stars…"):
            gaia_stars, src = gaia(round(ra0, 4), round(dec0, 4), round(ss.fov / 60 * 0.75, 3))
        gaia_note = f"{len(gaia_stars)} Gaia DR3 stars (G < {config.GAIA_MAX_G:g}, via {src})"
    except Exception:
        gaia_note = "Gaia catalogue unavailable right now."
gaia_text = [catalogs.gaia_hover(s) for s in gaia_stars]


def gaia_mark(mjd):
    r, d = catalogs.gaia_at(gaia_stars, mjd)
    x, y = wgrid.world_to_pixel_values(r, d) if len(r) else ([], [])
    return dict(x=list(np.atleast_1d(x)), y=list(np.atleast_1d(y)), name="Gaia DR3", symbol="circle-open",
                color="#38d0ff", size=9, text=gaia_text)


def marks_for(mjd):
    return [target_mark(mjd)] + ([gaia_mark(mjd)] if ss.gaia and gaia_stars else [])


tab_view, tab_exp = st.tabs(["Blink", "Exposures"])
with tab_view:
    mode = ss.mode
    diverging = False
    if mode == "blink" or (mode in ("add", "sub") and len(frames) < 2 and mode != "add"):
        shown = frames
        if mode == "sub":
            st.info("Subtract needs at least two epochs in this band.")
    elif mode == "add":
        shown = [stack.add(frames)]
    elif mode == "sub":
        labels = [f.label.split(":")[0] for f in frames]
        c1, c2 = st.columns(2)
        ia = c1.selectbox("Epoch A (older)", range(len(frames)), format_func=lambda i: labels[i])
        ib = c2.selectbox("Epoch B (newer)", range(len(frames)), index=len(frames) - 1,
                          format_func=lambda i: labels[i])
        shown = [stack.subtract(frames[ib], frames[ia])]
        diverging = True
    else:  # movie
        order = st.radio("Order", ["time", "wavelength"], horizontal=True)
        al = sorted(by_uri.values(), key=lambda a: a.stamp.mjd if order == "time" else a.stamp.wavelength)
        al = archive.evenly(al, config.MOVIE_CAP)
        shown = [stack.single_frame(a, i + 1) for i, a in enumerate(al)]

    marks = [marks_for(f.mjd_mid) for f in shown]
    fig = render.animation(shown, ss.stretch, marks, diverging, arcsec_px, ss.speed, show_flags=ss.showflags)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    c1, c2, c3 = st.columns([2, 2, 3])
    if not load_all and n_band > len(todo):
        c1.button(f"Load all {band} exposures ({min(n_band, config.LOAD_ALL_CAP)} of {n_band})",
                  on_click=lambda: ss.__setitem__(load_key, True))
    c2.download_button("Download GIF", render.to_gif(shown, ss.stretch, diverging, ss.speed),
                       file_name=f"spherex_{tgt.name.replace(' ', '_')}_{band}_{mode}.gif", mime="image/gif")
    with c3.expander("What am I looking at?"):
        st.markdown(
            "Each frame is a SPHEREx image of the same patch of sky, taken months apart and averaged per "
            "*epoch* (one sky survey pass). North is up, east is left; the bar is 1 arcminute. "
            "The orange circle is where the target should be on that date, using its catalogue proper motion. "
            "A star that moves (like Barnard's Star) shifts against the background; a variable star changes "
            "brightness. **Subtract** shows the change directly: a moving star leaves a dark/light pair. "
            "Motion between epochs is often smaller than one SPHEREx pixel (6.15″), so the image is shown "
            "2× finer than the detector.")

with tab_exp:
    rows = [dict(date=targets.mjd_to_date(a.stamp.mjd), mjd=round(a.stamp.mjd, 4), band=a.stamp.band,
                 wavelength_um=round(a.stamp.wavelength, 4), flags_at_target=", ".join(a.bad_near_target),
                 file=a.stamp.uri and f"{config.IRSA}/{a.stamp.uri}") for a in by_uri.values()]
    st.dataframe(pd.DataFrame(rows).sort_values("mjd"), hide_index=True, use_container_width=True,
                 column_config={"file": st.column_config.LinkColumn("file", display_text="FITS")})
    st.caption(f"{len(exps)} SPHEREx exposures cover this position in all bands; {n_band} in {band}.")

last, versions = archive.data_through(exps)
st.caption(f"{tgt.name}: RA {ra0:.5f}, Dec {dec0:+.5f} on 2026-01-01 ({tgt.source}). "
           f"SPHEREx data through {targets.mjd_to_date(last)} · processing {', '.join(versions)} · "
           f"{gaia_note} · Data: NASA/IPAC Infrared Science Archive (IRSA), ESA Gaia.")

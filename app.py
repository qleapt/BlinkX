"""SPHEREx Blink Viewer: type a sky position or object name, watch SPHEREx images of it blink."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from astropy.time import Time

from spx import archive, catalogs, config, cutouts, photometry, render, stack, targets

st.set_page_config(page_title="SPHEREx Blink Viewer", page_icon="🔭", layout="wide")

PRESET_BAND = targets.PRESET_BAND
MODES = {"blink": "Blink epochs", "add": "Add (all epochs)", "sub": "Subtract (B − A)", "movie": "Every exposure",
         "spectrum": "Spectrum & light curve", "motion": "Motion check"}
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
    st.toggle("Start with older surveys (2MASS ~2000, WISE 2010)", key="survey",
              help="J → D1/D2, Ks → D3, W1 → D4, W2 → D5/D6. Shown for comparison only (other telescope).")
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
st.caption("NASA's SPHEREx telescope maps the whole sky every six months in 102 infrared colours. Pick a target "
           "in the sidebar (or type any name or RA Dec) and press ▶ Play: things that move or change brightness "
           "stand out. Try the views in the sidebar for subtraction, spectra and a motion check.")
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


def fetch(todo, label):
    with st.status(f"Fetching {len(todo)} {label} cutouts…", expanded=False) as status:
        bar = st.progress(0.0)
        out, nfail = cutouts.fetch_many(todo, ra0, dec0, ss.fov * 1.45,
                                        lambda d, n: bar.progress(d / n, f"{d}/{n} cutouts"))
        status.update(label=f"{len(out)} {label} cutouts ready" + (f" ({nfail} failed)" if nfail else ""),
                      state="complete")
    if not out:
        st.warning("No cutouts could be downloaded (archive busy?). Please try again in a minute.")
        st.stop()
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
    st.subheader(f"{tgt.name}: SPHEREx spectrum, 0.75–5.0 µm")
    close = {}
    try:  # warn when a Gaia neighbour falls inside the 12" aperture (SPHEREx pixels are 6.15")
        nb, _ = catalogs.gaia_cone(round(ra0, 4), round(dec0, 4), 0.02)
        mids = [min(r["mjd"] for r in rows), max(r["mjd"] for r in rows)]
        for m in mids:
            pr, pd_ = targets.position_at(tgt, m)
            gr, gd = catalogs.gaia_at(nb, m)
            sep = np.hypot((gr - pr) * np.cos(np.radians(pd_)), gd - pd_) * 3600
            for s, d in zip(nb, sep):
                if 2.0 < d < 15:  # < 2" is the target's own Gaia entry
                    close.setdefault(s["id"], (s["g"], []))[1].append(d)
        for g, ds in close.values():
            st.warning(f"A Gaia star (G = {g:.1f}) lies {min(ds):.0f}–{max(ds):.0f}″ from the target on these dates. "
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
    st.subheader(f"Light curve in {band} (choose the band in the sidebar)")
    st.plotly_chart(render.lightcurve_fig(by_epoch, band, logy), use_container_width=True)
    # plain-language change report per band: median clean flux, first vs last epoch
    notes = []
    for b in config.BANDS:
        per = {k: [r for r in v if r["band"] == b and not r["flags"] and np.isfinite(r["flux"])]
               for k, v in by_epoch.items()}
        per = {k: v for k, v in per.items() if len(v) >= 2}
        if len(per) >= 2:
            k1, k2 = min(per), max(per)
            f1, f2 = (np.median([r["flux"] for r in per[k]]) for k in (k1, k2))
            e = np.hypot(*(1.25 * np.std([r["flux"] for r in per[k]]) / np.sqrt(len(per[k])) for k in (k1, k2)))
            sig = abs(f2 - f1) / e if e > 0 else 0
            word = "brighter" if f2 > f1 else "fainter"
            notes.append(f"**{b}**: epoch {k2} is {abs(f2 / f1 - 1) * 100:.0f}% {word} than epoch {k1}"
                         + (" — a real change is likely" if sig > 3 else " — within the scatter") + f" ({sig:.1f}σ)")
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
stamps = fetch(todo, band)  # cutouts are 1.45x the field so rotated frames cover the square grid

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
sphx_frames = list(frames)


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def survey(ra, dec, b, size):
    return catalogs.survey_frames(ra, dec, b, size)


if ss.survey and ss.mode in ("blink", "sub", "motion") and frames:
    try:
        with st.spinner("Fetching older WISE / 2MASS images…"):
            items = survey(round(ra0, 5), round(dec0, 5), band, ss.fov * 1.45)
        ref = float(np.nanmax([np.nanpercentile(f.image, 99.5) for f in sphx_frames]))
        old = [stack.survey_frame(it, wgrid, shape, ref, band) for it in items]
        frames = sorted([f for f in old if f is not None], key=lambda f: f.mjd_start) + frames
    except Exception as e:
        st.caption(f"Older survey images unavailable ({type(e).__name__}).")


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


def sky_offset(px, py):
    r, d = wgrid.pixel_to_world_values(px, py)
    de = ((r - ra0 + 180) % 360 - 180) * np.cos(np.radians(dec0)) * 3600
    return float(de), float((d - dec0) * 3600)


if ss.mode == "motion":
    st.subheader(f"Motion check: {tgt.name} in {band}")
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
                        off=float(np.hypot(de - pe, dn - pn)), par_e=par[0], par_n=par[1]))
    if len(pts) < 2:
        st.info("Need the target in at least two frames. Try a band where it is bright, or a larger field of view.")
        st.stop()
    m_lo, m_hi = min(p["mjd"] for p in pts) - 60, max(p["mjd"] for p in pts) + 60
    grid_mjd = np.linspace(m_lo, m_hi, int(min(2000, max(200, (m_hi - m_lo) / 3))))
    tr = targets.predicted_offsets(tgt, grid_mjd, config.REFERENCE_MJD)
    st.plotly_chart(render.motion_fig(tr[:, 0], tr[:, 1], pts), use_container_width=True)
    yrs = [(p["mjd"] - config.REFERENCE_MJD) / 365.25 for p in pts]
    fit = photometry.fit_motion(yrs, [p["de"] - p["par_e"] for p in pts], [p["dn"] - p["par_n"] for p in pts],
                                [p["err"] for p in pts])
    cat_e, cat_n = tgt.pmra / 1000, tgt.pmdec / 1000
    if fit:
        pe, pn, ee, _ = fit
        mu, cat_mu = np.hypot(pe, pn), np.hypot(cat_e, cat_n)
        diff = np.hypot(pe - cat_e, pn - cat_n)
        c1, c2, c3 = st.columns(3)
        c1.metric("Measured motion", f"{mu:.2f} ± {ee:.2f} ″/yr")
        c2.metric("Catalogue motion", f"{cat_mu:.2f} ″/yr", help=tgt.source)
        c3.metric("Difference", f"{diff:.2f} ″/yr", f"{diff / ee:.1f} σ", delta_color="off")
        if tgt.has_motion:
            verdict = (f"✅ **Agrees** with the catalogue: SPHEREx sees {tgt.name} move {mu:.2f}″ per year, "
                       f"the catalogue says {cat_mu:.2f}″." if diff < 3 * ee + 0.1 * cat_mu else
                       f"⚠ **Differs** from the catalogue by {diff:.2f}″/yr ({diff / ee:.1f}σ). Saturation, a "
                       "blended neighbour or too few epochs can cause this; check the blink.")
        else:
            verdict = (f"🚀 **Moves**: {mu:.2f} ± {ee:.2f}″/yr, more than 3× the uncertainty." if mu > 3 * ee else
                       f"No motion detected: less than about {3 * ee:.2f}″/yr (3σ).")
        st.markdown(verdict)
    st.caption("Each point is where the target's light is centred in one frame (east to the left, as on the sky), "
               "relative to its catalogue position on 2026-01-01. The grey line is where it should be: proper "
               "motion plus the yearly parallax wobble seen from Earth. SPHEREx pixels are 6.15″; centring on many "
               "pixels lets us measure shifts well below one pixel. Error bars come from the spread of single "
               "exposures. Saturated stars and blended neighbours make centres less reliable.")
    st.dataframe(pd.DataFrame([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items()
                                if k not in ("epoch", "par_e", "par_n")} for p in pts]),
                 hide_index=True, use_container_width=True)
    st.stop()

tab_view, tab_exp = st.tabs(["Blink", "Exposures"])
with tab_view:
    mode = ss.mode
    diverging = False
    if mode == "blink" or (mode in ("add", "sub") and len(frames) < 2 and mode != "add"):
        shown = frames
        if mode == "sub":
            st.info("Subtract needs at least two epochs in this band.")
    elif mode == "add":
        shown = [stack.add(sphx_frames)]
    elif mode == "sub":
        labels = [f.label.split(":")[0] for f in frames]
        c1, c2 = st.columns(2)
        ia = c1.selectbox("Epoch A (older)", range(len(frames)), index=len(frames) - len(sphx_frames),
                          format_func=lambda i: labels[i])
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

"""Stretch, Plotly charts (spectrum, light curve, motion) and GIF export."""
from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw

STRETCHES = ("asinh", "linear", "sqrt", "log")


def limits(frames, diverging: bool = False):
    vals = np.concatenate([f.image[np.isfinite(f.image)].ravel() for f in frames] or [np.zeros(1)])
    if vals.size == 0:
        return (-1.0, 1.0)
    if diverging:
        m = float(np.percentile(np.abs(vals), 99.5)) or 1.0
        return (-m, m)
    lo, hi = np.percentile(vals, [1.0, 99.8])
    return float(lo), float(hi if hi > lo else lo + 1)


def stretch(img: np.ndarray, lo: float, hi: float, kind: str = "asinh", diverging: bool = False):
    if diverging:
        x = np.clip(img / hi, -1, 1)
        return np.sign(x) * np.arcsinh(8 * np.abs(x)) / np.arcsinh(8) if kind != "linear" else x
    x = np.clip((img - lo) / (hi - lo), 0, 1)
    if kind == "asinh":
        return np.arcsinh(10 * x) / np.arcsinh(10)
    if kind == "sqrt":
        return np.sqrt(x)
    if kind == "log":
        return np.log10(1 + 99 * x) / 2
    return x


# --- charts (dataviz: validated dark categorical order, fixed per epoch, never cycled) -------
EPOCH_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
INK, INK2, GRID = "#f0efec", "#c3c2b7", "rgba(195,194,183,0.15)"


def epoch_color(k: int) -> str:
    return EPOCH_COLORS[(k - 1) % len(EPOCH_COLORS)]


def _chart_layout(fig, xtitle, ytitle, height=420, logy=False):
    fig.update_layout(
        height=height, margin=dict(l=60, r=20, t=30, b=50), paper_bgcolor="#1a1a19", plot_bgcolor="#1a1a19",
        font=dict(color=INK2, size=13), hovermode="closest",
        hoverlabel=dict(bgcolor="#262624", font=dict(color=INK)),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=INK2)))
    fig.update_xaxes(title=xtitle, gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False, linecolor=GRID,
                     type="log" if logy else "linear")
    return fig


def _points(rows, k, x, name, show_legend=True):
    """One epoch's points: filled = clean, hollow = flagged pixels in the aperture."""
    traces = []
    col = epoch_color(k)
    for flagged in (False, True):
        sel = [r for r in rows if bool(r["flags"]) == flagged and np.isfinite(r["flux"])]
        if not sel:
            continue
        traces.append(go.Scatter(
            x=[r[x] for r in sel], y=[r["flux"] for r in sel], mode="markers",
            name=name if not flagged else f"{name} (flagged)", legendgroup=name,
            showlegend=show_legend,
            marker=dict(size=9, color="#1a1a19" if flagged else col, line=dict(width=2, color=col),
                        symbol="circle"),
            error_y=dict(type="data", array=[r["err"] for r in sel], color=col, thickness=1.2, width=0),
            customdata=[[r["date"], r["wavelength"], r["err"], ", ".join(r["flags"]) or "none", r["band"]]
                        for r in sel],
            hovertemplate=("%{customdata[0]} · %{customdata[4]}<br>λ %{customdata[1]:.3f} µm<br>"
                           "%{y:.4g} ± %{customdata[2]:.2g} mJy<br>flags: %{customdata[3]}<extra></extra>")))
    return traces


def spectrum_fig(rows_by_epoch: dict, logy: bool = False):
    fig = go.Figure()
    for k, rows in rows_by_epoch.items():
        if rows:
            label = f"Epoch {k}: {rows[0]['date']}"
            for t in _points(rows, k, "wavelength", label):
                fig.add_trace(t)
    return _chart_layout(fig, "Wavelength (µm)", "Flux (mJy, quick look)", logy=logy)


def lightcurve_fig(rows_by_epoch: dict, band: str, logy: bool = False, ytitle: str | None = None):
    fig = go.Figure()
    for k, rows in rows_by_epoch.items():
        rows = [r for r in rows if r["band"] == band]
        if rows:
            for t in _points(rows, k, "datetime", f"Epoch {k}"):
                fig.add_trace(t)
    fig.update_layout(showlegend=False)
    return _chart_layout(fig, "Date", ytitle or f"{band} flux (mJy)", height=340, logy=logy)


def motion_fig(track_e, track_n, pts):
    """Sky-plane path: predicted track (line) and measured positions (points with error bars).
    East is to the left, as on the sky."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=track_e, y=track_n, mode="lines", name="predicted (proper motion + parallax)",
                             line=dict(color=INK2, width=2), hoverinfo="skip"))
    for p in pts:
        col = epoch_color(p["epoch"]) if p["epoch"] else INK2
        fig.add_trace(go.Scatter(
            x=[p["de"]], y=[p["dn"]], mode="markers", name=p["label"],
            marker=dict(size=11, color=col, symbol="circle" if p["epoch"] else "diamond",
                        line=dict(width=2, color="#1a1a19")),
            error_x=dict(type="data", array=[p["err"]], color=col, thickness=1.2, width=0),
            error_y=dict(type="data", array=[p["err"]], color=col, thickness=1.2, width=0),
            customdata=[[p["date"], p["err"], p["off"]]],
            hovertemplate=(f"{p['label']}<br>east %{{x:.1f}}″, north %{{y:.1f}}″ ± %{{customdata[1]:.1f}}″<br>"
                           "%{customdata[2]:.1f}″ from prediction<extra></extra>")))
    _chart_layout(fig, "East offset (″, east is left)", "North offset (″)", height=520)
    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(legend=dict(orientation="v", y=1, x=1.02, yanchor="top"))
    return fig


def miss_fig(pts):
    """How far each measured position is from the predicted one (arcsec), photo by photo."""
    fig = go.Figure(go.Bar(
        x=[p["label"] + "<br>" + p["date"] for p in pts], y=[p["off"] for p in pts],
        marker_color=[epoch_color(p["epoch"]) if p["epoch"] else INK2 for p in pts],
        error_y=dict(type="data", array=[p["err"] for p in pts], color=INK2, thickness=1.2, width=4),
        hovertemplate="%{x}<br>%{y:.1f}″ from the prediction<extra></extra>"))
    fig.add_hline(y=3.0, line=dict(color=INK2, dash="dot", width=1),
                  annotation_text="half a SPHEREx pixel (3″)", annotation_font_color=INK2)
    return _chart_layout(fig, "Photo", "Miss: measured − predicted (″)", height=300)


ACCENT, FUTURE = "#c98500", "#d55181"


def _cone(e, n, se, sn):
    """Polygon around a path, 1 sigma wide across it (the sideways uncertainty of the prediction)."""
    e, n, se, sn = (np.asarray(v, float) for v in (e, n, se, sn))
    if len(e) < 2:
        return None, None
    te, tn = np.gradient(e), np.gradient(n)
    ln = np.hypot(te, tn)
    ln[ln == 0] = 1
    pe, pn = -tn / ln, te / ln                       # unit vector across the path
    s = np.hypot(se * pe, sn * pn)
    return np.concatenate([e + s * pe, (e - s * pe)[::-1]]), np.concatenate([n + s * pn, (n - s * pn)[::-1]])


def future_fig(marks, path, future, catalogue=None, tick_label="year"):
    """Sky-plane chart of one object: the marks (with error bars), the path fitted through all of them, its
    continuation into the future with a 1-sigma uncertainty cone and dated ticks, and the catalogue path if any.
    East is to the left, as on the sky. All values are offsets in arcsec from one reference point.
    marks: dicts e, n, err, label, date, kind; path / future / catalogue: dicts of arrays e, n (future also se,
    sn and ticks: list of (index, text))."""
    fig = go.Figure()
    if future is not None and len(future["e"]) > 1:
        ce, cn = _cone(future["e"], future["n"], future["se"], future["sn"])
        if ce is not None:
            fig.add_trace(go.Scatter(x=ce, y=cn, fill="toself", mode="lines", line=dict(width=0),
                                     fillcolor="rgba(213,81,129,0.18)", name="prediction uncertainty (1σ)",
                                     hoverinfo="skip"))
    if catalogue is not None:
        fig.add_trace(go.Scatter(x=catalogue["e"], y=catalogue["n"], mode="lines", name="catalogue prediction",
                                 line=dict(color=INK2, width=1.5, dash="dot"), hoverinfo="skip"))
    if path is not None:
        fig.add_trace(go.Scatter(x=path["e"], y=path["n"], mode="lines", name="path fitted through the marks",
                                 line=dict(color=ACCENT, width=2.5), hoverinfo="skip"))
    if future is not None:
        fig.add_trace(go.Scatter(x=future["e"], y=future["n"], mode="lines", name="predicted path (future)",
                                 line=dict(color=FUTURE, width=2, dash="dash"), hoverinfo="skip"))
        ticks = future.get("ticks") or []
        if ticks:
            idx = [i for i, _ in ticks]
            fig.add_trace(go.Scatter(
                x=[future["e"][i] for i in idx], y=[future["n"][i] for i in idx], mode="markers+text",
                text=[t for _, t in ticks], textposition="top center", textfont=dict(color=FUTURE, size=11),
                marker=dict(size=7, color=FUTURE, line=dict(width=1, color="#1a1a19")),
                name=f"predicted position each {tick_label}",
                customdata=[[t, future["se"][i], future["sn"][i]] for i, t in ticks],
                hovertemplate=("%{customdata[0]}: east %{x:.1f}″, north %{y:.1f}″<br>± %{customdata[1]:.1f}″ east, "
                               "± %{customdata[2]:.1f}″ north (1σ)<extra></extra>")))
    for k, m in enumerate(marks):
        col = epoch_color(k + 1)
        found = m.get("kind") == "found"
        fig.add_trace(go.Scatter(
            x=[m["e"]], y=[m["n"]], mode="markers+text", text=[m["label"]], textposition="bottom right",
            textfont=dict(color=col, size=11), name=f"{m['label']} · {m['date']}" + (" (found)" if found else ""),
            marker=dict(size=11, color="#1a1a19" if found else col, line=dict(width=2, color=col),
                        symbol="diamond" if found else "circle"),
            error_x=dict(type="data", array=[m["err"]], color=col, thickness=1.2, width=0),
            error_y=dict(type="data", array=[m["err"]], color=col, thickness=1.2, width=0),
            hovertemplate=(f"{m['label']} · {m['date']}<br>east %{{x:.1f}}″, north %{{y:.1f}}″ ± {m['err']:.1f}″"
                           "<extra></extra>")))
    _chart_layout(fig, "East offset (″, east is left)", "North offset (″)", height=560)
    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(legend=dict(orientation="v", y=1, x=1.02, yanchor="top"))
    return fig


def track_time_fig(marks, model, today=None):
    """East and north offsets against time: the marks, the fitted model and its prediction with a 1-sigma band.
    marks: dicts e, n, err, label, datetime; model: dict datetime, e, n, se, sn, future (bool array)."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("East offset (″)", "North offset (″)"))
    t = list(model["datetime"])
    fut = np.asarray(model["future"], bool)
    for row, key, skey in ((1, "e", "se"), (2, "n", "sn")):
        v, s = np.asarray(model[key]), np.asarray(model[skey])
        fig.add_trace(go.Scatter(x=t + t[::-1], y=list(v + s) + list((v - s)[::-1]), fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor="rgba(213,81,129,0.18)", showlegend=row == 1,
                                 name="uncertainty (1σ)", hoverinfo="skip"), row=row, col=1)
        for sel, name, col, dash in ((~fut, "fitted path", ACCENT, "solid"), (fut, "prediction", FUTURE, "dash")):
            if sel.any():
                fig.add_trace(go.Scatter(x=[x for x, k in zip(t, sel) if k], y=v[sel], mode="lines", name=name,
                                         line=dict(color=col, width=2, dash=dash), showlegend=row == 1,
                                         hoverinfo="skip"), row=row, col=1)
        for k, m in enumerate(marks):
            col = epoch_color(k + 1)
            fig.add_trace(go.Scatter(
                x=[m["datetime"]], y=[m[key]], mode="markers", name=m["label"], showlegend=False,
                marker=dict(size=9, color=col, line=dict(width=1.5, color="#1a1a19")),
                error_y=dict(type="data", array=[m["err"]], color=col, thickness=1.2, width=0),
                hovertemplate=f"{m['label']}<br>%{{x}}<br>%{{y:.1f}}″ ± {m['err']:.1f}″<extra></extra>"),
                row=row, col=1)
    if today:
        fig.add_vline(x=today, line=dict(color=INK2, dash="dot", width=1))
    _chart_layout(fig, "", "", height=520)
    fig.update_annotations(font=dict(color=INK2, size=12))
    return fig


def class_fig(prob: dict, names: dict):
    """How strongly the evidence points to each kind of object (horizontal bars, best first)."""
    order = sorted(prob, key=prob.get)
    colors = {"static": "#3987e5", "pm_star": ACCENT, "solar_system": "#199e70", "mismatch": "#e66767"}
    fig = go.Figure(go.Bar(x=[100 * prob[k] for k in order], y=[names[k] for k in order], orientation="h",
                           marker_color=[colors.get(k, INK2) for k in order],
                           text=[f"{100 * prob[k]:.0f}%" for k in order], textposition="outside",
                           hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
    _chart_layout(fig, "How strongly the evidence points to it (%)", "", height=230)
    fig.update_xaxes(range=[0, 115])
    fig.update_layout(margin=dict(l=10, r=20, t=10, b=45))
    fig.update_yaxes(automargin=True, type="category")
    return fig


def _rgb(z: np.ndarray, diverging: bool) -> np.ndarray:
    z = np.nan_to_num(z, nan=0.0)
    if diverging:
        t = (z + 1) / 2
        stops = np.array([[33, 102, 172], [103, 169, 207], [247, 247, 247], [239, 138, 98], [178, 24, 43]], float)
        pos = np.linspace(0, 1, 5)
        rgb = np.stack([np.interp(t, pos, stops[:, c]) for c in range(3)], -1)
    else:
        rgb = np.repeat((z * 255)[..., None], 3, -1)
    return rgb.clip(0, 255).astype(np.uint8)


def to_gif(frames, kind: str = "asinh", diverging: bool = False, duration_ms: int = 700, size: int = 480) -> bytes:
    lo, hi = limits(frames, diverging)
    imgs = []
    for f in frames:
        rgb = _rgb(stretch(f.image, lo, hi, kind, diverging), diverging)[::-1]  # north up
        im = Image.fromarray(rgb).resize((size, size), Image.NEAREST)
        canvas = Image.new("RGB", (size, size + 24), (14, 17, 23))
        canvas.paste(im, (0, 24))
        ImageDraw.Draw(canvas).text((6, 5), f.label[:80], fill=(230, 230, 230))
        imgs.append(canvas)
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)
    return buf.getvalue()

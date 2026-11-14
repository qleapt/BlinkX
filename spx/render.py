"""Stretch, colour maps, Plotly animation and GIF export."""
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


GRAY = [[0, "#000000"], [1, "#ffffff"]]
DIVERGING = [[0, "#2166ac"], [0.25, "#67a9cf"], [0.5, "#f7f7f7"], [0.75, "#ef8a62"], [1, "#b2182b"]]


def _marks_traces(marks):
    """marks: list of dicts {x, y, name, symbol, color, size, text}. One scatter per mark kind."""
    out = []
    for m in marks:
        out.append(go.Scatter(
            x=m.get("x", []), y=m.get("y", []), mode="markers", name=m["name"],
            marker=dict(symbol=m.get("symbol", "circle-open"), size=m.get("size", 18),
                        color=m.get("color", "#ffb000"), line=dict(width=2, color=m.get("color", "#ffb000"))),
            text=m.get("text"), hovertemplate="%{text}<extra></extra>" if m.get("text") is not None else None,
            hoverinfo=None if m.get("text") is not None else "skip", showlegend=True))
    return out


def animation(frames, kind: str = "asinh", marks_per_frame=None, diverging: bool = False,
              arcsec_per_px: float = 3.075, duration_ms: int = 700, height: int = 640):
    lo, hi = limits(frames, diverging)
    ny, nx = frames[0].image.shape
    marks_per_frame = marks_per_frame or [[] for _ in frames]

    def data(i):
        f = frames[i]
        z = stretch(f.image, lo, hi, kind, diverging)
        heat = go.Heatmap(z=z, customdata=f.image, colorscale=DIVERGING if diverging else GRAY,
                          zmin=-1 if diverging else 0, zmax=1, showscale=False,
                          hovertemplate="x %{x}, y %{y}<br>%{customdata:.4g} MJy/sr<extra></extra>")
        return [heat] + _marks_traces(marks_per_frame[i])

    def title(i):
        f = frames[i]
        warn = f"  ⚠ flags at target: {', '.join(f.flags)}" if f.flags else ""
        return f"{f.label}{warn}"

    fig = go.Figure(data=data(0), frames=[go.Frame(data=data(i), name=str(i), layout=go.Layout(title_text=title(i)))
                                          for i in range(len(frames))])
    bar_px = 60 / arcsec_per_px
    x0, y0 = nx * 0.05, ny * 0.05
    fig.update_layout(
        title=dict(text=title(0), font=dict(size=14), x=0.01),
        height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="#0e1117", plot_bgcolor="#111111",
        font=dict(color="#e6e6e6"),
        xaxis=dict(visible=False, range=[-0.5, nx - 0.5], constrain="domain"),
        yaxis=dict(visible=False, range=[-0.5, ny - 0.5], scaleanchor="x", constrain="domain"),
        legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        shapes=[dict(type="line", x0=x0, x1=x0 + bar_px, y0=y0, y1=y0, line=dict(color="#ffffff", width=3))],
        annotations=[
            dict(x=x0 + bar_px / 2, y=y0 + ny * 0.03, text="1′", showarrow=False, font=dict(color="#fff")),
            dict(x=nx * 0.93, y=ny * 0.93, ax=0, ay=40, text="N", showarrow=True, arrowcolor="#fff",
                 font=dict(color="#fff"), arrowhead=2, xref="x", yref="y", axref="pixel", ayref="pixel"),
            dict(x=nx * 0.83, y=ny * 0.80, ax=40, ay=0, text="E", showarrow=True, arrowcolor="#fff",
                 font=dict(color="#fff"), arrowhead=2, xref="x", yref="y", axref="pixel", ayref="pixel"),
        ],
    )
    if len(frames) > 1:
        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="left", x=0, y=1.0, xanchor="left", yanchor="bottom",
                              pad=dict(t=0, r=8), showactive=False, buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=duration_ms, redraw=True), fromcurrent=True,
                                          transition=dict(duration=0), mode="immediate")]),
                    dict(label="❚❚ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
            sliders=[dict(active=0, y=0, x=0.12, len=0.88, pad=dict(t=30), currentvalue=dict(visible=False),
                          steps=[dict(method="animate", label=str(i + 1),
                                      args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                                 for i in range(len(frames))])])
        fig.layout.updatemenus[0].x = 0
        fig.layout.updatemenus[0].y = -0.02
        fig.layout.updatemenus[0].yanchor = "top"
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

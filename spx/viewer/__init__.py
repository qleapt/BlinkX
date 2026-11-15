"""The sky viewer: one in-browser canvas for every image view (Blink, Subtract, Every exposure).

Python sends the pixel values themselves (not pictures), packed losslessly enough into PNGs: the
16-bit code of each pixel sits in the red and green channels, flagged pixels in blue. The code is
asinh-compressed, so faint sky keeps fine steps while bright stars still fit. The browser decodes the
values and applies brightness, contrast and stretch itself, so those controls never reload the page.

The viewer is a two-way Streamlit component (plain HTML + postMessage, no build step): it reports the
user's marks back to Python, which can then look them up in Gaia ("Check my find").
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from .. import targets

_FRONTEND = Path(__file__).parent / "frontend"
_component = None


def encode_frames(images, flag_masks=None) -> dict:
    """images: 2-D arrays (row 0 = south). Returns {'frames': [base64 PNG], 'lo', 's', 'a'}.
    Decode: q = 256 R + G; q == 0 means no data; value = lo + s * sinh((q - 1) / 65534 * a)."""
    fin = np.concatenate([im[np.isfinite(im)].ravel() for im in images] or [np.zeros(1)])
    if fin.size == 0:
        fin = np.zeros(1)
    lo, hi = float(fin.min()), float(fin.max())
    med = float(np.median(fin))
    s = float(1.4826 * np.median(np.abs(fin - med))) or (hi - lo) / 1000 or 1.0
    a = float(np.arcsinh((hi - lo) / s)) or 1.0
    out = []
    for i, im in enumerate(images):
        ok = np.isfinite(im)
        t = np.arcsinh(np.clip(np.where(ok, im, lo) - lo, 0, None) / s) / a
        q = np.where(ok, 1 + np.rint(np.clip(t, 0, 1) * 65534), 0).astype(np.uint32)
        fm = flag_masks[i] if flag_masks is not None and flag_masks[i] is not None else np.zeros(im.shape, bool)
        rgb = np.stack([(q >> 8).astype(np.uint8), (q & 255).astype(np.uint8),
                        np.where(fm, 255, 0).astype(np.uint8)], -1)[::-1]  # first PNG row = north (top)
        buf = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(buf, format="PNG", compress_level=6)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return dict(frames=out, lo=lo, s=s, a=a)


def photo_uid(f) -> str:
    """One id per photo, so marks on two single exposures taken minutes apart are never mixed up."""
    stamps = getattr(f, "stamps", None) or []
    if getattr(f, "kind", "") == "single" and stamps and getattr(stamps[0], "uri", ""):
        return "exp:" + stamps[0].uri
    return f"{getattr(f, 'kind', '')}:{f.label.split(' · ')[0]}:{float(f.mjd_mid):.6f}"


def payload(frames, target=None, gaia=None, scale: float = 3.075, center=(0.0, 0.0), radec=(0.0, 0.0),
            diverging: bool = False, title: str = "", mode: str = "blink", stars=None, changes=None) -> dict:
    """frames: stack.Frame list. target / gaia / stars: one entry per frame (dict x, y, label / list of
    dicts x, y, text / list of dicts x, y) in grid pixels, or None. changes: helper results (dicts x, y,
    x2, y2, kind, label, text). center: grid pixel of (ra0, dec0) = radec."""
    enc = encode_frames([f.image for f in frames], [getattr(f, "flag_mask", None) for f in frames])
    ny, nx = frames[0].image.shape
    out = []
    for i, f in enumerate(frames):
        out.append(dict(png=enc["frames"][i], label=f.label, mjd=float(f.mjd_mid), uid=photo_uid(f),
                        date=targets.mjd_to_date(f.mjd_mid), kind=getattr(f, "kind", ""),
                        flags=list(getattr(f, "flags", []) or []),
                        target=target[i] if target else None, gaia=gaia[i] if gaia else [],
                        stars=[dict(x=round(s["x"], 2), y=round(s["y"], 2)) for s in stars[i]] if stars else []))
    keep = ("x", "y", "x2", "y2", "kind", "label", "text")
    return dict(frames=out, nx=nx, ny=ny, lo=enc["lo"], s=enc["s"], a=enc["a"], scale=scale,
                cx=float(center[0]), cy=float(center[1]), ra0=float(radec[0]), dec0=float(radec[1]),
                diverging=diverging, title=title, mode=mode,
                changes=[{k: c[k] for k in keep if k in c} for c in (changes or [])])


def show(data: dict, key: str, store_key: str, height: int = 640, settings: dict | None = None):
    """Draw the viewer. Returns the user's marks ({'marks': [...], 'lines': [...]}) or None."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components
        _component = components.declare_component("spx_viewer", path=str(_FRONTEND))
    return _component(data=data, store_key=store_key, height=height, settings=settings or {}, key=key,
                      default=None)

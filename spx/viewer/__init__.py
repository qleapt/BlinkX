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
import json
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


# Band changes at once. The page hands the bands it has lined up in the background to the browser (alt_html, into
# window.__spxAlts); when a band is then chosen in the sidebar, this script tells the viewer, which shows it straight
# away (spxShowBand in frontend/index.html), and the page catches up a moment later with the same picture. Bands not
# handed over yet simply load the usual way. It also hides the page's invisible hand-over box.
# (No angle-bracket tags in the script: st.html's sanitizer drops a script whose text contains one.)
BAND_SCRIPT = r"""<script>
(() => {
  const W = window;
  if (W.__spxBandHook) return;
  W.__spxBandHook = true;
  const css = document.createElement("style");
  css.textContent = ".st-key-spxalts{display:none!important}";
  document.head.appendChild(css);
  // The band box is a combobox: while its list is open the focus stays in its input, the option under the keyboard is
  // the input's aria-activedescendant, and a mouse choice is a press (down and up on the same option).
  const input = () => document.querySelector(".st-key-band input");
  const open = () => { const i = input(); return !!i && document.activeElement === i; };
  const bandIn = text => { const m = (text || "").match(/^\s*D([1-6])\b/); return m ? "D" + m[1] : null; };
  const viewers = () => Array.from(document.querySelectorAll("iframe")).map(f => {
    try { return f.contentWindow; } catch (e) { return null; } }).filter(w => w && w.spxShowBand);
  const chosen = () => { const v = (input() || {}).value || ""; return v.length > 3 ? bandIn(v) : null; };   // a full label
  const show = b => {
    if (!b) return;
    viewers().forEach(w => w.spxShowBand(b));   // (a viewer already showing b does nothing)
    // the box shows the choice right away; if it did not take after all, the viewer goes back
    setTimeout(() => { const c = chosen(); if (c && c !== b) viewers().forEach(w => w.spxRestore && w.spxRestore()); }, 600);
  };
  let down = null;
  document.addEventListener("pointerdown", e => {
    down = open() && e.target.closest ? e.target.closest('[role="option"]') : null; }, true);
  document.addEventListener("pointerup", e => {
    const o = e.target.closest && e.target.closest('[role="option"]');
    if (o && o === down && open()) show(bandIn(o.textContent));
    down = null;
  }, true);
  document.addEventListener("keydown", e => {
    if (e.key !== "Enter" || !open()) return;
    const id = input().getAttribute("aria-activedescendant"), shown = document.querySelectorAll('[role="option"]');
    const o = (id && document.getElementById(id)) || document.querySelector('[role="option"][data-focused]')
      || (shown.length === 1 ? shown[0] : null);
    if (o) show(bandIn(o.textContent));
  }, true);
})();
</script>"""


def alt_html(view_key: str, title: str, record: dict) -> str:
    """One band lined up ahead, for the browser: record = {data (payload()), store_key, settings}. view_key: the view
    it belongs to (target, field, mode, choices); a new view starts the browser's list afresh."""
    js = json.dumps({"k": view_key, "t": title, "r": record}, separators=(",", ":")).replace("<", "\\u003c")
    return ("<script>(() => { const m = " + js + "; let A = window.__spxAlts; "
            "if (!A || A.key !== m.k) A = window.__spxAlts = {key: m.k, items: {}}; A.items[m.t] = m.r; })();</script>")


def show(data: dict, key: str, store_key: str, height: int = 640, settings: dict | None = None):
    """Draw the viewer. Returns the user's marks ({'marks': [...], 'lines': [...]}) or None."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components
        _component = components.declare_component("spx_viewer", path=str(_FRONTEND))
    return _component(data=data, store_key=store_key, height=height, settings=settings or {}, key=key,
                      default=None)

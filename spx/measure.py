"""'Mark & measure': an in-browser tool to mark objects and measure distances on the blink frames.

Everything runs in the page (JavaScript on a canvas), so marking never reloads the app. Python only
packs the frames as small PNGs plus the numbers needed to turn pixels into sky positions.
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
from PIL import Image

from . import render, targets


def _png(z: np.ndarray, diverging: bool) -> str:
    rgb = render._rgb(z, diverging)[::-1]  # north up: first PNG row is the top of the sky
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def payload(frames, kind: str, diverging: bool, wgrid, arcsec_px: float, target=None) -> dict:
    """frames: stack.Frame list. target: optional dict(x, y, label) in grid pixels."""
    lo, hi = render.limits(frames, diverging)
    ny, nx = frames[0].image.shape
    out = []
    for f in frames:
        z = render.stretch(f.image, lo, hi, kind, diverging)
        out.append(dict(png=_png(z, diverging), label=f.label, mjd=float(f.mjd_mid),
                        date=targets.mjd_to_date(f.mjd_mid)))
    cx, cy = (float(v) - 1 for v in wgrid.wcs.crpix)
    ra0, dec0 = (float(v) for v in wgrid.wcs.crval)
    return dict(frames=out, nx=nx, ny=ny, scale=arcsec_px, cx=cx, cy=cy, ra0=ra0, dec0=dec0,
                diverging=diverging, target=target)


def html(data: dict, store_key: str, height: int = 640) -> str:
    return (TEMPLATE.replace("%DATA%", json.dumps(data)).replace("%KEY%", json.dumps(store_key))
            .replace("%H%", str(height)))


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><style>
:root{--bg:#0e1117;--panel:#161a23;--line:#2a2f3c;--ink:#e8eaf0;--ink2:#9aa3b5;--acc:#ffb000;--blue:#38d0ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:8px}
button,select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font:inherit;cursor:pointer}
button:hover{border-color:#4a5163}button.on{background:#2b2410;border-color:var(--acc);color:#ffd27a}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--blue);outline-offset:1px}
.sep{width:1px;height:24px;background:var(--line);margin:0 4px}
.wrap{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}
.cv{position:relative;flex:0 0 auto}
canvas{display:block;border-radius:8px;background:#000;cursor:crosshair;touch-action:none}
.read{font:12px ui-monospace,Consolas,monospace;color:var(--ink2);margin-top:6px;min-height:18px}
.side{flex:1 1 260px;min-width:240px;max-height:%H%px;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.side h4{margin:12px 0 4px;font-size:12px;color:var(--ink2);text-transform:uppercase;letter-spacing:.05em}
.tip{background:#1d2230;border-left:3px solid var(--blue);padding:6px 8px;border-radius:6px;color:#cdd3df;font-size:13px}
.empty{color:var(--ink2);font-style:italic;font-size:13px}
.row{padding:4px 0;border-bottom:1px dashed #262b37;font-size:13px}
.row b{color:var(--acc)} .m{color:var(--blue)}
.small{color:var(--ink2);font-size:12px}
label.ck{display:inline-flex;gap:5px;align-items:center;color:var(--ink2);font-size:13px;cursor:pointer}
input[type=number]{width:90px;background:#0f131b;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 6px;font:inherit}
.k{display:inline-block;min-width:1.4em;text-align:center;border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:11px;color:var(--ink2)}
</style></head><body>
<div class="bar" role="toolbar" aria-label="Mark and measure tools">
  <button id="tMark" title="Click on the image to drop a numbered marker on an object (a star, a smudge, anything)">📍 Mark object</button>
  <button id="tRuler" title="Click once where the line starts and once where it ends: shows the length on the sky">📏 Measure (line)</button>
  <button id="tLook" title="Just move the mouse to read positions; clicks do nothing">✋ Look only</button>
  <span class="sep"></span>
  <button id="prev" title="Previous image (key ←)">◀</button>
  <select id="frame" aria-label="Image shown" title="Which image (date) is shown"></select>
  <button id="next" title="Next image (key →)">▶</button>
  <button id="play" title="Flip through the images automatically, so things that move seem to jump">▶ Blink</button>
  <span class="sep"></span>
  <button id="undo" title="Remove the last marker or line (key Z)">↶ Undo</button>
  <button id="clear" title="Remove all markers and lines">🗑 Clear all</button>
  <label class="ck" title="Moves your click to the light-weighted centre of the nearest bright spot, so marks land on the middle of a star"><input type="checkbox" id="snap" checked> Snap to star centre</label>
</div>
<div class="wrap">
  <div class="cv"><canvas id="c" width="560" height="560" aria-label="Sky image: click to mark or measure"></canvas>
    <div class="read" id="read">Move the mouse over the image to read the sky position.</div></div>
  <div class="side">
    <div class="tip" id="tip"></div>
    <h4>Marked objects</h4><div id="marks"></div>
    <h4>Distances between marks</h4><div id="pairs"></div>
    <h4>Measured lines</h4><div id="lines"></div>
    <h4>Real size (optional)</h4>
    <div class="small">If you know how far away the object is, the angle gives its real size:
      size in AU (Earth–Sun distances) = angle in ″ × distance in parsecs.</div>
    <div style="margin-top:6px">Distance <input type="number" id="dist" min="0" step="any" placeholder="e.g. 1.8">
      <select id="dunit" aria-label="Distance unit"><option value="pc">parsecs (pc)</option><option value="ly">light-years</option></select></div>
    <div id="sizes" class="small" style="margin-top:6px"></div>
    <h4>Save</h4><button id="csv" title="Download your marks and lines as a spreadsheet (CSV) file">⬇ Download CSV</button>
    <p class="small">Keys: <span class="k">M</span> mark · <span class="k">L</span> line · <span class="k">←</span><span class="k">→</span> images · <span class="k">Space</span> blink · <span class="k">Z</span> undo.<br>
    1′ (arcminute) = 60″ (arcseconds); the full Moon is about 1,800″ wide. One SPHEREx pixel is 6.15″,
    so a hand-placed mark is good to about one pixel; “Snap” does better on stars.</p>
  </div>
</div>
<script>
const D = %DATA%, KEY = "spx-measure:" + %KEY%;
const cv = document.getElementById("c"), cx = cv.getContext("2d");
const N = D.frames.length, imgs = [], pix = [];
let cur = N - 1, tool = "mark", marks = [], lines = [], pending = null, hover = null, timer = null;
try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s) { marks = s.marks || []; lines = s.lines || []; } } catch (e) {}
// saved items remember their photo by date: re-attach them to the matching photo of this view
const nearest = (mjd) => D.frames.reduce((b, f, i) => Math.abs(f.mjd - mjd) < Math.abs(D.frames[b].mjd - mjd) ? i : b, 0);
marks = marks.filter(m => m.mjd !== undefined).map(m => Object.assign(m, {f: nearest(m.mjd)}));
lines = lines.filter(l => l.mjd !== undefined).map(l => Object.assign(l, {f: nearest(l.mjd)}));
function save() { try { localStorage.setItem(KEY, JSON.stringify({marks, lines})); } catch (e) {} }

function fit() {  // canvas fills the width up to 560 px; image pixels stay square
  const w = Math.max(260, Math.min(560, document.body.clientWidth - 8));
  cv.width = cv.height = Math.floor(w); draw();
}
const K = () => cv.width / D.nx;   // screen px per image px
const toImg = (ev) => { const r = cv.getBoundingClientRect();
  const sx = (ev.clientX - r.left) * cv.width / r.width, sy = (ev.clientY - r.top) * cv.height / r.height;
  return {x: sx / K() - 0.5, y: (D.ny - sy / K()) - 0.5}; };   // image coords, y up (north)
const toScr = (p) => ({x: (p.x + 0.5) * K(), y: (D.ny - p.y - 0.5) * K()});
function sky(p) {   // grid pixel -> RA, Dec (tangent plane; exact enough over a few arcminutes)
  const e = -(p.x - D.cx) * D.scale, n = (p.y - D.cy) * D.scale;   // arcsec; east is left
  const dec = D.dec0 + n / 3600, ra = ((D.ra0 + e / 3600 / Math.cos(D.dec0 * Math.PI / 180)) % 360 + 360) % 360;
  return {ra, dec, e, n};
}
const sepArc = (a, b) => Math.hypot(a.x - b.x, a.y - b.y) * D.scale;
const fmtAng = (as) => as >= 120 ? (as / 60).toFixed(2) + "′ (" + as.toFixed(0) + "″)" : as.toFixed(1) + "″";
const fmtRD = (s) => "RA " + s.ra.toFixed(5) + "°, Dec " + (s.dec >= 0 ? "+" : "") + s.dec.toFixed(5) + "°";
const pa = (a, b) => ((Math.atan2(-(b.x - a.x), b.y - a.y) * 180 / Math.PI) + 360) % 360;  // degrees east of north
const compass = (deg) => ["north","north-east","east","south-east","south","south-west","west","north-west"][Math.round(deg / 45) % 8];

function snap(p) {   // light-weighted centre of the brightest spot within 3 px (on the image shown)
  if (!document.getElementById("snap").checked || !pix[cur]) return p;
  const d = pix[cur], val = (x, y) => { if (x < 0 || y < 0 || x >= D.nx || y >= D.ny) return -1;
    const i = ((D.ny - 1 - y) * D.nx + x) * 4, v = (d[i] + d[i+1] + d[i+2]) / 3;
    return D.diverging ? Math.abs(v - 247) : v; };
  let q = {x: Math.round(p.x), y: Math.round(p.y)}, bv = -1;
  for (let y = q.y - 3; y <= q.y + 3; y++) for (let x = q.x - 3; x <= q.x + 3; x++) { const v = val(x, y); if (v > bv) { bv = v; p = {x, y}; } }
  let sx = 0, sy = 0, sw = 0, base = 1e9;
  for (let y = p.y - 3; y <= p.y + 3; y++) for (let x = p.x - 3; x <= p.x + 3; x++) { const v = val(x, y); if (v >= 0) base = Math.min(base, v); }
  for (let y = p.y - 2; y <= p.y + 2; y++) for (let x = p.x - 2; x <= p.x + 2; x++) {
    const w = Math.max(0, val(x, y) - base); sx += w * x; sy += w * y; sw += w; }
  return sw > 0 ? {x: sx / sw, y: sy / sw} : p;
}

function draw() {
  const k = K(); cx.imageSmoothingEnabled = false; cx.fillStyle = "#000"; cx.fillRect(0, 0, cv.width, cv.height);
  if (imgs[cur] && imgs[cur].complete) cx.drawImage(imgs[cur], 0, 0, cv.width, cv.height);
  cx.font = "bold 13px system-ui"; cx.lineWidth = 2;
  if (D.target) { const s = toScr(D.target); cx.strokeStyle = "rgba(255,176,0,.7)"; cx.beginPath();
    cx.moveTo(s.x - 10, s.y); cx.lineTo(s.x - 4, s.y); cx.moveTo(s.x + 4, s.y); cx.lineTo(s.x + 10, s.y);
    cx.moveTo(s.x, s.y - 10); cx.lineTo(s.x, s.y - 4); cx.moveTo(s.x, s.y + 4); cx.lineTo(s.x, s.y + 10); cx.stroke(); }
  if (marks.length > 1) { cx.strokeStyle = "rgba(255,176,0,.45)"; cx.setLineDash([4, 4]); cx.beginPath();
    marks.forEach((m, i) => { const s = toScr(m); i ? cx.lineTo(s.x, s.y) : cx.moveTo(s.x, s.y); }); cx.stroke(); cx.setLineDash([]); }
  marks.forEach((m, i) => { const s = toScr(m);
    cx.globalAlpha = m.f === cur ? 1 : 0.55; cx.strokeStyle = "#ffb000"; cx.beginPath(); cx.arc(s.x, s.y, Math.max(7, 2.2 * k), 0, 7); cx.stroke();
    cx.fillStyle = "#ffb000"; cx.fillText(String(i + 1), s.x + 9, s.y - 8); cx.globalAlpha = 1; });
  const ln = (a, b, col, label) => { const s = toScr(a), t = toScr(b); cx.strokeStyle = col; cx.beginPath(); cx.moveTo(s.x, s.y); cx.lineTo(t.x, t.y); cx.stroke();
    [s, t].forEach(q => { cx.beginPath(); cx.arc(q.x, q.y, 3, 0, 7); cx.fillStyle = col; cx.fill(); });
    if (label) { const w = cx.measureText(label).width + 8;   // keep the label inside the image
      const mx = Math.min(Math.max((s.x + t.x) / 2 + 6, 2), cv.width - w - 2), my = Math.min(Math.max((s.y + t.y) / 2, 40), cv.height - 30);
      cx.fillStyle = "rgba(0,0,0,.7)"; cx.fillRect(mx, my - 16, w, 18); cx.fillStyle = col; cx.fillText(label, mx + 4, my - 3); } };
  lines.forEach((l, i) => ln(l.a, l.b, "#38d0ff", "L" + (i + 1) + ": " + fmtAng(sepArc(l.a, l.b))));
  if (pending) { ln(pending, hover || pending, "rgba(56,208,255,.75)", hover ? fmtAng(sepArc(pending, hover)) : ""); }
  const bar = 60 / D.scale * k; cx.strokeStyle = "#fff"; cx.lineWidth = 3; cx.beginPath();   // 1 arcminute
  cx.moveTo(12, cv.height - 14); cx.lineTo(12 + bar, cv.height - 14); cx.stroke(); cx.fillStyle = "#fff"; cx.fillText("1′", 12 + bar / 2 - 5, cv.height - 20);
  cx.fillStyle = "rgba(0,0,0,.6)"; cx.fillRect(0, 0, cv.width, 22); cx.fillStyle = "#e8eaf0"; cx.font = "12px system-ui";
  cx.fillText((cur + 1) + "/" + N + "  " + D.frames[cur].label.slice(0, 70), 8, 15);
  cx.fillText("N ↑  E ←", cv.width - 64, 15);
}

function panel() {
  const tips = {mark: "📍 <b>Mark</b>: click on anything in the image to put a numbered marker on it. Mark the <i>same</i> object on two dates (switch with ◀ ▶) and the tool works out how far — and how fast — it moved.",
    ruler: "📏 <b>Measure</b>: click where the line starts, then where it ends. The length is the size of (or gap between) things <i>as seen on the sky</i>.",
    look: "✋ <b>Look only</b>: move the mouse to read the sky position (RA, Dec) of any pixel."};
  document.getElementById("tip").innerHTML = tips[tool] + (pending ? "<br><b>Now click the end point.</b>" : "");
  document.getElementById("marks").innerHTML = marks.length ? marks.map((m, i) =>
      `<div class="row"><b>#${i + 1}</b> · ${D.frames[m.f] ? D.frames[m.f].date : ""}<br><span class="small">${fmtRD(sky(m))}</span></div>`).join("")
    : '<div class="empty">No marked sky object. Choose 📍 Mark object and click on the image.</div>';
  let ph = "";
  for (let i = 1; i < marks.length; i++) { const a = marks[i - 1], b = marks[i], d = sepArc(a, b), ang = pa(a, b);
    const fa = D.frames[a.f], fb = D.frames[b.f], yrs = fa && fb ? Math.abs(fb.mjd - fa.mjd) / 365.25 : 0;
    ph += `<div class="row">#${i} → #${i + 1}: <span class="m">${fmtAng(d)}</span> toward the ${compass(ang)} (${ang.toFixed(0)}° from north)`;
    if (yrs > 0.02) ph += `<br><span class="small">${yrs.toFixed(2)} years apart → moved <b>${(d / yrs).toFixed(2)}″ per year</b>${d < 3 ? " — less than half a pixel, so probably no real motion" : ""}</span>`;
    ph += "</div>"; }
  document.getElementById("pairs").innerHTML = ph || '<div class="empty">' + (marks.length
    ? "Mark a second object (or the same object on another date) to measure between them."
    : "No marked sky object, so there is nothing to measure between.") + "</div>";
  document.getElementById("lines").innerHTML = lines.length ? lines.map((l, i) => { const d = sepArc(l.a, l.b);
      return `<div class="row"><span class="m">L${i + 1}</span>: ${fmtAng(d)} long, pointing ${compass(pa(l.a, l.b))}<br><span class="small">${(d / 6.15).toFixed(1)} SPHEREx pixels</span></div>`; }).join("")
    : '<div class="empty">No lines yet. Choose 📏 Measure (line) and click twice.</div>';
  const dist = parseFloat(document.getElementById("dist").value), unit = document.getElementById("dunit").value;
  const all = lines.map((l, i) => ["L" + (i + 1), sepArc(l.a, l.b)]).concat(marks.slice(1).map((m, i) => ["#" + (i + 1) + "→#" + (i + 2), sepArc(marks[i], m)]));
  const S = document.getElementById("sizes");
  if (!(dist > 0)) S.innerHTML = all.length ? "Type a distance to turn the angles into real sizes." : "";
  else { const pc = unit === "ly" ? dist / 3.26156 : dist;
    const num = (v, d) => v.toLocaleString("en-US", {maximumFractionDigits: d, minimumFractionDigits: d});
    S.innerHTML = all.length ? all.map(([n, as]) => { const au = as * pc, ly = au / 63241.1;
      return `${n}: <b>${num(au, au < 10 ? 2 : 0)} AU</b> ` + (au < 1e4
        ? `<span>(${num(au * 149.6, au < 10 ? 1 : 0)} million km)</span>`
        : `<span>(${num(ly, ly < 10 ? 2 : 1)} light-years)</span>`); }).join("<br>")
      : "Measure a line or mark two objects first."; }
}

function render() { draw(); panel(); save(); }
function setTool(t) { tool = t; pending = null; ["Mark", "Ruler", "Look"].forEach(n => document.getElementById("t" + n).classList.toggle("on", t === n.toLowerCase())); render(); }
function setFrame(i) { cur = (i + N) % N; document.getElementById("frame").value = cur; draw(); }

cv.addEventListener("click", ev => { let p = toImg(ev);
  const mjd = D.frames[cur].mjd;
  if (tool === "mark") { p = snap(p); marks.push({x: p.x, y: p.y, f: cur, mjd, t: Date.now()}); }
  else if (tool === "ruler") { p = snap(p); if (!pending) pending = p; else { lines.push({a: pending, b: p, f: cur, mjd, t: Date.now()}); pending = null; } }
  render(); });
cv.addEventListener("mousemove", ev => { hover = toImg(ev); const s = sky(hover);
  document.getElementById("read").textContent = fmtRD(s) + "  ·  " + Math.abs(s.e).toFixed(1) + "″ " + (s.e >= 0 ? "east" : "west") + ", " + Math.abs(s.n).toFixed(1) + "″ " + (s.n >= 0 ? "north" : "south") + " of the centre";
  if (pending) draw(); });
cv.addEventListener("mouseleave", () => { hover = null; if (pending) draw(); });
document.getElementById("tMark").onclick = () => setTool("mark");
document.getElementById("tRuler").onclick = () => setTool("ruler");
document.getElementById("tLook").onclick = () => setTool("look");
document.getElementById("prev").onclick = () => setFrame(cur - 1);
document.getElementById("next").onclick = () => setFrame(cur + 1);
document.getElementById("frame").onchange = e => setFrame(+e.target.value);
const play = document.getElementById("play");
play.onclick = () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ Blink"; play.classList.remove("on"); }
  else if (N > 1) { timer = setInterval(() => setFrame(cur + 1), 700); play.textContent = "❚❚ Stop"; play.classList.add("on"); } };
document.getElementById("undo").onclick = () => {
  if (pending) pending = null;
  else { const lm = marks.length ? marks[marks.length - 1].t || 0 : -1, ll = lines.length ? lines[lines.length - 1].t || 0 : -1;
    if (ll > lm) lines.pop(); else marks.pop(); }
  render(); };
document.getElementById("clear").onclick = () => { marks = []; lines = []; pending = null; render(); };
document.getElementById("dist").oninput = panel; document.getElementById("dunit").onchange = panel;
document.getElementById("csv").onclick = () => {
  let t = "kind,name,date,ra_deg,dec_deg,ra2_deg,dec2_deg,length_arcsec\n";
  marks.forEach((m, i) => { const s = sky(m); t += `mark,#${i + 1},${D.frames[m.f] ? D.frames[m.f].date : ""},${s.ra.toFixed(6)},${s.dec.toFixed(6)},,,\n`; });
  lines.forEach((l, i) => { const a = sky(l.a), b = sky(l.b); t += `line,L${i + 1},${D.frames[l.f] ? D.frames[l.f].date : ""},${a.ra.toFixed(6)},${a.dec.toFixed(6)},${b.ra.toFixed(6)},${b.dec.toFixed(6)},${sepArc(l.a, l.b).toFixed(2)}\n`; });
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([t], {type: "text/csv"})); a.download = "spherex_marks.csv"; a.click(); };
document.addEventListener("keydown", e => { if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const k = e.key.toLowerCase();
  if (k === "arrowleft") setFrame(cur - 1); else if (k === "arrowright") setFrame(cur + 1);
  else if (k === "m") setTool("mark"); else if (k === "l") setTool("ruler");
  else if (k === "z") document.getElementById("undo").click(); else if (k === " ") { e.preventDefault(); play.click(); } });

const sel = document.getElementById("frame");
D.frames.forEach((f, i) => { const o = document.createElement("option"); o.value = i; o.textContent = (i + 1) + ". " + f.label.slice(0, 60); sel.appendChild(o);
  const im = new Image(); im.onload = () => { const c = document.createElement("canvas"); c.width = D.nx; c.height = D.ny;
    const g = c.getContext("2d"); g.drawImage(im, 0, 0); pix[i] = g.getImageData(0, 0, D.nx, D.ny).data; if (i === cur) draw(); };
  im.src = "data:image/png;base64," + f.png; imgs[i] = im; });
sel.value = cur;
// grow the host frame to fit (side panel below the image on narrow screens)
function fitFrame() {   // the frame is same-origin (srcdoc), so it can size itself
  const h = Math.ceil(document.body.scrollHeight) + 6;
  if (h < 200) return;   // hidden tab: nothing laid out yet
  try { if (window.frameElement && Math.abs(window.frameElement.clientHeight - h) > 2) window.frameElement.style.height = h + "px"; } catch (e) {}
}
setInterval(fitFrame, 1000);   // also catches the tab becoming visible
if (window.ResizeObserver) new ResizeObserver(fitFrame).observe(document.body);
window.addEventListener("resize", () => { fit(); fitFrame(); }); setTool("mark"); fit(); fitFrame();
</script></body></html>"""

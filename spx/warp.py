"""Loading screen: a flight through a star field over the whole window, gone once the photos are ready.

Streamlit rebuilds a markdown block whenever its text changes, which would restart any animation
inside it. So the animation is split in two:

* SCRIPT is installed once into the page (st.html with JavaScript). It keeps the star field and the
  speed on `window`, finds the `.spx-warp` placeholder on every frame and draws into it. If Streamlit
  replaces the placeholder, the flight carries on where it was.
* card() is plain HTML (title, detail, steps, progress bar) that the app updates freely. The stage
  number in `data-stage` sets the speed: it speeds up from stage 0 to 3; stage 4 slows down and fades.

The fade-out at stage 4 does not hold the page: the app removes the card HANDOVER seconds after showing
stage 4, and SCRIPT puts that same card back on top (it lets the mouse through) to finish slowing down and
fading while the photos are already drawn underneath.
"""
from __future__ import annotations

import html as _html
import json as _json

STAGES = ["Find target", "Search NASA archive", "Download photos", "Line up & average"]
HANDOVER = 0.08   # s between showing stage 4 and removing the card: enough for the browser to see it arrive

CSS = """<style>
.spx-warp{position:fixed;inset:0;z-index:1000002;overflow:hidden;
  background:radial-gradient(ellipse at 50% 42%,#101a3c 0%,#060a18 50%,#020308 85%);
  animation:spx-in .35s ease .25s both;transition:opacity .45s ease}
@keyframes spx-in{from{opacity:0}to{opacity:1}}
.spx-warp canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.spx-card{position:absolute;left:50%;bottom:12vh;transform:translateX(-50%);width:min(580px,92%);padding:16px 20px 13px;
  border-radius:12px;background:rgba(8,11,20,.78);border:1px solid #2c3346;backdrop-filter:blur(4px);text-align:center;
  transition:opacity .4s ease}
.spx-warp[data-stage="4"]{opacity:0}
.spx-warp[data-stage="4"] .spx-card{opacity:0}
.spx-card h4{margin:0 0 4px;padding:0;font-size:1.02rem;color:#eef1f7}
.spx-card p{margin:0;color:#a7afc0;font-size:.86rem}
.spx-steps{display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-top:9px;font-size:.78rem;color:#6f7890}
.spx-steps .done{color:#5ad19a}.spx-steps .now{color:#ffcf66;font-weight:600}
.spx-prog{height:4px;margin-top:9px;border-radius:2px;background:#20263a;overflow:hidden}
.spx-prog i{display:block;height:100%;background:linear-gradient(90deg,#4a6cff,#9fd8ff);transition:width .3s ease}
.spx-prog-t{font-size:.74rem;color:#7d869b;margin-top:3px;min-height:1em}
</style>"""

SCRIPT = r"""<script>
(() => {
  if (window.__spxWarp) return;            // installed once per page; reruns reuse it
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const COLS = ["255,255,255", "223,233,255", "188,212,255", "159,216,255", "255,226,176", "255,201,138"];
  const W = window.__spxWarp = {speed: 0.25, fade: 1, last: 0};
  const N = 560;
  const star = (z) => ({x: (Math.random() * 2 - 1) * 1.6, y: (Math.random() * 2 - 1) * 1.1,
    z: z === undefined ? 0.05 + Math.random() * 0.95 : z,
    c: COLS[Math.random() < 0.6 ? 0 : 1 + Math.floor(Math.random() * (COLS.length - 1))],
    s: 0.6 + Math.random() * 1.3});
  const stars = Array.from({length: N}, () => star());
  const SPEED = [0.30, 0.55, 0.95, 1.35, 0.0];    // per stage; 4 = arrived
  // next frame: requestAnimationFrame, with a timer as a fallback (some embedded views never fire rAF)
  let queued = false;
  function next(delay) {
    if (queued) return; queued = true; let ran = false;
    const go = () => { if (ran) return; ran = true; queued = false; frame(performance.now()); };
    const arm = () => { requestAnimationFrame(go); setTimeout(go, 120); };
    delay ? setTimeout(arm, delay) : arm();
  }

  // An arrived card (stage 4) outlives its Streamlit element: when the app removes it, the same card goes
  // back on top of the page (letting the mouse through) and finishes slowing down and fading out while the
  // photos are drawn underneath. Its look is installed page-wide here, since its own style tag leaves with it.
  // (No angle-bracket tags in this script: st.html's sanitizer drops a script whose text contains one.)
  const css = document.createElement("style");
  css.textContent = __CSS__ + ".spx-warp.spx-ghost{animation:none!important;transition:none!important;pointer-events:none}";
  document.head.appendChild(css);
  let seen = null, ghost = null;
  const bury = () => { if (ghost) { ghost.remove(); ghost = null; } };
  new MutationObserver(() => {
    const live = document.querySelector(".spx-warp:not(.spx-ghost)");
    if (live) { seen = live; bury(); return; }
    if (seen && !seen.isConnected) {
      if (seen.dataset.stage === "4" && !reduce && !document.hidden) {
        bury(); ghost = seen; ghost.style.opacity = "1"; ghost.classList.add("spx-ghost");
        document.body.appendChild(ghost); W.gstart = performance.now(); next(0);
        const mine = ghost; setTimeout(() => { if (ghost === mine) bury(); }, 1500);   // never lingers
      }
      seen = null;
    }
  }).observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ["data-stage"]});

  function frame(now) {
    const host = document.querySelector(".spx-warp:not(.spx-ghost)") || ghost;
    if (!host) { W.last = 0; W.fade = 1; W.speed = 0.25; next(200); return; }
    let cv = host.querySelector("canvas");
    if (!cv) { cv = document.createElement("canvas"); cv.setAttribute("aria-hidden", "true"); host.prepend(cv); }
    const dpr = Math.min(2, window.devicePixelRatio || 1), w = host.clientWidth, h = host.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const dt = W.last ? Math.min(0.05, (now - W.last) / 1000) : 0; W.last = now;
    const stage = Math.max(0, Math.min(4, parseInt(host.dataset.stage || "0", 10)));
    const target = reduce ? 0 : SPEED[stage];
    W.speed += (target - W.speed) * (1 - Math.exp(-dt * (stage === 4 ? 4 : 1.6)));   // ease in / out
    W.fade = stage === 4 ? Math.max(0, W.fade - dt * 2.2) : Math.min(1, W.fade + dt * 3);
    const g = cv.getContext("2d"), cx = cv.width / 2, cy = cv.height * 0.42, f = Math.min(cv.width, cv.height) * 0.55;
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv.width, cv.height);
    g.globalAlpha = W.fade;
    const glow = g.createRadialGradient(cx, cy, 0, cx, cy, f * 0.9);   // faint glow ahead, brighter when fast
    glow.addColorStop(0, `rgba(140,170,255,${0.10 + 0.18 * W.speed})`); glow.addColorStop(0.35, "rgba(70,100,255,0.05)");
    glow.addColorStop(1, "rgba(0,0,0,0)"); g.fillStyle = glow; g.fillRect(0, 0, cv.width, cv.height);
    g.lineCap = "round";
    const dz = W.speed * dt * 0.42, trail = 0.012 + W.speed * 0.05;
    for (const s of stars) {
      s.z -= dz;
      let px = cx + s.x / s.z * f, py = cy + s.y / s.z * f;
      if (s.z < 0.02 || px < -40 || px > cv.width + 40 || py < -40 || py > cv.height + 40) { Object.assign(s, star(1)); continue; }
      const z2 = Math.min(1.2, s.z + trail), qx = cx + s.x / z2 * f, qy = cy + s.y / z2 * f;
      const near = 1 - s.z, a = Math.min(1, 0.15 + near * 1.1);
      g.strokeStyle = `rgba(${s.c},${a})`; g.lineWidth = Math.max(0.6, s.s * near * 2.4) * dpr;
      g.beginPath(); g.moveTo(qx, qy); g.lineTo(px + 0.01, py); g.stroke();
    }
    g.globalAlpha = 1;
    if (host === ghost) {   // the arrived card dissolves into the page in 0.45 s by the clock, then goes
      const left = Math.max(0, 1 - (now - W.gstart) / 450);
      host.style.opacity = left.toFixed(3);
      if (left <= 0) bury();
    }
    const still = document.hidden || (reduce && W.drawn === cv);   // hidden tab / static field: check now and then
    if (reduce) W.drawn = cv;
    next(still ? 250 : 0);
  }
  next(0);
})();
</script>""".replace("__CSS__", _json.dumps(CSS.replace("<style>", "").replace("</style>", "")))


def card(title: str, detail: str = "", stage: int = 0, frac: float | None = None, prog_text: str = "",
         stages=None) -> str:
    """The placeholder the script draws into, with the text on top. stage 0-3 = working, 4 = done."""
    steps = "".join(f'<span class="{"done" if i < stage else "now" if i == stage else ""}">{i + 1}. {s}</span>'
                    for i, s in enumerate(stages or STAGES))
    bar = (f'<div class="spx-prog"><i style="width:{max(0.0, min(1.0, frac)) * 100:.1f}%"></i></div>'
           if frac is not None else "")
    return (f'{CSS}<div class="spx-warp" data-stage="{int(stage)}" role="status" aria-live="polite">'
            f'<div class="spx-card"><h4>{_html.escape(title)}</h4><p>{_html.escape(detail)}</p>'
            f'<div class="spx-steps">{steps}</div>{bar}<div class="spx-prog-t">{_html.escape(prog_text)}</div>'
            f'</div></div>')

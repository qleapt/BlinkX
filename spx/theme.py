"""The page's look: deep-space navy, launch-blue accents, a glowing planet rim under the sidebar.

.streamlit/config.toml sets Streamlit's own colours and fonts; CSS here styles what the theme cannot reach
(gradients, the Go button's glow, the challenge-target list, tabs). The sky viewer (viewer/frontend/index.html)
uses the same tokens, so the two read as one page. brand() and section() are the sidebar's heading rows.
"""
from __future__ import annotations

import base64 as _b64
import html as _html

# one source for the colours (the viewer's index.html repeats them in its :root)
BG, PANEL, PANEL2, LINE, LINE2 = "#050a17", "#0b1630", "#0e1a36", "#1d2d52", "#2a4275"
INK, INK2, BLUE, BLUE_HI, BLUE_DEEP = "#e8eefb", "#a3b3d1", "#2b8cff", "#5fb0ff", "#1f6fe0"

CSS = """<style>
:root{--spx-bg:%(BG)s;--spx-panel:%(PANEL)s;--spx-panel2:%(PANEL2)s;--spx-line:%(LINE)s;--spx-line2:%(LINE2)s;
  --spx-ink:%(INK)s;--spx-ink2:%(INK2)s;--spx-blue:%(BLUE)s;--spx-blue-hi:%(BLUE_HI)s;--spx-blue-deep:%(BLUE_DEEP)s}
/* page: navy with a soft nebula glow top right and a sprinkle of faint stars */
.stApp{background:
  radial-gradient(1px 1px at 12%% 18%%,rgba(255,255,255,.35),transparent 60%%),
  radial-gradient(1px 1px at 68%% 9%%,rgba(255,255,255,.28),transparent 60%%),
  radial-gradient(1.2px 1.2px at 84%% 34%%,rgba(190,215,255,.30),transparent 60%%),
  radial-gradient(1px 1px at 41%% 72%%,rgba(255,255,255,.18),transparent 60%%),
  radial-gradient(1px 1px at 93%% 81%%,rgba(255,255,255,.22),transparent 60%%),
  radial-gradient(900px 520px at 100%% -8%%,rgba(43,140,255,.16),transparent 62%%),
  radial-gradient(700px 480px at 30%% 110%%,rgba(31,111,224,.08),transparent 60%%),
  var(--spx-bg)}
[data-testid="stHeader"]{background:transparent}
/* the browser's own surfaces in the palette: selection, caret, scrollbars, numerals in data */
.stApp ::selection{background:rgba(43,140,255,.38);color:#fff}
.stApp input,.stApp textarea{caret-color:var(--spx-blue-hi)}
.stApp,.stApp *{scrollbar-width:thin;scrollbar-color:var(--spx-line2) transparent}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stMetricValue"],.stApp [data-testid="stDataFrame"]{font-variant-numeric:tabular-nums}
[data-testid="stMainBlockContainer"]{padding-top:2.6rem}
/* sidebar: darker navy, and the rim of a planet glowing at the bottom */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#071230 0%%,#050d22 60%%,#040a1b 100%%);
  border-right:1px solid var(--spx-line);overflow:hidden}
[data-testid="stSidebar"]::after{content:"";position:absolute;left:calc(25%% - 450px);bottom:-795px;width:900px;height:900px;
  border-radius:50%%;pointer-events:none;z-index:0;
  background:radial-gradient(circle at 50%% 0%%,#1a4ea8 0%%,#0b2560 7%%,#06143a 14%%,#040b1f 24%%);
  box-shadow:0 -1px 0 rgba(170,215,255,.55),0 -3px 26px rgba(95,176,255,.75),0 -22px 90px rgba(43,140,255,.42),inset 0 4px 14px rgba(150,205,255,.65)}
[data-testid="stSidebarContent"]{position:relative;z-index:1}
[data-testid="stSidebarUserContent"]{padding-bottom:7.5rem}
/* headings in the sidebar */
.spx-brand{display:flex;align-items:center;gap:10px;margin:-.4rem 0 .2rem}
.spx-brand img{flex:0 0 auto;filter:drop-shadow(0 3px 6px rgba(2,8,30,.6))}
.spx-brand b{font-size:1.28rem;font-weight:700;letter-spacing:-.01em;color:var(--spx-ink)}
.spx-brand-sub{margin:0 0 .3rem;color:var(--spx-ink2);font-size:.82rem;line-height:1.45}
.spx-sec{display:flex;align-items:center;gap:9px;margin:.35rem 0 -.2rem;font-weight:600;font-size:1rem;color:var(--spx-ink)}
.spx-sec img{flex:0 0 auto}
/* buttons */
.stApp button[data-testid="stBaseButton-secondary"],.stApp [data-testid="stDownloadButton"] button{
  background:var(--spx-panel2);border:1px solid var(--spx-line);color:var(--spx-ink);
  transition:background-color .15s ease,border-color .15s ease,box-shadow .15s ease}
.stApp button[data-testid="stBaseButton-secondary"]:hover,.stApp [data-testid="stDownloadButton"] button:hover{
  background:#13254d;border-color:rgba(95,176,255,.6);color:#fff}
.stApp button[data-testid="stBaseButton-primary"]{min-height:46px;border:0;font-weight:600;font-size:1.02rem;
  background:linear-gradient(90deg,#1a6fe0 0%%,#2b8cff 55%%,#3aa8ff 100%%);color:#fff;
  box-shadow:0 6px 22px rgba(43,140,255,.38),inset 0 1px 0 rgba(255,255,255,.22);
  transition:filter .15s ease,box-shadow .15s ease,transform .1s ease}
.stApp button[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.08);box-shadow:0 8px 28px rgba(43,140,255,.5),inset 0 1px 0 rgba(255,255,255,.25);color:#fff}
.stApp button[data-testid="stBaseButton-primary"]:active{transform:translateY(1px)}
.stApp button:focus-visible{outline:2px solid var(--spx-blue-hi)!important;outline-offset:2px}
/* the challenge targets: a list with chevrons, like menu rows */
.st-key-spxpresets{gap:0!important;border:1px solid var(--spx-line);border-radius:10px;overflow:hidden;background:var(--spx-panel)}
.st-key-spxpresets [data-testid="stElementContainer"]{width:100%%}
.st-key-spxpresets button[data-testid="stBaseButton-secondary"]{width:100%%;justify-content:flex-start;border:0;border-radius:0;
  background:transparent;min-height:42px;padding:0 2.2rem 0 .85rem;position:relative;border-bottom:1px solid rgba(29,45,82,.75)}
.st-key-spxpresets [data-testid="stElementContainer"]:last-child button{border-bottom:0}
.st-key-spxpresets button[data-testid="stBaseButton-secondary"]::after{content:"";position:absolute;right:.95rem;top:50%%;
  width:7px;height:7px;border-right:2px solid var(--spx-ink2);border-top:2px solid var(--spx-ink2);transform:translateY(-50%%) rotate(45deg);
  transition:right .15s ease,border-color .15s ease}
.st-key-spxpresets button[data-testid="stBaseButton-secondary"]:hover{background:rgba(43,140,255,.10)}
.st-key-spxpresets button[data-testid="stBaseButton-secondary"]:hover::after{right:.75rem;border-color:var(--spx-blue-hi)}
.st-key-spxpresets button>div{width:100%%;justify-content:flex-start}
.st-key-spxpresets button p{text-align:left}
/* segmented controls (language, view, field of view): pills, the chosen one solid blue */
.stApp button[data-variant="segmented_control"]{background:var(--spx-panel2);border-color:var(--spx-line);color:var(--spx-ink2);
  transition:background-color .15s ease,border-color .15s ease,color .15s ease}
.stApp button[data-variant="segmented_control"]:hover{color:#fff;border-color:rgba(95,176,255,.6);background:#13254d}
.stApp button[data-variant="segmented_control"][aria-checked="true"]{background:var(--spx-blue-deep);border-color:var(--spx-blue);color:#fff;
  box-shadow:0 4px 14px rgba(43,140,255,.30);position:relative;z-index:1}
.stApp button[data-variant="segmented_control"][aria-checked="true"] p{color:#fff}
.st-key-fov,.st-key-lang{width:100%%!important}
.st-key-fov [data-testid="stButtonGroup"],.st-key-lang [data-testid="stButtonGroup"]{width:100%%}
.st-key-fov [data-testid="stButtonGroup"]>div:last-child,.st-key-lang [data-testid="stButtonGroup"]>div:last-child{width:100%%;display:flex;flex-wrap:nowrap}
.st-key-fov button[data-variant],.st-key-lang button[data-variant]{flex:1 1 auto;min-width:0}
/* inputs */
.stApp [data-baseweb="input"],.stApp [data-baseweb="select"]>div,.stApp [data-baseweb="textarea"]{background:var(--spx-panel);border-color:var(--spx-line)}
.stApp [data-baseweb="input"]:focus-within,.stApp [data-baseweb="select"]>div:focus-within{border-color:var(--spx-blue);box-shadow:0 0 0 3px rgba(43,140,255,.18)}
.stApp [data-baseweb="input"] input{background:transparent}
/* tabs: blue text and underline on the open one */
.stApp [data-baseweb="tab-list"]{gap:1.4rem}
.stApp [data-baseweb="tab"]{font-weight:600;color:var(--spx-ink2);padding-left:.15rem;padding-right:.15rem}
.stApp [data-baseweb="tab"][aria-selected="true"]{color:var(--spx-blue-hi)}
.stApp [data-baseweb="tab"]:hover{color:#fff}
.stApp [data-baseweb="tab-highlight"]{background:var(--spx-blue);height:2px;border-radius:2px}
.stApp [data-baseweb="tab-border"]{background:var(--spx-line)}
/* cards */
.stApp [data-testid="stExpander"] details{background:rgba(11,22,48,.55);border-color:var(--spx-line);border-radius:12px}
.stApp [data-testid="stExpander"] summary:hover{color:var(--spx-blue-hi)}
.stApp [data-testid="stAlert"]>div{border-radius:10px}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stWidgetLabel"] p{color:var(--spx-ink2)}
.stApp [data-testid="stMetric"]{background:var(--spx-panel);border:1px solid var(--spx-line);border-radius:12px;padding:.7rem .9rem}
.stApp [data-testid="stDataFrame"]{border:1px solid var(--spx-line);border-radius:10px;overflow:hidden}
.stApp hr{border-color:var(--spx-line)}
/* the view switch at the top of the page */
.st-key-mode button[data-variant]{min-height:40px;padding-left:1.05rem;padding-right:1.05rem}
@media (prefers-reduced-motion: reduce){.stApp *{transition:none!important}}
</style>""" % dict(BG=BG, PANEL=PANEL, PANEL2=PANEL2, LINE=LINE, LINE2=LINE2, INK=INK, INK2=INK2, BLUE=BLUE,
                   BLUE_HI=BLUE_HI, BLUE_DEEP=BLUE_DEEP)

# small line icons (24-unit grid, drawn with the current text colour)
ICONS = {
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
    "palette": '<path d="M12 3a9 9 0 1 0 0 18c1.1 0 1.6-.9 1.2-1.8-.5-1.1.2-2.2 1.4-2.2H17a4 4 0 0 0 4-4c0-5-4-10-9-10z"/>'
               '<circle cx="7.5" cy="11" r="1"/><circle cx="10" cy="7" r="1"/><circle cx="14.5" cy="7" r="1"/>',
    "sparkles": '<path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/>'
                '<path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
}


def _img(svg: str, size: int) -> str:
    """An SVG as an <img> (st.html's sanitizer drops inline SVG but keeps data-URI images)."""
    uri = "data:image/svg+xml;base64," + _b64.b64encode(svg.encode()).decode()
    return f'<img src="{uri}" width="{size}" height="{size}" alt="" aria-hidden="true">'


def _svg(name: str, size: int = 18) -> str:
    return _img(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{BLUE_HI}" '
                f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>', size)


LOGO = _img('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><defs>'
            '<radialGradient id="spxg" cx="38%" cy="32%" r="70%"><stop offset="0" stop-color="#9fd2ff"/>'
            '<stop offset=".45" stop-color="#2b8cff"/><stop offset="1" stop-color="#0a2a6b"/></radialGradient></defs>'
            '<circle cx="20" cy="20" r="11" fill="url(#spxg)"/>'
            '<ellipse cx="20" cy="20" rx="18" ry="6.2" fill="none" stroke="#5fb0ff" stroke-width="1.6" '
            'transform="rotate(-24 20 20)" opacity=".9"/>'
            '<circle cx="34.6" cy="13.6" r="1.9" fill="#e8eefb"/></svg>', 34)


def brand(name: str, sub: str = "") -> str:
    """The sidebar's name row (no agency logo: an orbiting sphere of our own)."""
    out = f'<div class="spx-brand">{LOGO}<b>{_html.escape(name)}</b></div>'
    return out + (f'<p class="spx-brand-sub">{_html.escape(sub)}</p>' if sub else "")


def section(icon: str, text: str) -> str:
    """A sidebar section heading with a line icon."""
    return f'<div class="spx-sec" role="heading" aria-level="3">{_svg(icon)}<span>{_html.escape(text)}</span></div>'

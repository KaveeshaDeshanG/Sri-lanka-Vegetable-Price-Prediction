"""Shared visual design for every dashboard page: colours, CSS, header, cards and chart styling.

Colours and fonts that Streamlit's theme supports are set in .streamlit/config.toml. This module
adds only what the theme can't express (cards, hover effects, responsive spacing) and keeps it in
one place, so every page looks the same. It changes appearance only, never data or predictions.
"""

from html import escape
from pathlib import Path

import matplotlib as mpl
import streamlit as st

from app.i18n import current_language, init_language, language_selector, t

COLORS = {
    "background": "#0B0F0E",
    "sidebar": "#111916",
    "card": "#17221D",
    "card_raised": "#1C2A23",
    "border": "#24352C",
    "primary": "#22C55E",
    "secondary": "#86EFAC",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "forecast": "#FBBF24",
    "info": "#38BDF8",
}
# Chart series, in a fixed order so a series keeps its colour wherever it appears.
SERIES = {"actual": COLORS["primary"], "forecast": COLORS["forecast"], "target": COLORS["secondary"], "estimate": COLORS["muted"]}
WEATHER_COLORS = {"rainfall_mm": COLORS["info"], "temperature_c": COLORS["forecast"], "humidity_pct": COLORS["secondary"]}
# CBSL markets: each market keeps its colour whichever markets are shown.
MARKET_COLORS = {"Dambulla": COLORS["primary"], "Pettah": COLORS["info"], "Narahenpita": COLORS["forecast"], "Marandagahamula": COLORS["secondary"]}

_ASSETS = Path(__file__).resolve().parent / "assets"

_CSS = """
<style>
:root {
  --vp-bg: #0B0F0E; --vp-sidebar: #111916; --vp-card: #17221D; --vp-card-raised: #1C2A23;
  --vp-border: #24352C; --vp-primary: #22C55E; --vp-secondary: #86EFAC;
  --vp-text: #F8FAFC; --vp-muted: #94A3B8;
  --vp-radius: 16px; --vp-shadow: 0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.25);
}
@keyframes vp-rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

/* Page frame */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 420px at 85% -10%, rgba(34,197,94,.10), transparent 60%),
    radial-gradient(900px 380px at -10% 0%, rgba(134,239,172,.05), transparent 55%),
    var(--vp-bg);
}
[data-testid="stHeader"] { background: transparent; }
/* Streamlit's own developer prompt about AI coding tools isn't part of this app. */
[data-testid="stSkillsNudge"], [data-testid="stSkillsNudgeAnchor"] { display: none !important; }
[data-testid="stMainBlockContainer"] { padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1180px; animation: vp-rise .35s ease-out; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--vp-muted); }

/* Headings: section titles get an accent bar */
[data-testid="stMain"] h2 { font-size: 1.45rem; letter-spacing: -.01em; padding-top: .6rem; }
[data-testid="stMain"] h2::before {
  content: ""; display: inline-block; width: 4px; height: 1.05em; margin-right: .6rem;
  vertical-align: -.12em; border-radius: 4px; background: linear-gradient(180deg, var(--vp-secondary), var(--vp-primary));
}
[data-testid="stMain"] h3 { font-size: 1.12rem; color: var(--vp-text); }

/* Hero header */
.vp-hero { position: relative; padding: 1.6rem 1.75rem 1.5rem; margin: 0 0 1.4rem; border-radius: 20px;
  background: linear-gradient(135deg, rgba(34,197,94,.16), rgba(23,34,29,.92) 45%, rgba(17,25,22,.96));
  border: 1px solid var(--vp-border); box-shadow: var(--vp-shadow); overflow: hidden; }
.vp-hero::after { content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px; border-radius: 50%;
  background: radial-gradient(circle, rgba(134,239,172,.18), transparent 65%); pointer-events: none; }
.vp-eyebrow { display: inline-flex; align-items: center; gap: .45rem; font-size: .72rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: var(--vp-secondary); background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.28);
  padding: .28rem .65rem; border-radius: 999px; }
.vp-hero h1 { margin: .7rem 0 .35rem; padding: 0; font-size: 2.1rem; line-height: 1.15; letter-spacing: -.02em; color: var(--vp-text); }
.vp-hero p { margin: 0; max-width: 760px; color: var(--vp-muted); font-size: 1rem; line-height: 1.55; }

/* Statistic cards */
.vp-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .9rem; margin: .2rem 0 1.2rem; }
.vp-stat { background: var(--vp-card); border: 1px solid var(--vp-border); border-radius: var(--vp-radius); padding: 1rem 1.1rem;
  box-shadow: var(--vp-shadow); transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
.vp-stat:hover { transform: translateY(-2px); border-color: rgba(34,197,94,.45); box-shadow: 0 10px 28px rgba(0,0,0,.35); }
.vp-stat-label { font-size: .74rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--vp-muted); }
.vp-stat-value { margin-top: .35rem; font-size: 1.65rem; font-weight: 700; color: var(--vp-text); letter-spacing: -.02em; }
.vp-stat-value.vp-accent { color: var(--vp-primary); }
.vp-stat-sub { margin-top: .2rem; font-size: .82rem; color: var(--vp-muted); }
.vp-stats.vp-compact { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)) !important; gap: .6rem; margin: .4rem 0 .6rem; }
.vp-compact .vp-stat { padding: .7rem .85rem; box-shadow: none; background: var(--vp-card-raised); }
.vp-compact .vp-stat-value { font-size: 1.25rem; margin-top: .2rem; }

/* Step cards on Home */
.vp-steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .9rem; margin: .4rem 0 1rem; }
.vp-step { background: var(--vp-card); border: 1px solid var(--vp-border); border-radius: var(--vp-radius); padding: 1rem 1.1rem;
  transition: border-color .18s ease, transform .18s ease; }
.vp-step:hover { border-color: rgba(34,197,94,.45); transform: translateY(-2px); }
.vp-step-num { display: inline-grid; place-items: center; width: 1.8rem; height: 1.8rem; border-radius: 9px; font-weight: 700; font-size: .85rem;
  color: var(--vp-bg); background: linear-gradient(135deg, var(--vp-secondary), var(--vp-primary)); }
.vp-step-title { margin: .6rem 0 .25rem; font-weight: 650; color: var(--vp-text); }
.vp-step-text { font-size: .88rem; line-height: 1.5; color: var(--vp-muted); }

/* Streamlit metrics as cards */
[data-testid="stMetric"] { background: var(--vp-card); border: 1px solid var(--vp-border); border-radius: var(--vp-radius);
  padding: .9rem 1.1rem; box-shadow: var(--vp-shadow); transition: border-color .18s ease, transform .18s ease; }
[data-testid="stMetric"]:hover { border-color: rgba(34,197,94,.45); transform: translateY(-1px); }
[data-testid="stMetricLabel"] p { color: var(--vp-muted); font-weight: 600; }

/* Prediction result: the headline card */
[class*="st-key-vp_result"] { background: linear-gradient(135deg, rgba(34,197,94,.18), rgba(23,34,29,.95) 55%);
  border: 1px solid rgba(34,197,94,.45); border-radius: 20px; padding: 1.1rem 1.25rem; box-shadow: 0 12px 34px rgba(0,0,0,.35);
  animation: vp-rise .35s ease-out; }
[class*="st-key-vp_result"] [data-testid="stMetric"] { background: transparent; border: none; box-shadow: none; padding: 0; }
[class*="st-key-vp_result"] [data-testid="stMetricValue"] { color: var(--vp-primary); font-size: 2.4rem; font-weight: 700; }

/* Bordered containers, expanders, alerts, tables */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stLayoutWrapper"] > div[style*="border"] { border-radius: var(--vp-radius); }
[data-testid="stExpander"] details { background: var(--vp-card); border: 1px solid var(--vp-border); border-radius: var(--vp-radius); }
[data-testid="stExpander"] summary:hover { color: var(--vp-secondary); }
[data-testid="stAlert"] > div { border-radius: 14px; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; box-shadow: var(--vp-shadow); }
[data-testid="stVegaLiteChart"], [data-testid="stImage"] img { border-radius: 14px; }
[data-testid="stVegaLiteChart"] { background: var(--vp-card); border: 1px solid var(--vp-border); padding: .6rem .4rem .2rem; }

/* Buttons and inputs */
[data-testid="stBaseButton-primary"] { font-weight: 650; box-shadow: 0 6px 18px rgba(34,197,94,.25);
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease; }
[data-testid="stBaseButton-primary"]:hover { transform: translateY(-1px); filter: brightness(1.06); box-shadow: 0 10px 24px rgba(34,197,94,.35); }
[data-testid="stBaseButton-secondary"] { transition: border-color .15s ease, color .15s ease; }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--vp-primary); color: var(--vp-secondary); }
[data-testid="stRadioOption"] { transition: color .15s ease; }
[data-testid="stSpinner"] { color: var(--vp-secondary); }

/* Page links on Home as navigation cards */
[data-testid="stPageLink"] a { background: var(--vp-card); border: 1px solid var(--vp-border); border-radius: 14px; padding: .75rem 1rem;
  transition: border-color .18s ease, transform .18s ease, background .18s ease; }
[data-testid="stPageLink"] a:hover { border-color: var(--vp-primary); background: var(--vp-card-raised); transform: translateX(2px); }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid #1E2B25; }
[data-testid="stSidebarNav"] { padding-top: .4rem; }
[data-testid="stSidebarNavLink"] { border-radius: 10px; margin: 2px 8px; padding: .45rem .75rem; transition: background .15s ease, color .15s ease; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(34,197,94,.10); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(34,197,94,.16); box-shadow: inset 3px 0 0 var(--vp-primary); }
[data-testid="stSidebarNavLink"][aria-current="page"] p { color: var(--vp-secondary); font-weight: 650; }
.vp-side-card { margin: 1rem .2rem 0; padding: .85rem .95rem; border-radius: 14px; background: var(--vp-card); border: 1px solid var(--vp-border); }
.vp-side-title { font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; color: var(--vp-muted); font-weight: 700; }
.vp-side-row { display: flex; align-items: center; gap: .5rem; margin-top: .5rem; font-size: .84rem; color: var(--vp-text); }
.vp-dot { width: .5rem; height: .5rem; border-radius: 50%; background: var(--vp-primary); box-shadow: 0 0 0 3px rgba(34,197,94,.18); flex: none; }
.vp-side-foot { margin: .9rem .3rem 0; font-size: .74rem; color: var(--vp-muted); line-height: 1.5; }

/* Price comparison (Cultivation Planner): today's actual price next to the harvest-time price */
.vp-cmp { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .9rem; margin: .3rem 0 .8rem; }
.vp-cmp-card { background: var(--vp-card); border: 1px solid var(--vp-border); border-top: 4px solid var(--vp-cmp-accent, var(--vp-border));
  border-radius: var(--vp-radius); padding: 1.1rem 1.25rem; box-shadow: var(--vp-shadow); }
.vp-cmp-tag { display: inline-block; padding: .18rem .6rem; border-radius: 999px; font-size: .72rem; font-weight: 650; letter-spacing: .04em;
  text-transform: uppercase; color: var(--vp-bg); background: var(--vp-cmp-accent, var(--vp-muted)); }
.vp-cmp-title { margin: .65rem 0 .1rem; font-size: .95rem; font-weight: 650; color: var(--vp-text); }
.vp-cmp-value { font-size: 2.3rem; font-weight: 750; letter-spacing: -.02em; color: var(--vp-text); line-height: 1.15; }
.vp-cmp-value small { font-size: 1rem; font-weight: 600; color: var(--vp-muted); }
.vp-cmp-value.vp-cmp-none { font-size: 1.35rem; color: var(--vp-muted); }
.vp-cmp-meta { margin: .45rem 0 0; padding: 0; list-style: none; font-size: .86rem; line-height: 1.6; color: var(--vp-muted); }
.vp-cmp-meta b { color: var(--vp-text); font-weight: 600; }
.vp-cmp .vp-cmp-meta { padding-left: 0 !important; }
.vp-cmp .vp-cmp-meta li { margin: 0 0 .1rem !important; padding: 0 !important; list-style: none; }
.vp-cmp-delta { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem 1.1rem; margin: 0 0 1rem; padding: .9rem 1.25rem;
  border-radius: var(--vp-radius); border: 1px solid var(--vp-delta, var(--vp-border)); background: var(--vp-card-raised); }
.vp-cmp-delta-main { font-size: 1.35rem; font-weight: 750; color: var(--vp-delta, var(--vp-text)); }
.vp-cmp-delta-text { font-size: .92rem; color: var(--vp-text); }
.vp-cmp-delta-note { flex-basis: 100%; font-size: .82rem; color: var(--vp-muted); }

/* Data freshness: one card per source, with a coloured status chip */
.vp-fresh { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: .7rem; margin: .3rem 0 .8rem; }
.vp-fresh-card { background: var(--vp-card); border: 1px solid var(--vp-border); border-left: 4px solid var(--vp-fresh, var(--vp-border));
  border-radius: 12px; padding: .75rem .9rem; font-size: .84rem; line-height: 1.5; color: var(--vp-muted); }
.vp-fresh-name { font-weight: 650; color: var(--vp-text); font-size: .92rem; }
.vp-fresh-chip { display: inline-block; margin: .3rem 0 .35rem; padding: .12rem .55rem; border-radius: 999px; font-size: .72rem; font-weight: 700;
  color: var(--vp-bg); background: var(--vp-fresh, var(--vp-muted)); }
.vp-fresh-card b { color: var(--vp-text); font-weight: 600; }
.vp-fresh-note { margin-top: .3rem; font-size: .78rem; }

/* Language switcher, top right of every page */
[class*="st-key-vp_lang"] { align-items: flex-end; margin-bottom: -.4rem; }
[class*="st-key-vp_lang"] > div { width: auto !important; }
[class*="st-key-vp_lang"] [data-testid="stButtonGroup"] { justify-content: flex-end; }

/* Responsive */
@media (max-width: 1100px) {
  .vp-stats, .vp-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 1024px) {
  [data-testid="stMainBlockContainer"] { padding-left: 1.4rem; padding-right: 1.4rem; }
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding: 1.6rem .9rem 3rem; }
  .vp-hero { padding: 1.15rem 1.1rem; border-radius: 16px; }
  .vp-hero h1 { font-size: 1.55rem; }
  .vp-hero p { font-size: .92rem; }
  .vp-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .6rem; }
  .vp-steps { grid-template-columns: 1fr; }
  .vp-stat { padding: .8rem .85rem; }
  .vp-stat-value { font-size: 1.3rem; }
  .vp-cmp { grid-template-columns: 1fr; }
  .vp-cmp-value { font-size: 1.9rem; }
  [class*="st-key-vp_result"] [data-testid="stMetricValue"] { font-size: 1.9rem; }
  [data-testid="stMain"] h2 { font-size: 1.25rem; }
}
</style>
"""

_MATPLOTLIB_STYLE = {
    "figure.facecolor": COLORS["card"],
    "axes.facecolor": COLORS["card"],
    "savefig.facecolor": COLORS["card"],
    "axes.edgecolor": COLORS["border"],
    "axes.labelcolor": COLORS["muted"],
    "axes.titlecolor": COLORS["text"],
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=[COLORS["primary"], COLORS["forecast"], COLORS["secondary"], COLORS["info"]]),
    "grid.color": COLORS["border"],
    "grid.linewidth": 0.8,
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "text.color": COLORS["text"],
    "legend.facecolor": COLORS["sidebar"],
    "legend.edgecolor": COLORS["border"],
    "legend.labelcolor": COLORS["text"],
    "legend.framealpha": 0.95,
    "lines.linewidth": 2.0,
    "font.family": ["Segoe UI", "Nirmala UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
}


def apply_theme(page_sources: tuple[str, ...] | None = None) -> None:
    """Call once per page, right after st.set_page_config().

    page_sources: the sources (names from _SOURCE_FILES) this page actually uses. The sidebar then
    lists only those, under "Used on this page", so a page never appears to use a source it doesn't.
    Omitted, it lists every source the app has data for.
    """
    init_language()  # the chosen language (app/i18n.py), before any text is drawn
    mpl.rcParams.update(_MATPLOTLIB_STYLE)
    st.markdown(_CSS, unsafe_allow_html=True)
    if (_ASSETS / "logo.svg").exists():
        st.logo(str(_ASSETS / "logo.svg"), icon_image=str(_ASSETS / "logo_icon.svg"), size="large")
    _navigation()
    names = available_sources()
    title = t("Data sources")
    if page_sources is not None:
        names = [name for name in names if name in page_sources]
        title = t("Used on this page")
    rows = "".join(f'<div class="vp-side-row"><span class="vp-dot"></span>{escape(t(name))}</div>' for name in names)
    st.sidebar.markdown(
        f"""
        <div class="vp-side-card">
          <div class="vp-side-title">{escape(title)}</div>
          {rows}
        </div>
        <div class="vp-side-foot">{escape(t("Dambulla wholesale market · Ridge direct forecaster, 1–30 days ahead"))}</div>
        """,
        unsafe_allow_html=True,
    )


# Every page, with the name the sidebar shows (translated when drawn). .streamlit/config.toml turns off
# Streamlit's own list, which can only show the file names in English.
PAGES = (
    ("Home.py", "Home", "🏠"),
    ("pages/1_Predict.py", "Predict", "📈"),
    ("pages/2_Model_Performance.py", "Model Performance", "📊"),
    ("pages/3_Price_History.py", "Price History", "🗓️"),
    ("pages/4_Cultivation_Planner.py", "Cultivation Planner", "🌱"),
)


def _navigation() -> None:
    for path, label, icon in PAGES:
        try:
            st.sidebar.page_link(path, label=t(label), icon=icon)
        except Exception:  # noqa: BLE001 -- a page run on its own (tests) has no multipage app to link within
            return


# Source name -> the file that must exist for the app to be using it.
SOURCE_HARTI = "HARTI daily bulletins (wholesale)"
SOURCE_HORDI = "DOA HORDI crop guidance"
SOURCE_WEATHER = "Open-Meteo weather"
SOURCE_FUEL = "CEYPETCO fuel prices"
_SOURCE_FILES = (
    ("HARTI daily bulletins (wholesale)", "paths.raw_files.harti_prices"),
    ("CBSL daily report (wholesale & retail)", "paths.raw_files.cbsl_daily_prices"),
    ("DCS weekly prices (retail)", "paths.raw_files.dcs_weekly_retail"),
    ("DOA HORDI crop guidance", "data/reference/doa_harvest_durations_v2.csv"),
    ("Open-Meteo weather", "paths.raw_files.weather_data_by_region"),
    ("CEYPETCO fuel prices", "paths.raw_files.fuel_prices"),
)


def available_sources(config_path: str = "config/config.yaml") -> list[str]:
    """The data sources whose datasets are actually present, so the app never lists one it isn't using."""
    from config.config_loader import ConfigLoader

    config = ConfigLoader(config_path)
    names = []
    for name, key in _SOURCE_FILES:
        path = config.get(key) if key.startswith("paths.") else key
        if path and Path(path).exists():
            names.append(name)
    return names


def page_header(title: str, subtitle: str, eyebrow: str) -> None:
    """The page's hero banner, with its title as the page heading, and the language switcher above it.

    title, subtitle and eyebrow are English; they are translated here.
    """
    with st.container(key="vp_lang"):
        language_selector()
    title, subtitle, eyebrow = t(title), t(subtitle), t(eyebrow)
    st.markdown(
        f'<div class="vp-hero"><span class="vp-eyebrow">{escape(eyebrow)}</span>'
        f"<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def stat_cards(cards: list[dict], compact: bool = False) -> None:
    """A responsive row of statistic cards: each dict has label, value and optional sub and accent.

    compact: smaller cards that wrap to fit a narrow column, such as the weather details on Predict.
    """
    items = "".join(
        f'<div class="vp-stat"><div class="vp-stat-label">{escape(c["label"])}</div>'
        f'<div class="vp-stat-value{" vp-accent" if c.get("accent") else ""}">{escape(c["value"])}</div>'
        + (f'<div class="vp-stat-sub">{escape(c["sub"])}</div>' if c.get("sub") else "")
        + "</div>"
        for c in cards
    )
    st.markdown(f'<div class="vp-stats{" vp-compact" if compact else ""}">{items}</div>', unsafe_allow_html=True)


def data_source_panel(source: str, market: str, price_type: str, latest, today=None, extra: str = "") -> None:
    """Which data a page is showing, its latest date, and a freshness warning when it is old.

    source and market are names, shown as given; price_type is translated by the caller.
    """
    from app.catalogue import freshness

    fresh = freshness(latest, today)
    stat_cards(
        [
            {"label": t("Data source"), "value": source},
            {"label": t("Market"), "value": market},
            {"label": t("Price type"), "value": price_type},
            {"label": t("Latest data"), "value": str(fresh.latest.date()), "sub": t("{days} day(s) before today", days=fresh.age_days), "accent": fresh.level == "current"},
        ],
        compact=True,
    )
    if fresh.level == "stale":
        st.warning(fresh.message + (f" {extra}" if extra else ""))
    elif fresh.level == "ageing":
        st.info(fresh.message + (f" {extra}" if extra else ""))
    elif extra:
        st.caption(extra)


# Direction of a price change: colour, arrow and words, so the meaning never rests on colour alone.
PRICE_CHANGE_STYLE = {
    "higher": (COLORS["primary"], "▲"),
    "lower": ("#F87171", "▼"),
    "similar": (COLORS["muted"], "◆"),
}


def price_comparison(cards: list[dict], delta: dict | None) -> None:
    """Two price cards side by side (one column on a phone), then the change between them.

    Each card: tag, title, value (None for no price), value_text (shown instead when value is None),
    accent (a colour), meta (list of (label, text)). delta: direction, headline, text, note.
    """
    items = []
    for card in cards:
        meta = "".join(f"<li><b>{escape(label)}:</b> {escape(text)}</li>" for label, text in card.get("meta", []))
        value = (
            f'<div class="vp-cmp-value">Rs. {card["value"]:,.2f} <small>/ {escape(t("kg"))}</small></div>'
            if card.get("value") is not None
            else f'<div class="vp-cmp-value vp-cmp-none">{escape(card.get("value_text", t("No price")))}</div>'
        )
        items.append(
            f'<div class="vp-cmp-card" style="--vp-cmp-accent: {card["accent"]}">'
            f'<span class="vp-cmp-tag">{escape(card["tag"])}</span>'
            f'<div class="vp-cmp-title">{escape(card["title"])}</div>{value}<ul class="vp-cmp-meta">{meta}</ul></div>'
        )
    html = f'<div class="vp-cmp">{"".join(items)}</div>'
    if delta:
        colour, arrow = PRICE_CHANGE_STYLE.get(delta.get("direction"), (COLORS["muted"], "–"))
        note = f'<div class="vp-cmp-delta-note">{escape(delta["note"])}</div>' if delta.get("note") else ""
        html += (
            f'<div class="vp-cmp-delta" style="--vp-delta: {colour}">'
            f'<span class="vp-cmp-delta-main">{arrow} {escape(delta["headline"])}</span>'
            f'<span class="vp-cmp-delta-text">{escape(delta["text"])}</span>{note}</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


FRESHNESS_COLORS = {
    "Latest Available Data": COLORS["primary"],
    "Recently Updated": COLORS["info"],
    "Update Pending": COLORS["forecast"],
    "Source Unavailable": "#F87171",
    "Historical Data": COLORS["muted"],
}


def data_freshness_panel(sources: list, compact: bool = False) -> None:
    """One card per data source: status label, category, location, newest stored data and last checks.

    sources: app.catalogue.data_freshness() rows. compact leaves out the publishing schedule and notes.
    """
    def when(value) -> str:
        return f"{value:%Y-%m-%d %H:%M}" if value is not None else t("never")

    cards = []
    for row in sources:
        source = row.source
        lines = [
            f"<div><b>{escape(t('Type'))}:</b> {escape(t(source.category))}</div>",
            f"<div><b>{escape(t('Location'))}:</b> {escape(t(source.location))}</div>",
            f"<div><b>{escape(t('Latest data'))}:</b> {escape(row.latest_stored or t('none stored'))}</div>",
        ]
        if row.newest_published and row.newest_published != row.latest_stored:
            lines.append(f"<div><b>{escape(t('Newest found at last check'))}:</b> {escape(str(row.newest_published))}</div>")
        lines.append(f"<div><b>{escape(t('Last successful sync'))}:</b> {escape(when(row.last_success))}</div>")
        if not compact:
            lines.append(f"<div><b>{escape(t('Publishes'))}:</b> {escape(t(source.schedule))}</div>")
            lines.append(f'<div class="vp-fresh-note">{escape(row.note)}</div>')
        cards.append(
            f'<div class="vp-fresh-card" style="--vp-fresh: {FRESHNESS_COLORS.get(row.label, COLORS["muted"])}">'
            f'<div class="vp-fresh-name">{escape(t(source.name))}</div>'
            f'<span class="vp-fresh-chip">{escape(t(row.label))}</span>{"".join(lines)}</div>'
        )
    st.markdown(f'<div class="vp-fresh">{"".join(cards)}</div>', unsafe_allow_html=True)


def freshness_summary(sources: list) -> str:
    """One line for an expander title, e.g. 'Data freshness: 4 up to date, 1 update pending'."""
    counts: dict[str, int] = {}
    for row in sources:
        counts[row.label] = counts.get(row.label, 0) + 1
    up_to_date = counts.get("Latest Available Data", 0) + counts.get("Recently Updated", 0)
    parts = [t("{n} of {total} sources up to date", n=up_to_date, total=len(sources))]
    if counts.get("Update Pending"):
        parts.append(t("{n} update pending", n=counts["Update Pending"]))
    if counts.get("Source Unavailable"):
        parts.append(t("{n} source unavailable", n=counts["Source Unavailable"]))
    return t("Data freshness") + ": " + ", ".join(parts)


# Chart dates: the chart library writes month names in English ("Jul", "October"), so in Sinhala and Tamil
# dates are shown as numbers instead. English charts are unchanged.
NUMERIC_DATE = "%Y-%m-%d"


def date_axis(**kwargs):
    """An Altair date axis: as before in English, numeric dates (2026-11-17) in Sinhala and Tamil."""
    import altair as alt

    if current_language() != "en":
        kwargs.setdefault("format", NUMERIC_DATE)
    return alt.Axis(**kwargs)


def date_tooltip(field: str, title: str):
    import altair as alt

    return alt.Tooltip(field, title=title, **({} if current_language() == "en" else {"format": NUMERIC_DATE}))


def line_chart(series, y_title: str, color: str, height: int | None = None) -> None:
    """A date-indexed series as a line. In English this is st.line_chart, exactly as before; in Sinhala
    and Tamil an Altair line with numeric dates. Gaps marked by an empty point (services.break_long_gaps)
    break the line in both."""
    import altair as alt

    if current_language() == "en":
        st.line_chart(series, x_label=t("Date"), y_label=y_title, color=color, **({"height": height} if height else {}))
        return
    data = series.rename("value").rename_axis("date").reset_index()
    data["segment"] = data["value"].isna().cumsum()
    data = data.dropna(subset=["value"])
    chart = alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=t("Date"), axis=date_axis()),
        y=alt.Y("value:Q", title=y_title),
        detail="segment:N",
        color=alt.value(color),
        tooltip=[date_tooltip("date:T", t("Date")), alt.Tooltip("value:Q", title=y_title, format=",.2f")],
    )
    st.altair_chart(chart.properties(height=height or 300), width="stretch")


def translated_columns(frame):
    """A copy of a table for display, with its English column headers in the current language.

    Only the headers change; the values (prices, dates, vegetable and market names as stored) don't.
    """
    return frame.rename(columns={column: t(column) for column in frame.columns if isinstance(column, str)})


def step_cards(steps: list[tuple[str, str]]) -> None:
    """Numbered explanation cards: (title, text) pairs."""
    items = "".join(
        f'<div class="vp-step"><span class="vp-step-num">{number}</span>'
        f'<div class="vp-step-title">{escape(title)}</div><div class="vp-step-text">{escape(text)}</div></div>'
        for number, (title, text) in enumerate(steps, start=1)
    )
    st.markdown(f'<div class="vp-steps">{items}</div>', unsafe_allow_html=True)

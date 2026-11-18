"""Editable source of the hand-laid-out diagrams (architecture, context/DFD 0, use case, DFD 1, multilingual).

Each diagram is a list of shapes with explicit coordinates, written as SVG into ../rendered/. Edit a coordinate or
a label here and re-run:  python structural_diagrams.py   (PNG copies are made from the SVGs in headless Edge by
the report build; the SVG files open in any browser, Inkscape or draw.io).
"""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "rendered"
FONT = "Arial, 'Nirmala UI', sans-serif"
INK, LINE = "#1d2733", "#44515e"
GREEN, GREEN_EDGE = "#e8f3ec", "#2f7d4f"
BLUE, BLUE_EDGE = "#e9f0f8", "#3b5f8a"
GREY, GREY_EDGE = "#f4f6f8", "#8a97a5"
AMBER, AMBER_EDGE = "#fff4e0", "#a8730c"


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def text(self, x, y, lines, size=15, weight="normal", anchor="middle", color=INK, italic=False):
        lines = [lines] if isinstance(lines, str) else lines
        lh = size * 1.25
        y0 = y - (len(lines) - 1) * lh / 2
        style = ' font-style="italic"' if italic else ""
        for i, line in enumerate(lines):
            self.parts.append(f'<text x="{x}" y="{y0 + i * lh:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
                              f'text-anchor="{anchor}" dominant-baseline="central" fill="{color}"{style}>{escape(line)}</text>')

    def box(self, x, y, w, h, lines, fill=GREEN, edge=GREEN_EDGE, size=15, rx=6, weight="normal", dash=False):
        d = ' stroke-dasharray="6 4"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{edge}" stroke-width="1.6"{d}/>')
        if lines:
            self.text(x + w / 2, y + h / 2, lines, size, weight)

    def band(self, x, y, w, h, title, fill=GREY, edge=GREY_EDGE):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{edge}" stroke-width="1.4"/>')
        self.text(x + 12, y + 16, title, 14, "bold", "start", "#3a4652")

    def circle(self, cx, cy, r, lines, size=15, fill=GREEN, edge=GREEN_EDGE):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{edge}" stroke-width="1.8"/>')
        self.text(cx, cy, lines, size)

    def ellipse(self, cx, cy, rx, ry, lines, size=14):
        self.parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{GREEN}" stroke="{GREEN_EDGE}" stroke-width="1.6"/>')
        self.text(cx, cy, lines, size)

    def store(self, x, y, w, h, ident, lines, size=14):
        """Gane-Sarson data store: open-ended rectangle with an identifier compartment."""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{BLUE}" stroke="none"/>')
        self.parts.append(f'<path d="M{x + w} {y} H{x} V{y + h} H{x + w}" fill="none" stroke="{BLUE_EDGE}" stroke-width="1.6"/>')
        self.parts.append(f'<line x1="{x + 40}" y1="{y}" x2="{x + 40}" y2="{y + h}" stroke="{BLUE_EDGE}" stroke-width="1.6"/>')
        self.text(x + 20, y + h / 2, ident, size, "bold")
        self.text(x + 40 + (w - 40) / 2, y + h / 2, lines, size)

    def actor(self, cx, top, label, size=14):
        p = self.parts
        p.append(f'<circle cx="{cx}" cy="{top + 10}" r="10" fill="#fff" stroke="{INK}" stroke-width="1.8"/>')
        p.append(f'<path d="M{cx} {top + 20} V{top + 48} M{cx - 18} {top + 30} H{cx + 18} M{cx} {top + 48} L{cx - 14} {top + 68} '
                 f'M{cx} {top + 48} L{cx + 14} {top + 68}" fill="none" stroke="{INK}" stroke-width="1.8"/>')
        self.text(cx, top + 86, label, size)

    def arrow(self, pts, label=None, at=0.5, dx=0, dy=-10, dash=False, size=13, head=True, both=False, width=1.6, anchor="middle"):
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        dashed = ' stroke-dasharray="6 4"' if dash else ""
        end = ' marker-end="url(#a)"' if head else ""
        start = ' marker-start="url(#s)"' if both else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="{width}"{dashed}{end}{start}/>')
        if label:
            # label at fraction `at` along the first segment that holds it
            seg = pts if len(pts) == 2 else pts[:2]
            for i in range(len(pts) - 1):
                if label and at <= 1: seg = pts[i:i + 2]; break
            (x1, y1), (x2, y2) = seg
            lx, ly = x1 + (x2 - x1) * at + dx, y1 + (y2 - y1) * at + dy
            lines = label if isinstance(label, list) else [label]
            wmax = max(len(s) for s in lines) * size * 0.53 + 8
            hh = len(lines) * size * 1.25 + 4
            bx = lx - wmax / 2 if anchor == "middle" else (lx - 4 if anchor == "start" else lx - wmax + 4)
            self.parts.append(f'<rect x="{bx:.1f}" y="{ly - hh / 2:.1f}" width="{wmax:.1f}" height="{hh:.1f}" fill="#ffffff" fill-opacity="0.92"/>')
            self.text(lx, ly, lines, size, anchor=anchor, color="#2c3743")

    def save(self, name):
        defs = (f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
                f'<path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/></marker>'
                f'<marker id="s" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
                f'<path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/></marker></defs>')
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">{defs}'
               f'<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.parts) + "</svg>")
        (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")


def architecture():
    s = Svg(1000, 900)
    W, X = 960, 20
    s.box(380, 12, 240, 46, ["User (web browser)", "http://localhost:8501"], AMBER, AMBER_EDGE, 14)
    # Presentation
    s.band(X, 82, W, 170, "1  Presentation layer: Streamlit multipage app (app/)")
    names = ["Home", "Predict", ["Model", "Performance"], ["Price", "History"], ["Cultivation", "Planner"]]
    for i, n in enumerate(names):
        s.box(40 + i * 188, 112, 170, 54, n, size=15, weight="bold")
    s.box(40, 182, 450, 54, ["Shared logic: services.py · forecasting.py", "cultivation.py · catalogue.py · source_records.py"], "#ffffff", GREEN_EDGE, 14)
    s.box(510, 182, 450, 54, ["ui.py (theme, cards, freshness panel) · i18n.py", "+ translations/en.json, si.json, ta.json"], "#ffffff", GREEN_EDGE, 14)
    # Processing
    s.band(X, 294, W, 132, "2  Processing and model layer (src/, scripts/)")
    items = [["Preprocessing", "clean, standardise,", "merge"], ["ForecastFeatureBuilder", "as-of-origin", "features"],
             ["DirectForecaster", "30 Ridge models", "(1 to 30 days)"], ["Earlier models", "LR · RF · XGBoost", "(uploaded CSV)"],
             ["Training scripts", "train_forecaster.py", "retrain_if_better.py"]]
    for i, n in enumerate(items):
        s.box(40 + i * 188, 326, 170, 82, n, "#ffffff", GREEN_EDGE, 13)
    # Storage
    s.band(X, 468, W, 132, "3  File storage (CSV, JSON, joblib; no database server)")
    items = [["data/raw/*.csv", "prices · weather · fuel"], ["data_manifest.json", "update_log.jsonl", "harti_import_log.csv"],
             ["models/*.joblib", "reports/metrics/*.json"], ["Backups folder", "(outside the project)"],
             ["data/reference/", "DOA durations · local", "names · item map"]]
    for i, n in enumerate(items):
        s.box(40 + i * 188, 500, 170, 82, n, BLUE, BLUE_EDGE, 13)
    # Collection
    s.band(X, 642, W, 132, "4  Data collection layer (src/data_collection/, scripts/)")
    s.box(40, 674, 450, 82, ["daily_update.py: new records only, validation,", "backups, retries, lock, manifest and run log", "run every 3 hours by Windows Task Scheduler or on request"], "#ffffff", GREEN_EDGE, 13)
    s.box(510, 674, 210, 82, ["Parsers: HARTI, CBSL", "and DCS PDF bulletins"], "#ffffff", GREEN_EDGE, 13)
    s.box(740, 674, 220, 82, ["Clients: Open-Meteo,", "CEYPETCO · update_data.py", "(full rebuild since 2015)"], "#ffffff", GREEN_EDGE, 13)
    # Sources
    srcs = [["HARTI", "daily · PDF"], ["CBSL", "working days · PDF"], ["DCS", "weekly · PDF"], ["Open-Meteo", "daily · JSON API"],
            ["CEYPETCO", "on revision · JSON"], ["DOA / HORDI", "static web pages"]]
    for i, n in enumerate(srcs):
        s.box(20 + i * 163, 818, 145, 64, n, AMBER, AMBER_EDGE, 13)
    # arrows between layers (data flows upwards)
    s.arrow([(500, 818), (500, 774)], "downloads (PDF, JSON)", dx=95, dy=0)
    s.arrow([(500, 642), (500, 600)], "validated new rows appended", dx=118, dy=0)
    s.arrow([(500, 468), (500, 426)], "datasets, saved models, metrics", dx=128, dy=0)
    s.arrow([(500, 294), (500, 252)], "forecasts, plans, history, freshness", dx=142, dy=0)
    s.arrow([(500, 82), (500, 58)], both=True)
    s.arrow([(912, 818), (912, 798), (992, 798), (992, 541), (962, 541)], dash=True)
    s.text(905, 806, "collected once", 12, anchor="end", italic=True)
    s.save("01_system_architecture")


def context():
    import math
    s = Svg(1080, 620)
    cx, cy, r = 545, 310, 112

    def edge(x, y):
        ang = math.atan2(y - cy, x - cx)
        return cx + r * math.cos(ang), cy + r * math.sin(ang)
    s.circle(cx, cy, r, ["0", "Vegetable Price", "Prediction and", "Planning System"], 16)
    left = [("HARTI", "daily wholesale bulletins (PDF)"), ("CBSL", "daily price reports (PDF)"), ("DCS", "weekly retail bulletins (PDF)"),
            ("Open-Meteo", "rainfall, temperature, humidity"), ("CEYPETCO", "fuel price revisions"), ("DOA / HORDI", "harvest durations")]
    for i, (name, flow) in enumerate(left):
        y = 52 + i * 103
        s.box(20, y - 26, 140, 52, name, BLUE, BLUE_EDGE, 15, weight="bold")
        s.arrow([(160, y), (370, y), edge(370, y)])
        s.text(166, y - 12, flow, 13, anchor="start", color="#2c3743")
    right = [("User", 150, ["vegetable, period, planting", "date, filters, language"], ["forecasts, history, plans,", "freshness, accuracy"]),
             ("Data maintainer", 330, ["rebuild and retrain", "commands"], ["update log, retraining", "decision"]),
             ("Windows Task|Scheduler", 500, ["start an update", "every 3 hours"], None)]
    for name, y, inflow, outflow in right:
        s.box(900, y - 40, 160, 80, name.split("|"), BLUE, BLUE_EDGE, 15, weight="bold")
        if outflow:
            yi, yo = y - 22, y + 22
            s.arrow([(900, yi), (730, yi), edge(730, yi)])
            s.text(894, yi - 22, inflow, 13, anchor="end", color="#2c3743")
            s.arrow([edge(730, yo), (730, yo), (900, yo)])
            s.text(894, yo + 22, outflow, 13, anchor="end", color="#2c3743")
        else:
            s.arrow([(900, y), (730, y), edge(730, y)])
            s.text(894, y - 22, inflow, 13, anchor="end", color="#2c3743")
    s.save("02_context_dfd_level0")


def use_case():
    s = Svg(1000, 730)
    s.band(190, 20, 640, 650, "Vegetable Price Prediction and Planning System", "#fbfcfd", "#6c7a89")
    left = [["View data overview", "and freshness"], ["Request a price", "forecast (1-30 days)"], ["Predict from an", "uploaded CSV"],
            ["View model", "performance"], ["Explore price history", "(HARTI, CBSL, DCS)"], ["Choose interface", "language"],
            ["Generate a", "cultivation plan"], ["Check official", "sources now"]]
    ys = [80 + i * 70 for i in range(8)]
    for y, lab in zip(ys, left):
        s.ellipse(360, y, 135, 29, lab)
    s.actor(80, ys[3] - 20, [])
    s.text(80, ys[3] - 62, ["User (farmer, trader,", "student, analyst)"], 14)
    for y in ys:
        s.arrow([(98, ys[3] + 18), (225, y)], head=False)
    s.ellipse(680, ys[6], 128, 29, ["Compare current and", "harvest-time price"])
    s.ellipse(680, ys[7], 128, 29, ["Update datasets from", "official sources"])
    s.ellipse(680, ys[7] + 80, 128, 29, ["Retrain the forecaster", "only if it is better"])
    for y in (ys[6], ys[7]):
        s.arrow([(495, y), (552, y)], "«include»", dy=-14, dash=True, size=12)
    s.actor(915, ys[6] - 36, ["Task", "Scheduler"])
    s.arrow([(897, ys[6] + 8), (800, ys[7] - 10)], head=False)
    s.actor(915, ys[7] + 26, ["Data", "maintainer"])
    s.arrow([(897, ys[7] + 60), (806, ys[7] + 6)], head=False)
    s.arrow([(897, ys[7] + 68), (806, ys[7] + 80)], head=False)
    s.save("03_use_case")


def dfd1():
    s = Svg(1000, 615)
    L = lambda x, y, t, anchor="start": s.text(x, y, t, 13, anchor=anchor, color="#2c3743")
    a = s.arrow
    # external entities
    s.box(20, 28, 240, 54, ["HARTI · CBSL · DCS", "Open-Meteo · CEYPETCO"], AMBER, AMBER_EDGE, 14, weight="bold")
    s.box(920, 196, 75, 48, "User", AMBER, AMBER_EDGE, 15, weight="bold")
    s.box(680, 476, 120, 44, "DOA / HORDI", AMBER, AMBER_EDGE, 14, weight="bold")
    # processes
    P = {1: (110, 220), 2: (280, 220), 4: (470, 220), 5: (650, 220), 6: (830, 220), 3: (330, 430)}
    labels = {1: ["1.0", "Collect and", "validate"], 2: ["2.0", "Build", "features"], 3: ["3.0", "Train and", "evaluate"],
              4: ["4.0", "Produce", "forecast"], 5: ["5.0", "Build", "cultivation plan"], 6: ["6.0", "Present", "results"]}
    for k, (x, y) in P.items():
        s.circle(x, y, 56, labels[k], 13)
    # data stores
    s.store(20, 378, 190, 44, "D1", ["Price, weather", "and fuel data"], 13)
    s.store(300, 40, 190, 44, "D2", ["Manifest and", "update logs"], 13)
    s.store(410, 408, 170, 44, "D3", ["Models and", "metrics"], 13)
    s.store(680, 380, 120, 44, "D4", ["Reference", "data"], 13)
    # flows
    a([(110, 82), (110, 164)]); L(102, 123, ["bulletins,", "reports,", "weather, fuel"], "end")
    a([(150, 181), (300, 66)]); L(232, 142, ["run status,", "hashes"])
    a([(110, 276), (110, 378)]); L(102, 327, ["validated", "new rows"], "end")
    a([(190, 378), (247, 266)]); L(224, 330, "history")
    a([(336, 220), (414, 220)]); L(375, 180, ["latest", "features"], "middle")
    a([(300, 273), (318, 375)]); L(318, 300, ["origin", "features"])
    a([(386, 430), (410, 430)]); L(392, 470, "models, scores")
    a([(495, 408), (482, 275)]); L(500, 340, ["Ridge", "models"])
    a([(526, 220), (594, 220)]); L(560, 180, ["forecast", "path"], "middle")
    a([(470, 164), (470, 125), (795, 125), (795, 177)]); L(632, 112, "forecast (Predict page)", "middle")
    a([(706, 220), (774, 220)]); L(740, 180, ["plan, price", "comparison"], "middle")
    a([(490, 62), (830, 62), (830, 164)]); L(838, 105, "freshness")
    a([(700, 380), (674, 270)]); L(704, 330, ["crop", "timing"])
    a([(740, 476), (740, 424)]); L(748, 450, "durations")
    a([(110, 422), (110, 570), (812, 570), (812, 274)]); L(300, 558, "history (D1)", "middle"); L(804, 310, "history", "end")
    a([(540, 452), (540, 596), (852, 596), (852, 268)]); L(548, 500, "metrics (D3)"); L(860, 420, "metrics")
    a([(630, 570), (630, 273)]); L(622, 360, ["recorded", "prices"], "end")
    a([(886, 208), (920, 208)]); L(903, 190, "pages", "middle")
    a([(920, 234), (886, 234)]); L(912, 258, "requests", "middle")
    s.save("04_dfd_level1")


def multilingual():
    s = Svg(1000, 560)
    s.box(20, 40, 220, 56, ["Address bar", "…/Predict?lang=si"], AMBER, AMBER_EDGE, 14)
    s.box(20, 160, 220, 64, ["Switcher (top right)", "English · සිංහල · தமிழ்"], AMBER, AMBER_EDGE, 14)
    s.box(300, 40, 230, 56, ["init_language(): session,", "then address, then English"], size=14)
    s.box(300, 160, 230, 64, ["st.session_state['lang']", "kept across reruns and pages"], size=14)
    s.box(300, 300, 230, 56, ["set_language(): current", "language for this run"], size=14)
    s.box(600, 40, 380, 56, ["Page code:  t('Price history')", "t('{n} of {total}', n=3, total=5)"], "#ffffff", GREEN_EDGE, 14)
    s.box(600, 160, 380, 64, ["t(text, **values)", "English text is the key"], size=15, weight="bold")
    s.box(600, 290, 380, 76, ["app/translations/en.json · si.json · ta.json", "864 texts each, same keys and placeholders", "checked by tests/app/test_i18n.py"], BLUE, BLUE_EDGE, 14)
    s.box(600, 420, 380, 64, ["Translated text; values (prices, dates) filled", "in after translation; missing key → English"], "#ffffff", GREEN_EDGE, 14)
    s.box(20, 300, 220, 64, ["vegetable_names.csv", "(DOA Sinhala and Tamil names)"], BLUE, BLUE_EDGE, 14)
    s.box(20, 420, 220, 64, ["display_name():", "'Beans · බෝංචි'"], size=14)
    s.box(300, 420, 230, 64, ["Data, model inputs, prices,", "source and market names"], GREY, GREY_EDGE, 14)
    a = s.arrow
    a([(240, 56), (300, 56)])
    a([(300, 82), (240, 82)], dash=True)
    s.text(270, 110, "written back", 12, italic=True)
    a([(240, 192), (300, 192)], "on_change", dy=-13)
    a([(415, 96), (415, 160)])
    a([(415, 224), (415, 300)])
    a([(530, 328), (565, 328), (565, 200), (600, 200)])
    a([(790, 96), (790, 160)])
    a([(790, 290), (790, 224)])
    a([(980, 192), (990, 192), (990, 452), (980, 452)])
    a([(130, 364), (130, 420)])
    s.text(415, 510, "not translated: shown as stored", 12, italic=True)
    a([(240, 452), (300, 452)], head=False, dash=True)
    s.text(790, 520, "→ headings, buttons, messages, tables, chart titles, axes, tooltips", 13, italic=True)
    s.save("09_multilingual_architecture")


if __name__ == "__main__":
    architecture(); context(); use_case(); dfd1(); multilingual()
    print("written to", OUT)

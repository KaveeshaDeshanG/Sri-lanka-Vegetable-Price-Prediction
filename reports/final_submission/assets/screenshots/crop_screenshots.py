"""Crop report screenshots from the full-page captures, using the heading positions recorded during capture.
Only cropping: no pixel is edited. argv: shots folder (with full/ and capture.json), output folder."""
import json, sys
from pathlib import Path
from PIL import Image

SHOTS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
CAP = json.loads((SHOTS / "capture.json").read_text(encoding="utf-8"))
P = Path(__file__).resolve().parents[4] / "app" / "translations"
TR = {c: json.loads((P / f"{c}.json").read_text(encoding="utf-8")) for c in ("si", "ta")}
tr = lambda c, s: s if c == "en" else TR[c][s]


def marks(name):
    return CAP["captures"][name]["marks"]


def y_of(name, text, after=0, tags=("H1", "H2", "H3", "H4")):
    for m in marks(name):
        if (tags is None or m["tag"] in tags) and m["text"].startswith(text) and m["y"] >= after:
            return m["y"]
    return None


def main_box(name):
    m = next(m for m in marks(name) if m["id"] == "stMainBlockContainer")
    return m["x"], m["x"] + m["w"]


def crop(name, out, start=None, end=None, pad_top=14, pad_bottom=6, sidebar=False, end_extra=None, start_y=None, tags=("H1", "H2", "H3", "H4")):
    cap = CAP["captures"][name]; s = cap["scale"]
    img = Image.open(SHOTS / "full" / f"{name}.png")
    y0 = start_y if start_y is not None else (y_of(name, start, tags=tags) if start else 0)
    if y0 is None: raise SystemExit(f"{name}: start '{start}' not found")
    y0 = max(0, y0 - pad_top)
    if end_extra is not None:
        y1 = y0 + end_extra
    elif end:
        y1 = y_of(name, end, after=y0 + 30, tags=tags)
        if y1 is None: raise SystemExit(f"{name}: end '{end}' not found")
        y1 -= pad_bottom
    else:
        y1 = cap["height_css"]
    x0, x1 = (0, 1280) if sidebar else main_box(name)
    x0 = max(0, x0 - 8); x1 = min(1280, x1 + 8)
    box = tuple(int(v * s) for v in (x0, y0, x1, min(y1, cap["height_css"])))
    im = img.crop(box)
    im.save(OUT / out, optimize=True)
    print(out, im.size)


crop("home_en", "01_home_en.png", start_y=0, end="Data freshness", sidebar=True)
crop("home_en", "02_home_data_freshness.png", "Data freshness", "Datasets")
crop("home_en", "03_home_update_status.png", "Datasets", "How it works")
crop("switcher_si", "04_language_switcher.png", start_y=0, end=tr("si", "Data freshness"), sidebar=True)
crop("home_si", "05_home_si.png", start_y=0, end=tr("si", "Data freshness"), sidebar=True)
crop("home_ta", "06_home_ta.png", start_y=0, end=tr("ta", "Data freshness"), sidebar=True)
crop("predict_default", "07_predict_source_status.png", "1. Official source status", "2. Latest collected data")
crop("predict_result_beans_7d", "09_predict_result_beans_7d.png", "📈 Future Vegetable Price Forecast", end_extra=800, tags=None)
crop("predict_result_beans_7d", "09b_predict_horizon_table.png", "Forecasts at 1, 7, 14 and 30 days", "5. Data freshness and warnings", tags=None)
crop("predict_upload_missing_price_column", "10_predict_upload_error.png", "1. Your data", end_extra=430)
crop("performance_en", "11_model_performance.png", "Ridge direct forecaster", "Next-day comparison on the same rows")
crop("history_harti_dambulla", "12_price_history_harti.png", start_y=0, end="Weather", sidebar=False)
crop("history_cbsl_retail", "13_price_history_cbsl_markets.png", start_y=0, end="View as a table", tags=None)
crop("history_dcs", "14_price_history_dcs.png", start_y=0, end="View as a table", tags=None)
crop("planner_input", "15_planner_input.png", "1. Your crop", None)
crop("planner_within_reach", "17_planner_price_comparison.png", "2. Your cultivation plan", "Price type", tags=None)
crop("planner_beyond_reach", "18_planner_seasonal_reference.png", "2. Your cultivation plan", "Price type", tags=None)
crop("planner_no_price", "19_planner_no_price.png", "2. Your cultivation plan", end_extra=900)
crop("history_si", "20_price_history_si.png", start_y=0, end=tr("si", "Weather"))
crop("planner_ta", "21_planner_ta.png", tr("ta", "2. Your cultivation plan"), tr("ta", "Price type"), tags=None)
crop("performance_ta", "22_performance_ta.png", tr("ta", "Ridge direct forecaster (used for real data on the Predict page)"), end=tr("ta", "Next-day comparison on the same rows"))
Image.open(SHOTS / "full" / "home_si_phone.png").save(OUT / "23_home_si_phone.png", optimize=True)
print("23_home_si_phone.png", Image.open(OUT / "23_home_si_phone.png").size)

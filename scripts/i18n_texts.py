"""Lists every English UI text the app translates, and checks the translation files against it.

The texts come from three places:
1. every t("...") / tn(..., "...", "...") literal in app/ (read from the source code);
2. the constant tables whose values are shown through t() (crop timings, page names, data
   sources, freshness labels, months, units...);
3. texts only known at run time, recorded by app.i18n while every page is run through
   Streamlit's test harness (--run-pages).

Usage:
    .venv312/Scripts/python.exe scripts/i18n_texts.py                 # report missing translations
    .venv312/Scripts/python.exe scripts/i18n_texts.py --run-pages     # also run every page (slower)
    .venv312/Scripts/python.exe scripts/i18n_texts.py --write-en      # rewrite app/translations/en.json
"""

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRANSLATIONS = ROOT / "app" / "translations"


def source_texts(folder: Path = ROOT / "app") -> tuple[set[str], list[str]]:
    """t()/tn() literals in the code, and any call that passes an f-string (which can't be translated)."""
    texts, problems = set(), []
    for path in sorted(folder.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in ("t", "tn"):
                continue
            args = node.args[:1] if node.func.id == "t" else node.args[1:3]
            for arg in args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    texts.add(arg.value)
                elif isinstance(arg, ast.JoinedStr):
                    problems.append(f"{path.relative_to(ROOT)}:{node.lineno}: an f-string passed to {node.func.id}()")
    return texts, problems


def constant_texts() -> set[str]:
    """Values of the tables the app shows through t(), so a text not reached in a test run is still listed."""
    from app import ui
    from app.catalogue import DATA_SOURCES, FRESHNESS_LABELS, HARTI_SERIES, NO_DURATION_REASONS
    from app.cultivation import EVENTS, PRICE_KINDS, START_STAGES, UNITS, load_crop_timings
    from app.i18n import MONTHS

    texts = set(FRESHNESS_LABELS) | set(MONTHS) | set(PRICE_KINDS.values()) | set(NO_DURATION_REASONS.values())
    texts |= {label for _, label, _ in ui.PAGES} | {name for name, _ in ui._SOURCE_FILES}
    for source in DATA_SOURCES:
        texts |= {source.name, source.category, source.location, source.schedule}
    for stage, (label, note) in START_STAGES.items():
        texts |= {stage, label, note}
    texts |= set(EVENTS) | set(UNITS) | {unit.rstrip("s") for unit in UNITS}
    for timing in load_crop_timings(ROOT / "data" / "reference" / "doa_harvest_durations_v2.csv"):
        texts |= {timing.scope}
        if "guidance" in timing.variety.lower() or timing.variety.lower().startswith(("local", "recommended")):
            texts.add(timing.variety)  # generic labels; released variety names are proper names, not translated
    # Crops the DOA's name list doesn't cover get their local name from the translation files.
    texts |= {name for name, (_, crop, _) in HARTI_SERIES.items() if crop is None}
    return texts


def page_texts() -> set[str]:
    """Run every page (and the planner and price history in several states) and record what t() saw."""
    import os

    import pandas as pd
    from streamlit.testing.v1 import AppTest

    from app import i18n
    from app.catalogue import HARTI_SERIES, crop_label, display_name
    from app.cultivation import load_crop_timings

    os.chdir(ROOT)
    i18n.recorded_texts = set()
    app = ROOT / "app"

    def run(page: str) -> AppTest:
        at = AppTest.from_file(str(app / page), default_timeout=300).run()
        assert not at.exception, (page, at.exception)
        return at

    for page in ("Home.py", "pages/2_Model_Performance.py"):
        run(page)

    history = run("pages/3_Price_History.py")
    for source in history.radio[0].options:
        at = run("pages/3_Price_History.py")
        at.radio[0].set_value(source)
        at.run()

    predict = run("pages/1_Predict.py")
    for label in ("7 days", "Another date (up to {days} days)"):
        at = run("pages/1_Predict.py")
        next(r for r in at.radio if r.label == "Forecast period").set_value(label)
        at.run()
        next(b for b in at.button if b.label == "Predict").click()
        at.run()
    next(r for r in predict.radio if r.label == "Data source").set_value("Upload CSV")
    predict.run()

    plans = [
        ("Beans", "Sanjaya", "2026-10-15", None),
        ("Beans", "Sanjaya", "2026-09-01", None),
        ("Beans", "Lanka Butter", "2026-09-25", None),
        ("Carrot", None, None, None),
        ("Manioc", None, "2026-02-20", None),
        ("Cauliflower", None, None, None),
        ("Cabbage", None, None, "Cabbage (Nuwara Eliya)"),
        ("Potato", None, None, None),
        ("Pumpkin", "A.N.K.", None, None),
        ("Leeks", None, None, None),
        ("Capsicum", None, None, None),
    ]
    for crop, variety, start, series in plans:
        at = run("pages/4_Cultivation_Planner.py")
        at.selectbox(key="planner_crop").set_value(crop)
        at.run()
        if variety:
            at.selectbox(key=f"planner_variety_{crop}").set_value(variety)
            at.run()
        if series:
            at.selectbox(key=f"planner_series_{crop}").set_value(series)
            at.run()
        if start:
            at.date_input[0].set_value(pd.Timestamp(start).date())
        next(b for b in at.button if b.label == "Generate Cultivation Plan").click()
        at.run()
        assert not at.exception, (crop, at.exception)

    # Every vegetable and crop name, in a language that looks up local names.
    i18n.set_language("si")
    for series in HARTI_SERIES:
        display_name(series)
    for timing in load_crop_timings(ROOT / "data" / "reference" / "doa_harvest_durations_v2.csv"):
        crop_label(timing.crop)
    i18n.set_language("en")
    texts, i18n.recorded_texts = i18n.recorded_texts, None
    return texts


def all_texts(run_pages: bool = False) -> set[str]:
    texts, _ = source_texts()
    texts |= constant_texts()
    if run_pages:
        texts |= page_texts()
    return {text for text in texts if text and text.strip()}


def load(code: str) -> dict[str, str]:
    path = TRANSLATIONS / f"{code}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-pages", action="store_true")
    parser.add_argument("--write-en", action="store_true")
    parser.add_argument("--dump", help="write the full list of English texts to this JSON file")
    args = parser.parse_args(argv)
    texts = all_texts(args.run_pages)
    _, problems = source_texts()
    for problem in problems:
        print("PROBLEM:", problem)
    if args.dump:
        Path(args.dump).write_text(json.dumps(sorted(texts), ensure_ascii=False, indent=1), encoding="utf-8")
    if args.write_en:
        TRANSLATIONS.mkdir(exist_ok=True)
        (TRANSLATIONS / "en.json").write_text(json.dumps({text: text for text in sorted(texts)}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(texts)} English texts")
    for code in ("si", "ta"):
        missing = sorted(texts - set(load(code)))
        print(f"{code}: {len(missing)} missing")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

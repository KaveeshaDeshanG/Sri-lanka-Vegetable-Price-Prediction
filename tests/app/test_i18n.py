"""Language support (app/i18n.py): complete, consistent translation files, and every page in all three languages.

The translation files are checked against the texts the code actually passes to t(); every page is run
in Sinhala and Tamil through Streamlit's test harness; and prices and forecasts are checked to be the
same in every language.
"""

import ast
import json
import re
import string
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import i18n
from app.i18n import LANGUAGES, format_date, set_language, t
from scripts.i18n_texts import constant_texts, source_texts

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
FILES = {code: json.loads((APP / "translations" / f"{code}.json").read_text(encoding="utf-8")) for code in LANGUAGES}
PAGES = ["Home.py", "pages/1_Predict.py", "pages/2_Model_Performance.py", "pages/3_Price_History.py", "pages/4_Cultivation_Planner.py"]


@pytest.fixture(autouse=True)
def english_afterwards():
    yield
    set_language("en")


def _placeholders(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}


# --- the translation files -------------------------------------------------------------------------


def test_the_three_files_have_exactly_the_same_texts():
    assert set(FILES["en"]) == set(FILES["si"]) == set(FILES["ta"])
    assert all(key == value for key, value in FILES["en"].items())  # English is the key itself
    assert len(FILES["en"]) > 800


@pytest.mark.parametrize("code", ["si", "ta"])
def test_every_translation_is_present_and_keeps_its_placeholders(code):
    script = {"si": re.compile("[඀-෿]"), "ta": re.compile("[஀-௿]")}[code]
    for english, translated in FILES[code].items():
        assert translated.strip(), english
        assert _placeholders(translated) == _placeholders(english), (code, english)
        if english != translated:
            assert script.search(translated), (code, english)  # really in Sinhala / Tamil script


def test_every_text_the_code_translates_has_a_translation():
    texts, problems = source_texts()
    assert not problems  # an f-string passed to t() could never be translated
    missing = sorted((texts | constant_texts()) - set(FILES["si"]))
    assert not missing, missing[:10]


_DISPLAY_CALLS = {
    "header", "subheader", "caption", "info", "warning", "error", "success", "button", "expander", "metric",
    "selectbox", "radio", "date_input", "checkbox", "spinner", "file_uploader", "form_submit_button", "page_link",
}


def test_no_page_shows_an_untranslated_literal():
    """A display call's text must go through t(): a plain or f-string literal would stay English."""
    offenders = []
    for path in [APP / "Home.py", APP / "ui.py", *sorted((APP / "pages").glob("*.py"))]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _DISPLAY_CALLS):
                continue
            # page_link's first argument is a file path, not text.
            texts = ([] if node.func.attr == "page_link" else list(node.args[:1])) + [kw.value for kw in node.keywords if kw.arg in ("label", "help")]
            for arg in texts:
                if isinstance(arg, ast.JoinedStr):  # an f-string may only join translated parts with punctuation or markup
                    fixed = "".join(part.value for part in arg.values if isinstance(part, ast.Constant))
                elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    fixed = arg.value
                else:
                    continue
                if re.search("[A-Za-z]{2,}", fixed):
                    offenders.append(f"{path.name}:{node.lineno} {node.func.attr}")
    assert not offenders, offenders


# --- t() itself --------------------------------------------------------------------------------------


def test_english_is_the_default_and_unknown_languages_fall_back_to_it():
    assert i18n.current_language() == "en" and t("Market") == "Market"
    assert set_language("fr") == "en" and t("Market") == "Market"


def test_translations_fill_in_values_without_changing_them():
    set_language("si")
    assert t("Market") == "වෙළඳපොළ"
    assert t("Rs. {low:,.2f} to {high:,.2f}", low=1234.5, high=2000) == "Rs. 1,234.50 සිට 2,000.00 දක්වා"
    set_language("ta")
    assert t("{n} of {total}", n=3, total=5) == "5 இல் 3"


def test_a_text_with_no_translation_falls_back_to_english():
    set_language("ta")
    assert t("A sentence nobody translated, {n}", n=2) == "A sentence nobody translated, 2"


def test_dates_keep_their_numbers_and_translate_only_the_month():
    day = pd.Timestamp("2026-11-17")
    assert format_date(day) == "17 Nov 2026"  # unchanged English, as the pages showed before
    set_language("si")
    assert format_date(day) == "17 නොවැ 2026"
    set_language("ta")
    assert format_date(day) == "17 நவ 2026"


def test_vegetable_names_keep_english_and_add_the_local_name():
    from app.catalogue import display_name, localized

    assert display_name("Beans") == "Beans" and localized("Brinjal") == "Brinjal"  # English unchanged
    set_language("si")
    assert display_name("Beans") == "Beans · බෝංචි"
    assert display_name("Green Chillies") == "Green Chillies · අමු මිරිස්"  # from the translation file
    set_language("ta")
    assert display_name("Carrot") == "Carrot · கரட்"


# --- every page in Sinhala and Tamil -------------------------------------------------------------------


def _page(page: str, code: str) -> AppTest:
    at = AppTest.from_file(str(APP / page), default_timeout=300)
    at.session_state["lang"] = code
    at.run()
    assert not at.exception, (page, code, at.exception)
    return at


def _text(at) -> str:
    return " ".join(
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [h.value for h in at.header]
        + [h.value for h in at.subheader]
        + [e.label for e in at.expander]
    )


_HEADINGS = {
    "Home.py": "Sri Lankan Vegetable Price Prediction",
    "pages/1_Predict.py": "1. Official source status",
    "pages/2_Model_Performance.py": "Model performance",
    "pages/3_Price_History.py": "Price history",
    "pages/4_Cultivation_Planner.py": "1. Your crop",
}


@pytest.mark.parametrize("code", ["si", "ta"])
@pytest.mark.parametrize("page", PAGES)
def test_every_page_runs_in_each_language(page, code):
    at = _page(page, code)
    text = _text(at)
    heading = _HEADINGS[page]
    assert FILES[code][heading] in text and heading not in text
    assert FILES[code]["Data freshness"] in text  # the freshness panel follows the language too


def test_the_language_comes_from_the_url_and_is_kept_in_the_session():
    # (A page whose radios show translated labels can't be rerun by the test harness, which looks options up by
    # label; browsers send the option's position, so they're unaffected. Model Performance has no such radio.)
    at = AppTest.from_file(str(APP / "pages/2_Model_Performance.py"), default_timeout=300)
    at.query_params["lang"] = "ta"
    at.run()
    assert at.session_state["lang"] == "ta" and FILES["ta"]["Model performance"] in _text(at)
    at.run()  # a rerun keeps it
    assert at.session_state["lang"] == "ta" and FILES["ta"]["Model performance"] in _text(at)


def test_the_language_survives_page_navigation():
    at = _page("Home.py", "si")
    at.switch_page("pages/4_Cultivation_Planner.py").run()
    assert not at.exception
    assert at.session_state["lang"] == "si" and FILES["si"]["1. Your crop"] in _text(at)


def test_choosing_a_language_in_the_switcher_applies_it():
    at = _page("pages/2_Model_Performance.py", "en")
    switcher = at.button_group[0]
    assert list(switcher.options) == ["English", "සිංහල", "தமிழ்"] or len(switcher.options) == 3
    switcher.set_value("ta").run()
    assert at.session_state["lang"] == "ta" and FILES["ta"]["Model performance"] in _text(at)
    at.button_group[0].set_value("en").run()
    assert at.session_state["lang"] == "en" and "Model performance" in _text(at)


def _numbers(text: str) -> list[str]:
    return re.findall(r"Rs\.? ?[\d,]+\.\d\d", text)


def _plan(code: str) -> AppTest:
    # The harness formats widget options on this thread, so it needs the same language as the page run.
    set_language(code)
    at = AppTest.from_file(str(APP / "pages/4_Cultivation_Planner.py"), default_timeout=300)
    at.session_state["lang"] = code
    at.session_state["planner_crop"] = "Carrot"  # chosen up front: the harness can't set a selectbox by a translated label
    at.run()
    latest = pd.read_csv(ROOT / "data/raw/harti_prices.csv", usecols=["date", "vegetable"]).query("vegetable == 'Carrot'")["date"].max()
    at.date_input[0].set_value((pd.Timestamp(latest) + pd.Timedelta(days=7 - 110)).date())  # harvest 7 days after the latest price
    next(b for b in at.button if b.label == ("Generate Cultivation Plan" if code == "en" else FILES[code]["Generate Cultivation Plan"])).click()
    at.run()
    assert not at.exception, at.exception
    return at


def test_planner_prices_and_comparison_are_identical_in_every_language():
    comparisons = {}
    for code in LANGUAGES:
        html = next(m.value for m in _plan(code).markdown if m.value.startswith('<div class="vp-cmp"'))
        comparisons[code] = _numbers(html)
    assert comparisons["en"] and comparisons["en"] == comparisons["si"] == comparisons["ta"]

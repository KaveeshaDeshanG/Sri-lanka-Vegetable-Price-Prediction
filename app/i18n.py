"""Language support for the whole app: English (default), Sinhala and Tamil.

One centralised mechanism, used by every page and shared module:

- t(text, **values) returns `text` in the current language. The key is the English text itself
  (the gettext convention), so English output is exactly what the code says, and a missing
  translation falls back to English instead of showing a key. Values are filled in with
  str.format after translating, so numbers, dates and prices are never translated or altered:
      t("Latest data {date} ({age} day(s) old).", date="2026-11-17", age=0)
- Translations live in app/translations/si.json and ta.json ({English text: translation});
  en.json lists every key. tests/app/test_i18n.py checks that all three hold the same keys, that
  each translation keeps the same {placeholders}, and that every t("...") in the code has one.
- The current language is per script run (set by init_language() at the top of every page), so
  modules without Streamlit (app/cultivation.py, app/catalogue.py) translate with the same t().
- Data, model inputs, source names, file names and internal identifiers are never translated.
"""

import json
import threading
from functools import cache
from pathlib import Path

LANGUAGES = {"en": "English", "si": "සිංහල", "ta": "தமிழ்"}
DEFAULT_LANGUAGE = "en"
TRANSLATIONS_DIR = Path(__file__).resolve().parent / "translations"
SESSION_KEY = "lang"  # the chosen language, kept in st.session_state and in the URL (?lang=si)
_WIDGET_KEY = "lang_selector"
_state = threading.local()
# When a set, every text passed to t() is added to it. Tests use it to find texts that are only
# known at run time (option values, crop timings, column headers) and check each has a translation.
recorded_texts: set[str] | None = None


def set_language(code: str | None) -> str:
    """Use `code` for this script run (an unknown code means English). Returns the code used."""
    _state.code = code if code in LANGUAGES else DEFAULT_LANGUAGE
    return _state.code


def current_language() -> str:
    return getattr(_state, "code", DEFAULT_LANGUAGE)


@cache
def load_translations(code: str) -> dict[str, str]:
    """{English text: translation} for a language; empty for English or a missing file."""
    if code == DEFAULT_LANGUAGE:
        return {}
    path = TRANSLATIONS_DIR / f"{code}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def t(text: str, /, **values) -> str:
    """`text` in the current language, with {placeholders} filled from `values`."""
    if recorded_texts is not None:
        recorded_texts.add(text)
    code = current_language()
    template = (load_translations(code).get(text) or text) if code != DEFAULT_LANGUAGE else text
    if not values:
        return template
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):  # a broken translation never hides the information
        return text.format(**values)


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def format_date(day) -> str:
    """A date as "17 Nov 2026", with the month in the current language. Day and year stay as numbers."""
    return f"{day.day} {t(MONTHS[day.month - 1])} {day.year}"


def tn(count: int, one: str, many: str, /, **values) -> str:
    """Singular or plural English text, translated: tn(n, "{n} record", "{n} records", n=n)."""
    return t(one if count == 1 else many, **values)


# --- Streamlit: remembering and switching the language ---------------------------------------------


def init_language() -> str:
    """Set this run's language from the session, else from the URL (?lang=si), else English.

    Called at the top of every page (ui.apply_theme does it). The choice is kept in
    st.session_state, which survives reruns and page navigation, and written back to the URL so a
    reload or a bookmark keeps it too.
    """
    import streamlit as st

    if SESSION_KEY not in st.session_state:
        requested = st.query_params.get(SESSION_KEY)
        st.session_state[SESSION_KEY] = requested if requested in LANGUAGES else DEFAULT_LANGUAGE
    code = set_language(st.session_state[SESSION_KEY])
    if st.query_params.get(SESSION_KEY) != code:
        st.query_params[SESSION_KEY] = code
    return code


def _on_change() -> None:
    import streamlit as st

    choice = st.session_state.get(_WIDGET_KEY)
    if choice in LANGUAGES:
        st.session_state[SESSION_KEY] = choice
        st.query_params[SESSION_KEY] = choice
        set_language(choice)


def language_selector() -> None:
    """English / සිංහල / தமிழ், as three buttons. Applies at once; no restart."""
    import streamlit as st

    # The widget's own state is cleared when the user changes page, so the choice is kept in
    # SESSION_KEY and the widget is re-seeded from it on every run.
    st.session_state[_WIDGET_KEY] = current_language()
    st.segmented_control(
        "Language / භාෂාව / மொழி",
        options=list(LANGUAGES),
        format_func=LANGUAGES.get,
        key=_WIDGET_KEY,
        on_change=_on_change,
        label_visibility="collapsed",
        help="English · සිංහල · தமிழ்",
    )

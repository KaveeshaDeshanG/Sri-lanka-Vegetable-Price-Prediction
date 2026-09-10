"""Unit tests for src.data_collection.harti_live_client.HartiLiveClient.

Network calls are mocked except for the two "real structure" tests at the bottom,
which parse tests/fixtures/harti_sample_bulletin.pdf (one real page extracted from an
actual HARTI bulletin) through pymupdf's real find_tables() -- proving the parsing
logic works against genuine PDF table structure, not just hand-built fixtures.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data_collection.harti_live_client import (
    DEFAULT_ARCHIVE_URL,
    HartiLiveClient,
    HartiLiveDataError,
)

FIXTURE_PDF_PATH = "tests/fixtures/harti_sample_bulletin.pdf"


def _client(**overrides) -> HartiLiveClient:
    kwargs = {"archive_url": DEFAULT_ARCHIVE_URL, "timeout_seconds": 5.0, "target_market": "Dambulla"}
    kwargs.update(overrides)
    return HartiLiveClient(**kwargs)


def _archive_html(rows=(("2026-08-18", "English", "2026.08.18"), ("2026-08-17", "English", "2026.08.17"))) -> str:
    trs = "".join(
        f'<tr><td>{date}</td><td>{medium}</td><td>'
        f'<a href="assets/pdf/food_price/daily/eng/2026/August/Vegetable Pricenew ex1({tag}).pdf" '
        f'target="_blank"><img src="assets/img/common/pdf.jpg"></a></td></tr>'
        for date, medium, tag in rows
    )
    return f"""
    <html><body>
    <table class="table table-bordered">
      <thead><tr><th>Date</th><th>Medium</th><th>Download</th></tr></thead>
      <tbody>{trs}</tbody>
    </table>
    </body></html>
    """


def _fake_response(status_code=200, text="", content=b"") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.content = content or text.encode("utf-8")
    return response


def _table(rows, dambulla_date="2026.08.18", other_date="2026.08.17"):
    """Hand-built table_data mimicking pymupdf Table.extract()'s shape:
    row 0 = per-column reporting date, row 1 = per-column market label, rows 2+ = data."""
    date_row = ["Variety", other_date, other_date, dambulla_date, other_date]
    market_row = [None, "Peliyagoda\nMarket", "Kandy\nMarket", "Dambulla\nMarket", "Meegoda\nMarket"]
    return [date_row, market_row, *rows]


# 1. _parse_price_cell -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("150 - 180", 165.0),
        ("300-450", 375.0),
        ("  60 -  70 ", 65.0),
        ("225.50 - 230.00", 227.75),
        ("175", 175.0),
        ("-", None),
        ("", None),
        (None, None),
        ("   ", None),
        ("N/A", None),
    ],
)
def test_parse_price_cell(raw, expected):
    assert HartiLiveClient._parse_price_cell(raw) == expected


# 2. _normalize_label / _normalize_market_header ---------------------------------


def test_normalize_label_collapses_whitespace_and_newlines_and_lowercases():
    assert HartiLiveClient._normalize_label(" Beet   root (N Eliya)\n") == "beet root (n eliya)"


def test_normalize_market_header_strips_trailing_market_word():
    assert HartiLiveClient._normalize_market_header("Dambulla\nMarket") == "dambulla"
    assert HartiLiveClient._normalize_market_header("Dambulla") == "dambulla"


# 3. _parse_table_rows (pure -- no PDF/network involved) -------------------------


def test_parse_table_rows_selects_correct_market_column_and_maps_vegetable_names():
    table = _table(
        [
            ["Carrot", "150 - 180", "200 - 220", "150 - 200", "190 - 230"],
            ["Beet root (N Eliya)", "-", "240 - 250", "230 - 250", "-"],
        ]
    )
    df = _client()._parse_table_rows(table)

    assert set(df["vegetable"]) == {"Carrot", "Beetroot (Nuwara Eliya)"}
    carrot_row = df[df["vegetable"] == "Carrot"].iloc[0]
    assert carrot_row["price"] == 175.0  # midpoint of Dambulla's "150 - 200"
    assert carrot_row["market"] == "Dambulla"
    assert carrot_row["source"] == "HARTI_LIVE"
    assert carrot_row["date"] == pd.Timestamp("2026-08-18")


def test_parse_table_rows_uses_the_target_columns_own_reported_date():
    """Dambulla's column can carry a different reporting date than other markets --
    the parsed date must come from Dambulla's own column, not the bulletin's cover date."""
    table = _table([["Carrot", "150 - 180", "200 - 220", "150 - 200", "190 - 230"]], dambulla_date="2026.08.16")
    df = _client()._parse_table_rows(table)
    assert df.iloc[0]["date"] == pd.Timestamp("2026-08-16")


def test_parse_table_rows_skips_unknown_vegetable_labels():
    table = _table(
        [
            ["Carrot", "150 - 180", "200 - 220", "150 - 200", "190 - 230"],
            ["Up Country Vegetable", "", "", "", ""],  # a section header, not a real vegetable
            ["Papaya (Rs/Kg)", "50 - 60", "50 - 60", "50 - 60", "50 - 60"],  # fruit, not tracked
        ]
    )
    df = _client()._parse_table_rows(table)
    assert list(df["vegetable"]) == ["Carrot"]


def test_parse_table_rows_raises_when_target_market_column_is_entirely_missing():
    date_row = ["Variety", "2026.08.18", "2026.08.18"]
    market_row = [None, "Peliyagoda\nMarket", "Kandy\nMarket"]  # no Dambulla column at all
    table = [date_row, market_row, ["Carrot", "150 - 180", "200 - 220"]]

    with pytest.raises(HartiLiveDataError, match="No 'Dambulla' market column"):
        _client()._parse_table_rows(table)


def test_parse_table_rows_raises_when_no_vegetable_has_a_reading_for_target_market():
    table = _table([["Cabbage (N'Eliya)", "140 - 150", "-", "-", "120 - 140"]])  # dash for Dambulla
    with pytest.raises(HartiLiveDataError, match="no usable"):
        _client()._parse_table_rows(table)


def test_parse_table_rows_raises_on_unparseable_column_date():
    table = _table(
        [["Carrot", "150 - 180", "200 - 220", "150 - 200", "190 - 230"]], dambulla_date="not-a-date"
    )
    with pytest.raises(HartiLiveDataError, match="Could not parse a reporting date"):
        _client()._parse_table_rows(table)


# 4. _validate --------------------------------------------------------------------


def test_validate_raises_on_duplicate_vegetable_rows():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-18")] * 2,
            "vegetable": ["Carrot", "Carrot"],
            "market": ["Dambulla", "Dambulla"],
            "price": [175.0, 180.0],
            "source": ["HARTI_LIVE", "HARTI_LIVE"],
        }
    )
    with pytest.raises(HartiLiveDataError, match="duplicate vegetable"):
        _client()._validate(df)


def test_validate_raises_on_non_positive_price():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-18")],
            "vegetable": ["Carrot"],
            "market": ["Dambulla"],
            "price": [0.0],
            "source": ["HARTI_LIVE"],
        }
    )
    with pytest.raises(HartiLiveDataError, match="failed validation"):
        _client()._validate(df)


# 5. _fetch_archive_entries ---------------------------------------------------------


@patch("src.data_collection.harti_live_client.requests.get")
def test_fetch_archive_entries_parses_dates_medium_and_resolves_relative_urls(mock_get):
    mock_get.return_value = _fake_response(text=_archive_html())
    entries = _client()._fetch_archive_entries()

    assert len(entries) == 2
    assert entries[0].date == pd.Timestamp("2026-08-18")
    assert entries[0].medium == "English"
    assert entries[0].pdf_url == (
        "https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2026/August/"
        "Vegetable Pricenew ex1(2026.08.18).pdf"
    )


@patch("src.data_collection.harti_live_client.requests.get")
def test_fetch_archive_entries_raises_when_no_table_found(mock_get):
    mock_get.return_value = _fake_response(text="<html><body>no bulletins here</body></html>")
    with pytest.raises(HartiLiveDataError, match="no bulletin table found"):
        _client()._fetch_archive_entries()


@patch("src.data_collection.harti_live_client.requests.get")
def test_fetch_archive_entries_raises_on_http_error(mock_get):
    mock_get.return_value = _fake_response(status_code=500)
    with pytest.raises(HartiLiveDataError, match="HTTP 500"):
        _client()._fetch_archive_entries()


@patch("src.data_collection.harti_live_client.requests.get")
def test_fetch_archive_entries_raises_on_network_failure(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("no route to host")
    with pytest.raises(HartiLiveDataError, match="request failed"):
        _client()._fetch_archive_entries()


@patch("src.data_collection.harti_live_client.requests.get")
def test_fetch_archive_entries_raises_on_timeout(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout("timed out")
    with pytest.raises(HartiLiveDataError, match="timed out"):
        _client()._fetch_archive_entries()


# 6. _download_pdf ------------------------------------------------------------------


@patch("src.data_collection.harti_live_client.requests.get")
def test_download_pdf_raises_on_empty_content(mock_get):
    mock_get.return_value = _fake_response(content=b"")
    with pytest.raises(HartiLiveDataError, match="empty"):
        _client()._download_pdf("https://www.harti.gov.lk/some.pdf")


@patch("src.data_collection.harti_live_client.requests.get")
def test_download_pdf_raises_on_http_error(mock_get):
    mock_get.return_value = _fake_response(status_code=404, content=b"x")
    with pytest.raises(HartiLiveDataError, match="HTTP 404"):
        _client()._download_pdf("https://www.harti.gov.lk/missing.pdf")


# 7. fetch_latest_prices() / fetch_prices_for_date() -- full flow, PDF parsing mocked out --


@patch("src.data_collection.harti_live_client.HartiLiveClient._extract_vegetable_table")
@patch("src.data_collection.harti_live_client.requests.get")
def test_fetch_latest_prices_picks_the_most_recent_english_entry(mock_get, mock_extract):
    mock_get.side_effect = [
        _fake_response(text=_archive_html()),  # archive index
        _fake_response(content=b"%PDF-fake"),  # pdf bytes (unused, extraction mocked)
    ]
    mock_extract.return_value = _table(
        [["Carrot", "150 - 180", "200 - 220", "150 - 200", "190 - 230"]]
    )

    snapshot = _client().fetch_latest_prices()

    assert snapshot.bulletin_date == pd.Timestamp("2026-08-18")
    assert snapshot.source_url.endswith("(2026.08.18).pdf")
    assert list(snapshot.data["vegetable"]) == ["Carrot"]
    assert isinstance(snapshot.retrieved_at, pd.Timestamp)


@patch("src.data_collection.harti_live_client.HartiLiveClient._extract_vegetable_table")
@patch("src.data_collection.harti_live_client.requests.get")
def test_fetch_prices_for_date_selects_the_exact_requested_date(mock_get, mock_extract):
    mock_get.side_effect = [
        _fake_response(text=_archive_html()),
        _fake_response(content=b"%PDF-fake"),
    ]
    mock_extract.return_value = _table(
        [["Carrot", "150 - 180", "200 - 220", "150 - 200", "190 - 230"]], dambulla_date="2026.08.17"
    )

    snapshot = _client().fetch_prices_for_date("2026-08-17")

    assert snapshot.source_url.endswith("(2026.08.17).pdf")


@patch("src.data_collection.harti_live_client.requests.get")
def test_fetch_prices_for_date_raises_when_no_bulletin_exists_for_that_date(mock_get):
    mock_get.return_value = _fake_response(text=_archive_html())
    with pytest.raises(HartiLiveDataError, match="No HARTI bulletin found for 2020-01-01"):
        _client().fetch_prices_for_date("2020-01-01")


@patch("src.data_collection.harti_live_client.requests.get")
def test_fetch_latest_prices_raises_when_no_english_entries_exist(mock_get):
    mock_get.return_value = _fake_response(
        text=_archive_html(rows=[("2026-08-18", "Sinhala", "2026.08.18")])
    )
    with pytest.raises(HartiLiveDataError, match="no English-language"):
        _client().fetch_latest_prices()


# 8. from_config --------------------------------------------------------------------


def test_from_config_reads_harti_section(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "harti:\n"
        '  archive_url: "https://example.test/daily-price.php"\n'
        "  request_timeout_seconds: 20\n"
        '  target_market: "Kandy"\n',
        encoding="utf-8",
    )
    client = HartiLiveClient.from_config(str(config_path))
    assert client._archive_url == "https://example.test/daily-price.php"
    assert client._timeout_seconds == 20
    assert client._target_market == "Kandy"


def test_from_config_falls_back_to_documented_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  name: x\n", encoding="utf-8")
    client = HartiLiveClient.from_config(str(config_path))
    assert client._archive_url == DEFAULT_ARCHIVE_URL
    assert client._target_market == "Dambulla"


# 9. Real PDF structure (integration-style, no network) ------------------------------


def test_extract_vegetable_table_finds_dambulla_column_in_a_real_bulletin_page():
    with open(FIXTURE_PDF_PATH, "rb") as file:
        pdf_bytes = file.read()

    table_data = _client()._extract_vegetable_table(pdf_bytes)
    df = _client()._parse_table_rows(table_data)

    assert len(df) > 20
    assert "Carrot" in set(df["vegetable"])
    assert (df["market"] == "Dambulla").all()
    assert (df["price"] > 0).all()


def test_extract_vegetable_table_raises_for_a_market_absent_from_the_real_bulletin():
    with open(FIXTURE_PDF_PATH, "rb") as file:
        pdf_bytes = file.read()

    with pytest.raises(HartiLiveDataError, match="no table with a 'Colombo Fort' market column"):
        _client(target_market="Colombo Fort")._extract_vegetable_table(pdf_bytes)


def test_extract_vegetable_table_raises_on_corrupt_pdf_bytes():
    with pytest.raises(HartiLiveDataError, match="Could not open"):
        _client()._extract_vegetable_table(b"this is not a pdf")

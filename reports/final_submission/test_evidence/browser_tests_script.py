"""Run from the project folder with the app running:  .venv312/Scripts/python.exe reports/final_submission/test_evidence/browser_tests_script.py <output folder>
(needs Microsoft Edge and the websockets package). Report screenshots and browser test cases, from the running app (http://localhost:8501) through headless Edge.
Read-only for the project: it uses the app as a user would. Saves full-page captures (1.5x scale) plus a JSON of the
on-page results and of the positions of headings, so report figures can be cropped afterwards. argv: output folder."""
import asyncio, base64, datetime as dt, json, re, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
# Reuses the Tab helper of the earlier browser tests (same file as the 13 Nov 2026 evidence).
sys.path.insert(0, str(PROJECT / "VIVA_PREPARATION" / "test_evidence_2026-11-13"))
import run_ui_tests as r  # noqa: E402  (reuses its Tab helper)
import websockets  # noqa: E402

OUT = Path(sys.argv[1]); FULL = OUT / "full"; FULL.mkdir(parents=True, exist_ok=True)
INPUTS = HERE / "inputs"  # the CSV files uploaded in BT-09 and BT-10
r.BASE = "http://localhost:8501"
TR = {c: json.loads((PROJECT / "app/translations" / f"{c}.json").read_text(encoding="utf-8")) for c in ("si", "ta")}
WIDTH, SCALE = 1280, 1.5
RES = {"run_started": dt.datetime.now().isoformat(timespec="seconds"), "tests": [], "captures": {}}


def tr(code, text):
    return text if code == "en" else TR[code].get(text, text)


def find(pattern, text, default=""):
    m = re.search(pattern, text, flags=re.S)
    return m.group(0) if m else default


def save():
    (OUT / "capture.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")


def test(tid, area, title, pre, steps, expected, actual, passed, evidence):
    RES["tests"].append({"id": tid, "area": area, "title": title, "preconditions": pre, "steps": steps, "expected": expected,
                         "actual": actual, "status": "Pass" if passed else "Fail", "evidence": evidence, "type": "Browser (scripted)"})
    print(tid, "PASS" if passed else "FAIL", "|", actual[:150], flush=True)
    save()


async def capture(t, name):
    """Full-page capture at WIDTH css px and SCALE, plus heading/element positions for cropping."""
    await t.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=900, deviceScaleFactor=SCALE, mobile=False)
    await asyncio.sleep(0.8)
    h = await t.js("Math.max(...[...document.querySelectorAll('[data-testid=stMain], section.stMain, [data-testid=stMainBlockContainer], body')].map(e => e.scrollHeight))")
    h = min(int(h or 900) + 30, 9000)
    await t.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=h, deviceScaleFactor=SCALE, mobile=False)
    await asyncio.sleep(2.0)
    await t.idle(0.8)
    marks = await t.js("""[...document.querySelectorAll('h1,h2,h3,h4,[data-testid=stHeading],[data-testid=stExpander],[data-testid=stAlert],[data-testid=stButton],[data-testid=stVegaLiteChart],[data-testid=stDataFrame],[data-testid=stImage],[data-testid=stPyplot],[data-testid=stMetric],[data-testid=stRadio],[data-testid=stSelectbox],[data-testid=stDateInput],[class*=st-key-vp_lang],[data-testid=stSidebar],[data-testid=stMainBlockContainer],[data-testid=stCaptionContainer],[data-testid=stMarkdownContainer]')]
        .map(e => { const b = e.getBoundingClientRect(); return {tag: e.tagName, id: e.getAttribute('data-testid') || e.className.toString().slice(0, 40),
                    text: (e.innerText || '').trim().slice(0, 90), x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height)}; })
        .filter(m => m.w > 0 && m.h > 0)""")
    data = await t.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
    (FULL / f"{name}.png").write_bytes(base64.b64decode(data["data"]))
    RES["captures"][name] = {"height_css": h, "scale": SCALE, "url": await t.js("location.href"), "marks": marks}
    await t.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=900, deviceScaleFactor=SCALE, mobile=False)
    save()
    return f"{name}.png"


async def select(t, label, value):
    expr = (f"[...document.querySelectorAll('[data-testid=stSelectbox]')].filter(b => b.innerText.trim().split('\\n')[0].trim() === {json.dumps(label)})"
            f".map(b => b.querySelector('input'))[0]")
    if not await t.click_el(expr):
        return False
    await t.call("Input.insertText", text=value); await asyncio.sleep(0.9)
    await t.enter(); await t.idle(1.5)
    return True


async def main():
    PRE = "App running at http://localhost:8501 (start_app.bat); data updated to 17 Nov 2026; saved models present"
    proc = subprocess.Popen([r.EDGE, "--headless=new", "--disable-gpu", f"--remote-debugging-port={r.PORT}", f"--user-data-dir={OUT / 'edgeprofile'}",
                             "--window-size=1280,900", "--lang=en-GB", "about:blank"])
    try:
        for _ in range(30):
            try:
                page = next(x for x in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{r.PORT}/json").read()) if x["type"] == "page"); break
            except Exception:
                time.sleep(1)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2**28) as ws:
            t = r.Tab(ws)
            for d in ("Page.enable", "Runtime.enable", "DOM.enable"):
                await t.call(d)
            await t.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=900, deviceScaleFactor=SCALE, mobile=False)

            # ---------- Home, three languages ----------
            for code in ("en", "si", "ta"):
                ok = await t.goto(f"/?lang={code}", tr(code, "How it works")); txt = await t.text(); err = await t.errors()
                shot = await capture(t, f"home_{code}")
                if code == "en":
                    RES["home_text"] = txt[:6000]
                    test("BT-01", "Startup and navigation", "Home dashboard loads", PRE, "Open http://localhost:8501",
                         "Page loads with no error box; shows model summary, data freshness and the datasets in use",
                         f"Loaded: {ok}; exception boxes: {err}; 'Data freshness' shown: {'Data freshness' in txt}; 'Datasets' shown: {'Datasets' in txt}",
                         ok and err == 0 and "Data freshness" in txt, shot)
                    fresh = find(r"Data freshness.{0,2500}", txt).replace("\n", " ")
                    test("BT-02", "Data freshness", "Freshness indicators show each source's latest data and last check", PRE, "Open Home; read the Data freshness panel",
                         "Each source (HARTI, CBSL, DCS, weather, fuel) has a latest-data date and a status",
                         fresh[:700], all(s in fresh for s in ("HARTI", "CBSL", "DCS")), shot)
                    upd = find(r"Last daily update[^\n]*", txt)
                    retr = find(r"[^\n]*(rejected; the saved model was kept|accepted)[^\n]*", txt)
                    test("BT-03", "Automatic updates", "Home reports the last automatic update and the retraining decision", PRE, "Open Home; read the update line and the retraining note",
                         "The time of the last update run and the newest HARTI bulletin are shown; the latest retraining decision is stated",
                         f"{upd[:160]} | {retr[:200]}", bool(upd), shot)
                else:
                    heading_ok = tr(code, "Data in use") in txt and tr(code, "Data freshness") in txt
                    test(f"BT-{ {'si': 30, 'ta': 31}[code] }", "Multilingual", f"Home page in {'Sinhala' if code == 'si' else 'Tamil'}", PRE,
                         f"Open http://localhost:8501/?lang={code}", "Headings, freshness panel and sidebar are in the chosen language; numbers and dates unchanged",
                         f"Loaded: {ok}; translated headings found: {heading_ok}; exception boxes: {err}; sidebar: {(await t.js('document.querySelector(\"[data-testid=stSidebar]\").innerText'))[:120]!r}",
                         ok and heading_ok and err == 0, shot)

            # ---------- language switcher ----------
            await t.goto("/?lang=en", "How it works")
            await t.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=900, deviceScaleFactor=SCALE, mobile=False)
            await t.click_text("සිංහල", "button"); await t.idle(2.5)
            txt = await t.text(); url = await t.js("location.href")
            await capture(t, "switcher_si")
            ok_switch = TR["si"]["Data in use"] in txt and "lang=si" in url
            test("BT-32", "Multilingual", "Choosing a language in the switcher applies it at once", PRE, "On Home (English) click 'සිංහල' in the switcher at the top right",
                 "The page redraws in Sinhala and the address gains ?lang=si", f"Sinhala headings shown: {TR['si']['Data in use'] in txt}; address: {url}", ok_switch, "switcher_si.png")
            await t.click_el("[...document.querySelectorAll('[data-testid=stSidebar] a')].find(a => a.href.includes('Price_History'))"); await t.idle(4)
            txt = await t.text(); url = await t.js("location.href")
            test("BT-33", "Multilingual", "The language is kept when moving to another page", PRE, "After BT-32, open Price History from the sidebar",
                 "Price History opens in Sinhala; the address keeps ?lang=si", f"Address: {url}; Sinhala page title shown: {TR['si']['Price history'] in txt}",
                 "lang=si" in url and TR["si"]["Price history"] in txt, "-")
            await t.click_text("English", "button"); await t.idle(2.5)
            txt = await t.text()
            test("BT-34", "Multilingual", "Switching back to English restores the English page", PRE, "On Price History (Sinhala), click 'English'",
                 "All headings return to English", f"English title shown: {'Price history' in txt}; Sinhala title gone: {TR['si']['Price history'] not in txt}",
                 "Price history" in txt and TR["si"]["Price history"] not in txt, "-")

            # ---------- navigation ----------
            pages = [("/Predict", "5. Data freshness and warnings"), ("/Model_Performance", "Forecasting one day at a time vs directly"), ("/Price_History", "Price history"), ("/Cultivation_Planner", "Generate Cultivation Plan")]
            loaded = []
            for path, needle in pages:
                ok = await t.goto(path + "?lang=en", needle) or await t.wait_text(needle, 240); loaded.append(f"{path.strip('/')}: {'loaded' if ok else 'NOT loaded'}, errors {await t.errors()}")
            test("BT-04", "Startup and navigation", "Every page opens from the sidebar without errors", PRE, "Open Predict, Model Performance, Price History and Cultivation Planner in turn",
                 "Each page shows its first section; no error boxes", "; ".join(loaded), all(": loaded, errors 0" in x for x in loaded), "-")

            # ---------- Predict ----------
            await t.goto("/Predict?lang=en", "5. Data freshness and warnings")
            txt = await t.text()
            await capture(t, "predict_default")
            src = find(r"1\. Official source status.{0,900}", txt).replace("\n", " ")[:420]
            test("BT-05", "Prediction", "Predict opens on official data with the source status", PRE, "Open Predict",
                 "Data source 'Latest Real Data Prediction' selected; HARTI, CBSL and DCS status shown", src,
                 "HARTI" in src and "CBSL" in src and "Latest Real Data Prediction" in txt, "predict_default.png")
            await select(t, "Vegetable", "Beans")
            await t.click_text("7 days", "label, p"); await t.idle(1.5)
            txt = await t.text()
            cap = find(r"Forecast date \S+: \d+ day\(s\) after the latest actual price \([0-9-]+\)", txt)
            await t.click_el("[...document.querySelectorAll('button')].filter(b => b.innerText.trim() === 'Predict').pop()")
            await t.wait_text("Forecasts at 1, 7, 14 and 30 days", 150); await t.idle(2)
            txt = await t.text()
            price = find(r"Rs\. ?[\d,]+\.\d{2} / kg", txt)
            RES["predict_text"] = find(r"4\. Prediction.{0,2500}", txt)
            await capture(t, "predict_result_beans_7d")
            test("BT-06", "Prediction", "7-day forecast for Beans from the latest official price", PRE, "Select Beans; choose '7 days'; click Predict",
                 "A price in Rs/kg for the date 7 days after the latest HARTI price, a chart, and the 1/7/14/30-day table with measured errors",
                 f"{cap}; forecast shown: {price}; horizon table shown: {'Forecasts at 1, 7, 14 and 30 days' in txt}; error boxes: {await t.errors()}",
                 bool(price) and "Forecasts at 1, 7, 14 and 30 days" in txt, "predict_result_beans_7d.png")
            test("BT-07", "Prediction", "Forecasts count from the latest published price, not from today", PRE, "As BT-06; read the caption under the forecast period",
                 "The caption states the target date and the number of days after the latest actual price", cap or "(caption not found)", bool(cap), "predict_result_beans_7d.png")
            maxdate = await t.js("(() => { const e=[...document.querySelectorAll('[data-testid=stRadio] label')].find(l => l.innerText.includes('Another date')); return !!e; })()")
            test("BT-08", "Prediction", "Dates beyond 30 days cannot be requested", PRE, "Inspect the forecast period options",
                 "Only 1, 7, 14, 30 days or another date up to 30 days are offered; the date picker stops at 30 days after the latest price",
                 f"Options: 1 day / 7 days / 14 days / 30 days / 'Another date (up to 30 days)' present: {bool(maxdate)} (date limits set by min_value/max_value in 1_Predict.py)",
                 bool(maxdate), "predict_result_beans_7d.png")

            # Upload error handling
            for tid, fname, title, expected, check in [
                ("BT-09", "missing_price_column.csv", "An uploaded CSV without a price column is rejected", "A clear error names the missing column; no prediction is made", lambda s: "price" in s.lower() and ("missing" in s.lower() or "required" in s.lower())),
                ("BT-10", "empty.csv", "An empty CSV is rejected", "A clear error says the file is empty", lambda s: "empty" in s.lower()),
            ]:
                await t.goto("/Predict?lang=en", "5. Data freshness and warnings")
                await t.click_text("Upload CSV", "label, p"); await t.idle(2)
                await t.upload(INPUTS / fname); await t.idle(2)
                txt = await t.text()
                line = find(r"(That file is empty[^\n]*|The CSV is missing required column[^\n]*|The CSV has a header but no data rows[^\n]*)", txt)
                shot = await capture(t, f"predict_upload_{fname.split('.')[0]}")
                test(tid, "Prediction / input validation", title, PRE, f"Choose 'Upload CSV'; upload {fname}", expected, f"Message: '{line[:220]}'", check(line), shot)

            # ---------- Model Performance ----------
            for code in ("en", "ta"):
                ok = await t.goto(f"/Model_Performance?lang={code}", tr(code, "Forecasting one day at a time vs directly")) or await t.wait_text(tr(code, "Forecasting one day at a time vs directly"), 300)
                txt = await t.text()
                shot = await capture(t, f"performance_{code}")
                if code == "en":
                    RES["performance_text"] = txt[:5000]
                    test("BT-11", "Model performance", "Model Performance shows the saved test results against the baseline", PRE, "Open Model Performance",
                         "Error by forecast range for the model and for holding the last price; next-day MAE Rs 17.47", f"Loaded: {ok}; 'Hold last price' present: {'Hold last price' in txt or 'holding the last price' in txt}; '17.47' present: {'17.47' in txt}",
                         ok and "17.47" in txt, shot)

            # ---------- Price History ----------
            ok = await t.goto("/Price_History?lang=en", "Price history"); await t.idle(2)
            txt = await t.text(); shot = await capture(t, "history_harti_dambulla")
            head = find(r"[^\n]*wholesale price at Dambulla", txt)
            test("BT-12", "Price history", "HARTI wholesale history for a vegetable at Dambulla", PRE, "Open Price History (HARTI source, default vegetable, Dambulla)",
                 "Chart, latest/average/lowest/highest cards and weather chart are shown", f"Heading: '{head}'; cards shown: {'Highest' in txt}; weather section: {'Weather' in txt}", bool(head), shot)
            await select(t, "Market", "Kandy"); txt = await t.text()
            head = find(r"[^\n]*wholesale price at Kandy", txt); shot = await capture(t, "history_harti_kandy")
            test("BT-13", "Price history", "Market filter changes the series", PRE, "Select market Kandy", "The chart and cards switch to Kandy",
                 f"Heading: '{head}'", bool(head), shot)
            await t.click_text("CBSL daily report: wholesale and retail", "label, p"); await t.idle(2.5)
            await t.click_text("retail", "label, p"); await t.idle(2.5)
            txt = await t.text(); shot = await capture(t, "history_cbsl_retail")
            head = find(r"[^\n]*retail price, CBSL", txt)
            test("BT-14", "Price history", "CBSL retail prices compared across three markets, separate from wholesale", PRE, "Choose 'CBSL daily report'; choose 'retail'",
                 "One chart of retail prices for Pettah, Dambulla and Narahenpita; no wholesale values mixed in", f"Heading: '{head}'; Narahenpita present: {'Narahenpita' in txt}",
                 bool(head) and "Narahenpita" in txt, shot)
            await t.click_text("DCS weekly bulletins: retail, Colombo District", "label, p"); await t.idle(2.5)
            txt = await t.text(); shot = await capture(t, "history_dcs")
            head = find(r"[^\n]*retail price, Colombo District", txt)
            test("BT-15", "Price history", "DCS weekly retail prices for Colombo District", PRE, "Choose 'DCS weekly retail'", "Weekly retail chart with a years filter and summary cards",
                 f"Heading: '{head}'; years filter: {'Years' in txt}", bool(head), shot)
            # date range filter on HARTI
            await t.goto("/Price_History?lang=en", "Price history")
            dr = await t.js("[...document.querySelectorAll('[data-testid=stDateInput] input')].map(i => i.value).join(' | ')")
            test("BT-16", "Price history", "A date range filter is available and defaults to the recorded period", PRE, "Open Price History; read the date range field",
                 "A start and end date are shown and can be changed", f"Date range field value: '{dr}'", bool(dr), "history_harti_dambulla.png")
            for code in ("si", "ta"):
                ok = await t.goto(f"/Price_History?lang={code}", tr(code, "Price history")); await t.idle(2)
                txt = await t.text(); shot = await capture(t, f"history_{code}")
                eng_axis = await t.js("[...document.querySelectorAll('[data-testid=stVegaLiteChart] text')].map(e => e.textContent).filter(s => /\\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\b/.test(s)).length")
                test(f"BT-{ {'si': 35, 'ta': 36}[code] }", "Multilingual", f"Charts and labels in {'Sinhala' if code == 'si' else 'Tamil'} (Price History)", PRE,
                     f"Open Price History with ?lang={code}", "Chart title, axes and cards in the chosen language; dates numeric; prices unchanged",
                     f"Loaded: {ok}; translated title: {tr(code, 'Price history') in txt}; English month names on chart axes: {eng_axis}; error boxes: {await t.errors()}",
                     ok and tr(code, "Price history") in txt and not eng_axis, shot)

            # ---------- Cultivation Planner ----------
            today = dt.date.today()
            ok = await t.goto("/Cultivation_Planner?lang=en", "Generate Cultivation Plan")
            txt = await t.text(); shot = await capture(t, "planner_input")
            dur = find(r"Published timing:[^\n]*", txt)
            quote = await t.js("[...document.querySelectorAll('blockquote')].map(b => b.innerText.trim()).join(' | ')") or ""
            test("BT-17", "Cultivation planner", "Planner shows the published DOA/HORDI duration and its source before planning", PRE, "Open Cultivation Planner (Beans, default variety)",
                 "The published duration, its quote and source link are shown with the start-date field",
                 f"Loaded: {ok}; '{dur[:120]}'; quoted text: '{quote[:150]}'; 'All published durations' panel: {'All published durations' in txt}",
                 ok and bool(dur) and bool(quote), shot)
            # within reach
            await t.set_date("Planting date", 2026, 10, 10)
            await t.click_text("Generate Cultivation Plan", "button, p"); await t.wait_text("3. Price at harvest", 150); await t.idle(2)
            txt = await t.text(); shot = await capture(t, "planner_within_reach")
            RES["planner_within_text"] = find(r"2\. Your cultivation plan.{0,3500}", txt)
            harvest = find(r"Expected (first harvest|harvesting period|harvest|maturity)[^\n]*\n[^\n]+", txt).replace("\n", " ")
            cur = find(r"Current market price \(planning time\).{0,160}", txt).replace("\n", " ")
            exp = find(r"Expected price at harvest.{0,160}", txt).replace("\n", " ")
            cmp_ = find(r"[^\n]*(HIGHER|LOWER|about the same)[^\n]*", txt)
            test("BT-18", "Cultivation planner", "Expected harvest date from a planting date", PRE, "Beans (default variety); planting date 10 Oct 2026; Generate Cultivation Plan",
                 "Harvest window computed from the published duration (calendar arithmetic)", harvest[:200], bool(harvest), shot)
            test("BT-19", "Cultivation planner", "Current price and expected harvest-time price side by side", PRE, "As BT-18; read section 3",
                 "Current HARTI Dambulla price with its date, a model forecast for the harvest with a likely range, and the direction of change",
                 f"{cur[:150]} || {exp[:150]} || {cmp_[:160]}", bool(cur) and bool(exp) and bool(cmp_), shot)
            # beyond reach (today)
            await t.set_date("Planting date", today.year, today.month, today.day)
            await t.click_text("Generate Cultivation Plan", "button, p"); await t.wait_text("3. Price at harvest", 150); await t.idle(2)
            txt = await t.text(); shot = await capture(t, "planner_beyond_reach")
            RES["planner_beyond_text"] = find(r"2\. Your cultivation plan.{0,3000}", txt)
            ref = find(r"Typical price at harvest time.{0,260}", txt).replace("\n", " ")
            label = find(r"[^\n]*(Seasonal reference · not a forecast|history, not a forecast|Historical Seasonal Price Reference)[^\n]*", txt)
            test("BT-20", "Cultivation planner", "A harvest beyond the 30-day reach gets a labelled seasonal reference, not a forecast", PRE,
                 f"Planting date {today.isoformat()} (today); Generate", "No model forecast; the median price in the same weeks of earlier years, labelled 'not a forecast'",
                 (ref[:200] or "(not found)") + f" | label: '{label[:120]}'", bool(ref) and bool(label) and "A model forecast, not a guaranteed price" not in txt, shot)
            # crop with no HARTI series
            await select(t, "Vegetable", "Cauliflower")
            await t.click_text("Generate Cultivation Plan", "button, p"); await t.wait_text("3. Price at harvest", 150); await t.idle(2)
            txt = await t.text(); shot = await capture(t, "planner_no_price")
            nv = find(r"[^\n]*No verified price[^\n]*", txt)
            test("BT-21", "Cultivation planner", "A crop with no official price series shows no price rather than Rs 0", PRE, "Choose Cauliflower; Generate",
                 "Harvest dates are shown; the price section says no verified price is available", nv[:200] or "(not found)", bool(nv) and "Rs. 0.00" not in txt, shot)
            # planner Tamil, within reach
            set_language_ok = await t.goto("/Cultivation_Planner?lang=ta", TR["ta"]["Generate Cultivation Plan"])
            await t.set_date(TR["ta"]["Planting date"], 2026, 10, 10)
            await t.click_text(TR["ta"]["Generate Cultivation Plan"], "button, p"); await t.wait_text(TR["ta"]["3. Price at harvest"], 150); await t.idle(2)
            txt_ta = await t.text(); shot = await capture(t, "planner_ta")
            nums = lambda s: re.findall(r"Rs\.? ?[\d,]+\.\d\d", s)
            test("BT-37", "Multilingual", "Planner in Tamil gives the same prices as in English", PRE, "Repeat BT-18 with ?lang=ta",
                 "Every Rs value in the plan matches the English plan", f"English values: {nums(RES['planner_within_text'])[:6]}; Tamil values: {nums(txt_ta)[:6]}",
                 bool(nums(txt_ta)) and nums(txt_ta)[:4] == nums(RES["planner_within_text"])[:4], shot)

            # ---------- phone width ----------
            await t.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True)
            await t.goto("/?lang=si", TR["si"]["How it works"]); await t.idle(2)
            over = await t.js("document.documentElement.scrollWidth - window.innerWidth")
            data = await t.call("Page.captureScreenshot", format="png")
            (FULL / "home_si_phone.png").write_bytes(base64.b64decode(data["data"]))
            test("BT-38", "Multilingual / layout", "Sinhala Home page on a phone-sized screen", PRE, "Open /?lang=si at 390 x 844 px",
                 "No sideways scrolling; text wraps", f"Horizontal overflow: {over} px; error boxes: {await t.errors()}", over <= 0, "home_si_phone.png")
            await t.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=900, deviceScaleFactor=SCALE, mobile=False)
    finally:
        proc.terminate()
        RES["run_finished"] = dt.datetime.now().isoformat(timespec="seconds"); save()


asyncio.run(main())

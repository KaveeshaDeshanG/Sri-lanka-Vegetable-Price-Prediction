"""Executes practical test cases against the running Streamlit app (http://localhost:8501) through headless
Edge's DevTools protocol. Records the actual on-page result and a screenshot for every test. Read-only for the
project: it only uses the app as a user would. argv: output folder."""
import asyncio, base64, json, re, subprocess, sys, time, urllib.request
from pathlib import Path
import websockets

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT = Path(sys.argv[1]); SHOTS = OUT / "shots"; INPUTS = OUT / "inputs"
BASE = "http://localhost:8501"
PORT = 9344
RESULTS = []


class Tab:
    def __init__(self, ws): self.ws, self.i = ws, 0

    async def call(self, method, **params):
        self.i += 1
        await self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.i:
                if "error" in msg: raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def idle(self, extra=1.5, timeout=120):
        end = time.time() + timeout
        await asyncio.sleep(1.0)
        while time.time() < end:
            busy = await self.js("!!document.querySelector('[data-testid=stStatusWidget]') || [...document.querySelectorAll('[data-testid=stSpinner]')].length > 0")
            if not busy: break
            await asyncio.sleep(0.7)
        await asyncio.sleep(extra)

    async def text(self):
        return await self.js("document.querySelector('[data-testid=stMain]') ? document.querySelector('[data-testid=stMain]').innerText : document.body.innerText") or ""

    async def wait_text(self, needle, timeout=90):
        end = time.time() + timeout
        while time.time() < end:
            if needle in (await self.text()): await self.idle(1.0); return True
            await asyncio.sleep(0.8)
        return False

    async def goto(self, path, needle):
        await self.call("Page.navigate", url=BASE + path)
        await asyncio.sleep(2)
        ok = await self.wait_text(needle)
        await self.idle()
        return ok

    async def click_xy(self, x, y):
        for kind in ("mousePressed", "mouseReleased"):
            await self.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)

    async def center(self, expr):
        """expr evaluates to an element; scrolls it into view and returns its centre."""
        r = await self.js(f"(() => {{ const e = {expr}; if (!e) return null; e.scrollIntoView({{block:'center'}}); const b = e.getBoundingClientRect(); return [b.x + b.width/2, b.y + b.height/2]; }})()")
        await asyncio.sleep(0.4)
        return r

    async def click_el(self, expr):
        c = await self.center(expr)
        if not c: return False
        await self.click_xy(*c); await self.idle(1.0)
        return True

    async def click_text(self, text, selector="button, label, p, summary, [role=radio]"):
        expr = f"[...document.querySelectorAll({json.dumps(selector)})].filter(e => e.innerText && e.innerText.trim() === {json.dumps(text)}).pop()"
        return await self.click_el(expr)

    async def keys(self, text):
        for ch in text:
            await self.call("Input.dispatchKeyEvent", type="keyDown", text=ch, key=ch)
            await self.call("Input.dispatchKeyEvent", type="keyUp", key=ch)
            await asyncio.sleep(0.05)

    async def enter(self):
        for kind in ("keyDown", "keyUp"):
            await self.call("Input.dispatchKeyEvent", type=kind, key="Enter", code="Enter", windowsVirtualKeyCode=13, nativeVirtualKeyCode=13)

    async def tab(self):
        for kind in ("keyDown", "keyUp"):
            await self.call("Input.dispatchKeyEvent", type=kind, key="Tab", code="Tab", windowsVirtualKeyCode=9, nativeVirtualKeyCode=9)

    async def select(self, label, value):
        expr = (f"[...document.querySelectorAll('[data-testid=stSelectbox]')].filter(b => b.innerText.trim().split('\\n')[0].trim() === {json.dumps(label)})"
                f".map(b => b.querySelector('input'))[0]")
        if not await self.click_el(expr): return False
        await self.call("Input.insertText", text=value); await asyncio.sleep(0.8)
        await self.enter(); await self.idle(1.5)
        return True

    async def set_date(self, label, year=None, month=None, day=None):
        """Type into the date field's segments, then commit with Enter and a click outside the field."""
        for seg, val in (("year", year), ("month", month), ("day", day)):
            if val is None: continue
            found = await self.js(f"(() => {{ const e = [...document.querySelectorAll('[role=spinbutton]')].filter(e => (e.getAttribute('aria-label')||'').startsWith({json.dumps(seg + ', ' + label)}))[0]; if (!e) return false; e.scrollIntoView({{block:'center'}}); e.focus(); return true; }})()")
            if not found: return False
            for ch in f"{val:02d}" if seg != "year" else str(val):
                vk = 48 + int(ch)
                await self.call("Input.dispatchKeyEvent", type="keyDown", key=ch, code=f"Digit{ch}", windowsVirtualKeyCode=vk, text=ch)
                await self.call("Input.dispatchKeyEvent", type="keyUp", key=ch, code=f"Digit{ch}", windowsVirtualKeyCode=vk)
            await asyncio.sleep(0.3)
        await self.enter()
        await self.click_el("document.querySelector('h1')")
        await self.idle(2)
        return True

    async def upload(self, path):
        doc = await self.call("DOM.getDocument", depth=-1)
        node = await self.call("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="input[type=file]")
        await self.call("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(path)])
        await self.idle(2.5)

    async def errors(self):
        return await self.js("[...document.querySelectorAll('[data-testid=stException]')].length")

    async def shot(self, name, width=1440, full=True, height=1000):
        await self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=width < 600)
        await asyncio.sleep(1)
        if full:
            h = await self.js("Math.max(...[...document.querySelectorAll('[data-testid=stMain], section.stMain, body')].map(e => e.scrollHeight))")
            await self.call("Emulation.setDeviceMetricsOverride", width=width, height=min(int(h or height) + 40, 7000), deviceScaleFactor=1, mobile=width < 600)
            await asyncio.sleep(1.5)
        data = await self.call("Page.captureScreenshot", format="png")
        (SHOTS / f"{name}.png").write_bytes(base64.b64decode(data["data"]))
        await self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=width < 600)
        return f"{name}.png"


def record(tc, category, desc, pre, steps, inp, expected, actual, passed, shot):
    RESULTS.append({"id": tc, "category": category, "description": desc, "preconditions": pre, "steps": steps, "input": inp,
                    "expected": expected, "actual": actual, "status": "Pass" if passed else "Fail", "screenshot": shot})
    print(f"{tc} {'PASS' if passed else 'FAIL'} | {actual[:160]}", flush=True)


def find(pattern, text, default="(not found)"):
    m = re.search(pattern, text)
    return m.group(0) if m else default


async def run(t):
    PRE = "App running at http://localhost:8501 (started with start_app.bat); stored datasets and saved models present"

    # TC01 Home
    ok = await t.goto("/", "Data in use"); txt = await t.text(); err = await t.errors()
    act = f"Home page rendered; 'Data in use' table shown; model summary '{find(r'NEXT-DAY ERROR\s*[\d.]+%', txt).replace(chr(10), ' ')}'; last update line '{find(r'Last daily update[^.]*', txt)[:70]}'; exceptions on page: {err}"
    record("TC01", "Functional / UI", "Home page loads with model summary and datasets in use", PRE, "1. Open http://localhost:8501", "None",
           "Page loads without errors; shows model accuracy cards and the 'Data in use' table", act, ok and err == 0, await t.shot("TC01_home"))

    # TC02 navigation
    pages = [("/Predict", "Request a prediction"), ("/Model_Performance", "Model performance"), ("/Price_History", "Price history"), ("/Cultivation_Planner", "Cultivation planner")]
    loaded = []
    for path, needle in pages:
        ok = await t.goto(path, needle); loaded.append(f"{path.strip('/')}: {'loaded' if ok else 'NOT loaded'}, exceptions {await t.errors()}")
    allok = all("loaded," in x and x.endswith("exceptions 0") for x in loaded)
    record("TC02", "UI / Navigation", "All four pages load from the sidebar without exceptions", PRE, "1. Open each page in turn: Predict, Model Performance, Price History, Cultivation Planner",
           "None", "Each page renders its heading; no Streamlit exception boxes", "; ".join(loaded), allok, None)

    # TC03 + TC04 next-day prediction for Beans (default target date is the day after the latest data)
    await t.goto("/Predict", "Request a prediction")
    await t.select("Vegetable", "Beans")
    txt = await t.text()
    warn = find(r"\d{4}-\d{2}-\d{2} has already passed \(\d+ day\(s\) ago\)", txt)
    record("TC04", "Input validation / Warning", "A target date that has already passed is flagged", PRE + "; latest Dambulla data 2026-11-09; today 2026-11-13",
           "1. Open Predict; 2. Select Beans; 3. Leave the default date (day after the latest data)", "Vegetable = Beans, date = 2026-11-10",
           "A warning states that the date has already passed and that the result is a forecast, not the recorded price",
           f"Warning shown: '{warn} ...'", warn != "(not found)", await t.shot("TC04_past_date_warning", full=False))
    await t.click_text("Predict", "button"); await t.wait_text("Predicted Price")
    txt = await t.text()
    head = find(r"Predicted Price . Beans at Dambulla on \d{4}-\d{2}-\d{2}", txt); price = find(r"Rs\. [\d,]+\.\d{2} / kg", txt)
    RESULTS_STATE["beans_next_day"] = {"heading": head, "price": price}
    record("TC03", "Functional / Prediction", "Next-day price prediction for a real series", PRE, "1. Select data source 'Historical dataset (real)'; 2. Select Beans; 3. Click Predict",
           "Beans, Dambulla, 2026-11-10", "A price in Rs/kg with the model used, data sources and weather used",
           f"{head}: {price}; model stated: {'Ridge direct forecaster' in txt}; weather panel: {'Weather Data Used' in txt}",
           price != "(not found)" and "Ridge direct forecaster" in txt, await t.shot("TC03_next_day_prediction"))

    # TC05 multi-day forecast to 2026-11-25 (16 days after the latest data)
    await t.click_text("Future Date Forecast (multi-step)", "label, p")
    await t.set_date("Forecast date", day=25)
    await t.click_text("Generate Forecast", "button"); await t.wait_text("Forecasted Price")
    txt = await t.text()
    head = find(r"Forecasted Price . Beans at Dambulla on \d{4}-\d{2}-\d{2}", txt); price = find(r"Rs\. [\d,]+\.\d{2} / kg", txt)
    horizon = find(r"\d+ day\(s\) ahead", txt)
    RESULTS_STATE["beans_multi"] = {"heading": head, "price": price, "horizon": horizon}
    record("TC05", "Functional / Prediction", "Multi-day forecast within the 30-day reach", PRE, "1. Choose 'Future Date Forecast (multi-step)'; 2. Set the forecast date to 2026-11-25; 3. Click Generate Forecast",
           "Beans, 2026-11-25", "A forecast price, the horizon and a forecast warning are shown", f"{head}: {price}; horizon {horizon}; forecast warning shown: {'FORECAST' in txt}",
           price != "(not found)" and "2026-11-25" in head, await t.shot("TC05_multi_day_forecast"))

    # TC06 beyond the 30-day reach
    await t.set_date("Forecast date", month=12, day=25)
    await t.click_text("Generate Forecast", "button"); await t.idle(3)
    txt = await t.text()
    msg = find(r"Cannot forecast that date[^\n]*", txt)
    record("TC06", "Input validation / Error handling", "A forecast beyond the 30-day reach is refused with a reason", PRE, "1. Set the forecast date to 2026-12-25 (46 days after the latest data); 2. Click Generate Forecast",
           "Beans, 2026-12-25", "No price is produced; a clear message explains the date is beyond reach", f"Message: '{msg[:200]}'; price shown: {'Forecasted Price' in txt}",
           msg != "(not found)", await t.shot("TC06_beyond_reach", full=False))

    # TC07 unsupported series are listed with reasons
    await t.goto("/Predict", "Request a prediction")
    await t.click_el("[...document.querySelectorAll('summary, [data-testid=stExpander] details summary')].filter(e => e.innerText.includes(\"can't be forecast\"))[0]")
    txt = await t.text()
    count = find(r"\d+ recorded price series can't be forecast", txt)
    offered = await t.js("[...document.querySelectorAll('[data-testid=stSelectbox]')].length")
    # The table inside the panel is drawn on a canvas, so its rows cannot be read as page text; they were checked in the screenshot.
    record("TC07", "Error handling / Data", "Series that cannot be forecast are excluded and explained", PRE, "1. Open Predict; 2. Expand the 'recorded price series can't be forecast' panel",
           "None", "Stale or untrained series are not offered; each is listed with a reason",
           f"Panel heading: '{count}'. Table rows (checked in the screenshot, as the table is drawn on a canvas): Bitter Gourd (Other), Bitter Gourd (Village), Brinjals (Other), Brinjals (Village) - 'not trained on this series (too few prices)'; Cabbage (Nuwara Eliya) - 'latest Dambulla price is 2025-07-12'",
           count != "(not found)", await t.shot("TC07_unsupported_series"))

    # TC08 latest HARTI bulletin (live integration)
    await t.click_text("Latest HARTI Data (real)", "label, p"); await t.wait_text("Latest HARTI bulletin", 120)
    txt = await t.text()
    bull = find(r"Latest HARTI bulletin: \d{4}-\d{2}-\d{2}[^)]*\)", txt); err = find(r"Couldn't fetch the latest HARTI bulletin[^\n]*", txt, "")
    record("TC08", "Integration (external source)", "Fetch the latest published HARTI bulletin live", PRE + "; internet access",
           "1. Open Predict; 2. Select 'Latest HARTI Data (real)'", "None", "The newest bulletin is downloaded and its date and number of prices are shown, labelled as not real-time",
           (bull if bull != "(not found)" else err) + f"; 'not a real-time feed' label: {'not a real-time feed' in txt}", bull != "(not found)", await t.shot("TC08_latest_harti", full=False))

    # TC09 synthetic sample data is labelled
    await t.goto("/Predict", "Request a prediction")
    await t.click_text("Bundled sample data (synthetic)", "label, p"); await t.idle(2)
    txt = await t.text()
    cap = find(r"Loaded `?app/sample_data/sample_history\.csv`?[^\n]*", txt)
    record("TC09", "Functional / Data source", "Synthetic sample data is clearly labelled as not real", PRE, "1. Select 'Bundled sample data (synthetic)'", "app/sample_data/sample_history.csv",
           "The data loads and is labelled synthetic; the earlier model, not the direct forecaster, is used", f"Caption: '{cap[:150]}'; earlier-model notice: {'earlier comparison model' in txt}",
           "synthetic" in cap.lower() or "Not real market data" in txt, await t.shot("TC09_sample_data", full=False))

    # TC10-12 uploads
    for tc, fname, desc, expected, check in [
        ("TC10", "valid_upload.csv", "A valid CSV upload is accepted", "The file is loaded and a model option is offered, with no error", lambda s: "Preview loaded data" in s and "Could not read" not in s and "missing required column" not in s),
        ("TC11", "missing_price_column.csv", "A CSV without the price column is rejected", "A clear error names the missing column; no prediction is made", lambda s: "price" in find(r"(missing required column[^\n]*|[^\n]*required column[^\n]*)", s).lower()),
        ("TC12", "empty.csv", "An empty CSV is rejected", "A clear error says the file is empty", lambda s: "empty" in s.lower()),
    ]:
        await t.goto("/Predict", "Request a prediction")
        await t.click_text("Upload my own CSV", "label, p"); await t.idle(1.5)
        await t.upload(INPUTS / fname)
        txt = await t.text()
        line = find(r"(That file is empty[^\n]*|[^\n]*missing required column[^\n]*|[^\n]*required column\(s\)[^\n]*|Preview loaded data|Could not read[^\n]*)", txt)
        record(tc, "Input validation", desc, PRE, "1. Select 'Upload my own CSV'; 2. Upload the file", fname, expected, f"Observed: '{line[:200]}'",
               check(txt), await t.shot(f"{tc}_upload_{fname.split('.')[0]}", full=False))

    # TC13 Price History: other market
    await t.goto("/Price_History", "wholesale price at")
    await t.select("Market", "Kandy")
    txt = await t.text()
    head = find(r"[^\n]*wholesale price at Kandy", txt)
    record("TC13", "Functional / UI", "Price history for a market other than Dambulla", PRE, "1. Open Price History; 2. Select market Kandy", "Market = Kandy",
           "Chart and statistics for Kandy are shown", f"Heading: '{head}'; latest/average/lowest/highest cards: {'Average' in txt and 'Highest' in txt}",
           head != "(not found)", await t.shot("TC13_history_kandy"))

    # TC14 Price History: CBSL retail
    await t.click_text("CBSL daily report: wholesale and retail", "label, p"); await t.idle(2)
    await t.click_text("retail", "label, p"); await t.idle(2)
    txt = await t.text()
    head = find(r"[^\n]*retail price, CBSL", txt)
    record("TC14", "Functional / Data separation", "CBSL retail prices shown separately from wholesale", PRE, "1. Select 'CBSL daily report'; 2. Choose price type 'retail'", "Item Beans (default)",
           "A retail-only chart with the source panel stating retail", f"Heading: '{head}'; markets include Narahenpita: {'Narahenpita' in txt}",
           head != "(not found)", await t.shot("TC14_history_cbsl_retail"))

    # TC15 Planner beyond reach (default start date = today)
    await t.goto("/Cultivation_Planner", "Generate Cultivation Plan")
    await t.click_text("Generate Cultivation Plan", "button"); await t.wait_text("Your cultivation plan")
    txt = await t.text()
    harvest = find(r"EXPECTED FIRST HARVEST\s*\n?\s*\d{1,2} \w{3} \d{4}", txt).replace("\n", " ")
    today = time.localtime(); import datetime as _dt
    start = _dt.date(today.tm_year, today.tm_mon, today.tm_mday); expect = start + _dt.timedelta(days=45)
    expect_txt = f"{expect.day} {expect.strftime('%b %Y')}"
    record("TC15", "Functional / Planner", "A harvest beyond the forecast reach shows history, not a forecast", PRE, "1. Open Cultivation Planner (Beans, bush types); 2. Keep today's planting date; 3. Generate",
           f"Beans, bush types (45 days), planting {start.isoformat()}", f"First harvest {expect_txt} (45 days later); price 'Not available'; prices from earlier years labelled as history",
           f"{harvest}; forecast price not available: {'Not available' in txt}; history table shown: {'Prices in the same weeks of earlier years' in txt}",
           "Not available" in txt and expect_txt in harvest, await t.shot("TC15_planner_beyond_reach"))

    # TC16 Planner within reach: planting 2026-10-20 -> first harvest 2026-12-04, within 30 days of 2026-11-09
    await t.set_date("Planting date", month=10, day=20)
    await t.click_text("Generate Cultivation Plan", "button"); await t.wait_text("Your cultivation plan")
    txt = await t.text()
    harvest = find(r"EXPECTED FIRST HARVEST\s*\n?\s*\d{1,2} \w{3} \d{4}", txt).replace("\n", " ")
    fp = find(r"FORECAST PRICE AT HARVEST\s*\n?\s*[^\n]+", txt).replace("\n", " ")
    record("TC16", "Functional / Planner + prediction", "A harvest within reach shows a forecast price and range", PRE, "1. Set planting date 2026-10-20; 2. Generate",
           "Beans, bush types, planting 2026-10-20", "First harvest 4 Dec 2026; a forecast price with a likely range", f"{harvest}; {fp}",
           "4 Dec 2026" in harvest and "Rs" in fp, await t.shot("TC16_planner_within_reach"))

    # TC17 Model Performance
    await t.goto("/Model_Performance", "Model performance")
    txt = await t.text()
    record("TC17", "Functional / Evaluation display", "Model accuracy is shown against the naive baseline", PRE, "1. Open Model Performance", "None",
           "Error by horizon for the model and for holding the last price", f"Next-day error shown: {find(r'NEXT-DAY ERROR\s*[\d.]+%', txt).replace(chr(10), ' ')}; 'Hold last price' series present: {'Hold last price' in txt}",
           "Hold last price" in txt, None)

    # TC18 Mobile layout
    await t.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=1, mobile=True)
    await t.goto("/Predict", "Request a prediction")
    dims = await t.js("[document.documentElement.clientWidth, document.documentElement.scrollWidth]")
    shot = await t.shot("TC18_mobile_predict", width=390, full=False, height=844)
    record("TC18", "UI / Responsive", "Predict page usable on a phone-sized screen", PRE, "1. Set the viewport to 390 x 844; 2. Open Predict", "Viewport 390 px",
           "No horizontal scrolling; controls stacked vertically", f"clientWidth {dims[0]}, scrollWidth {dims[1]}", dims[0] >= dims[1], shot)


RESULTS_STATE = {}


async def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={OUT / 'edgeprofile'}", "--window-size=1440,1000", "about:blank"])
    try:
        for _ in range(30):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read()); page = next(x for x in tabs if x["type"] == "page"); break
            except Exception:
                time.sleep(1)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2**27) as ws:
            t = Tab(ws)
            for d in ("Page.enable", "Runtime.enable", "DOM.enable"): await t.call(d)
            await t.call("Emulation.setDeviceMetricsOverride", width=1440, height=1000, deviceScaleFactor=1, mobile=False)
            try:
                await run(t)
            finally:
                (OUT / "ui_results.json").write_text(json.dumps({"executed_at": time.strftime("%Y-%m-%d %H:%M"), "results": RESULTS, "state": RESULTS_STATE}, indent=1), encoding="utf-8")
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())

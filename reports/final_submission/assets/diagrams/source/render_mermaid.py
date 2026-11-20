"""Render every *.mmd in this folder with Mermaid 10.9.1 (loaded from cdn.jsdelivr.net) in headless Edge.
Writes an SVG (vector) and a PNG (2x) per diagram into ../rendered/. argv: optional name filter."""
import asyncio, base64, json, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
import websockets

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9355
ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "source", ROOT / "rendered"
TMP = Path(tempfile.gettempdir()) / "vegprice_mermaid"; TMP.mkdir(exist_ok=True)
FILTER = sys.argv[1] if len(sys.argv) > 1 else ""

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<style>body{margin:0;background:#fff;font-family:Arial, 'Nirmala UI', sans-serif} #d{display:inline-block;padding:14px}</style></head>
<body><div id="d"></div><script>
mermaid.initialize({startOnLoad:false, theme:'base', securityLevel:'loose',
  flowchart:{htmlLabels:true, curve:'basis', nodeSpacing:34, rankSpacing:44, padding:10, useMaxWidth:false},
  themeVariables:{fontFamily:"Arial, 'Nirmala UI', sans-serif", fontSize:'15px', primaryColor:'#e8f3ec', primaryBorderColor:'#2f7d4f',
    primaryTextColor:'#13261a', lineColor:'#44515e', clusterBkg:'#f7f9fb', clusterBorder:'#9aa7b4', edgeLabelBackground:'#ffffff',
    tertiaryColor:'#f7f9fb'}});
window.renderIt = async (code) => { const {svg} = await mermaid.render('g', code); document.getElementById('d').innerHTML = svg; return svg; };
</script></body></html>"""


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
        if "exceptionDetails" in r: raise RuntimeError(r["exceptionDetails"])
        return r.get("result", {}).get("value")


async def main():
    (TMP / "page.html").write_text(PAGE, encoding="utf-8")
    proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", f"--remote-debugging-port={PORT}", f"--user-data-dir={TMP / 'profile'}", "about:blank"])
    try:
        for _ in range(30):
            try:
                page = next(x for x in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read()) if x["type"] == "page"); break
            except Exception: time.sleep(1)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2**28) as ws:
            t = Tab(ws)
            await t.call("Page.enable"); await t.call("Runtime.enable")
            await t.call("Emulation.setDeviceMetricsOverride", width=1800, height=1400, deviceScaleFactor=2, mobile=False)
            await t.call("Page.navigate", url=(TMP / "page.html").as_uri())
            for _ in range(60):
                if await t.js("typeof mermaid !== 'undefined' && typeof window.renderIt === 'function'"): break
                await asyncio.sleep(0.5)
            await t.js("document.fonts.ready.then(() => true)")
            for src in sorted(SRC.glob("*.mmd")):
                if FILTER and FILTER not in src.name: continue
                code = src.read_text(encoding="utf-8")
                svg = await t.js(f"window.renderIt({json.dumps(code)})")
                await asyncio.sleep(0.6)
                box = await t.js("(() => { const b = document.getElementById('d').getBoundingClientRect(); return [b.x, b.y, b.width, b.height]; })()")
                if box[2] > 1800 or box[3] > 1400:
                    await t.call("Emulation.setDeviceMetricsOverride", width=int(box[2]) + 40, height=int(box[3]) + 40, deviceScaleFactor=2, mobile=False)
                    await asyncio.sleep(0.5)
                shot = await t.call("Page.captureScreenshot", format="png", clip={"x": box[0], "y": box[1], "width": box[2], "height": box[3], "scale": 1})
                (OUT / f"{src.stem}.png").write_bytes(base64.b64decode(shot["data"]))
                (OUT / f"{src.stem}.svg").write_text(svg, encoding="utf-8")
                await t.call("Emulation.setDeviceMetricsOverride", width=1800, height=1400, deviceScaleFactor=2, mobile=False)
                print(src.stem, [round(v) for v in box], flush=True)
    finally:
        proc.terminate()

asyncio.run(main())

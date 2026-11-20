"""Rasterise SVG files to PNG at 2x in headless Edge. argv: svg files, e.g. ../rendered/01_system_architecture.svg"""
import asyncio, base64, json, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
import websockets

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9366


async def main(files):
    tmp = Path(tempfile.gettempdir()) / "vegprice_svg2png_profile"
    proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", f"--remote-debugging-port={PORT}", f"--user-data-dir={tmp}", "about:blank"])
    try:
        for _ in range(30):
            try:
                page = next(x for x in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read()) if x["type"] == "page"); break
            except Exception: time.sleep(1)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2**28) as ws:
            i = 0
            async def call(method, **params):
                nonlocal i
                i += 1
                await ws.send(json.dumps({"id": i, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == i: return msg.get("result", {})
            await call("Page.enable")
            for f in files:
                f = Path(f).resolve()
                head = f.read_text(encoding="utf-8")[:400]
                w, h = (int(v) for v in re.search(r'width="(\d+)" height="(\d+)"', head).groups())
                await call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=2, mobile=False)
                await call("Page.navigate", url=f.as_uri()); await asyncio.sleep(1.2)
                shot = await call("Page.captureScreenshot", format="png", clip={"x": 0, "y": 0, "width": w, "height": h, "scale": 1})
                f.with_suffix(".png").write_bytes(base64.b64decode(shot["data"]))
                print("png", f.name, w, h, flush=True)
    finally:
        proc.terminate()

asyncio.run(main(sys.argv[1:]))

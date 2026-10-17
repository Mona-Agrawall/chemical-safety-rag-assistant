"""Browser smoke test: drive the running app in headless Chrome and take screenshots.

Uses the Chrome DevTools protocol through tornado (already installed with Streamlit), so no
extra test dependencies are needed. Start the app first, then:

    python scripts/ui_smoke.py http://localhost:8501 ui-shots
    python scripts/ui_smoke.py http://localhost:8501 ui-shots --ask "What is the flash point of acetone?"

Checks the initial render (title, safety notice, indexed chemicals, model status, examples,
export), horizontal overflow on desktop and phone widths, light/dark backgrounds and keyboard
focus outlines; with ``--ask`` it submits a question to the real model, records whether text
streamed in progressively, expands the sources and saves the answer. Results go to
``<out_dir>/report.json``. Set CHROME_PATH if Chrome is not in a standard location.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from tornado.websocket import websocket_connect

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
)
CHROME = os.environ.get("CHROME_PATH") or next((c for c in CHROME_CANDIDATES if Path(c).exists()), "chrome")
PORT = 9333


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self.ids = itertools.count(1)

    async def call(self, method, **params):
        mid = next(self.ids)
        await self.ws.write_message(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    async def wait_for(self, expr, timeout, label):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if await self.js(expr):
                return time.time() - t0
            await asyncio.sleep(0.5)
        raise TimeoutError(f"timed out waiting for {label}")

    async def shot(self, path: Path, full=True):
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=full)
        path.write_bytes(base64.b64decode(r["data"]))
        print("screenshot", path)


async def main(url: str, out: Path, ask: str | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="cdp-")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--no-first-run", "--window-size=1366,900", "about:blank"])
    try:
        for _ in range(50):
            try:
                targets = httpx.get(f"http://127.0.0.1:{PORT}/json/list").json()
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                await asyncio.sleep(0.2)
        ws = await websocket_connect(page["webSocketDebuggerUrl"], max_message_size=200 * 1024 * 1024)
        cdp = CDP(ws)
        await cdp.call("Page.enable")
        await cdp.call("Runtime.enable")
        report: dict[str, object] = {}

        async def load(width, height, mobile=False, dark=False):
            await cdp.call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                           deviceScaleFactor=1, mobile=mobile)
            await cdp.call("Emulation.setEmulatedMedia",
                           features=[{"name": "prefers-color-scheme", "value": "dark" if dark else "light"}])
            await cdp.call("Page.navigate", url=url)
            await cdp.wait_for("document.body && document.body.innerText.includes('Ask your Safety Data Sheets')",
                               90, "app header")
            await cdp.wait_for("!!document.querySelector('[data-testid=\"stChatInputTextArea\"]')", 180, "chat input")
            await cdp.wait_for("!document.querySelector('[data-testid=\"stSpinner\"]')", 90, "spinners gone")
            await asyncio.sleep(2.5)

        # Desktop, light
        await load(1366, 900)
        body = await cdp.js("document.body.innerText")
        report["desktop_text_checks"] = {k: k in body for k in (
            "Ask your Safety Data Sheets", "Not a substitute for the official SDS", "Knowledge base",
            "Acetone", "Benzene", "Ethanol", "Methanol", "connected", "Try asking", "Export as Markdown",
        )}
        report["horizontal_overflow_desktop"] = await cdp.js(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth")
        report["bg_light"] = await cdp.js("getComputedStyle(document.querySelector('.stApp')).backgroundColor")
        await cdp.shot(out / "01-desktop-light-empty.png", full=False)

        if ask:
            await cdp.js("document.querySelector('[data-testid=\"stChatInputTextArea\"]').focus()")
            await cdp.call("Input.insertText", text=ask)
            await asyncio.sleep(0.5)
            await cdp.js("document.querySelector('[data-testid=\"stChatInputSubmitButton\"]').click()")
            await cdp.wait_for(
                "[...document.querySelectorAll('[data-testid=\"stChatMessage\"]')].length >= 2", 30, "assistant bubble")
            # First streamed text: the assistant bubble has markdown with any text besides the caption.
            streamed = None
            t0 = time.time()
            saw_partial = False
            while time.time() - t0 < 400:
                txt = await cdp.js("""(() => { const m=[...document.querySelectorAll('[data-testid="stChatMessage"]')];
                    return m.length ? m[m.length-1].innerText : '' })()""") or ""
                done = await cdp.js("document.body.innerText.includes('Sources ·') || "
                                    "document.body.innerText.includes('Retrieved excerpts') || "
                                    "document.body.innerText.includes('Could not generate')")
                if not done and len(txt) > 80 and streamed is None:
                    streamed = time.time() - t0
                    saw_partial = True
                    await cdp.shot(out / "02-streaming.png", full=False)
                if done:
                    break
                await asyncio.sleep(0.4)
            report["answer_seconds"] = round(time.time() - t0, 1)
            report["saw_partial_stream"] = saw_partial
            await asyncio.sleep(1.5)
            # Expand the sources panel.
            await cdp.js("""(() => {
                const s = [...document.querySelectorAll('summary')].find(x => x.innerText.includes('Sources'));
                if (s) s.click(); return !!s })()""")
            await asyncio.sleep(1.5)
            last = await cdp.js("""(() => { const m=[...document.querySelectorAll('[data-testid="stChatMessage"]')];
                return m[m.length-1].innerText })()""")
            report["answer_text"] = last
            await cdp.shot(out / "03-answer-with-sources.png", full=True)

        # Dark mode
        await load(1366, 900, dark=True)
        report["bg_dark"] = await cdp.js("getComputedStyle(document.querySelector('.stApp')).backgroundColor")
        await cdp.shot(out / "04-desktop-dark.png", full=False)

        # Phone width
        await load(390, 844, mobile=True)
        report["horizontal_overflow_mobile"] = await cdp.js(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth")
        await cdp.shot(out / "05-mobile.png", full=False)

        # Keyboard: Tab focus reaches interactive elements and a visible outline is applied.
        await load(1366, 900)
        for _ in range(3):
            await cdp.call("Input.dispatchKeyEvent", type="keyDown", key="Tab", code="Tab", windowsVirtualKeyCode=9)
            await cdp.call("Input.dispatchKeyEvent", type="keyUp", key="Tab", code="Tab", windowsVirtualKeyCode=9)
        report["focused_after_tab"] = await cdp.js(
            "document.activeElement ? document.activeElement.tagName + ' ' + "
            "(document.activeElement.innerText || '').slice(0, 40) : null")
        report["focus_outline"] = await cdp.js("getComputedStyle(document.activeElement).outlineStyle")

        print(json.dumps(report, indent=2, ensure_ascii=False))
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        proc.terminate()


if __name__ == "__main__":
    ask = sys.argv[sys.argv.index("--ask") + 1] if "--ask" in sys.argv else None
    asyncio.run(main(sys.argv[1], Path(sys.argv[2]), ask))

"""Run the real page in owned headless Chrome against an in-memory fixture server.

No human measurements or user profiles. Usage: python tests/browser_stopping.py OUTDIR
Chrome may be selected with CHROME_BIN. This is deliberately separate from pytest.
"""

import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import websocket

STATIC = Path(__file__).resolve().parents[1] / "theme/static"
CHROME = os.environ.get(
    "CHROME_BIN", "/nix/store/fadd2q4zxdfs2s5pf5dzm7b5fjwpwn19-google-chrome-152.0.7977.82/share/google/chrome/chrome"
)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    state = {"mode": "valid", "requests": [], "responses": []}
    release = threading.Event()

    def trial(n=0, polarity="day", complete=False):
        return {
            "n": n,
            "polarity": polarity,
            "mode": "duel",
            "is_duel": True,
            "chip": "synthetic fixture",
            "keys": "arrows",
            "progress": "fixture",
            "page_bg": "#f1ece7",
            "chrome_ink": "#3a3634",
            "prompt_html": "Fixture choices",
            "cards": [{"ground": "#f1ece7", "html": "<p>Fixture page</p>"}] * 2,
            "gate": True,
            "gate_text": "Synthetic fixture — no measurement saved",
            "block_complete": complete,
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, payload, kind="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            # An aborted pause fetch is the expected path under test.
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/status":
                return self.send({"responses": 0})
            if self.path.startswith("/api/trial/"):
                return self.send(trial())
            if self.path.startswith("/api/stopping/"):
                polarity = self.path.rsplit("/", 1)[1]
                mode = state["mode"]
                state["requests"].append({"polarity": polarity, "mode": mode})
                if mode == "late":
                    release.wait(5)
                if mode == "missing":
                    return self.send(
                        {
                            "status": "unknown",
                            "recommendation": "Pause and review the analysis.",
                            "reason": "No usable analysis summary is available.",
                        }
                    )
                return self.send(
                    {
                        "status": "review",
                        "polarity": polarity,
                        "created": "2026-09-10T00:00:00+00:00",
                        "age_seconds": 600,
                        "recommendation": "Review the reading-speed tradeoff.",
                        "reason": "Predictive validity and sustained comfort still need review.",
                        "observation": {
                            "duels": 40,
                            "leading_group_mass": 0.3,
                            "alternatives": 3,
                            "progress": {
                                "duels": 80,
                                "back": 25,
                                "lead_then": 0.4,
                                "lead_now": 0.3,
                                "set_then": 5,
                                "set_now": 3,
                            },
                        },
                    }
                )
            path = STATIC / ("index.html" if self.path == "/" else self.path.removeprefix("/static/"))
            if path.name in ("index.html", "app.js", "app.css"):
                return self.send(
                    path.read_bytes(),
                    {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[path.suffix],
                )
            self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["responses"].append(body)
            self.send({"ok": True, "next": trial(32, "night", complete=True)})

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    proc = None
    ws = None
    try:
        with (
            tempfile.TemporaryDirectory(prefix="calibration-stopping-browser-") as profile,
            (output / "chrome.log").open("w") as log,
        ):
            proc = subprocess.Popen(
                [
                    CHROME,
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--no-first-run",
                    "--disable-background-networking",
                    "--remote-debugging-port=0",
                    "--remote-allow-origins=http://localhost",
                    "--user-data-dir=" + profile,
                    "about:blank",
                ],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            port_file = Path(profile) / "DevToolsActivePort"
            deadline = time.monotonic() + 15
            while not port_file.exists():
                assert proc.poll() is None and time.monotonic() < deadline
                time.sleep(0.05)
            port = int(port_file.read_text().splitlines()[0])
            tabs = json.load(urlopen(f"http://127.0.0.1:{port}/json/list", timeout=3))
            target = next(t for t in tabs if t["type"] == "page")
            ws = websocket.create_connection(target["webSocketDebuggerUrl"], origin="http://localhost", timeout=5)
            number = 0

            def call(method, params=None):
                nonlocal number
                number += 1
                ws.send(json.dumps({"id": number, "method": method, "params": params or {}}))
                while True:
                    reply = json.loads(ws.recv())
                    if reply.get("id") == number:
                        assert "error" not in reply, reply
                        return reply.get("result", {})

            def js(expression):
                reply = call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
                assert "exceptionDetails" not in reply, reply
                return reply["result"].get("value")

            def until(expression):
                deadline = time.monotonic() + 5
                while not js(expression):
                    assert time.monotonic() < deadline, expression
                    time.sleep(0.02)

            call("Page.enable")
            call(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": "window.fixtureClock=1000; performance.now=()=>window.fixtureClock;"},
            )
            call(
                "Emulation.setDeviceMetricsOverride",
                {"width": 1280, "height": 900, "deviceScaleFactor": 1, "mobile": False},
            )
            call("Page.navigate", {"url": f"http://127.0.0.1:{http.server_port}/"})
            until("document.querySelector('#cover-text')?.textContent.includes('Synthetic fixture')")
            assert not state["requests"], "initial gate must not fetch evidence"
            js("document.querySelector('#go').click(); document.querySelector('#pause').click()")
            until("document.querySelector('#stopping').textContent.includes('reading-speed')")
            text = js("document.querySelector('#stopping').textContent")
            assert "day" in text and "10 minutes ago" in text and "50% mass" in text
            assert "sustained comfort" in text and "55 → 80" in text
            screenshot = call("Page.captureScreenshot", {"format": "png"})["data"]
            (output / "pause-valid.png").write_bytes(base64.b64decode(screenshot))
            js("window.fixtureClock=4000; document.querySelector('#go').click()")
            assert js("document.querySelector('#stopping').hidden && document.querySelector('#cover').hidden")
            state["mode"] = "missing"
            js("document.querySelector('#pause').click()")
            until("document.querySelector('#stopping').textContent.includes('No usable')")
            js("document.querySelector('#go').click()")
            state["mode"] = "late"
            js("document.querySelector('#pause').click()")
            deadline = time.monotonic() + 5
            while not any(r["mode"] == "late" for r in state["requests"]):
                assert time.monotonic() < deadline
                time.sleep(0.02)
            js("window.fixtureClock=6000; document.querySelector('#go').click()")
            assert js("document.querySelector('#stopping').hidden")
            state["mode"] = "valid"
            js("window.fixtureClock=6500; document.querySelector('.card').click()")
            until("document.querySelector('#stopping').textContent.includes('night')")
            release.set()
            time.sleep(0.15)
            assert "night" in js("document.querySelector('#stopping').textContent")
            assert "day · analysis" not in js("document.querySelector('#stopping').textContent")
            body = state["responses"][0]
            assert body["t_render"] == 6000 and body["t_click"] == 6500 and body["pauses"] == 3
            js("document.querySelector('#go').click()")
            assert js("document.querySelector('#stopping').hidden && document.querySelector('#cover').hidden")
            (output / "receipt.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "chromePid": proc.pid,
                        "tests": [
                            "valid cached pause",
                            "missing cache pause",
                            "age and polarity",
                            "late response after resume and polarity switch",
                            "block-complete recommendation",
                            "resume hides recommendation",
                            "clock rebaseline and pause count",
                        ],
                        "response": body,
                        "humanRowsWritten": 0,
                        "serverPort": http.server_port,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        release.set()
        if ws is not None:
            ws.close()
        if proc is not None:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        http.shutdown()
        http.server_close()


if __name__ == "__main__":
    run(Path(sys.argv[1]))

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["websocket-client>=1.7"]
# ///
"""Drive the sample page in headless Chrome over CDP: load model, run 20 warm decisions,
report the execution provider actually used, latency, and the answers. Saves a screenshot.

Usage: uv run verify_webgpu.py [--port 8765] [--chrome PATH] [--shot out.png]
Exit code 0 only if the model loads and the preset answers come back.
"""
from __future__ import annotations

import argparse
import base64
import http.server
import json
import logging
import os
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import websocket

log = logging.getLogger("verify_webgpu")
HERE = Path(__file__).resolve().parent
DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def serve(port: int) -> socketserver.TCPServer:
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(HERE), **k)  # noqa: E731
    http.server.SimpleHTTPRequestHandler.log_message = lambda *a, **k: None
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("serving %s on http://127.0.0.1:%d/", HERE, port)
    return srv


class CDP:
    def __init__(self, ws_url: str) -> None:
        self.ws = websocket.create_connection(ws_url, timeout=900, suppress_origin=True)
        self.i = 0
        self.console: list[str] = []

    def call(self, method: str, **params: object) -> dict:
        self.i += 1
        my = self.i
        self.ws.send(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("method") == "Runtime.consoleAPICalled":
                args = msg["params"]["args"]
                self.console.append(" ".join(str(a.get("value", a.get("description", ""))) for a in args))
            if msg.get("id") == my:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str) -> object:
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
        return r["result"].get("value")


def wait_for(cdp: CDP, expr: str, timeout: float, what: str) -> object:
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cdp.js(expr)
        if v:
            return v
        time.sleep(1)
    raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--chrome", default=os.environ.get("CHROME", DEFAULT_CHROME))
    ap.add_argument("--shot", default=str(HERE / "screenshot.png"))
    ap.add_argument("--preset", type=int, default=0, help="preset index to bench (0 = EN billing)")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--force-wasm", action="store_true", help="hide navigator.gpu so the encoder runs on WASM (baseline)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not (HERE / "model" / "encoder.onnx").exists():
        log.error("no model at %s — run ./export_model.sh first", HERE / "model")
        return 2

    srv = serve(args.port)
    profile = tempfile.mkdtemp(prefix="laya-chrome-")
    flags = [args.chrome, f"--user-data-dir={profile}", "--remote-debugging-port=0",
             "--enable-unsafe-webgpu", "--no-first-run", "--no-default-browser-check",
             "--window-size=1280,900", "about:blank"]
    if not args.headed:
        flags.insert(1, "--headless=new")
    proc = subprocess.Popen(flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.info("chrome pid %d profile %s", proc.pid, profile)
    try:
        port_file = Path(profile) / "DevToolsActivePort"
        for _ in range(100):
            if port_file.exists() and port_file.read_text().strip():
                break
            time.sleep(0.2)
        dbg_port = port_file.read_text().split()[0]
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json"))
        page = next(t for t in targets if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Runtime.enable")
        cdp.call("Page.enable")
        if args.force_wasm:
            cdp.call("Page.addScriptToEvaluateOnNewDocument", source="delete Navigator.prototype.gpu;")
        cdp.call("Page.navigate", url=f"http://127.0.0.1:{args.port}/index.html")
        wait_for(cdp, "document.readyState === 'complete' && !!document.getElementById('load')", 30, "page")
        gpu = wait_for(cdp, "(() => { const t = document.getElementById('chip-gpu').textContent; return t.includes('?') ? '' : t })()", 20, "GPU probe")
        log.info("GPU probe: %s", gpu)

        t0 = time.time()
        cdp.js("document.getElementById('load').click()")
        wait_for(cdp, "!document.getElementById('run').disabled || document.getElementById('log').textContent.startsWith('加载失败')", 900, "model load")
        status = cdp.js("document.getElementById('log').textContent")
        log.info("load: %s (%.1fs wall)", status, time.time() - t0)
        if str(status).startswith("加载失败"):
            return 1

        cdp.js(f"document.getElementById('presets').children[{args.preset}].click()")
        cdp.js("document.getElementById('bench').click()")
        bench = wait_for(cdp, "window.__lastBench || (document.getElementById('qerr').textContent.includes('失败') && document.getElementById('qerr').textContent)", 600, "bench")
        answers = cdp.js("[...document.querySelectorAll('.ans-head')].map(e => e.innerText.replace(/\\n/g, '  '))")
        ep_chip = cdp.js("document.getElementById('chip-ep').textContent")
        shot = cdp.call("Page.captureScreenshot", format="png")
        Path(args.shot).write_bytes(base64.b64decode(shot["data"]))

        report = {"gpu_probe": gpu, "ep": ep_chip, "load": status, "bench": bench, "answers": answers,
                  "console_tail": [c for c in cdp.console if "[laya-sample]" in c or "laya:" in c][-8:]}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        log.info("screenshot -> %s", args.shot)
        return 0 if isinstance(bench, dict) and answers else 1
    finally:
        proc.terminate()
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    raise SystemExit(main())

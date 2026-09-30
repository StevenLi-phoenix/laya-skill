#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["websocket-client>=1.7"]
# ///
"""Check the sample works when embedded like an itch.io HTML game.

itch serves the page from html-classic.itch.zone inside an iframe on *.itch.io (cross-site), and caps
uploads at 200 MB/file, so the ~1.3 GB model has to come from a third, CORS-enabled origin. This
reproduces that with three loopback origins:

    parent  http://127.0.0.1:<p>   page with itch's exact <iframe ... allow="..."> tag
    game    http://localhost:<g>   this directory (index.html, app.js, vendor/) — cross-site to parent
    model   http://127.0.0.1:<m>   ./model with/without Access-Control-Allow-Origin

Scenarios: itch (itch allow attr + CORS model), noallow (bare iframe), nocors (model without CORS).
Usage: uv run verify_iframe.py [--scenario itch|noallow|nocors|all]
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
from functools import partial
from pathlib import Path

import websocket

log = logging.getLogger("verify_iframe")
HERE = Path(__file__).resolve().parent
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
ITCH_ALLOW = ("autoplay; fullscreen *; geolocation; microphone; camera; midi; monetization; "
              "xr-spatial-tracking; gamepad; gyroscope; accelerometer; xr; cross-origin-isolated; web-share")


class Handler(http.server.SimpleHTTPRequestHandler):
    cors = False

    def end_headers(self) -> None:
        if self.cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *a: object) -> None:
        pass


def serve(directory: Path, port: int, cors: bool) -> socketserver.TCPServer:
    cls = type("H", (Handler,), {"cors": cors})
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), partial(cls, directory=str(directory)))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Browser:
    """Browser-level CDP connection; commands can target an attached iframe via session id."""

    def __init__(self, ws_url: str) -> None:
        self.ws = websocket.create_connection(ws_url, timeout=900, suppress_origin=True)
        self.i = 0
        self.console: list[str] = []

    def call(self, method: str, session: str | None = None, **params: object) -> dict:
        self.i += 1
        my = self.i
        msg = {"id": my, "method": method, "params": params}
        if session:
            msg["sessionId"] = session
        self.ws.send(json.dumps(msg))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("method") == "Runtime.consoleAPICalled":
                self.console.append(" ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"]))
            if m.get("id") == my:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    def js(self, session: str, expr: str) -> object:
        r = self.call("Runtime.evaluate", session, expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
        return r["result"].get("value")

    def wait(self, session: str, expr: str, timeout: float, what: str) -> object:
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = self.js(session, expr)
            if v:
                return v
            time.sleep(0.5)
        raise TimeoutError(f"{what}: timed out after {timeout:.0f}s")


def run_scenario(name: str, ports: dict[str, int], shot_dir: Path) -> dict:
    allow = "" if name == "noallow" else f' allow="{ITCH_ALLOW}"'
    model_url = f"http://127.0.0.1:{ports['model_nocors' if name == 'nocors' else 'model']}/"
    if name != "nocors" and os.environ.get("MODEL_URL"):
        model_url = os.environ["MODEL_URL"]  # e.g. the public Hugging Face mirror
    game_url = f"http://localhost:{ports['game']}/index.html?model={model_url}"
    parent_dir = Path(tempfile.mkdtemp(prefix="itch-parent-"))
    (parent_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>itch embed sim</title>"
        "<body style='margin:0;background:#222'>"
        f'<iframe id="game_drop" src="{game_url}"{allow} allowtransparency="true" frameborder="0" '
        'allowfullscreen="true" scrolling="no" style="width:1100px;height:860px;border:0"></iframe>')
    parent = serve(parent_dir, ports["parent"], cors=False)

    profile = tempfile.mkdtemp(prefix="laya-iframe-chrome-")
    proc = subprocess.Popen([CHROME, "--headless=new", f"--user-data-dir={profile}", "--remote-debugging-port=0",
                             "--no-first-run", "--no-default-browser-check", "--window-size=1140,900", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result: dict = {"scenario": name, "game_url": game_url, "allow_attr": bool(allow)}
    try:
        pf = Path(profile) / "DevToolsActivePort"
        for _ in range(100):
            if pf.exists() and pf.read_text().strip():
                break
            time.sleep(0.2)
        port = pf.read_text().split()[0]
        b = Browser(json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version"))["webSocketDebuggerUrl"])
        page = next(t for t in b.call("Target.getTargets")["targetInfos"] if t["type"] == "page")
        ps = b.call("Target.attachToTarget", targetId=page["targetId"], flatten=True)["sessionId"]
        b.call("Page.enable", ps)
        b.call("Page.navigate", ps, url=f"http://127.0.0.1:{ports['parent']}/index.html")

        # The cross-site iframe is an out-of-process frame: find its target and attach separately.
        t0 = time.time()
        frame = None
        while time.time() - t0 < 20 and frame is None:
            frame = next((t for t in b.call("Target.getTargets")["targetInfos"]
                          if t["type"] == "iframe" and "localhost" in t["url"]), None)
            time.sleep(0.3)
        if frame is None:
            raise RuntimeError("iframe target not found (was it rendered in-process?)")
        fs = b.call("Target.attachToTarget", targetId=frame["targetId"], flatten=True)["sessionId"]
        b.call("Runtime.enable", fs)
        result["oopif"] = True
        b.wait(fs, "document.readyState==='complete' && !!document.getElementById('load')", 30, "iframe page")
        result["embedded"] = b.js(fs, "window.top !== window.self")
        result["gpu"] = b.wait(fs, "(()=>{const t=document.getElementById('chip-gpu').textContent;return t.includes('?')?'':t})()", 20, "gpu probe")
        result["cache_storage"] = b.js(fs, "caches.open('probe').then(()=> 'ok').catch(e => 'blocked: '+e.message)")

        t = time.time()
        b.js(fs, "document.getElementById('load').click()")
        b.wait(fs, "!document.getElementById('run').disabled || document.getElementById('log').textContent.startsWith('加载失败')", 600, "load")
        result["load"] = b.js(fs, "document.getElementById('log').textContent")
        result["load_s"] = round(time.time() - t, 1)
        if not str(result["load"]).startswith("加载失败"):
            b.js(fs, "document.getElementById('bench').click()")
            result["bench"] = b.wait(fs, "window.__lastBench || (document.getElementById('qerr').textContent.includes('失败') && document.getElementById('qerr').textContent)", 300, "bench")
            result["answers"] = b.js(fs, "[...document.querySelectorAll('.ans-head')].map(e=>e.innerText.replace(/\\n/g,'  '))")
            result["ep"] = b.js(fs, "document.getElementById('chip-ep').textContent")
        shot = b.call("Page.captureScreenshot", ps, format="png")
        out = shot_dir / f"iframe-{name}.png"
        out.write_bytes(base64.b64decode(shot["data"]))
        result["screenshot"] = str(out)
        result["console"] = [c for c in b.console if "[laya-sample]" in c or "laya:" in c][-6:]
    finally:
        proc.terminate()
        parent.shutdown()
        parent.server_close()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=["itch", "noallow", "nocors", "all"])
    ap.add_argument("--shot-dir", default=str(HERE))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not (HERE / "model" / "encoder.onnx").exists():
        log.error("no model at %s — run ./export_model.sh first", HERE / "model")
        return 2
    ports = {"parent": 8866, "game": 8868, "model": 8867, "model_nocors": 8869}
    servers = [serve(HERE, ports["game"], cors=False), serve(HERE / "model", ports["model"], cors=True),
               serve(HERE / "model", ports["model_nocors"], cors=False)]
    ok = True
    try:
        for name in (["itch", "noallow", "nocors"] if args.scenario == "all" else [args.scenario]):
            log.info("scenario %s", name)
            r = run_scenario(name, ports, Path(args.shot_dir))
            print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
            works = isinstance(r.get("bench"), dict)
            ok &= works == (name != "nocors")  # nocors is expected to fail
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

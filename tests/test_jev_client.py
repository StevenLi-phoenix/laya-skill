"""Jev client against a local mock of POST /v1/systemone (no real key, no network)."""
import http.server
import json
import threading
from pathlib import Path

import jev_predict
import pytest
from conftest import SKILL

FIXTURE = json.loads((Path(__file__).parent / "fixtures/jev_response.json").read_text())
QUESTIONS = json.loads((SKILL / "assets/examples/questions.json").read_text())


class Mock:
    def __init__(self):
        self.calls = []
        self.script = []  # list of (status, body_text); last one repeats

    def handler(self):
        mock = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                mock.calls.append({"path": self.path, "auth": self.headers.get("authorization"), "body": body})
                status, text = mock.script[min(len(mock.calls), len(mock.script)) - 1]
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(text.encode())

            def log_message(self, *a):
                pass

        return H


@pytest.fixture
def server():
    mock = Mock()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), mock.handler())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    mock.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield mock
    srv.shutdown()


BODY = {"state": "billed twice", "model": "jev-latest", "questions": QUESTIONS}


def test_request_shape_auth_and_parse(server, tmp_path):
    server.script = [(200, json.dumps(FIXTURE))]
    resp, meta = jev_predict.call_systemone(BODY, base_url=server.url, api_key="sk-test", raw_dir=tmp_path)
    call = server.calls[0]
    assert call["path"] == "/v1/systemone"
    assert call["auth"] == "Bearer sk-test"
    assert call["body"]["model"] == "jev-latest" and set(call["body"]["questions"]) == set(QUESTIONS)
    assert resp["answers"]["department"]["choice"] == "billing"
    assert meta["cached"] is False and Path(meta["raw_path"]).exists()


def test_cache_hit_avoids_second_call(server, tmp_path):
    server.script = [(200, json.dumps(FIXTURE))]
    jev_predict.call_systemone(BODY, base_url=server.url, api_key="k", raw_dir=tmp_path)
    resp, meta = jev_predict.call_systemone(BODY, base_url=server.url, api_key="k", raw_dir=tmp_path)
    assert len(server.calls) == 1 and meta["cached"] is True and resp == FIXTURE
    jev_predict.call_systemone(BODY, base_url=server.url, api_key="k", raw_dir=tmp_path, use_cache=False)
    assert len(server.calls) == 2


def test_raw_saved_before_parse(server, tmp_path):
    server.script = [(200, "not json {")]
    with pytest.raises(json.JSONDecodeError):
        jev_predict.call_systemone(BODY, base_url=server.url, api_key="k", raw_dir=tmp_path)
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["status"] == 200 and saved["body"] == "not json {"


def test_retries_429_then_succeeds(server, tmp_path, monkeypatch):
    monkeypatch.setattr(jev_predict.time, "sleep", lambda s: None)
    server.script = [(429, '{"error":"rate"}'), (529, '{"error":"busy"}'), (200, json.dumps(FIXTURE))]
    resp, _ = jev_predict.call_systemone(BODY, base_url=server.url, api_key="k", raw_dir=tmp_path)
    assert len(server.calls) == 3 and resp["model"] == "jev-1.13.0"


def test_401_is_not_retried_and_not_cached(server, tmp_path):
    server.script = [(401, '{"error":"invalid key"}')]
    with pytest.raises(jev_predict.JevError) as e:
        jev_predict.call_systemone(BODY, base_url=server.url, api_key="bad", raw_dir=tmp_path)
    assert e.value.status == 401 and len(server.calls) == 1
    server.script = [(200, json.dumps(FIXTURE))]
    jev_predict.call_systemone(BODY, base_url=server.url, api_key="good", raw_dir=tmp_path)
    assert len(server.calls) == 2


def test_cli_requires_key_for_typesafe(monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    rc = jev_predict.main(["--state", "x", "--questions", json.dumps(QUESTIONS), "--base-url", "https://api.typesafe.ai"])
    assert rc == 2


def test_cli_end_to_end_against_mock(server, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-env")
    server.script = [(200, json.dumps(FIXTURE))]
    rc = jev_predict.main(["--state", "billed twice", "--questions", json.dumps(QUESTIONS),
                           "--base-url", server.url, "--raw-dir", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["answers"]["churn_risk"]["noul"] == 0.97
    assert server.calls[0]["auth"] == "Bearer sk-env"
    assert "sk-env" not in json.dumps(out)

"""Weight-loading tests. Run with LAYA_INTEGRATION=1 (downloads ~1.5 GB on first run)."""
import json
import subprocess
import sys

import pytest
from conftest import SCRIPTS, SKILL

pytestmark = pytest.mark.integration
Q = str(SKILL / "assets/examples/questions.json")
STATE = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."


def run(*args, timeout=900):
    p = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, timeout=timeout, check=False)
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout


def test_laya_predict_real_answer():
    out = json.loads(run(SCRIPTS / "laya_predict.py", "--state", STATE, "--questions", Q))
    ans = out["answers"]
    assert ans["department"]["choice"] == "billing"
    assert ans["department"]["probabilities"]["billing"] > 0.5
    assert 0 <= ans["urgency"]["score"] <= 2 and 0 <= ans["churn_risk"]["noul"] <= 1
    assert out["routing"]["model"] == "english"


def test_laya_predict_multilingual_batch(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps({"state": s}, ensure_ascii=False) for s in
                              ["我打开设置页面应用就闪退。", "La aplicación se cierra cada vez que abro la configuración."]))
    lines = run(SCRIPTS / "laya_predict.py", "--batch", rows, "--questions", Q).strip().splitlines()
    outs = [json.loads(line) for line in lines]
    assert [o["routing"]["model"] for o in outs] == ["multilingual", "multilingual"]
    assert all(o["answers"]["department"]["choice"] == "technical" for o in outs)


def test_finetune_smoke_then_load_and_predict(tmp_path):
    out_dir = tmp_path / "ft"
    report = json.loads(run(SCRIPTS / "finetune_laya.py", "--smoke", "--skip-base-eval", "--out", out_dir))
    assert report["finetuned_metrics"]["n_failed"] == 0 and report["finetuned_metrics"]["n_decisions"] > 0
    ckpt = out_dir / "checkpoint"
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    assert cfg["fine_tuned"] is True and "temperature_by_options" not in cfg and len(cfg["temperature"]) == 3
    out = json.loads(run(SCRIPTS / "laya_predict.py", "--model", ckpt, "--state", STATE, "--questions", Q))
    assert set(out["answers"]) == {"department", "urgency", "churn_risk"}

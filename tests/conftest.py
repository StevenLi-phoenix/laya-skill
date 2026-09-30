import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "system-one"
SCRIPTS = SKILL / "scripts"
sys.path.insert(0, str(SCRIPTS))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: downloads Laya weights / runs models (set LAYA_INTEGRATION=1)")


def pytest_collection_modifyitems(config, items):
    import os

    if os.environ.get("LAYA_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="set LAYA_INTEGRATION=1 to run weight-loading tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)

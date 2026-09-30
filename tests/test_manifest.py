import json
import re

from conftest import ROOT, SKILL


def test_plugin_and_marketplace_json():
    plugin = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())
    market = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    assert plugin["name"] == "system-one" and re.fullmatch(r"\d+\.\d+\.\d+", plugin["version"])
    assert market["plugins"][0]["name"] == plugin["name"] and market["plugins"][0]["source"] == "./"


def test_skill_frontmatter_and_references_exist():
    text = (SKILL / "SKILL.md").read_text()
    m = re.match(r"^---\nname: (.+)\ndescription: (.+)\n---\n", text)
    assert m and m.group(1) == "system-one"
    assert 100 < len(m.group(2)) <= 1024
    for ref in re.findall(r"`(references/[\w-]+\.md)`", text):
        assert (SKILL / ref).is_file(), ref


def test_changelog_version_matches_plugin():
    plugin = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())
    assert f"## [{plugin['version']}]" in (ROOT / "CHANGELOG.md").read_text()

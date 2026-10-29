"""scripts/d62_screen.py (docs/DECISIONS.md D62): the selection rule, and a toy run read end to end."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def d62():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("d62_screen", SCRIPTS / "d62_screen.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["d62_screen"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_selection_order_matches_the_stage(d62):
    from directions.screen import SELECTION_ORDER

    assert d62.SELECTION_ORDER == SELECTION_ORDER


def _passed(**forms):
    """forms: form -> {size: shots that pass}."""
    out = {s: {} for s in ("4b", "8b", "14b")}
    for form, by_size in forms.items():
        for s, f in out.items():
            f[form] = {k: k in by_size.get(s, ()) for k in ("8", "16")}
    return out


def test_select(d62):
    p = _passed(if_mixed={"8b": ("8",), "14b": ("8",)}, if_given={"8b": ("8",), "14b": ("16",)},
                python={"4b": ("8",), "8b": ("8",), "14b": ("8",)}, if_sum={"14b": ("8",)})
    s = d62.select(p)
    assert (s["shots"], s["task"], s["models"]) == (8, "if_mixed", ["8b", "14b"])
    assert s["given_arm"] == ["8b"] and s["order_arm"] == ["8b", "14b"] and s["if_sum_passes"] == ["14b"]
    # if_mixed fails on the 14B: the next in order that passes on both
    p = _passed(if_mixed={"8b": ("8",)}, if_gt={"4b": ("8",), "8b": ("8",), "14b": ("8",)})
    assert (d62.select(p)["task"], d62.select(p)["models"]) == ("if_gt", ["4b", "8b", "14b"])
    # nothing at 8: the rule at 16
    p = _passed(ternary={"8b": ("16",), "14b": ("16",)}, if_gt={"8b": ("8",)})
    assert (d62.select(p)["shots"], d62.select(p)["task"]) == (16, "ternary")
    # python selected: no separate token-order arm
    assert d62.select(_passed(python={"8b": ("8",), "14b": ("8",)}))["order_arm"] == []
    assert d62.select(_passed(if_given={"8b": ("8",), "14b": ("8",)}))["task"] is None  # if_given is never the task


def test_toy_run(d62, tmp_path):
    from directions.config import load_screen_config
    from directions.screen import run_screen

    cfg = load_screen_config("configs/smoke_toy_screen.yaml")
    cfg.output_dir = str(tmp_path / "runs")
    run_screen(cfg, run_id="screen_qwen3_8b_toy")
    assert d62.main([str(tmp_path / "runs"), "--out", str(tmp_path / "o.json")]) == 0
    s = json.loads((tmp_path / "o.json").read_text())
    assert set(s["table"]["qwen3_8b"]) == set(cfg.formats) and s["selection"]["task"] is None

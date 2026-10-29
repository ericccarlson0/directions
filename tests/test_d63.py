"""scripts/d63_cond.py (docs/DECISIONS.md D63): the Q1 and Q3 readings, and a toy run read end to end."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
LABELS = ["IF", "(", "<g>", "<cmp>", "<h>", ",", " ", "<a>", "+", "<b>", ",", " ", "<c>", "+", "<d>", ")", " =", "␣"]


@pytest.fixture(scope="module")
def d63():
    spec = importlib.util.spec_from_file_location("d63_cond", SCRIPTS / "d63_cond.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["d63_cond"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_q1(d63):
    rps = list(range(37))
    assert d63.q1({"R_C": 24, "value_C": {"<h>": 10, "␣": 25}}, LABELS, rps)["reading"] == "formed at the condition"
    assert d63.q1({"R_C": 24, "value_C": {",": 12}}, LABELS, rps)["reading"] == "formed at the condition"
    assert d63.q1({"R_C": 24, "value_C": {"<h>": 26}, "digit_C": {"<h>": [0, 22]}}, LABELS, rps)["reading"] == "deferred"
    assert d63.q1({"R_C": 24, "value_C": {" =": 20}, "digit_C": {"<h>": [0, 19]}}, LABELS, rps)["reading"] == "deferred"
    assert d63.q1({"R_C": 24, "value_C": {"<c>": 15}, "digit_C": {"<h>": [0, 19]}}, LABELS, rps)["reading"] == "undecided"
    assert d63.q1({"R_C": None, "value_C": {}, "digit_C": {}}, LABELS, rps)["reading"] == "undecided"
    # g's "value" is no test (the same-outcome source keeps the g crossing): it neither forms nor blocks a reading
    assert d63.q1({"R_C": 24, "value_C": {"<g>": 0, "␣": 25}, "digit_C": {"<h>": [0, 22]}}, LABELS, rps)["reading"] == "deferred"


def test_q3(d63):
    probe = {"probes": {"A/Sun|Y": {"carried": [False] * 20 + [True] * 5 + [False] * 12}}}
    rps = list(range(37))
    assert d63.q3(probe, rps, 24, 28)["reading"] == "the unchosen sum is computed"  # carried 20-24, window 22-28
    assert d63.q3(probe, rps, 30, 33)["reading"] == "not decodable"
    assert d63.q3(None, rps, 24, 28)["reading"].startswith("undecided")


def test_toy_run(d63, tmp_path):
    from directions.cond import run_cond
    from directions.config import load_cond_config

    cfg = load_cond_config("configs/smoke_toy_cond.yaml")
    cfg.output_dir = str(tmp_path / "runs")
    for part in ("map", "pairs"):
        run_cond(cfg, part, run_id=f"cond_{part}_qwen3_8b_toy")
    assert d63.main([str(tmp_path / "runs"), "--out", str(tmp_path / "o.json"), "--figures", str(tmp_path / "fig")]) == 0
    s = json.loads((tmp_path / "o.json").read_text())["qwen3_8b"]
    assert {"Q1", "Q2", "Q3", "map", "pairs"} <= set(s) and (tmp_path / "fig" / "qwen3_8b_cond.png").exists()

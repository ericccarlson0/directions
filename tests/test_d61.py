"""scripts/d61_interact.py (docs/DECISIONS.md D61): the after-failure summary, the ratio by group, and a toy run."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def d61():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("d61_interact", SCRIPTS / "d61_interact.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["d61_interact"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_after_failure(d61):
    cats = ["11", "10", "01", "00", "other"]
    rps = [0, 1, 2, 3]
    m = {"11": [1, 0.8, 0.2, 0.1], "10": [0, 0.1, 0.7, 0.2], "01": [0, 0, 0, 0.6], "00": [0, 0, 0.1, 0], "other": [0, 0.1, 0, 0.1]}
    rates = {"additive": {c: {"mean": np.array(v)} for c, v in m.items()}}
    a = d61.after_failure(rates, cats, rps, 2)
    assert a["by_read_point"] == {2: "10", 3: "01"} and a["most_frequent_from_K"] == {"10": 1, "01": 1}
    assert a["at_K"]["10"] == 0.7
    assert d61.after_failure(rates, cats, rps, None) == {}


def test_ratio_by_group(d61):
    ratio = [[0.0, 0.0, 0.0], [0.1, 0.3, 0.9]]
    assert d61.ratio_by_group(ratio, {"O": [0], "F": [1, 2], "S": []}, [4, 8], 8) == {"O": 0.1, "F": 0.6}
    assert d61.ratio_by_group(ratio, {"O": [0]}, [4, 8], None) == {}


def test_toy_run(d61, tmp_path):
    from directions.config import load_interact_config
    from directions.interact import run_interact

    cfg = load_interact_config("configs/smoke_toy_interact.yaml")
    cfg.output_dir = str(tmp_path / "runs")
    run_interact(cfg, run_id="interact_qwen3_8b_toy")
    d60 = tmp_path / "d60.json"
    d60.write_text(json.dumps({"two_site": {"per_model": {"8b": {"T2": 20, "G2": 24}}}}))
    assert d61.main([str(tmp_path / "runs"), "--d60", str(d60), "--out", str(tmp_path / "o.json")]) == 0
    s = json.loads((tmp_path / "o.json").read_text())["qwen3_8b"]
    assert set(s["pairs"]) == set(cfg.pairs) and s["landmarks"]["T2"] == 20
    assert "control_combined" in s["reading"] and "null_clear" in s["reading"]

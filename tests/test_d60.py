"""scripts/d60_14b.py (docs/DECISIONS.md D60): the take-up test, the replication predictions, and a toy run."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def d60():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("d60_14b", SCRIPTS / "d60_14b.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["d60_14b"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_take_up(d60):
    assert d60.take_up(30, 26, 31)["reading"] == "follows the gather"  # [24, 28]
    assert d60.take_up(30, 20, 24)["reading"] == "fixed depth"  # [18, 22]
    assert d60.take_up(30, 23, None)["reading"] == "neither"
    assert d60.take_up(28, 22, 25)["reading"] == "not discriminating (the ranges overlap)"
    assert d60.take_up(None, 20, 22)["reading"] == "undecided (no G*)"
    assert d60.take_up(30, 26, 27)["A_beside"] == "follows the gather and fixed depth"  # 27 in both ranges
    assert d60.take_up(30, 26, 20)["A_beside"] == "neither"


def test_predictions(d60):
    rps = [8, 9, 10, 11]

    def r(lo, hi):
        return {"flip": {"lo": np.array(lo), "hi": np.array(hi)}}

    arm = {"gap": [9, 10, 11], "rates": {"FS": r([0, 0, 0, 0], [0.1, 0.2, 0.1, 0.0]),
                                         "OA": r([0.9, 0.6, 0.55, 0.7], [1, 1, 1, 1]),
                                         "EA": r([0, 0.1, 0.6, 0.8], [0.1, 0.3, 0.8, 0.9])}}
    p = d60.predictions(arm, rps)
    assert p["a_operands_not_involved"] and p["b_two_ends_hold_it"] and p["c_eq_joins_the_answer"]
    assert p["EA_first_in_gap"] == 10 and p["EA_first"] == 10 and p["OA_missing_in_gap"] == []
    arm["rates"]["FS"] = r([0, 0, 0, 0], [0.1, 0.6, 0.1, 0.0])
    arm["rates"]["OA"] = r([0.9, 0.6, 0.4, 0.7], [1, 1, 1, 1])
    p = d60.predictions(arm, rps)
    assert not p["a_operands_not_involved"] and not p["b_two_ends_hold_it"] and p["OA_missing_in_gap"] == [10]


def test_toy_run(d60, tmp_path):
    import directions.varmap as vm
    from directions.config import ModelConfig, VarmapConfig, load_opset_config
    from directions.opset import run_opset

    vcfg = VarmapConfig(output_dir=str(tmp_path / "d57"), model=ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=64),
                        expressions=["maxmin"], n_bases=8, n_shots=2, n_demo_pool=8, read_points=[0, 1, 2, 4], n_boot=30,
                        min_accuracy=0.0, gate=False)
    vm.run_varmap(vcfg, run_id="varmap_maxmin_qwen3_14b_toy")
    cfg = load_opset_config("configs/smoke_toy_opset.yaml")
    cfg.output_dir = str(tmp_path / "set")
    run_opset(cfg, run_id="opset_qwen3_14b_toy")
    assert d60.main(["--d57", str(tmp_path / "d57"), "--opset", str(tmp_path / "set"), "--out", str(tmp_path / "o.json")]) == 0
    s = json.loads((tmp_path / "o.json").read_text())
    assert set(s["arms"]) == {"natural", "added"} and "take_up" in s["arms"]["natural"]
    assert set(s["two_site"]["per_model"]) == {"14b"} and s["two_site"]["E+A"]["reading"].startswith("undecided")


def test_two_site(d60):
    base = {"4b": {"T2": 20, "G2": 22, "T1": 22, "G1": 24}, "8b": {"T2": 20, "G2": 22, "T1": 27, "G1": 24}}
    r = d60.two_site({**base, "14b": {"T2": 27, "G2": 29, "T1": 30, "G1": 32}})
    assert r["E+A"]["reading"] == "follows the gather" and r["E+A"]["dG"] == 7 and r["E+A"]["dT"] == 7
    assert d60.two_site({**base, "14b": {"T2": 21, "G2": 29, "T1": 25, "G1": 32}})["E+A"]["reading"] == "fixed depth"
    assert d60.two_site({**base, "14b": {"T2": 24, "G2": 23, "T1": 25, "G1": 25}})["E+A"]["reading"].startswith("not discriminating")
    assert d60.two_site({**base, "14b": {"T2": 25, "G2": 30, "T1": 25, "G1": 32}})["E+A"]["reading"] == "neither"
    assert d60.two_site({**base, "14b": {"T2": None, "G2": 30, "T1": 25, "G1": 32}})["E+A"]["reading"].startswith("undecided")

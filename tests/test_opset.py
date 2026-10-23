"""The operator's token-set swaps (docs/DECISIONS.md D59): the token groups, the family of sets, the readings, and a
toy run of the stage."""

from __future__ import annotations

import json

import numpy as np

from directions.config import load_opset_config
from directions.opset import ENDS, FAMILY, GROUPS, K_OPERANDS, K_STRUCTURE, QUERY, members, readings, run_opset, token_groups

QWEN = ["op", "(", "a", "+", "b", ",", " ", "c", "+", "d", ")", " =", "␣"]  # D57's query labels on Qwen3
TOY = ["op", "op", "op", "(", "a", "+", "b", ",", " ", "c", "+", "d", ")", " ", "=", "␣"]  # one token per character


def test_family():
    assert len(FAMILY) == 23 and len(set(FAMILY)) == 23
    assert all(list(k) == [g for g in GROUPS if g in k] for k in FAMILY)  # letters in GROUPS order
    assert {K_OPERANDS, K_STRUCTURE, ENDS, QUERY, "FS", "PE", "FSPE"} <= set(FAMILY)
    assert all("".join(g for g in GROUPS if g != x) in FAMILY for x in GROUPS)  # every group but one


def test_token_groups_partition_the_query():
    g = token_groups(QWEN)
    assert g == {"O": [0], "F": [2, 3, 4], "S": [7, 8, 9], "P": [1, 5, 6, 10], "E": [11], "A": [12]}
    assert members(QUERY, g) == list(range(13)) and members(K_STRUCTURE, g) == [0, 1, 5, 6, 10, 11, 12]
    t = token_groups(TOY)
    assert t["O"] == [0, 1, 2] and t["E"] == [14] and t["P"] == [3, 7, 8, 12, 13] and t["A"] == [15]


def _rates(spec: dict[str, list[tuple[float, float]]], M: int) -> dict[str, dict]:
    """Flip-rate intervals per set: unspecified sets read 0 (lower and upper bound) everywhere."""
    out = {}
    for k in FAMILY:
        lohi = spec.get(k, [(0.0, 0.0)] * M)
        out[k] = {"flip": {"lo": np.array([a for a, _ in lohi]), "hi": np.array([b for _, b in lohi]),
                           "mean": np.array([(a + b) / 2 for a, b in lohi])}}
    return out


def test_readings_classify_the_gap():
    rps = [10, 11, 12, 13, 14, 15]
    yes, no, mid = (0.8, 1.0), (0.0, 0.2), (0.4, 0.6)
    spec = {
        "O": [yes, no, no, no, no, no],          # the operator token alone through 10
        "A": [no, no, no, no, no, yes],          # the answer token alone from 15
        K_OPERANDS: [yes, yes, yes, no, yes, yes],
        K_STRUCTURE: [yes, no, yes, yes, no, yes],
        ENDS: [yes, no, no, no, no, yes],
        QUERY: [yes] * 6,
        "OFSPE": [yes] * 6,                      # all but A: sufficient (A not necessary)
        "FSPEA": [no, no, no, no, no, yes],      # all but O: not sufficient through 14 (O necessary)
    }
    r = readings(_rates(spec, len(rps)), rps, 0.5)
    assert r["gap"] == [11, 12, 13, 14]
    assert r["classes"] == {11: "operands", 12: "either side", 13: "structure", 14: "operands"}
    assert r["reading"] == "operands" and r["ends_sufficient"] == [10, 15]
    assert r["necessity"]["O"] == [10, 11, 12, 13, 14] and r["necessity"]["A"] == []
    # OFSPE (all but A) contains neither four-group set, so it is minimal in the family too
    assert r["minimal_sufficient"][10] == ["O"] and r["minimal_sufficient"][12] == [K_OPERANDS, K_STRUCTURE, "OFSPE"]
    spec[K_OPERANDS] = [yes, mid, no, no, no, yes]  # undecided, only together, structure (twice): a tie broken by count
    spec[K_STRUCTURE] = [yes, no, no, yes, yes, yes]
    r = readings(_rates(spec, len(rps)), rps, 0.5)
    assert r["classes"] == {11: "undecided", 12: "only together", 13: "structure", 14: "structure"}
    assert r["reading"] == "structure"
    spec[K_OPERANDS] = [yes, no, no, no, no, yes]  # only together twice, structure twice: a tie, reported as such
    spec[K_STRUCTURE] = [yes, no, no, yes, yes, yes]
    assert readings(_rates(spec, len(rps)), rps, 0.5)["reading"] == "tie: only together, structure"


def test_toy_run(tmp_path):
    cfg = load_opset_config("configs/smoke_toy_opset.yaml")
    cfg.output_dir = str(tmp_path)
    root = run_opset(cfg, run_id="smoke")
    r = json.loads((root / "core" / "opset.json").read_text())
    meta = json.loads((root / "metadata.json").read_text())
    assert set(r["arms"]) == {"natural", "added", "sums"} and r["family"] == FAMILY
    for a in r["arms"].values():
        assert set(a["rates"]) == set(FAMILY) and a["labels"] == TOY and a["n"] == cfg.n_bases
        assert a["query_check"] == 1.0  # the whole query swapped is the source's run
        assert set(a["classes"]) == {str(m) for m in a["gap"]} and set(a["necessity"]) == set(GROUPS)  # JSON keys
    assert "direction" in r["arms"]["added"] and all(meta["identity_check"].values())
    assert meta["determinism_check"]["swap"]["identical"] and meta["query_check"] == {"natural": 1.0, "added": 1.0, "sums": 1.0}
    assert set(r["arms"]["sums"]["first_mean_half"]) == set(FAMILY)  # D60: the sums' arrival at each set
    arrays = np.load(root / "core" / "opset_arrays.npz")
    assert arrays[f"natural__{QUERY}"].shape == (cfg.n_bases, len(cfg.read_points))
    assert (root / "figures" / "opset.png").exists()


def test_d59_reading_script(tmp_path, monkeypatch, capsys):
    """scripts/d59_opset.py reads a toy run named like a model's."""
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("d59_opset", scripts / "d59_opset.py")
    d59 = importlib.util.module_from_spec(spec)
    sys.modules["d59_opset"] = d59
    spec.loader.exec_module(d59)
    cfg = load_opset_config("configs/smoke_toy_opset.yaml")
    cfg.output_dir = str(tmp_path)
    run_opset(cfg, run_id="opset_qwen3_8b_toy")
    assert d59.main([str(tmp_path), "--out", str(tmp_path / "o.json")]) == 0
    s = json.loads((tmp_path / "o.json").read_text())["qwen3_8b"]
    assert set(s) == {"natural", "added", "sums", "natural_vs_added"} and s["natural"]["query_check"] == 1.0
    assert "qwen3_8b natural" in capsys.readouterr().out

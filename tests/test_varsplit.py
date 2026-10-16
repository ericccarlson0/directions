"""Splitting the block where the sums appear (docs/DECISIONS.md D47): the readouts and a toy run."""

import json

import numpy as np

from directions.config import load_varsplit_config
from directions.varsplit import (
    attention_share,
    read_heads,
    read_share,
    run_varsplit,
    state_sequence,
)


def test_state_sequence_interleaves_attention_and_mlp_steps():
    names, recipe = state_sequence([24, 25])
    assert names == ["h23", "mid24", "h24", "mid25", "h25"]
    assert recipe == [("h", 23), ("mid", 24), ("h", 24), ("mid", 25), ("h", 25)]


def test_attention_share_reads_which_sublayer_carries_the_gain():
    names, _ = state_sequence([24])
    rng = np.random.default_rng(0)
    n = 400
    before = rng.random(n) < 0.3
    gained = before | (rng.random(n) < 0.8)
    by_attention = np.stack([before, gained, gained])     # the gain arrives with attention
    by_mlp = np.stack([before, before, gained])           # ... or only after the MLP
    a = attention_share(by_attention, names, 24, np.random.default_rng(1), 200, 0.05)
    m = attention_share(by_mlp, names, 24, np.random.default_rng(1), 200, 0.05)
    assert a["share"] == 1.0 and read_share(a, 0.1) == "attention"
    assert m["share"] == 0.0 and read_share(m, 0.1) == "mlp"
    flat = attention_share(np.stack([before, before, before]), names, 24, np.random.default_rng(1), 50, 0.05)
    assert read_share(flat, 0.1) == "no gain"


def _types(s1_lo, s1_hi, s2_lo, s2_hi):
    """Per-head transfer intervals, with a trailing 'all heads' column."""
    def t(lo, hi):
        return {"transfer": {"lo": list(lo) + [0.9], "hi": list(hi) + [1.0], "mean": list(hi) + [0.95]}}
    return {"S1": t(s1_lo, s1_hi), "S2": t(s2_lo, s2_hi)}


def test_head_reading_lanes_one_channel_and_none():
    lanes = _types([0.3, 0.0, 0.0], [0.5, 0.02, 0.01], [0.0, 0.4, 0.0], [0.01, 0.6, 0.02])
    r = read_heads(lanes, 0.1, 0.05, 0.25)
    assert r["reading"] == "lanes" and r["S1_only"] == [0] and r["S2_only"] == [1]
    one = _types([0.3, 0.0], [0.5, 0.02], [0.2, 0.0], [0.4, 0.02])
    assert read_heads(one, 0.1, 0.05, 0.25)["reading"] == "one channel"
    none = _types([0.0, 0.0], [0.3, 0.2], [0.0, 0.0], [0.3, 0.2])
    assert read_heads(none, 0.1, 0.05, 0.25)["reading"] == "no carrier"


def test_head_reading_on_three_brackets():
    """Three brackets (D51): lanes need a carrier of its own for every bracket."""
    def t(lo, hi):
        return {"transfer": {"lo": list(lo) + [0.9], "hi": list(hi) + [1.0], "mean": list(hi) + [0.95]}}
    three = ("S1", "S2", "S3")
    lanes = {"S1": t([0.3, 0, 0], [0.5, 0.01, 0.01]), "S2": t([0, 0.3, 0], [0.01, 0.5, 0.01]), "S3": t([0, 0, 0.3], [0.01, 0.01, 0.5])}
    r = read_heads(lanes, 0.1, 0.05, 0.25, three)
    assert r["reading"] == "lanes" and (r["S1_only"], r["S2_only"], r["S3_only"]) == ([0], [1], [2])
    two_lanes = {"S1": t([0.3, 0, 0], [0.5, 0.01, 0.01]), "S2": t([0, 0.3, 0], [0.01, 0.5, 0.01]), "S3": t([0, 0.3, 0], [0.01, 0.5, 0.01])}
    assert read_heads(two_lanes, 0.1, 0.05, 0.25, three)["reading"] == "mixed"
    assert read_heads({"S1": lanes["S1"], "S2": lanes["S2"]}, 0.1, 0.05, 0.25, three)["reading"] == "undecided"


def test_toy_run_writes_both_parts(tmp_path):
    cfg = load_varsplit_config("configs/smoke_toy_varsplit.yaml")
    cfg.output_dir = str(tmp_path)
    root = run_varsplit(cfg, run_id="toy")
    meta = json.loads((root / "metadata.json").read_text())
    assert all(v["identical"] for v in meta["determinism_check"].values())
    assert set(meta["determinism_check"]) == {"capture", "head_patch", "noop_patch"}
    for task in ("two_lane", "one_lane"):
        r = json.loads((root / "core" / "tasks" / task / "varsplit.json").read_text())
        assert r["probe"]["states"] == ["h1", "mid2", "h2", "mid3", "h3"]
        assert r["probe"]["additivity"]["max"] < 1e-4  # float32 toy: a block's step is its two writes
        for t in r["probe"]["targets"].values():
            assert len(t["states"]["accuracy"]) == 5 and len(t["writes"]["accuracy"]) == 4
        H = r["heads"]["n_heads"]
        for b in ("2", "3"):
            s1 = r["heads"]["blocks"][b]["types"]["S1"]
            assert len(s1["transfer"]["mean"]) == H + 1  # every head, then all heads at once
        assert (root / "figures" / f"{task}.png").exists()


def test_three_lane_toy_run(tmp_path):
    cfg = load_varsplit_config("configs/smoke_toy_varsplit.yaml")
    cfg.output_dir, cfg.expressions = str(tmp_path), ["three_lane"]
    root = run_varsplit(cfg, run_id="toy3")
    r = json.loads((root / "core" / "tasks" / "three_lane" / "varsplit.json").read_text())
    assert set(r["probe"]["targets"]) == {"Y", "S1|Y", "S2|Y", "S3|Y", "a|S1", "c|S2", "e|S3"}
    for b in ("2", "3"):
        types = r["heads"]["blocks"][b]["types"]
        assert {"S1", "S2", "S3", "all", "S3_same"} <= set(types) and "carries_S3" in r["heads"]["blocks"][b]["reading"]
    assert (root / "figures" / "three_lane.png").exists()

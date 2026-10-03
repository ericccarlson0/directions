"""directions.mixing: the geometry test (D40) — the mix, the readout helpers and the verdict."""

import numpy as np

from directions.config import PromptConfig
from directions.mixing import _candidate_key, _candidate_prompts, mix, verdict
from directions.tasks import Item


def test_mix_is_the_normalised_convex_combination():
    u = np.array([1.0, 0.0, 0.0]); v = np.array([0.0, 1.0, 0.0])
    assert np.allclose(mix(u, v, 0.0), u) and np.allclose(mix(u, v, 1.0), v)
    m = mix(u, v, 0.5)
    assert abs(np.linalg.norm(m) - 1) < 1e-12 and np.allclose(m, [np.sqrt(0.5), np.sqrt(0.5), 0.0])


def test_candidate_prompts_list_words_and_operands():
    pc = PromptConfig(query_template="Q: {input}\nA:", target_template=" {output}")
    rows = _candidate_prompts(Item("cat, dog, owl", "dog"), "list_words", pc, [])
    assert [k for k, _ in rows] == ["pos_1", "pos_2", "pos_3"]
    assert [p.target for _, p in rows] == [" cat", " dog", " owl"] and rows[0][1].prompt == "Q: cat, dog, owl\nA:"
    rows = _candidate_prompts(Item("40", "43"), "operands", pc, [1, 2, 3, 5, 10])
    assert [k for k, _ in rows] == ["add_1", "add_2", "add_3", "add_5", "add_10"]
    assert [p.target for _, p in rows] == [" 41", " 42", " 43", " 45", " 50"]
    assert all(p.score_reference == " 40" for _, p in rows)  # changed-token scoring keeps the operation's digits


def _dist_curves(peak_interior: bool):
    # three candidates (pos_1, pos_2, pos_3), 11 weights, 20 items; the endpoints select their own label
    W = np.linspace(0, 1, 11)
    real = np.zeros((11, 20, 3))
    for j, t in enumerate(W):
        a = 1 - t; b = t
        mid = (1 - abs(2 * t - 1)) if peak_interior else 0.0  # a tent peaking at t=0.5, or nothing
        raw = np.array([a, 0.6 * mid, b]) + 0.05
        real[j, :] = raw / raw.sum()
    return list(W), real


def test_verdict_reads_a_tent_as_parameter_and_a_pure_switch_as_switch():
    rng = np.random.default_rng(0)
    W, real = _dist_curves(peak_interior=True)
    # the dilution null: no intermediate rise (the middle candidate stays near its floor)
    null = np.stack([np.stack([np.array([1 - t, 0.02, t]) / (1 - t + 0.02 + t) for t in W]) for _ in range(4)])
    null = null[:, :, None, :].repeat(20, axis=2)
    v = verdict(W, real, null, ["pos_1", "pos_2", "pos_3"], "pos_1", "pos_3", ["pos_2"], (0.2, 0.8), rng, n_boot=500)
    assert v["endpoints_cross"] and v["verdict"] == "parameter"
    assert v["intermediates"]["pos_2"]["over_endpoints"] and v["intermediates"]["pos_2"]["interior_max"]

    W, real = _dist_curves(peak_interior=False)
    null = real[None].repeat(4, axis=0)  # null identical to real: no excess
    v = verdict(W, real, null, ["pos_1", "pos_2", "pos_3"], "pos_1", "pos_3", ["pos_2"], (0.2, 0.8), rng, n_boot=500)
    assert v["endpoints_cross"] and v["verdict"] == "switch"


def test_verdict_neither_when_endpoints_do_not_cross():
    rng = np.random.default_rng(1)
    W = list(np.linspace(0, 1, 11))
    real = np.tile(np.array([0.6, 0.1, 0.3]), (11, 20, 1))  # pos_1 dominates at both ends
    null = real[None].repeat(2, axis=0)
    v = verdict(W, real, null, ["pos_1", "pos_2", "pos_3"], "pos_1", "pos_3", ["pos_2"], (0.2, 0.8), rng, n_boot=200)
    assert not v["endpoints_cross"] and v["verdict"] == "neither"


def test_candidate_key_maps_every_position_label_to_its_candidate():
    """The end labels of the exploratory pairs (1, 2) and (2, 3) read at their own candidates, not at pos_1/pos_3."""
    assert [_candidate_key(f"kth_{k}", "list_words") for k in (1, 2, 3)] == ["pos_1", "pos_2", "pos_3"]
    assert _candidate_key("add_5", "operands") == "add_5" and _candidate_key("antonym", "own_targets") == "antonym"


def test_within_reading_connected_partial_isolated():
    """The within-label path (D44): two solutions of one task are connected when their midpoint beats the
    dilution null and keeps half the endpoints' gain, isolated when it does no better than the null."""
    from directions.mixing import within_reading
    rng = np.random.default_rng(0)
    W = [i / 10 for i in range(11)]
    n = 40
    noise = rng.normal(scale=0.01, size=(11, n))
    null = np.stack([np.stack([np.full(n, 1.0 - 0.9 * t) for t in W]) + rng.normal(scale=0.01, size=(11, n)) for _ in range(4)])
    # connected: the path stays near the endpoints' value (1.0) all the way, the unsteered value is 0.1
    v = within_reading(W, np.ones((11, n)) + noise, null, 0.1, rng, 500)
    assert v["reading"] == "connected" and v["retained_mid"] > 0.9 and v["p_mid"] <= 0.05
    # isolated: the path falls to the null's value in the middle
    path = np.stack([np.full(n, 1.0 - 0.9 * (1 - abs(2 * t - 1)) * 0.6) for t in W]) + noise
    path[5] = null[:, 5].mean(axis=0) - 0.02
    v = within_reading(W, path, null, 0.1, rng, 500)
    assert v["reading"] == "isolated"
    # partial: above the null at the middle, but below half the gain
    path = path.copy()
    path[5] = 0.4 + noise[5]  # retains (0.4 - 0.1) / 0.9 of the gain
    low_null = null.copy()
    low_null[:, 5] = 0.25 + rng.normal(scale=0.01, size=(4, n))
    v = within_reading(W, path, low_null, 0.1, rng, 500)
    assert v["reading"] == "partial" and v["retained_mid"] < 0.5


def test_mixing_config_accepts_within_and_requires_the_midpoint(tmp_path):
    import pytest
    import yaml
    from directions.config import load_mixing_config
    cfg = load_mixing_config("configs/mixing_within.yaml")
    assert all(p.constructions == ["learned", "within"] for p in cfg.pairs) and cfg.within_retained_min == 0.5
    bad = yaml.safe_load(open("configs/mixing_within.yaml"))
    bad["weights"] = [0.0, 0.25, 0.75, 1.0]
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        load_mixing_config(f)


def test_mixing_within_on_the_toy_learned_runs(tmp_path):
    """The within-label path end to end on the toy model: the weight-decayed smoke run's two lexical labels, read as
    own targets, with the cross-label path beside it."""
    from directions.config import load_config, load_mixing_config
    from directions.mixing import run_mixing
    from directions.pipeline import run_pipeline
    from dataclasses import replace
    cfg = load_config("configs/smoke_toy_learned_wd.yaml")
    cfg.output_dir = str(tmp_path / "runs")
    root = run_pipeline(cfg, "pilot", config_path="configs/smoke_toy_learned_wd.yaml")
    mcfg = load_mixing_config("configs/mixing_within.yaml")
    lex = next(p for p in mcfg.pairs if p.family == "lexical")
    mcfg = replace(mcfg, output_dir=str(tmp_path / "mix"), n_prompts=4, n_null=2, n_boot=50, device="cpu",
                   pairs=[replace(lex, labels=["antonym", "number_to_words"])])
    out = run_mixing(mcfg, {"lexical": (str(root), None)}, run_id="mix_toy")
    import json
    summary = json.loads((out / "core" / "summary.json").read_text())
    within = next(p for p in summary["pairs"] if p["construction"] == "within")
    for label in ("antonym", "number_to_words"):
        paths = within["per_label"][label]["paths"]
        assert len(paths) == 1 and paths[0]["seed"] == 1 and len(paths[0]["curve"]) == 11
        assert paths[0]["reading"] in ("connected", "partial", "isolated") and -1 <= paths[0]["cos"] <= 1
    assert any(p["construction"] == "learned" for p in summary["pairs"])

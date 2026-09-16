import json
from pathlib import Path

import numpy as np
import pytest

from directions.cli import main
from directions.geometry import (
    _singular_values_sq,
    covariance_matched_unit_vector,
    d90,
    effective_rank,
    normalize,
    spectrum,
)
from directions.tasks import build_task, number_to_words, present_participle


def test_covariance_matched_control_is_unit_and_in_span():
    rng = np.random.default_rng(0)
    H = rng.standard_normal((20, 50)) @ np.diag(np.linspace(1, 0.1, 50))
    u = covariance_matched_unit_vector(rng, H)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    Hc = H - H.mean(axis=0)
    # u lies in the row space of the centered residuals
    proj = np.linalg.lstsq(Hc.T, u, rcond=None)[0]
    assert np.allclose(Hc.T @ proj, u, atol=1e-8)
    # and is (in expectation) covariance-weighted: heavier first coordinates
    us = np.stack([covariance_matched_unit_vector(rng, H) for _ in range(200)])
    assert np.mean(us[:, :10] ** 2) > np.mean(us[:, -10:] ** 2)


def test_gram_trick_matches_full_svd():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 200)) @ np.diag(np.linspace(1, 0.01, 200))
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    assert np.allclose(_singular_values_sq(Xc)[:30], s**2, rtol=1e-8, atol=1e-8)
    assert effective_rank(X) == pytest.approx((s**2).sum() ** 2 / (s**4).sum())
    sp = spectrum(X, frac=0.9)
    assert sp.d90 == d90(X)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    overlap = np.abs(sp.top_subspace @ vt[: sp.d90].T)
    assert np.allclose(overlap, np.eye(sp.d90), atol=1e-6)
    # tall matrices take the SVD path and agree too
    Y = rng.standard_normal((50, 8))
    assert spectrum(Y).top_subspace.shape[0] == d90(Y)
    assert np.isnan(effective_rank(np.ones((5, 10))))


def test_new_tasks():
    assert present_participle("run") == "running"
    assert present_participle("make") == "making"
    assert present_participle("see") == "seeing"
    assert present_participle("die") == "dying"
    assert present_participle("open") == "opening"
    assert present_participle("begin") == "beginning"
    assert present_participle("fix") == "fixing"
    assert number_to_words(0) == "zero"
    assert number_to_words(21) == "twenty-one"
    assert number_to_words(100) == "one hundred"
    assert number_to_words(347) == "three hundred forty-seven"
    for name, n_min in (("present_participle", 320), ("singular", 320), ("uppercase", 320),
                        ("number_to_words", 1000), ("add_two", 3000)):
        t = build_task(name)
        inputs = [i.input for i in t.items]
        assert len(inputs) == len(set(inputs)) and len(t.items) >= n_min, name
    sing = build_task("singular")
    assert all(i.input != i.output for i in sing.items)
    assert build_task("add_two", {"min": 1, "max": 3}).items[0].input == "1 + 1"
    with pytest.raises(ValueError):
        build_task("add_two", {"foo": 1})


def test_word_lists_fit_the_pilot_pools():
    for name in ("antonym", "plural", "past_tense", "en_fr"):
        assert len(build_task(name).items) >= 64 + 64 + 192, name


def test_aggregate_command(tmp_path):
    """Two fake runs -> per-task qualification rate and scalar mean/sd."""
    for i, (qual, cum) in enumerate(((True, 1.0), (False, 3.0))):
        root = tmp_path / f"run{i}"
        (root / "core").mkdir(parents=True)
        (root / "core" / "summary.json").write_text(json.dumps({
            "run_id": f"run{i}", "seed": i, "model": "m",
            "tasks": {"t": {"qualified": qual, "gates": {"fewshot": True, "steering": qual},
                            "selection": {"layer": 8, "rho": 0.5}, "labels": ["cascade"]}},
        }))
        (root / "core" / "cross_task.json").write_text(json.dumps({
            "signatures": {"t": {"labels": ["cascade"], "cumulative_log_gain": cum, "alignment_final": 0.1,
                                 "by_kind": {"isotropic": {"n": 4, "new_subspace_uncentered_z_mean": 0.5}}}},
            "table": {},
        }))
        (root / "metadata.json").write_text(json.dumps({"function_vector": {"head_count": {"chosen": 8 * (i + 1), "mean_effect_by_k": {"8": 0.5}}}}))
    # the qualified run has per-task files; the core quantities come from them (and the commitment curves are
    # re-summarised under the amended D29, so an older run without the hand-over keys still contributes)
    tdir = tmp_path / "run0" / "core" / "tasks" / "t"
    tdir.mkdir(parents=True)
    (tdir / "qualification.json").write_text(json.dumps({
        "selection": {"layer": 8, "rho": 0.5, "rho_layer_norm": 0.7},
        "fewshot": {"logprob_per_token_mean": -1.0}, "zeroshot_baseline": {"logprob_per_token_mean": -5.0},
        "damage": {"neutral_kl_mean": 0.2, "neutral_gate_excess_mean": 0.01},
        "decomposition": {"cos_with_common": 0.6, "common_part_mean_diff": 1.0, "residual_part_mean_diff": 2.0},
    }))
    (tdir / "evaluation.json").write_text(json.dumps({
        "steering_test": {"mean_diff": 3.0}, "excess_test": {"excess_mean": 2.5}, "steered": {"accuracy": 0.4},
    }))
    rows = lambda vals: [{"read_point": 9 + j, "retained": v} for j, v in enumerate(vals)]  # noqa: E731
    (tdir / "commitment.json").write_text(json.dumps({
        "intervention_layer": 8, "n_read_points": 13, "directions": ["injected", "task_pc1"], "curves": {
            "remove:injected": rows([0.1, 0.6, 0.95, 1.0]), "keep:injected": rows([0.9, 0.5, 0.2, 0.05]),
            "remove:task_pc1": rows([0.9, 0.8, 0.95, 1.0]), "keep:task_pc1": rows([0.1, 0.3, 0.1, 0.0]),
        }}))
    out = tmp_path / "agg.json"
    assert main(["aggregate", str(tmp_path / "run0"), str(tmp_path / "run1"), "--out", str(out)]) == 0
    agg = json.loads(out.read_text())
    t = agg["tasks"]["t"]
    assert t["n_runs"] == 2 and t["n_qualified"] == 1 and t["gates_failed"] == {"steering": 1}
    assert t["scalars"]["cumulative_log_gain"] == {"n": 2, "mean": 2.0, "sd": pytest.approx(np.sqrt(2))}
    assert t["by_kind"]["isotropic"]["new_subspace_uncentered_z_mean"]["mean"] == 0.5
    assert t["labels"] == {"cascade": 2}
    assert [h["chosen"] for h in agg["head_counts"]] == [8, 16]
    core = t["core"]
    assert core["held_out_effect"] == {"n": 1, "mean": 3.0, "sd": 0.0, "values": [3.0]}
    assert core["gap_fraction"]["mean"] == pytest.approx(3.0 / 4.0) and core["layer"]["values"] == [8]
    # hand-over: removal stays >= 50 % from read point 10 (2/4 of the downstream depth) and >= 90 % from 11 on;
    # the direction alone carries >= 50 % up to read point 10
    assert core["handed_over_50_fraction"]["mean"] == pytest.approx(0.5)
    assert core["handed_over_90_fraction"]["mean"] == pytest.approx(0.75)
    assert core["carried_alone_until_50_fraction"]["mean"] == pytest.approx(0.5)
    assert core["retained_after_removal_final"]["mean"] == 1.0 and core["carried_by_direction_final"]["mean"] == 0.05
    assert core["task_pc1_removal_min_retained"]["mean"] == 0.8 and core["task_pc1_alone_max_retained"]["mean"] == 0.3

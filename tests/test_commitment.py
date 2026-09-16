"""Depth of commitment (docs/DECISIONS.md D29): exact edits of the perturbation at later read points."""

import numpy as np
import pytest

from directions.commitment import depth_of_commitment, edit_vectors, random_unit_directions, summarise
from directions.config import CommitmentConfig, ModelConfig, PromptConfig
from directions.model import Intervention, ModelBackend
from directions.prompts import zero_shot_prompt
from directions.tasks import build_task


@pytest.fixture(scope="module")
def backend():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=4), run_seed=5)


@pytest.fixture(scope="module")
def prompts():
    task = build_task("antonym")
    return [zero_shot_prompt(PromptConfig(), it) for it in task.items[:6]]


def test_edit_vectors_split_the_perturbation():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal((5, 8))
    u = rng.standard_normal(8)
    rem, keep = edit_vectors(delta, u, "remove"), edit_vectors(delta, u, "keep")
    # remove + keep edits sum to -delta: applying both leaves the baseline
    assert np.allclose(rem + keep, -delta, atol=1e-5)
    # after the remove edit nothing is left along u; after the keep edit only the u component remains
    uu = u / np.linalg.norm(u)
    assert np.allclose((delta + rem) @ uu, 0.0, atol=1e-5)
    assert np.allclose(delta + keep, ((delta @ uu)[:, None] * uu[None, :]), atol=1e-5)
    assert np.allclose(np.linalg.norm(random_unit_directions(rng, 3, 8), axis=1), 1.0)
    with pytest.raises(ValueError):
        edit_vectors(delta, u, "drop")


def test_edits_land_on_the_captured_residuals_exactly(backend, prompts):
    """The edited run's residual at m equals the intended edit of the captured steered residual (determinism)."""
    layer, alpha = 1, 2.0
    v = np.random.default_rng(1).standard_normal(backend.hidden_size).astype(np.float32)
    v /= np.linalg.norm(v)
    base = backend.run(prompts, capture=True)
    steered = backend.run(prompts, interventions=[Intervention(layer, v, alpha)], capture=True)
    delta = steered.residuals.astype(np.float64) - base.residuals.astype(np.float64)
    m = layer + 1
    for variant in ("remove", "keep"):
        vec = edit_vectors(delta[m], v, variant)
        r = backend.run(prompts, interventions=[Intervention(layer, v, alpha), Intervention(m, vec, 1.0)], capture=True)
        target = steered.residuals[m].astype(np.float64) + vec.astype(np.float64)
        assert np.allclose(r.residuals[m], target, atol=1e-4)
        assert np.array_equal(r.residuals[layer], steered.residuals[layer])  # untouched before m
    # removing the direction at the injection point itself undoes the injection: the baseline comes back
    vec = edit_vectors(delta[layer], v, "remove")
    r = backend.run(prompts, interventions=[Intervention(layer, v, alpha), Intervention(layer, vec, 1.0)])
    assert np.allclose(r.logprob_per_token, base.logprob_per_token, atol=1e-4)


def test_depth_of_commitment_sweeps_every_later_read_point(backend, prompts):
    layer, alpha = 1, 2.0
    v = np.random.default_rng(2).standard_normal(backend.hidden_size).astype(np.float32)
    v /= np.linalg.norm(v)
    base = backend.run(prompts, capture=True)
    steered = backend.run(prompts, interventions=[Intervention(layer, v, alpha)], capture=True)
    pc1 = np.random.default_rng(3).standard_normal((backend.n_layers + 1, backend.hidden_size))
    pc1 /= np.linalg.norm(pc1, axis=1, keepdims=True)
    cfg = CommitmentConfig(n_prompts=0, n_controls=2, alpha=0.05, n_boot=50)
    res = depth_of_commitment(backend, prompts, base, steered, layer, v, alpha, pc1, cfg,
                              np.random.default_rng(10), np.random.default_rng(11))
    assert res["read_points"] == list(range(layer + 1, backend.n_layers + 1)) and res["n_prompts"] == 6
    assert set(res["curves"]) == {"remove:injected", "keep:injected", "remove:task_pc1", "keep:task_pc1"}
    for rows in res["curves"].values():
        assert [r["read_point"] for r in rows] == res["read_points"]
        for r in rows:
            assert set(r) >= {"effect", "retained", "ci_low", "ci_high", "cost_vs_full", "cost_test", "random_effect_mean", "excess_vs_random"}
            assert r["retained"] == pytest.approx(r["effect"] / res["full_effect"])
    # remove + keep at the same read point along the same direction: the two edited effects need not sum, but
    # removing everything but a random component leaves ~nothing, and removing a random component ~everything
    last = res["curves"]["remove:injected"][-1]
    assert abs(last["random_retained_mean"] - 1.0) < 0.5
    s = res["summary"]["injected"]
    assert set(s) >= {"handed_over_50", "handed_over_50_fraction", "handed_over_90", "carried_alone_until_50",
                      "carried_alone_until_90_fraction", "needed_until", "commitment_layer", "commitment_fraction",
                      "needed_at_end", "retained_after_removal_final", "carried_by_direction_final", "carried_until"}
    assert res["handover_shares"] == [0.5, 0.9]
    assert s["needed_until"] is None or layer < s["needed_until"] <= backend.n_layers
    # deterministic in its seeds
    again = depth_of_commitment(backend, prompts, base, steered, layer, v, alpha, pc1, cfg,
                                np.random.default_rng(10), np.random.default_rng(11))
    assert again["curves"]["keep:injected"][0]["excess_vs_random"] == res["curves"]["keep:injected"][0]["excess_vs_random"]


def test_summarise_reads_the_needed_and_carried_read_points():
    def row(m, remove_p, keep_p, retained):
        return {"read_point": m, "retained": retained,
                "excess_vs_random": {"excess_mean": 0.5, "p_value": remove_p}} if keep_p is None else \
               {"read_point": m, "retained": retained, "excess_vs_random": {"excess_mean": 0.5, "p_value": keep_p}}
    out = {"intervention_layer": 2, "n_read_points": 7, "directions": ["injected"], "curves": {
        "remove:injected": [row(3, 0.01, None, 0.2), row(4, 0.02, None, 0.5), row(5, 0.4, None, 0.9), row(6, 0.6, None, 0.95)],
        "keep:injected": [row(3, None, 0.01, 0.9), row(4, None, 0.03, 0.6), row(5, None, 0.5, 0.1), row(6, None, 0.7, 0.05)],
    }}
    s = summarise(out, 0.05)["injected"]
    # hand-over depths (effect sizes): removal leaves >= 50 % from read point 4 on and >= 90 % from 5 on; the
    # direction alone carries >= 50 % up to read point 4 and >= 90 % only at read point 3
    assert s["handed_over_50"] == 4 and s["handed_over_50_fraction"] == pytest.approx(2 / 4)
    assert s["handed_over_90"] == 5 and s["handed_over_90_fraction"] == pytest.approx(3 / 4)
    assert s["carried_alone_until_50"] == 4 and s["carried_alone_until_90"] == 3
    assert s["carried_alone_until_90_fraction"] == pytest.approx(1 / 4)
    assert s["needed_until"] == 4 and s["commitment_layer"] == 5 and s["commitment_fraction"] == pytest.approx(3 / 4)
    assert not s["needed_at_end"] and s["retained_after_removal_final"] == 0.95 and s["retained_after_removal_min"] == 0.2
    assert s["carried_until"] == 4 and s["carried_by_direction_final"] == 0.05 and s["carried_by_direction_max"] == 0.9
    # a dip below the share after a crossing moves the hand-over to after the dip; never crossing gives None
    out["curves"]["remove:injected"][2]["retained"] = 0.4
    s = summarise(out, 0.05)["injected"]
    assert s["handed_over_50"] == 6 and s["handed_over_90"] == 6
    out["curves"]["remove:injected"][3]["retained"] = 0.3
    out["curves"]["keep:injected"][0]["retained"] = 0.2
    s = summarise(out, 0.05, [0.5])["injected"]
    assert s["handed_over_50"] is None and s["carried_alone_until_50"] is None and "handed_over_90" not in s
    out["curves"]["remove:injected"][2]["retained"] = 0.9
    out["curves"]["remove:injected"][3]["retained"] = 0.95
    out["curves"]["keep:injected"][0]["retained"] = 0.9
    # needed at the very end: no commitment layer
    out["curves"]["remove:injected"][-1]["excess_vs_random"]["p_value"] = 0.01
    s = summarise(out, 0.05)["injected"]
    assert s["needed_until"] == 6 and s["commitment_layer"] is None and s["needed_at_end"]
    # never needed: commitment at the first read point after the injection
    for r in out["curves"]["remove:injected"]:
        r["excess_vs_random"]["p_value"] = 0.9
    s = summarise(out, 0.05)["injected"]
    assert s["needed_until"] is None and s["commitment_layer"] == 3 and s["commitment_fraction"] == pytest.approx(1 / 4)

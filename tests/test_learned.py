"""Learned single vector (docs/DECISIONS.md D31): the gradient it optimises, the fit, its null, the smoke run."""
import json
from pathlib import Path

import numpy as np
import pytest

from directions.config import LearnedVectorConfig, ModelConfig, PromptConfig, load_config
from directions.learned import fit_vector, learned_directions, permuted_target_vectors
from directions.model import Intervention, ModelBackend
from directions.pipeline import run_pipeline
from directions.prompts import zero_shot_prompt
from directions.tasks import build_task

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def backend():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16), run_seed=1)


@pytest.fixture(scope="module")
def prompts():
    task = build_task("antonym")
    return [zero_shot_prompt(PromptConfig(), x) for x in task.items[:12]]


def test_gradient_wrt_added_vector_matches_finite_differences(backend, prompts):
    """The activation gradient at the injection read point is the gradient of the summed scored log p with
    respect to the added vector: checked against a central finite difference along a random direction."""
    layer = 2
    rng = np.random.default_rng(0)
    v = rng.normal(size=backend.hidden_size)
    u = rng.normal(size=backend.hidden_size)
    u /= np.linalg.norm(u)
    grads, scores = backend.gradients_with_scores(prompts, interventions=[Intervention(layer, v, 1.0)], read_points=[layer])
    assert grads.shape == (1, len(prompts), backend.hidden_size) and scores.shape == (len(prompts),)
    # the scores of the same pass equal a plain steered forward pass
    plain = backend.run(prompts, interventions=[Intervention(layer, v, 1.0)])
    assert np.allclose(scores, plain.logprob_sum, atol=1e-5)
    eps = 1e-3
    up = backend.run(prompts, interventions=[Intervention(layer, v + eps * u, 1.0)]).logprob_sum
    down = backend.run(prompts, interventions=[Intervention(layer, v - eps * u, 1.0)]).logprob_sum
    fd = (up - down) / (2 * eps)
    analytic = grads[0].astype(np.float64) @ u
    assert np.allclose(analytic, fd, atol=2e-3, rtol=5e-2), (analytic[:3], fd[:3])
    # all read points by default, in order
    full = backend.gradients(prompts, interventions=[Intervention(layer, v, 1.0)])
    assert full.shape[0] == backend.n_layers + 1 and np.allclose(full[layer], grads[0])


def test_fit_lowers_the_loss_keeps_the_radius_and_is_deterministic(backend, prompts):
    cfg = LearnedVectorConfig(n_steps=12, lr_fraction=0.1, log_every=4)
    radius = 3.0
    fit = fit_vector(backend, prompts, layer=2, radius=radius, rng=np.random.default_rng(5), cfg=cfg)
    assert fit.loss_steps == [0, 4, 8, 12] and len(fit.losses) == 4 and fit.n_steps == 12
    assert np.linalg.norm(fit.vector) == pytest.approx(radius, rel=1e-9)
    assert fit.losses[-1] < fit.losses[0]  # the fitted vector raises the summed log p over the pool
    # the final loss is the pool's loss under the fitted vector
    steered = backend.run(prompts, interventions=[Intervention(2, fit.vector, 1.0)])
    assert fit.losses[-1] == pytest.approx(float(-np.mean(steered.logprob_sum)), abs=1e-5)
    again = fit_vector(backend, prompts, layer=2, radius=radius, rng=np.random.default_rng(5), cfg=cfg)
    assert np.array_equal(again.vector, fit.vector) and again.losses == fit.losses
    other = fit_vector(backend, prompts, layer=2, radius=radius, rng=np.random.default_rng(6), cfg=cfg)
    assert not np.array_equal(other.vector, fit.vector)
    # an explicit initialisation is normalised to the radius
    init = fit_vector(backend, prompts, layer=2, radius=radius, rng=np.random.default_rng(5),
                      cfg=LearnedVectorConfig(n_steps=0), init=np.ones(backend.hidden_size))
    assert np.allclose(init.vector, radius * np.ones(backend.hidden_size) / np.sqrt(backend.hidden_size))


def test_learned_directions_and_permuted_null(backend):
    task = build_task("antonym")
    pool = task.items[:10]
    cfg = LearnedVectorConfig(n_steps=3, lr_fraction=0.1, log_every=1)
    radii = {1: 2.0, 3: 2.5}
    dirs, fits = learned_directions(backend, PromptConfig(), pool, [1, 3], radii, run_seed=7, task_name="antonym", cfg=cfg, n_seeds=2)
    for l in (1, 3):
        d = dirs[l]
        assert d.kind == "learned_vector" and d.layer == l and d.mean_difference_norm == radii[l]
        assert np.linalg.norm(d.direction) == pytest.approx(1.0) and d.seed_directions.shape == (2, backend.hidden_size)
        assert -1.0 <= d.stability <= 1.0 and d.stability == pytest.approx(float(d.seed_directions[0] @ d.seed_directions[1]))
        assert len(fits[l]) == 2 and all(f.n_steps == 3 for f in fits[l])
    nulls = permuted_target_vectors(backend, PromptConfig(), pool, 1, radii[1], run_seed=7, task_name="antonym", cfg=cfg, n=2)
    assert len(nulls) == 2 and all(np.linalg.norm(u) == pytest.approx(1.0) for u in nulls)
    assert not np.allclose(nulls[0], nulls[1])


@pytest.fixture(scope="module")
def smoke_learned_run(tmp_path_factory):
    cfg = load_config(CONFIGS / "smoke_toy_learned.yaml")
    cfg.output_dir = str(tmp_path_factory.mktemp("results"))
    return run_pipeline(cfg, "pilot", config_path=str(CONFIGS / "smoke_toy_learned.yaml"))


def test_smoke_learned_vector_outputs(smoke_learned_run):
    root = smoke_learned_run
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["control"] == "learned_vector" and "function_vector" not in meta
    summary = json.loads((root / "core" / "summary.json").read_text())
    assert summary["control"] == "learned_vector"
    for task in ("antonym", "add_3", "number_to_words"):
        d = root / "core" / "tasks" / task
        ext = json.loads((d / "extraction.json").read_text())
        lv = ext["learned_vector"]
        n_seeds = lv["n_seeds"]
        assert lv["n_steps"] == 4 and n_seeds >= 2 and set(lv["layers"]) == set(lv["stability"])
        for l, info in lv["layers"].items():
            assert info["loss_steps"] == [0, 2, 4] and len(info["losses"]) == n_seeds and len(info["final_loss"]) == n_seeds
            assert info["radius"] > 0 and all(f < i for f, i in zip(info["final_loss"], info["initial_loss"]))
        assert "cos_with_pca" in lv and lv["demo_variation"]["construction"].startswith("vectors fitted to permuted targets")
        arrays = np.load(d / "directions.npz")
        assert "learned" in arrays.files and "learned_per_seed" in arrays.files and "learned_radii" in arrays.files
        assert arrays["learned"].shape[0] == len(arrays["layers"]) and "fv" not in arrays.files
        q = json.loads((d / "qualification.json").read_text())
        assert "head_support" not in q["gates"] and "stability" in q["gates"] and "pca_stability" in q
        assert set(q["stability"]) == {str(l) for l in arrays["layers"]}
        if q.get("selection"):
            lvq = q["learned_vector"]
            assert lvq["natural_norm"] > 0 and str(q["selection"]["layer"]) in lvq["natural_rho"]
            assert len(lvq["final_loss"]) == n_seeds and summary["tasks"][task]["learned_vector"] == lvq
            ev = json.loads((d / "evaluation.json").read_text())
            kinds = {c["kind"] for c in ev["controls"]} if "controls" in ev else set(ev["by_kind_excess"])
            assert "demo_variation" in kinds  # the permuted-target null was fitted at the selected layer
            if (d / "decomposition.json").exists():
                dec = json.loads((d / "decomposition.json").read_text())
                assert dec["common_direction"] is None or "cos_with_fv" in dec["common_direction"]

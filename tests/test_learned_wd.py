"""The weight-decayed learned vector (docs/DECISIONS.md D42): the optimiser against a known minimiser, the fit on the
toy model, its directions and null, the smoke run."""
import json
from pathlib import Path

import numpy as np
import pytest

from directions.config import LearnedVectorWDConfig, ModelConfig, PromptConfig, load_config
from directions.learned import (adam_regularised, fit_vector_wd, learned_directions_wd, permuted_target_vectors_wd,
                                regularised_objective, regularised_step_schedule, wd_init)
from directions.model import Intervention, ModelBackend
from directions.pipeline import run_pipeline
from directions.prompts import zero_shot_prompt
from directions.tasks import build_task

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_step_schedule_is_a_cosine_decay_from_the_scaled_step():
    lrs = regularised_step_schedule(0.02, 10.0, 400, 100)
    assert lrs.shape == (100,) and lrs[0] == pytest.approx(0.02 * 10.0 / 20.0)
    assert np.all(np.diff(lrs) < 0) and lrs[-1] < 1e-3 * lrs[0]


def test_adam_reaches_the_penalised_minimiser_of_a_quadratic_from_any_start():
    """On f(v) = 1/2 (v - a)^T H (v - a) the penalised objective has the unique minimiser
    (H + lambda / r^2 I)^{-1} H a; the optimiser reaches it from zero and from a random start, and the
    stationarity it reports is small there."""
    rng = np.random.default_rng(0)
    d, radius, lam = 24, 5.0, 2.0
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    H = Q @ np.diag(np.linspace(0.05, 1.0, d)) @ Q.T
    a = rng.normal(size=d) * 2.0

    def grad_fn(v):
        r = v - a
        return 0.5 * float(r @ H @ r), H @ r

    expected = np.linalg.solve(H + lam / radius ** 2 * np.eye(d), H @ a)
    n = 3000
    lrs = regularised_step_schedule(0.05, radius, d, n)
    for v0 in (np.zeros(d), 2.0 * rng.normal(size=d)):
        v, steps, losses, objectives, norms, stationarity = adam_regularised(grad_fn, v0, radius, lam, n, lrs, log_every=500)
        assert np.allclose(v, expected, atol=2e-3), np.abs(v - expected).max()
        assert stationarity < 1e-2
        assert steps == [0, 500, 1000, 1500, 2000, 2500, 3000]
        assert objectives[-1] == pytest.approx(regularised_objective(losses[-1], v, radius, lam))
        assert norms[-1] == pytest.approx(np.linalg.norm(v) / radius)
    # a larger penalty gives a shorter minimiser
    v_big, *_ = adam_regularised(grad_fn, np.zeros(d), radius, 20 * lam, n, lrs)
    assert np.linalg.norm(v_big) < np.linalg.norm(expected)


def test_init_is_zero_for_the_first_seed_and_random_at_the_fraction_otherwise():
    rng = np.random.default_rng(3)
    assert np.array_equal(wd_init(0, rng, 8, 4.0, 0.5), np.zeros(8))
    v = wd_init(1, rng, 8, 4.0, 0.5)
    assert np.linalg.norm(v) == pytest.approx(2.0)


@pytest.fixture(scope="module")
def backend():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16), run_seed=1)


@pytest.fixture(scope="module")
def prompts():
    task = build_task("antonym")
    return [zero_shot_prompt(PromptConfig(), x) for x in task.items[:12]]


def test_fit_from_zero_lowers_the_loss_finds_its_own_norm_and_is_deterministic(backend, prompts):
    cfg = LearnedVectorWDConfig(n_steps=20, step_fraction=0.1, weight_decay=0.5, log_every=5)
    radius = 3.0
    zero = np.zeros(backend.hidden_size)
    fit = fit_vector_wd(backend, prompts, layer=2, radius=radius, init=zero, cfg=cfg)
    assert fit.loss_steps == [0, 5, 10, 15, 20] and len(fit.losses) == len(fit.objectives) == len(fit.norms) == 5
    assert fit.norms[0] == 0.0 and fit.norm > 0 and fit.norm != pytest.approx(radius)
    assert fit.losses[-1] < fit.losses[0] and fit.objectives[-1] < fit.objectives[0]
    # the logged loss at the end is the pool's loss under the fitted vector, and the objective adds the penalty
    steered = backend.run(prompts, interventions=[Intervention(2, fit.vector, 1.0)])
    assert fit.losses[-1] == pytest.approx(float(-np.mean(steered.logprob_sum)), abs=1e-5)
    assert fit.objectives[-1] == pytest.approx(fit.losses[-1] + 0.5 * 0.5 * fit.norm ** 2 / radius ** 2)
    assert np.isfinite(fit.stationarity)
    again = fit_vector_wd(backend, prompts, layer=2, radius=radius, init=zero, cfg=cfg)
    assert np.array_equal(again.vector, fit.vector) and again.losses == fit.losses
    # a stronger penalty finds a shorter vector
    strong = fit_vector_wd(backend, prompts, layer=2, radius=radius, init=zero,
                           cfg=LearnedVectorWDConfig(n_steps=20, step_fraction=0.1, weight_decay=50.0, log_every=5))
    assert strong.norm < fit.norm


def test_learned_directions_wd_and_permuted_null(backend):
    task = build_task("antonym")
    pool = task.items[:10]
    cfg = LearnedVectorWDConfig(n_steps=4, step_fraction=0.1, weight_decay=1.0, log_every=1)
    radii = {1: 2.0, 3: 2.5}
    dirs, fits = learned_directions_wd(backend, PromptConfig(), pool, [1, 3], radii, run_seed=7, task_name="antonym",
                                       cfg=cfg, n_seeds=3)
    for l in (1, 3):
        d = dirs[l]
        assert d.kind == "learned_vector_wd" and d.layer == l
        assert d.mean_difference_norm == pytest.approx(fits[l][0].norm)  # the natural norm is the norm the fit found
        assert np.linalg.norm(d.direction) == pytest.approx(1.0) and d.seed_directions.shape == (3, backend.hidden_size)
        assert np.array_equal(d.direction, d.seed_directions[0])  # the carried direction is the zero-start fit
        assert np.array_equal(fits[l][0].init, np.zeros(backend.hidden_size))
        assert all(np.linalg.norm(f.init) == pytest.approx(0.5 * radii[l]) for f in fits[l][1:])
        cos = d.seed_directions @ d.seed_directions.T
        assert d.stability == pytest.approx(min(cos[0, 1], cos[0, 2], cos[1, 2]))
    nulls = permuted_target_vectors_wd(backend, PromptConfig(), pool, 1, radii[1], run_seed=7, task_name="antonym", cfg=cfg, n=2)
    assert len(nulls) == 2 and all(np.linalg.norm(u) == pytest.approx(1.0) for u in nulls)
    assert not np.allclose(nulls[0], nulls[1])


@pytest.fixture(scope="module")
def smoke_wd_run(tmp_path_factory):
    cfg = load_config(CONFIGS / "smoke_toy_learned_wd.yaml")
    cfg.output_dir = str(tmp_path_factory.mktemp("results"))
    return run_pipeline(cfg, "pilot", config_path=str(CONFIGS / "smoke_toy_learned_wd.yaml"))


def test_smoke_learned_vector_wd_outputs(smoke_wd_run):
    root = smoke_wd_run
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["control"] == "learned_vector_wd" and "function_vector" not in meta
    summary = json.loads((root / "core" / "summary.json").read_text())
    assert summary["control"] == "learned_vector_wd"
    n_selected = 0
    for task in ("antonym", "add_3", "number_to_words"):
        d = root / "core" / "tasks" / task
        ext = json.loads((d / "extraction.json").read_text())
        lv = ext["learned_vector"]
        assert lv["construction"] == "weight_decay" and lv["n_steps"] == 6 and lv["weight_decay"] == 1.0
        n_seeds = lv["n_seeds"]
        for l, info in lv["layers"].items():
            assert info["loss_steps"] == [0, 2, 4, 6] and len(info["losses"]) == n_seeds and len(info["norms"]) == n_seeds
            assert info["norms"][0][0] == 0.0 and info["fitted_norm_over_radius"][0] > 0
            assert info["final_loss"][0] < info["initial_loss"][0]
            assert info["cos_with_init"][0] is None and all(np.isfinite(info["cos_with_init"][1:]))  # nan is written as null
            assert lv["fitted_norm_over_radius"][l] == pytest.approx(info["fitted_norm_over_radius"][0])
        arrays = np.load(d / "directions.npz")
        assert {"learned", "learned_per_seed", "learned_radii", "learned_reference_radii"} <= set(arrays.files)
        # the natural strength downstream commands fall back to is the fitted norm, not the median residual norm
        assert np.allclose(arrays["learned_radii"] / arrays["learned_reference_radii"],
                           [lv["fitted_norm_over_radius"][str(int(l))] for l in arrays["layers"]])
        q = json.loads((d / "qualification.json").read_text())
        assert "head_support" not in q["gates"] and "stability" not in q["gates"]
        if q.get("selection"):
            n_selected += 1
            lvq = q["learned_vector"]
            assert lvq["construction"] == "weight_decay" and lvq["fitted_norm_over_radius"] > 0
            assert lvq["natural_norm"] == pytest.approx(lvq["fitted_norm_over_radius"] * arrays["learned_reference_radii"][
                list(arrays["layers"]).index(q["selection"]["layer"])])
            ev = json.loads((d / "evaluation.json").read_text())
            kinds = {c["kind"] for c in ev["controls"]} if "controls" in ev else set(ev["by_kind_excess"])
            assert "demo_variation" in kinds  # the permuted-target null fitted with the same penalty
    assert n_selected > 0


def test_learned_vector_d31_is_unchanged_by_d42():
    """The D31 construction keeps its defaults and its fixed norm: D42 adds a control, it does not refactor D31."""
    cfg = load_config(CONFIGS / "learned_qwen3_0.6b.yaml")
    assert cfg.extraction.control == "learned_vector"
    assert cfg.extraction.learned_vector.n_steps == 100 and cfg.extraction.learned_vector.lr_fraction == 0.05


def test_comparison_script_on_the_three_smoke_runs(smoke_wd_run, tmp_path):
    """scripts/learned_wd_compare.py reads the weight-decayed, the D31 and the head-mean smoke runs of one seed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("learned_wd_compare", Path(__file__).resolve().parents[1] / "scripts" / "learned_wd_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    roots = {}
    for key, fname in (("learned", "smoke_toy_learned.yaml"), ("fv", "smoke_toy_fv.yaml")):
        cfg = load_config(CONFIGS / fname)
        cfg.output_dir = str(tmp_path / key)
        roots[key] = run_pipeline(cfg, "pilot", config_path=str(CONFIGS / fname))
    res = mod.compare(smoke_wd_run, roots["learned"], roots["fv"])
    assert set(res["tasks"]) == {"antonym", "add_3", "number_to_words"}
    for row in res["tasks"].values():
        assert set(row) == {"wd", "learned", "fv", "geometry"}
        for g in row["geometry"].values():
            assert -1 <= g["cos_d31"] <= 1 and -1 <= g["cos_pca"] <= 1 and g["fitted_norm_over_radius"] > 0
            if "cos_fv" in g:
                assert -1 <= g["cos_fv"] <= 1 and -1 <= g["cos_d31_fv"] <= 1
    assert mod.markdown(res).count("\n") == len(res["tasks"]) + 1
    with pytest.raises(SystemExit):
        mod.compare(roots["learned"], smoke_wd_run, roots["fv"])  # the runs in the wrong roles

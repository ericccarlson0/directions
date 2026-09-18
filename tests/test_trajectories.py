"""All-to-all trajectory comparison (docs/DECISIONS.md D32): geometry, strengths, curve labels, the smoke run."""
import json
from pathlib import Path

import numpy as np
import pytest

from directions.cli import main
from directions.config import ModelConfig, PromptConfig, load_config, load_trajectories_config
from directions.stats import bootstrap_median_ci_rows
from directions.model import ModelBackend
from directions.pipeline import run_pipeline
from directions.prompts import deranged_prompt, few_shot_prompt
from directions.seeds import rng_for
from directions.tasks import build_task
from directions.trajectories import (
    Geometry,
    alpha_from_calibration,
    coherence_curve,
    mean_cosines,
    pair_cosines,
    projection_fraction,
    remove_direction,
    summarise_alignment,
    unit_rows,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _naive_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_pair_cosines_match_a_naive_loop_and_mark_undefined_points():
    rng = np.random.default_rng(0)
    L1, n, d = 5, 7, 6
    A, B = rng.normal(size=(L1, n, d)), rng.normal(size=(L1, n, d))
    A[:2] = 0.0  # below the injection layer the perturbation vanishes
    cos = pair_cosines(A, B, start=2)
    assert cos.shape == (L1, n) and np.isnan(cos[:2]).all() and not np.isnan(cos[2:]).any()
    for m in range(2, L1):
        for i in range(n):
            assert cos[m, i] == pytest.approx(_naive_cos(A[m, i], B[m, i]))
    # start=0 with vanishing rows: nan where either vector is zero, the same values elsewhere
    cos0 = pair_cosines(A, B, start=0)
    assert np.isnan(cos0[:2]).all() and np.allclose(cos0[2:], cos[2:])
    # with a unit direction removed from both vectors, the cosine is that of the projections
    U = rng.normal(size=(n, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    cos_na = pair_cosines(A, B, start=2, U=U)
    for i in range(n):
        a = A[3, i] - (A[3, i] @ U[i]) * U[i]
        b = B[3, i] - (B[3, i] @ U[i]) * U[i]
        assert cos_na[3, i] == pytest.approx(_naive_cos(a, b))
        assert abs(remove_direction(A[3], U)[i] @ U[i]) < 1e-12
    # one set of unit rows per read point removes, at read point m, that read point's rows
    U3 = np.stack([U, -U, U, U, -U])
    cos_na3 = pair_cosines(A, B, start=2, U=U3)
    assert np.allclose(cos_na3[2:], cos_na[2:])
    W = rng.normal(size=(L1, n, d))
    W[1, 0] = 0.0
    assert np.allclose(unit_rows(W)[1, 0], 0.0) and np.allclose(np.linalg.norm(unit_rows(W)[2], axis=1), 1.0)
    cos_w = pair_cosines(A, B, start=2, U=unit_rows(W))
    for i in range(n):
        u = W[4, i] / np.linalg.norm(W[4, i])
        assert cos_w[4, i] == pytest.approx(_naive_cos(A[4, i] - (A[4, i] @ u) * u, B[4, i] - (B[4, i] @ u) * u))
    frac = projection_fraction(A, A, start=2)
    assert np.allclose(frac[2:], 1.0) and np.isnan(frac[:2]).all()
    assert projection_fraction(2 * A, A, start=2)[3, 0] == pytest.approx(2.0)
    mc = mean_cosines(A.mean(axis=1), A.mean(axis=1), start=2)
    assert np.allclose(mc[2:], 1.0) and np.isnan(mc[:2]).all()


def test_coherence_curve_is_one_for_identical_and_near_zero_for_unrelated_pushes():
    rng = np.random.default_rng(3)
    L1, n, d = 4, 5, 200
    base = rng.normal(size=(L1, n, d))
    same = coherence_curve([base, base.copy(), base.copy()], start=1)
    assert same[0] is None and all(v == pytest.approx(1.0) for v in same[1:])
    unrelated = coherence_curve([rng.normal(size=(L1, n, d)) for _ in range(8)], start=1)
    assert all(0.2 < v < 0.5 for v in unrelated[1:])  # ~ 1/sqrt(8) for independent directions
    # a shared component plus noise: between the two
    shared = rng.normal(size=(L1, n, d))
    mixed = coherence_curve([shared + 0.5 * rng.normal(size=(L1, n, d)) for _ in range(8)], start=1)
    assert all(0.8 < v < 0.95 for v in mixed[1:])


def test_torch_geometry_matches_the_numpy_reference():
    """The device path (D33) reproduces the NumPy reference: cosines with and without removals, projection
    fractions, coherence, and the row-wise bootstrap (medians exactly, the CIs statistically)."""
    rng = np.random.default_rng(4)
    L1, n, d = 6, 40, 24
    A = rng.normal(size=(L1, n, d)).astype(np.float32)
    B = rng.normal(size=(L1, n, d)).astype(np.float32)
    A[:2] = 0.0
    U2 = unit_rows(rng.normal(size=(n, d)))
    U3 = unit_rows(rng.normal(size=(L1, n, d)))
    U3[1, 3] = 0.0
    geo = Geometry("cpu", seed=11)
    At, Bt = geo.tensor(A), geo.tensor(B)
    for U, Ut in ((None, None), (U2, geo.tensor(U2, geo.torch.float64)), (U3, geo.tensor(U3, geo.torch.float64))):
        ref = pair_cosines(A, B, 2, U)
        got = geo.pair_cosines(At, Bt, 2, Ut)
        assert np.isnan(got[:2]).all() and np.allclose(got[2:], ref[2:], atol=1e-9)
    assert np.allclose(geo.projection_fraction(At, Bt, 2)[2:], projection_fraction(A, B, 2)[2:], atol=1e-9)
    assert np.allclose(geo.unit_rows(geo.tensor(U3 * 3.0, geo.torch.float64)).cpu().numpy(), U3, atol=1e-12)
    deltas = [rng.normal(size=(L1, n, d)).astype(np.float32) for _ in range(5)]
    ref_c = coherence_curve(deltas, 1)
    got_c = geo.coherence([geo.tensor(x) for x in deltas], 1)
    assert got_c[0] is None and all(abs(a - b) < 1e-9 for a, b in zip(got_c[1:], ref_c[1:]))
    assert np.allclose(geo.mean_over_examples(At), A.astype(np.float64).mean(1))
    # bootstrap: identical medians and counts, CIs within sampling noise of the NumPy draws
    X = rng.normal(size=(L1, n))
    X[:2] = np.nan
    X[3, 5] = np.nan
    ref_b = bootstrap_median_ci_rows(X, np.random.default_rng(0), n_boot=3000, alpha=0.05)
    got_b = geo.bootstrap_rows(X, n_boot=3000, alpha=0.05)
    assert got_b["n"] == ref_b["n"] and np.allclose(np.nan_to_num(got_b["median"]), np.nan_to_num(ref_b["median"]))
    for k in ("low", "high"):
        assert np.allclose(np.nan_to_num(got_b[k]), np.nan_to_num(ref_b[k]), atol=0.12)
    assert all(lo <= med <= hi for lo, med, hi in zip(got_b["low"][2:], got_b["median"][2:], got_b["high"][2:]))
    # single-entry rows and the generator's determinism
    Y = np.full((2, 3), np.nan)
    Y[1, 0] = 0.7
    single = geo.bootstrap_rows(Y, 100, 0.05)
    assert single["median"][1] == 0.7 and single["low"][1] == 0.7 and single["n"] == [0, 1]
    g1, g2 = Geometry("cpu", seed=5), Geometry("cpu", seed=5)
    b1, b2 = g1.bootstrap_rows(X, 200, 0.05), g2.bootstrap_rows(X, 200, 0.05)
    assert all(np.array_equal(np.nan_to_num(b1[k]), np.nan_to_num(b2[k])) for k in ("median", "low", "high", "n"))


def test_alpha_from_calibration_follows_the_runs_rule():
    cal = {
        "selected": {"layer": 4, "alpha": 10.0, "rho": 1.0},
        "reference_rho": 1.0,
        "strength_units": {"2": 5.0, "4": 10.0, "6": 20.0},
        "grid": [
            {"layer": 2, "rho": 0.5, "alpha": 2.5, "reliable": True},
            {"layer": 2, "rho": 2.0, "alpha": 10.0, "reliable": True},  # same log distance to 1.0: the weaker wins
            {"layer": 2, "rho": 1.5, "alpha": 7.5, "reliable": False},
            {"layer": 6, "rho": 1.0, "alpha": 20.0, "reliable": False},
        ],
    }
    assert alpha_from_calibration(cal, 4) == {"alpha": 10.0, "rho": 1.0, "source": "selected"}
    assert alpha_from_calibration(cal, 2) == {"alpha": 2.5, "rho": 0.5, "source": "reliable_grid_point"}
    assert alpha_from_calibration(cal, 6) == {"alpha": 20.0, "rho": 1.0, "source": "natural_norm"}
    assert alpha_from_calibration(cal, 8) is None and alpha_from_calibration(None, 2) is None
    # without a reference rho the weakest reliable point is taken
    assert alpha_from_calibration({**cal, "reference_rho": None, "selected": None}, 2)["alpha"] == 2.5


def _curve(values: list[float], n: int, rng: np.random.Generator, noise: float = 0.05) -> np.ndarray:
    return np.array(values)[:, None] + noise * rng.normal(size=(len(values), n))


def test_summarise_alignment_labels_the_three_shapes():
    rng = np.random.default_rng(1)
    n, start = 60, 1
    floor = _curve([0.0] * 6, 4 * n, rng)
    floor[0] = np.nan
    converging = _curve([np.nan, 0.0, 0.2, 0.4, 0.6, 0.6], n, rng)
    transient = _curve([np.nan, 0.0, 0.5, 0.5, 0.2, 0.0], n, rng)
    flat = _curve([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0], n, rng)
    partly = _curve([np.nan, 0.0, 0.6, 0.6, 0.3, 0.3], n, rng)
    s = summarise_alignment(converging, floor, start, np.random.default_rng(2), 300, 0.05)
    assert s["label"] == "converges" and s["peak_read_point"] in (4, 5) and s["final"] == pytest.approx(0.6, abs=0.05)
    assert s["at_injection"] == pytest.approx(0.0, abs=0.05) and 0 <= s["peak_depth_fraction"] <= 1
    assert len(s["curve"]["median"]) == 6 and len(s["above_floor"]) == 6 and s["above_floor"][5]
    s = summarise_alignment(transient, floor, start, np.random.default_rng(2), 300, 0.05)
    assert s["label"] == "aligns_then_diverges" and s["peak_read_point"] in (2, 3) and s["decline_from_peak"]["p_value"] < 0.05
    s = summarise_alignment(flat, floor, start, np.random.default_rng(2), 300, 0.05)
    assert s["label"] == "never_aligns" and not s["above_floor_at_final"]
    s = summarise_alignment(partly, floor, start, np.random.default_rng(2), 300, 0.05)
    assert s["label"] == "aligns_then_partly_diverges"
    # an all-nan curve is undefined rather than an error
    assert summarise_alignment(np.full((6, n), np.nan), floor, start, rng, 50, 0.05)["label"] == "undefined"


@pytest.fixture(scope="module")
def backend():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16), run_seed=1)


def test_unembedding_directions_and_deranged_prompts(backend):
    task = build_task("antonym")
    cfg = PromptConfig(n_shots=3)
    prompts = [few_shot_prompt(cfg, task.items[:20], task.items[i], rng_for(1, i)) for i in range(4)]
    ids = backend.first_target_token_ids(prompts)
    assert ids.shape == (4,) and all(backend.tokenizer.encode(p.target, add_special_tokens=False)[0] == t for p, t in zip(prompts, ids))
    U = backend.unembedding_directions(ids)
    W = backend.model.lm_head.weight.detach().numpy()
    g = backend.model.model.norm.weight.detach().numpy()
    assert U.shape == (4, backend.hidden_size) and np.allclose(U, W[ids] * g[None, :])
    der = deranged_prompt(cfg, prompts[0], np.random.default_rng(0))
    assert der.query == prompts[0].query and der.target == prompts[0].target and der.prompt != prompts[0].prompt
    assert [d.input for d in der.demos] == [d.input for d in prompts[0].demos]
    assert all(a.output != b.output for a, b in zip(der.demos, prompts[0].demos))
    assert sorted(d.output for d in der.demos) == sorted(d.output for d in prompts[0].demos)


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    runs = {}
    for kind in ("fv", "learned"):
        path = CONFIGS / f"smoke_toy_{kind}.yaml"
        cfg = load_config(path)
        cfg.output_dir = str(out)
        runs[kind] = run_pipeline(cfg, "pilot", config_path=str(path), run_id=kind)
    return out, runs


def test_smoke_trajectories(smoke_runs):
    out, runs = smoke_runs
    cfg_path = CONFIGS / "smoke_toy_trajectories.yaml"
    cfg = load_trajectories_config(cfg_path)
    assert main(["trajectories", "--config", str(cfg_path), "--fv-run", str(runs["fv"]), "--learned-run",
                 str(runs["learned"]), "--output-dir", str(out), "--run-id", "traj"]) == 0
    root = out / "traj"
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["determinism_check"]["identical"] and meta["runs"]["learned"]["run_id"] == "learned"
    assert meta["run_config"]["model"]["backend"] == "toy" and meta["config"]["n_isotropic"] == cfg.n_isotropic
    summary = json.loads((root / "core" / "summary.json").read_text())
    tasks = summary["tasks"]
    assert set(tasks) == {"antonym", "add_3", "number_to_words"} and not summary["skipped"]
    for task, t in tasks.items():
        d = root / "core" / "tasks" / task
        res = json.loads((d / "trajectories.json").read_text())
        L1 = res["n_read_points"]
        assert res["primary_layer"] in res["layers"] and str(t["primary_layer"]) in t["per_layer"]
        # D33 amended: the nearest candidate layer on each side of the primary (the toy has candidates 1 and 2,
        # so one side is missing and noted), with its role recorded
        candidates = sorted(int(l) for l in np.load(runs["learned"] / "core" / "tasks" / task / "directions.npz")["layers"])
        assert set(res["layers"]) == set(candidates) and len(candidates) == 2
        roles = res["layer_roles"]
        assert roles[str(res["primary_layer"])][0] == "primary" and t["layer_roles"] == roles
        other = next(l for l in res["layers"] if l != res["primary_layer"])
        assert ("neighbour_above" if other > res["primary_layer"] else "neighbour_below") in roles[str(other)]
        assert any("of 1 candidate layers" in n for n in res["notes"])
        for cond in ("base", "icl", "icl2", "deranged"):
            assert "logprob_per_token_mean" in res["conditions"][cond]
        arrays = np.load(d / "trajectories_arrays.npz")
        for layer in res["layers"]:
            info = res["per_layer"][str(layer)]
            cons = info["constructions"]
            assert set(cons) == {"pca", "fv", "learned"}
            assert cons["pca"]["source"] == "calibration_pool_best" and len(cons["pca"]["grid"]) == len(cfg.pc1_rho_grid)
            assert cons["learned"]["source"] in ("selected", "reliable_grid_point", "natural_norm", "radius")
            assert all(c["alpha"] > 0 for c in cons.values())
            # every pair among the six named trajectories, per example, nan below the injection for constructions
            for a, b in (("icl", "learned"), ("fv", "learned"), ("pca", "fv"), ("icl", "icl2"), ("icl", "task")):
                cos = arrays[f"L{layer}_cos_{a}~{b}"]
                assert cos.shape == (L1, res["n_examples"]) and np.abs(np.nan_to_num(cos)).max() <= 1 + 1e-6
                natural = {"icl", "icl2", "task"}
                start = 1 if a in natural and b in natural else layer  # the embedding of the query token is context-free
                assert np.isnan(cos[:start]).all() and not np.isnan(cos[start:]).any()
                assert f"L{layer}_cos_noanswer_{a}~{b}" in arrays.files and f"L{layer}_cos_nogeneric_{a}~{b}" in arrays.files
                assert np.isnan(arrays[f"L{layer}_cos_nogeneric_{a}~{b}"][:start]).all()
                assert len(info["pairs"][f"{a}~{b}"]["mean_trajectory_cosine"]) == L1
            for c in ("pca", "fv", "learned"):
                for other in ("icl", "task", "icl2"):
                    s = info["summaries"][f"{c}->{other}"]
                    assert s["label"] in ("never_aligns", "converges", "aligns_then_diverges", "aligns_then_partly_diverges", "partial")
                    assert len(s["curve"]["median"]) == L1 and len(s["floor"]["median"]) == L1
                    assert s["answer_removed"]["label"] and s["generic_removed"]["label"] and s["start"] == layer
                    assert f"L{layer}_floor_{c}_vs_{other}" in arrays.files and f"L{layer}_floor_nogeneric_{c}_vs_{other}" in arrays.files
                    assert f"L{layer}_projection_{c}_on_icl" in arrays.files
                assert "alignment" in t["per_layer"][str(layer)]["constructions"][c]
                assert t["per_layer"][str(layer)]["constructions"][c]["alignment"]["icl"]["label"]
            assert set(info["ceilings"]) == {"icl~icl2", "icl~icl2_answer_removed", "icl~icl2_generic_removed",
                                             "icl~task", "icl~task_answer_removed", "icl~task_generic_removed"}
            assert info["variants"] == ["raw", "answer_removed", "generic_removed"]
            assert info["generic_response"]["n_controls"] == 3 * cfg.n_isotropic
            assert len(info["generic_response"]["cos_with_answer_direction"]) == L1
            coh = info["generic_response"]["coherence"]
            assert len(coh) == L1 and all(v is None for v in coh[:layer]) and all(0 <= v <= 1 + 1e-9 for v in coh[layer:])
            assert set(info["generic_response"]["coherence_by_construction"]) == {"pca", "fv", "learned"}
            assert arrays[f"L{layer}_mean_generic"].shape == (L1, meta["model"]["hidden_size"])
            # D33 (amended): the other strengths, each with its own floors and generic response, the cross-strength cosines
            assert set(info["other_strengths"]) == {"0.5", "0.25"} and info["strength_factor"] == 1.0
            for f, tag in ((0.5, "0.5"), (0.25, "0.25")):
                sub = info["other_strengths"][tag]
                assert sub["strength_factor"] == f and set(sub["constructions"]) == {"pca", "fv", "learned"}
                for c in ("pca", "fv", "learned"):
                    assert sub["constructions"][c]["alpha"] == pytest.approx(f * cons[c]["alpha"])
                    assert sub["constructions"][c]["canonical_alpha"] == pytest.approx(cons[c]["alpha"])
                    assert "generic_removed" in sub["summaries"][f"{c}->icl"]
                    x = info["cross_strength"][tag][c]
                    assert len(x["curve"]["median"]) == L1 and x["at_injection"] == pytest.approx(1.0, abs=1e-5)
                    assert f"L{layer}_r{tag}_cross_cos_{c}" in arrays.files and f"L{layer}_r{tag}_cos_icl~{c}" in arrays.files
                assert len(info["cross_strength"][tag]["generic_cosine"]) == L1
                assert len(sub["generic_response"]["coherence"]) == L1
            half = info["other_strengths"]["0.5"]
            # D33: what the generic response is
            diag = info["generic_diagnostics"]
            assert set(diag["per_factor"]) == {"1", "0.5", "0.25"}
            for f, e in diag["per_factor"].items():
                for k in diag["top_k"]:
                    own, mass = e["energy_in_own_top_k"][str(k)], e["energy_in_residual_top_k"][str(k)]
                    assert all(v is None for v in own[:layer]) and all(0 <= v <= 1 + 1e-9 for v in own[layer:])
                    assert all(0 <= v <= 1 + 1e-9 for v in mass[layer:])
                assert all(-1 - 1e-9 <= v <= 1 + 1e-9 for v in e["cos_with_residual_mean"][layer:])
                lens = e["logit_lens_last"]
                assert len(lens["promoted"]) == cfg.logit_lens_top and len(lens["demoted"]) == cfg.logit_lens_top
                assert all(isinstance(tok, str) and v >= 0 for tok, v in lens["promoted"])
                assert lens["mean_abs_logit_change"] > 0
            assert "_first_token_logprob" not in info and "_first_token_logprob" not in half
            # D33 (amended): the patch test at every compared layer (patch.layers: all), for the learned vector
            patch = info["patch"]
            assert patch["reference"] == "icl" and set(patch["constructions"]) == {"learned"}
            rows = patch["constructions"]["learned"]["rows"]
            assert [r["read_point"] for r in rows] == patch["read_points"] and all(layer < r["read_point"] <= L1 - 1 for r in rows)
            for r in rows:
                for e in ("remove", "keep", "patch"):
                    assert set(r[e]) >= {"effect", "retained", "random_effect_mean", "excess_vs_random", "effect_test",
                                         "first_token_effect", "first_token_retained"}
                    assert 0 <= r[e]["excess_vs_random"]["p_value"] <= 1
            assert t["per_layer"][str(layer)]["patch"]["learned"][0]["remove"]["p"] == rows[0]["remove"]["excess_vs_random"]["p_value"]
            assert info["isotropic"]["n"] == cfg.n_isotropic and info["isotropic"]["learned"]["alpha"] == cons["learned"]["alpha"]
            assert arrays[f"L{layer}_mean_delta_learned"].shape == (L1, meta["model"]["hidden_size"])
            assert (root / "figures" / f"{task}_trajectories_L{layer}.png").exists()
        assert arrays["mean_delta_icl"].shape == (L1, meta["model"]["hidden_size"])
    assert meta["linalg_device"] == "cpu"
    # the two runs must be the two controls of one model with one seed
    with pytest.raises(ValueError, match="control"):
        main(["trajectories", "--config", str(cfg_path), "--fv-run", str(runs["learned"]), "--learned-run",
              str(runs["learned"]), "--output-dir", str(out), "--run-id", "traj_bad"])

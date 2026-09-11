"""Canonical function-vector extraction (docs/DECISIONS.md D21): head hooks, effects, selection, composition."""

import json
from pathlib import Path

import numpy as np
import pytest

from directions.cli import main
from directions.config import ModelConfig, PromptConfig, load_config
from directions.function_vector import build_function_vector, compose, head_effects, mean_head_outputs, select_heads
from directions.model import HeadPatch, ModelBackend
from directions.pipeline import run_pipeline
from directions.prompts import few_shot_prompt, zero_shot_prompt
from directions.seeds import rng_for
from directions.tasks import build_task

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def backend():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=16), run_seed=1)


@pytest.fixture(scope="module")
def prompts():
    task = build_task("antonym")
    cfg = PromptConfig(n_shots=3)
    return [few_shot_prompt(cfg, task.items[:20], task.items[i], rng_for(1, i)) for i in range(5)]


def _query_positions(backend, prompts):
    return [len(backend.tokenizer.encode(p.prompt, add_special_tokens=True)) - 1 for p in prompts]


def test_head_capture_shape_and_identity_patch(backend, prompts):
    base = backend.run(prompts, capture=True, capture_heads=True)
    assert base.head_outputs.shape == (4, 5, backend.n_heads, backend.head_dim) and base.head_outputs.dtype == np.float32
    same = backend.run(prompts, capture=True, head_patches=[HeadPatch(1, 2, base.head_outputs[1, :, 2, :])])
    assert np.array_equal(same.residuals, base.residuals) and np.array_equal(same.logprob_sum, base.logprob_sum)


def test_head_to_residual_decomposes_the_attention_output(backend, prompts):
    """The attention output at the query token is the sum of the per-head terms (o_proj is linear, no bias)."""
    layer = 1
    got = {}
    handle = backend.attn_out[layer].register_forward_hook(lambda m, a, o: got.__setitem__("out", o.detach().float().cpu().numpy()))
    try:
        r = backend.run(prompts, capture_heads=True)
    finally:
        handle.remove()
    for i, q in enumerate(_query_positions(backend, prompts)):
        recon = sum(backend.head_to_residual(layer, h, r.head_outputs[layer, i, h]) for h in range(backend.n_heads))
        assert np.allclose(got["out"][i, q], recon, atol=1e-6)


def test_per_example_head_patches_match_shared_patches(backend, prompts):
    base = backend.run(prompts)
    z = np.full(backend.head_dim, 0.3, dtype=np.float32)
    shared = backend.run(prompts, head_patches=[HeadPatch(0, 3, z)])
    per = backend.run(prompts, head_patches=[HeadPatch(0, np.full(len(prompts), 3), np.tile(z, (len(prompts), 1)))])
    assert np.allclose(shared.logprob_sum, per.logprob_sum, atol=1e-6)
    assert not np.allclose(shared.logprob_sum, base.logprob_sum)
    # different heads per example: each example sees only its own head patched
    mixed_heads = np.array([0, 1, 2, 3, 0])
    mixed = backend.run(prompts, head_patches=[HeadPatch(0, mixed_heads, np.tile(z, (len(prompts), 1)))])
    for i, h in enumerate(mixed_heads):
        single = backend.run([prompts[i]], head_patches=[HeadPatch(0, int(h), z)])
        assert np.allclose(mixed.logprob_sum[i], single.logprob_sum[0], atol=1e-6)


def test_head_effects_batched_equals_one_run_per_head(backend, prompts):
    """The batched effect computation (several heads per forward pass) against one run per head."""
    pos = backend.run(prompts, capture_heads=True)
    means = mean_head_outputs(pos)
    assert means.shape == (4, backend.n_heads, backend.head_dim)
    neg = [zero_shot_prompt(PromptConfig(), p.query) for p in prompts]
    base = backend.run(neg)
    assert base.first_token_logprob.shape == (5,) and np.all(base.first_token_logprob <= 0)
    assert backend.cfg.batch_size // len(neg) > 1  # the batched path really packs several heads per pass
    for metric, values in (("target_probability", lambda r: np.exp(r.logprob_sum)),
                           ("first_token_probability", lambda r: np.exp(r.first_token_logprob)),
                           ("logprob_per_token", lambda r: r.logprob_per_token)):
        aie = head_effects(backend, neg, values(base), means, metric=metric)
        assert aie.shape == (4, backend.n_heads)
        for layer in range(4):
            for head in range(backend.n_heads):
                r = backend.run(neg, head_patches=[HeadPatch(layer, head, means[layer, head].astype(np.float32))])
                assert aie[layer, head] == pytest.approx(float(np.mean(values(r) - values(base))), abs=1e-6), metric
    assert np.all(np.abs(head_effects(backend, neg, np.exp(base.logprob_sum), means)) <= 1.0)  # a probability difference
    # the whole-target probability equals the first-token probability exactly for one-token targets
    one = [p for p in neg if backend.target_token_count(p.target) == 1]
    if one:
        r = backend.run(one)
        assert np.allclose(np.exp(r.logprob_sum), np.exp(r.first_token_logprob))
    with pytest.raises(ValueError):
        head_effects(backend, neg, base.logprob_per_token[:-1], means)
    with pytest.raises(ValueError):
        head_effects(backend, neg, base.logprob_per_token, means, metric="accuracy")


def test_select_heads_universal_and_per_task():
    a = np.array([[1.0, 0.0], [0.0, 5.0], [2.0, 0.0]])
    b = np.array([[0.0, 4.0], [0.0, 1.0], [2.0, 0.0]])
    uni = select_heads({"a": a, "b": b}, 2, "universal")
    assert [(l, h) for l, h, _, _ in uni["a"]] == [(1, 1), (0, 1)] == [(l, h) for l, h, _, _ in uni["b"]]
    assert uni["a"][0][2] == pytest.approx(3.0) and uni["a"][0][3] == 5.0 and uni["b"][0][3] == 1.0
    per = select_heads({"a": a, "b": b}, 2, "per_task")
    assert [(l, h) for l, h, _, _ in per["a"]] == [(1, 1), (2, 0)]
    assert [(l, h) for l, h, _, _ in per["b"]] == [(0, 1), (2, 0)]
    assert select_heads({}, 2, "universal") == {}
    with pytest.raises(ValueError):
        select_heads({"a": a}, 1, "random")


def test_compose_and_build_function_vector(backend, prompts):
    r1 = backend.run(prompts, capture_heads=True)
    r2 = backend.run(prompts[::-1], capture_heads=True)
    seed_means = np.stack([mean_head_outputs(r1), mean_head_outputs(r2)])
    selection = [(1, 2, 0.5, 0.4), (3, 0, 0.3, 0.3)]
    fv = build_function_vector(backend, seed_means, selection)
    expected = backend.head_to_residual(1, 2, seed_means.mean(0)[1, 2]) + backend.head_to_residual(3, 0, seed_means.mean(0)[3, 0])
    assert np.allclose(fv.vector, expected) and np.allclose(compose(backend, seed_means.mean(0), [(1, 2), (3, 0)]), expected)
    assert np.linalg.norm(fv.direction) == pytest.approx(1.0) and fv.natural_norm == pytest.approx(np.linalg.norm(expected))
    assert fv.seed_directions.shape == (2, backend.hidden_size) and 0 <= fv.stability <= 1
    assert [h["layer"] for h in fv.heads] == [1, 3] and fv.heads[0]["ranking_effect"] == 0.5 and fv.heads[0]["task_effect"] == 0.4
    # the two seeds see the same prompts in a different order, so their vectors coincide
    assert fv.stability == pytest.approx(1.0, abs=1e-6)


@pytest.fixture(scope="module")
def smoke_fv_run(tmp_path_factory):
    cfg = load_config(CONFIGS / "smoke_toy_fv.yaml")
    cfg.output_dir = str(tmp_path_factory.mktemp("results"))
    return run_pipeline(cfg, "pilot", config_path=str(CONFIGS / "smoke_toy_fv.yaml"))


def test_smoke_function_vector_outputs(smoke_fv_run):
    root = smoke_fv_run
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["control"] == "function_vector"
    assert len(meta["function_vector"]["universal_heads"]) == 3 and meta["function_vector"]["head_selection"] == "universal"
    # the run profile (D22): every stage of every task, with forward counts; the flat timings mirror it
    prof = meta["profile"]
    for key in ("model_load", "function_vectors", "figures", "total", "prepare:antonym", "extraction:antonym", "head_effects:antonym",
                "demo_variation:antonym", "calibration:antonym", "controls_forward:antonym", "controls_profiles:antonym",
                "layerwise:antonym", "strength_robustness:antonym", "block_ablation:antonym"):
        assert key in prof["sections"] and key in meta["timings_seconds"], key
    assert prof["sections"]["head_effects:antonym"]["forward_examples"] > 0 and prof["totals"]["forward_examples"] > 0
    assert prof["sections"]["controls_setup:antonym"]["gradient_examples"] == 6 and meta["linalg_device"] == "numpy"
    summary = json.loads((root / "core" / "summary.json").read_text())
    assert summary["control"] == "function_vector"
    heads = None
    for task in ("antonym", "arithmetic", "number_to_words"):
        d = root / "core" / "tasks" / task
        splits = json.loads((d / "splits.json").read_text())
        tok = splits["target_tokenisation"]
        assert tok["n_tokens_mean"] > 1 and tok["first_token_is_space_fraction"] == 1.0 and tok["examples"][0][0] == " "  # char tokens
        ext = json.loads((d / "extraction.json").read_text())["function_vector"]
        assert ext["aie_metric"] == "target_probability" and 0 <= ext["aie_baseline_mean"] <= ext["aie_baseline_first_token_probability_mean"] <= 1
        cal = json.loads((d / "calibration.json").read_text())
        assert cal["strength_unit"] == "natural" and cal["reference_rho"] == 4.0 and cal["layer_rule"] == "earliest"
        assert set(cal) >= {"weakest", "strongest", "middle", "strength_units"} and all("rho_layer_norm" in g for g in cal["grid"])
        for g in cal["grid"]:
            assert g["alpha"] == pytest.approx(g["rho"] * cal["strength_units"][str(g["layer"])])
            assert g["rho_layer_norm"] == pytest.approx(g["alpha"] / cal["layer_norms"][str(g["layer"])])
        rob = json.loads((root / "exploratory" / "tasks" / task / "strength_robustness.json").read_text())
        assert set(rob["profiles"]) == {"weakest", "middle", "strongest"} and rob["selected"]["rho"] == cal["selected"]["rho"] if cal["selected"] else True
        fv = json.loads((d / "function_vector.json").read_text())
        assert len(fv["heads"]) == 3 and np.array(fv["head_effects"]).shape == (4, 4) and fv["head_effects_n_prompts"] == 6
        assert set(fv["cos_with_pca"]) == {"1", "2"} and len(fv["cos_with_pca_all_layers"]) == 5
        assert len(fv["demo_variation_cos_with_control"]) == 1
        if heads is None:
            heads = [(h["layer"], h["head"]) for h in fv["heads"]]
        assert [(h["layer"], h["head"]) for h in fv["heads"]] == heads  # universal: the same set for every task
        arrays = np.load(d / "directions.npz")
        assert set(arrays) >= {"pooled", "per_seed", "all_layers", "fv", "fv_direction", "fv_per_seed", "fv_heads", "head_effects"}
        assert np.linalg.norm(arrays["fv_direction"]) == pytest.approx(1.0) and arrays["fv_heads"].shape == (3, 2)
        assert arrays["pooled"].shape == (2, 32)  # PC1 at the candidate layers is still extracted
        q = json.loads((d / "qualification.json").read_text())
        assert set(q["gates"]) == {"fewshot", "stability", "calibration", "steering", "random_controls"}
        assert "pca_stability" in q and q["stability"]["1"] == q["stability"]["2"] == q["function_vector"]["stability"]
        assert set(q["function_vector"]["natural_rho"]) == {"1", "2"} and "cos_with_pca_at_selected_layer" in q["function_vector"]
        assert q["function_vector"]["alpha_over_natural_norm"] == pytest.approx(q["selection"]["rho"])  # natural units
        assert q["selection"]["strength_unit"] == "natural" and "rho_layer_norm" in q["selection"]
        ev = json.loads((d / "evaluation.json").read_text())
        assert ev["control"] == "function_vector" and ev["function_vector"]["natural_norm"] > 0
        lw = json.loads((d / "layerwise.json").read_text())
        assert sorted({c["kind"] for c in lw["controls"]}) == ["covariance", "demo_variation", "isotropic", "orthogonal", "other_task"]
        assert lw["real"]["summaries"]["alignment"]["median"][lw["real"]["intervention_layer"]] == pytest.approx(1.0, abs=1e-4)
        assert summary["tasks"][task]["function_vector"]["natural_rho_at_selection"] > 0
        assert (root / "figures" / f"{task}_function_vector.png").exists()
    rejections = [json.loads(l) for l in (root / "rejections.jsonl").read_text().splitlines()]
    assert not any(r["stage"] == "error" for r in rejections), rejections


def test_smoke_function_vector_is_reproducible(smoke_fv_run, tmp_path):
    cfg = load_config(CONFIGS / "smoke_toy_fv.yaml")
    cfg.output_dir = str(tmp_path)
    second = run_pipeline(cfg, "pilot")
    assert main(["compare", str(smoke_fv_run), str(second)]) == 0

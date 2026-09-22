import numpy as np
import pytest
import torch

from directions.config import ModelConfig, PromptConfig, ToyModelConfig
from directions.model import Intervention, ModelBackend
from directions.prompts import few_shot_prompt
from directions.seeds import rng_for
from directions.tasks import build_task

FAMILIES = ["qwen3", "olmo3", "gemma4"]


@pytest.fixture(scope="module", params=FAMILIES)
def backend(request):
    """A tiny random model of each supported architecture (the backend must be exact on all of them)."""
    cfg = ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=4, toy=ToyModelConfig(family=request.param))
    return ModelBackend(cfg, run_seed=1)


@pytest.fixture(scope="module")
def prompts():
    task = build_task("antonym")
    cfg = PromptConfig(n_shots=3)
    return [few_shot_prompt(cfg, task.items[:20], task.items[i], rng_for(1, i)) for i in range(6)]


def test_metadata_and_capture_shape(backend, prompts):
    meta = backend.metadata()
    assert meta["n_layers"] == 4 and meta["hidden_size"] == 32
    res = backend.run(prompts, capture=True)
    assert res.residuals.shape == (5, 6, 32)
    assert res.residuals.dtype == np.float32
    assert res.logprob_per_token.shape == (6,)


def test_changed_token_scoring(backend):
    """With a score reference only the target tokens that differ from the input's (right-aligned) count (D30)."""
    from dataclasses import replace

    from directions.prompts import zero_shot_prompt
    from directions.tasks import Item

    tok = backend.tokenizer  # character-level: " 154" -> [" ", "1", "5", "4"]
    assert backend._scored_tokens(tok.encode(" 154"), " 151") == [False, False, False, True]
    assert backend._scored_tokens(tok.encode(" 201"), " 198") == [False, True, True, True]
    assert backend._scored_tokens(tok.encode(" 100"), " 97") == [False, True, True, True]  # longer target: leading tokens count
    assert backend._scored_tokens(tok.encode(" 7"), " 7") == [True, True]  # identity keeps every token
    assert backend._scored_tokens(tok.encode(" 7"), None) == [True, True]
    assert backend.scored_token_count(" 154", " 151") == 1 and backend.scored_token_count(" 154", None) == 4
    cfg = PromptConfig(n_shots=2, target_scoring="changed_tokens")
    p = zero_shot_prompt(cfg, Item("151", "154"))
    assert p.score_reference == " 151"
    scored = backend.run([p])
    plain = backend.run([replace(p, score_reference=None)])
    # the plain score averages four tokens; the changed-token score is the last digit's log-probability alone
    ids = tok.encode(p.prompt)
    tg = tok.encode(p.target)
    with torch.no_grad():
        out = backend.model(input_ids=torch.tensor([ids + tg]))
    lp = torch.log_softmax(out.logits[0].float(), -1)
    per_token = [float(lp[len(ids) - 1 + j, tg[j]]) for j in range(len(tg))]
    assert scored.logprob_sum[0] == pytest.approx(per_token[-1], abs=1e-5)
    assert scored.logprob_per_token[0] == pytest.approx(per_token[-1], abs=1e-5)
    assert plain.logprob_per_token[0] == pytest.approx(sum(per_token) / 4, abs=1e-5)
    assert scored.exact_match[0] == plain.exact_match[0] and scored.n_target_tokens[0] == 4


def test_logprob_matches_native_forward(backend, prompts):
    res = backend.run(prompts)
    for k in range(3):
        ids = backend.tokenizer.encode(prompts[k].prompt)
        tg = backend.tokenizer.encode(prompts[k].target)
        with torch.no_grad():
            out = backend.model(input_ids=torch.tensor([ids + tg]))
        lp = torch.log_softmax(out.logits[0].float(), -1)
        ref = sum(lp[len(ids) - 1 + j, tg[j]].item() for j in range(len(tg)))
        assert res.logprob_sum[k] == pytest.approx(ref, abs=1e-4)
        assert res.logprob_per_token[k] == pytest.approx(ref / len(tg), abs=1e-4)
        logits0 = out.logits[0, len(ids) - 1].float()
        gold = logits0[tg[0]].item()
        logits0[tg[0]] = -float("inf")
        assert res.first_token_margin[k] == pytest.approx(gold - logits0.max().item(), abs=1e-4)
        argmax_ok = all(out.logits[0, len(ids) - 1 + j].argmax().item() == tg[j] for j in range(len(tg)))
        assert bool(res.exact_match[k]) == argmax_ok


def test_batching_invariance(backend, prompts):
    a = backend.run(prompts, capture=True, batch_size=6)
    b = backend.run(prompts, capture=True, batch_size=2)
    assert np.allclose(a.residuals, b.residuals, atol=1e-5)
    assert np.allclose(a.logprob_sum, b.logprob_sum, atol=1e-5)


def test_intervention_is_exact_at_layer_and_zero_before(backend, prompts):
    base = backend.run(prompts, capture=True)
    v = np.random.default_rng(0).standard_normal(32)
    v /= np.linalg.norm(v)
    st = backend.run(prompts, interventions=[Intervention(2, v, 3.0)], capture=True)
    delta = st.residuals - base.residuals
    assert np.abs(delta[:2]).max() == 0.0
    assert np.allclose(delta[2], 3.0 * v[None, :], atol=1e-5)
    assert np.linalg.norm(delta[3], axis=1).min() > 0
    # per-example vectors give the same result as a shared vector
    per = backend.run(prompts, interventions=[Intervention(2, np.tile(v, (6, 1)), 3.0)], capture=True)
    assert np.allclose(per.residuals, st.residuals)
    # intervening at the final read point (L) modifies the block output
    last = backend.run(prompts, interventions=[Intervention(4, v, 1.0)], capture=True)
    assert np.allclose(last.residuals[4] - base.residuals[4], v[None, :], atol=1e-5)
    assert np.abs(last.residuals[:4] - base.residuals[:4]).max() == 0.0
    # interventions change behaviour
    assert not np.allclose(last.logprob_sum, base.logprob_sum)


def test_attention_output_norm_flag_matches_the_block(backend, prompts):
    """`attention_output_normed` (D35) says whether the block's post-attention norm is applied to the attention
    output (OLMo 3, Gemma) or to the residual before the MLP (Qwen/Llama): checked by hooking the norm's input."""
    block = backend.blocks[1]
    got = {}
    h1 = block.self_attn.o_proj.register_forward_hook(lambda m, a, o: got.__setitem__("attn", o.detach().clone()))
    h2 = block.post_attention_layernorm.register_forward_hook(lambda m, a, o: got.__setitem__("norm_in", a[0].detach().clone()))
    try:
        with torch.no_grad():
            backend.model(input_ids=torch.tensor([backend.tokenizer.encode(prompts[0].prompt)]))
    finally:
        h1.remove(), h2.remove()
    normed = torch.allclose(got["attn"], got["norm_in"])
    assert backend.attention_output_normed == normed
    assert normed == (backend.cfg.toy.family in ("olmo3", "gemma4"))
    assert backend.metadata()["attention_output_normed"] == normed


def test_hooks_are_inert_outside_run(backend, prompts):
    ids = torch.tensor([backend.tokenizer.encode(prompts[0].prompt)])
    with torch.no_grad():
        a = backend.model(input_ids=ids).logits
        b = backend.model(input_ids=ids).logits
    assert torch.equal(a, b)


def test_gradients_match_finite_differences_and_batching(backend, prompts):
    g = backend.gradients(prompts)
    assert g.shape == (5, 6, 32) and g.dtype == np.float32
    rng = np.random.default_rng(0)
    eps = 1e-3
    for layer in (0, 2, 4):
        u = rng.standard_normal(32)
        u /= np.linalg.norm(u)
        plus = backend.run(prompts, interventions=[Intervention(layer, u, eps)]).logprob_sum
        minus = backend.run(prompts, interventions=[Intervention(layer, u, -eps)]).logprob_sum
        fd = (plus - minus) / (2 * eps)
        assert np.allclose(fd, g[layer] @ u, atol=5e-3, rtol=1e-2), layer
    # batching invariance, and the gradient pass leaves the hooks inert
    assert np.allclose(g, backend.gradients(prompts, batch_size=2), atol=1e-5)
    a = backend.run(prompts, capture=True)
    assert np.array_equal(a.logprob_sum, backend.run(prompts).logprob_sum)


def test_sublayer_writes_sum_to_the_residual_increments(backend, prompts):
    """D38: the captured attention and MLP writes are exactly what each block adds to the residual stream on
    every family (pre-norm Qwen3; OLMo 3 and Gemma 4 with their post-sublayer norms)."""
    res = backend.run(prompts, capture=True, capture_sublayers=True)
    assert res.attn_writes.shape == (4, 6, 32) and res.mlp_writes.shape == (4, 6, 32)
    inc = res.residuals[1:] - res.residuals[:-1]
    err = np.abs(res.attn_writes + res.mlp_writes - inc).max()
    assert err < 1e-4 * (1 + np.abs(inc).max()), err
    # an intervention at a read point changes the writes of the blocks after it, not before
    steered = backend.run(prompts, interventions=[Intervention(2, np.ones(32, dtype=np.float32), 0.5)], capture=True, capture_sublayers=True)
    assert np.allclose(steered.attn_writes[:2], res.attn_writes[:2]) and np.allclose(steered.mlp_writes[:2], res.mlp_writes[:2])
    assert not np.allclose(steered.attn_writes[2:], res.attn_writes[2:])
    assert backend.block_scalars == [1.0] * 4
    if backend.cfg.toy.family == "gemma4":
        # Gemma 4's per-block output scalar: the writes land scaled and the residual is rescaled by (scalar - 1)
        for b, v in zip(backend.blocks, (0.5, 0.8, 1.0, 0.25)):
            b.layer_scalar.fill_(v)
        backend.block_scalars = [0.5, 0.8, 1.0, 0.25]
        try:
            r2 = backend.run(prompts, capture=True, capture_sublayers=True)
            inc = r2.residuals[1:] - r2.residuals[:-1]
            sc = np.array(backend.block_scalars)[:, None, None]
            err = np.abs(r2.attn_writes + r2.mlp_writes + (sc - 1) * r2.residuals[:-1] - inc).max()
            assert err < 1e-4 * (1 + np.abs(inc).max()), err
        finally:
            for b in backend.blocks:
                b.layer_scalar.fill_(1.0)
            backend.block_scalars = [1.0] * 4

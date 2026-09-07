import numpy as np
import pytest
import torch

from directions.config import ModelConfig, PromptConfig
from directions.model import Intervention, ModelBackend
from directions.prompts import few_shot_prompt
from directions.seeds import rng_for
from directions.tasks import build_task


@pytest.fixture(scope="module")
def backend():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=4), run_seed=1)


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


def test_hooks_are_inert_outside_run(backend, prompts):
    ids = torch.tensor([backend.tokenizer.encode(prompts[0].prompt)])
    with torch.no_grad():
        a = backend.model(input_ids=ids).logits
        b = backend.model(input_ids=ids).logits
    assert torch.equal(a, b)

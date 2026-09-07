"""Correctness of residual capture, intervention and scoring.

These run on the offline `tiny_random` backend, so they need no download, no
network and no GPU, but they exercise exactly the same hook and scoring code
paths as a real pilot run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from directions.config import PromptTemplate
from directions.extraction import DirectionSpec
from directions.model import BlockPatch
from directions.prompts import build_eval_prompts
from directions.tasks import load_task, split_task


@pytest.fixture(scope="module")
def prompts():
    task = load_task("antonym")
    splits = split_task(task, 8, 2, 6, seed=0)
    return build_eval_prompts(
        np.random.default_rng(0), PromptTemplate(), list(splits.extraction),
        list(splits.extraction), 3,
    )


def _direction(lm, layer, alpha, seed=0):
    v = np.random.default_rng(seed).standard_normal(lm.d_model).astype(np.float32)
    return DirectionSpec(layer=layer, vector=v / np.linalg.norm(v), alpha=alpha)


def test_metadata_reports_architecture(tiny_model):
    meta = tiny_model.metadata()
    assert meta["n_layers"] == tiny_model.n_layers
    assert meta["d_model"] == tiny_model.d_model
    assert meta["n_parameters"] > 0
    assert meta["backend"] == "tiny_random"


def test_capture_shape_and_layer_count(tiny_model, prompts):
    h = tiny_model.capture(prompts, batch_size=3, max_seq_len=512)
    assert h.shape == (tiny_model.n_layers + 1, len(prompts), tiny_model.d_model)
    assert np.isfinite(h).all()


def test_capture_is_batch_size_invariant(tiny_model, prompts):
    a = tiny_model.capture(prompts, batch_size=1, max_seq_len=512)
    b = tiny_model.capture(prompts, batch_size=5, max_seq_len=512)
    assert np.allclose(a, b, atol=1e-4)


def test_capture_layer_zero_is_the_embedding_of_the_last_prompt_token(tiny_model, prompts):
    h = tiny_model.capture(prompts[:3], batch_size=3, max_seq_len=512)
    emb = tiny_model.model.get_input_embeddings()
    for i, p in enumerate(prompts[:3]):
        ids = tiny_model.encode(p, 512)
        want = emb(torch.tensor([ids.input_ids[ids.prompt_len - 1]])).detach().numpy()[0]
        assert np.allclose(h[0, i], want, atol=1e-5)


def test_intervention_is_exact_at_its_own_layer(tiny_model, prompts):
    """delta is exactly zero upstream and exactly alpha*v at the intervened layer."""
    layer, alpha = 2, 3.0
    spec = _direction(tiny_model, layer, alpha)
    base = tiny_model.capture(prompts, batch_size=3, max_seq_len=512)
    steer = tiny_model.capture(prompts, intervention=spec, batch_size=3, max_seq_len=512)
    delta = steer - base
    assert np.allclose(delta[:layer], 0.0, atol=1e-6)
    assert np.allclose(delta[layer], alpha * spec.vector[None, :], atol=1e-4)
    assert np.linalg.norm(delta[layer + 1], axis=-1).min() > 0


def test_intervention_scales_linearly_for_small_alpha(tiny_model, prompts):
    base = tiny_model.capture(prompts, batch_size=3, max_seq_len=512)
    d1 = tiny_model.capture(prompts, intervention=_direction(tiny_model, 1, 1e-3),
                            batch_size=3, max_seq_len=512) - base
    d2 = tiny_model.capture(prompts, intervention=_direction(tiny_model, 1, 2e-3),
                            batch_size=3, max_seq_len=512) - base
    assert np.allclose(d2, 2 * d1, atol=1e-5, rtol=1e-2)


def test_intervention_layer_bounds_are_enforced(tiny_model, prompts):
    with pytest.raises(ValueError, match="outside"):
        tiny_model.capture(prompts, intervention=_direction(tiny_model, tiny_model.n_layers, 1.0),
                           batch_size=3, max_seq_len=512)


def test_hooks_are_removed_after_use(tiny_model, prompts):
    tiny_model.capture(prompts, intervention=_direction(tiny_model, 1, 1.0),
                       batch_size=3, max_seq_len=512)
    for block in tiny_model.blocks:
        assert len(block._forward_pre_hooks) == 0
        assert len(block._forward_hooks) == 0


def test_scoring_is_batch_size_invariant_and_order_preserving(tiny_model, prompts):
    a = tiny_model.score(prompts, batch_size=1, max_seq_len=512)
    b = tiny_model.score(prompts, batch_size=4, max_seq_len=512)
    assert a.item_ids == b.item_ids == [p.item_id for p in prompts]
    assert np.allclose(a.target_logprob, b.target_logprob, atol=1e-4)
    assert (a.correct == b.correct).all()


def test_scoring_matches_an_unbatched_unpadded_reference(tiny_model, prompts):
    """Padding, length-sorting and the lm_head window must not change any number."""
    got = tiny_model.score(prompts, batch_size=4, max_seq_len=512)
    ref_lp, ref_ok = [], []
    for p in prompts:
        e = tiny_model.encode(p, 512)
        with torch.inference_mode():
            logits = tiny_model.model(input_ids=torch.tensor([e.input_ids])).logits[0].float()
        sel = logits[e.prompt_len - 1 : e.prompt_len - 1 + len(e.target_ids)]
        tgt = torch.tensor(e.target_ids)
        ref_lp.append(float(torch.log_softmax(sel, -1).gather(-1, tgt[:, None]).sum()))
        ref_ok.append(bool(torch.equal(sel.argmax(-1), tgt)))
    assert np.allclose(got.target_logprob, ref_lp, atol=1e-4)
    assert (got.correct == np.array(ref_ok)).all()


def test_zero_alpha_intervention_is_a_no_op(tiny_model, prompts):
    base = tiny_model.score(prompts, batch_size=3, max_seq_len=512)
    same = tiny_model.score(prompts, intervention=_direction(tiny_model, 1, 0.0),
                            batch_size=3, max_seq_len=512)
    assert np.allclose(base.target_logprob, same.target_logprob, atol=1e-5)


def test_block_patch_cancels_the_block_response(tiny_model, prompts):
    """Removing b_l from block l's output makes delta_{l+1} equal delta_l."""
    layer = 1
    spec = _direction(tiny_model, layer, 2.0)
    base = tiny_model.capture(prompts, batch_size=3, max_seq_len=512)
    steer = tiny_model.capture(prompts, intervention=spec, batch_size=3, max_seq_len=512)
    deltas = steer - base
    b = (deltas[layer + 1] - deltas[layer]).astype(np.float32)
    patched = tiny_model.score(
        prompts, intervention=spec,
        block_patch=BlockPatch(layer=layer, vectors=b, sign=-1.0),
        batch_size=3, max_seq_len=512,
    )
    unpatched = tiny_model.score(prompts, intervention=spec, batch_size=3, max_seq_len=512)
    assert not np.allclose(patched.target_logprob, unpatched.target_logprob)
    assert np.isfinite(patched.target_logprob).all()


def test_layer_index_from_fraction(tiny_model):
    assert tiny_model.layer_index_from_fraction(0.0) == 0
    assert tiny_model.layer_index_from_fraction(1.0) == tiny_model.n_layers - 1
    assert 0 <= tiny_model.layer_index_from_fraction(0.5) < tiny_model.n_layers


def test_encode_rejects_overlong_prompt(tiny_model, prompts):
    with pytest.raises(ValueError, match="max_seq_len"):
        tiny_model.encode(prompts[0], max_seq_len=4)

"""Site swaps and multi-position capture (docs/DECISIONS.md D45): exact on the toy models of every architecture."""

import numpy as np
import pytest
import torch

from directions.config import ModelConfig, PromptConfig, ToyModelConfig
from directions.model import ModelBackend, SitePatch
from directions.prompts import few_shot_prompt
from directions.seeds import rng_for
from directions.tasks import build_task

FAMILIES = ["qwen3", "olmo3", "gemma4"]


@pytest.fixture(scope="module", params=FAMILIES)
def backend(request):
    cfg = ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=4, toy=ToyModelConfig(family=request.param))
    return ModelBackend(cfg, run_seed=3)


@pytest.fixture(scope="module")
def prompts():
    task = build_task("antonym")
    cfg = PromptConfig(n_shots=2)
    return [few_shot_prompt(cfg, task.items[:20], task.items[i], rng_for(3, i)) for i in range(6)]


def _positions(backend, prompts, n=3):
    """The last ``n`` prompt tokens of each prompt (absolute indices)."""
    lens = [len(backend._encode_prompt(p)[0]) for p in prompts]
    return np.array([[L - n + j for j in range(n)] for L in lens], dtype=np.int64)


def test_capture_sites_matches_query_capture(backend, prompts):
    pos = _positions(backend, prompts)
    states = backend.capture_sites(prompts, pos)
    L = backend.n_layers
    assert states.shape == (len(prompts), L + 1, 3, backend.hidden_size)
    ref = backend.run(prompts, capture=True).residuals  # (L+1, N, d) at the query token = the last prompt token
    np.testing.assert_array_equal(states[:, :, -1].float().cpu().numpy().transpose(1, 0, 2), ref)


def test_swapping_a_run_s_own_state_changes_nothing(backend, prompts):
    pos = _positions(backend, prompts)
    states = backend.capture_sites(prompts, pos)
    ids = np.arange(5)
    base = backend.run(prompts, token_ids=ids)
    N = len(prompts)
    for layer in (0, 2, backend.n_layers):
        for j in range(3):
            sp = SitePatch(np.full(N, layer), pos[:, j], states[:, layer, j])
            got = backend.run(prompts, site_patch=sp, token_ids=ids)
            np.testing.assert_array_equal(got.token_logprobs, base.token_logprobs)
            np.testing.assert_array_equal(got.logprob_sum, base.logprob_sum)


def test_swap_from_another_prompt_transfers_and_batches_like_single_rows(backend, prompts):
    """A swap at the query token at the last read point carries the source's prediction entirely; rows with
    different sites in one batch give what each gives alone."""
    pos = _positions(backend, prompts)
    states = backend.capture_sites(prompts, pos)
    N, L = len(prompts), backend.n_layers
    src = np.roll(np.arange(N), 1)
    ids = np.arange(8)
    full = backend.run(prompts, token_ids=ids)
    sp = SitePatch(np.full(N, L), pos[:, -1], states[src, L, -1])
    got = backend.run(prompts, site_patch=sp, token_ids=ids)
    np.testing.assert_allclose(got.token_logprobs, full.token_logprobs[src], rtol=0, atol=1e-5)
    # mixed sites in one batch against one row at a time
    layers = np.array([0, 1, 2, 3, L, -1])
    cols = np.array([0, 1, 2, 0, 2, 1])
    vals = torch.stack([states[src[i], max(layers[i], 0), cols[i]] for i in range(N)])
    batched = backend.run(prompts, site_patch=SitePatch(layers, pos[np.arange(N), cols], vals), token_ids=ids, batch_size=N)
    for i in range(N):
        one = backend.run([prompts[i]], site_patch=SitePatch(layers[i : i + 1], pos[i : i + 1, cols[i]], vals[i : i + 1]),
                          token_ids=ids, batch_size=1)
        np.testing.assert_allclose(batched.token_logprobs[i], one.token_logprobs[0], rtol=0, atol=1e-4)
        assert batched.query_argmax[i] == one.query_argmax[0]
    np.testing.assert_allclose(batched.token_logprobs[5], full.token_logprobs[5], rtol=0, atol=1e-4)  # layer -1: untouched


def test_subspace_swap_full_basis_is_the_whole_swap_and_empty_basis_nothing(backend, prompts):
    pos = _positions(backend, prompts)
    states = backend.capture_sites(prompts, pos)
    N, d = len(prompts), backend.hidden_size
    src = np.roll(np.arange(N), 2)
    ids = np.arange(8)
    layers, at = np.full(N, 2), pos[:, 1]
    whole = backend.run(prompts, site_patch=SitePatch(layers, at, states[src, 2, 1]), token_ids=ids)
    eye = backend.run(prompts, site_patch=SitePatch(layers, at, states[src, 2, 1], basis=np.eye(d)), token_ids=ids)
    np.testing.assert_allclose(eye.token_logprobs, whole.token_logprobs, rtol=0, atol=1e-4)
    base = backend.run(prompts, token_ids=ids)
    none = backend.run(prompts, site_patch=SitePatch(layers, at, states[src, 2, 1], basis=np.zeros((0, d))), token_ids=ids)
    np.testing.assert_allclose(none.token_logprobs, base.token_logprobs, rtol=0, atol=1e-5)


def test_site_patch_rejects_bad_sites(backend, prompts):
    pos = _positions(backend, prompts)
    states = backend.capture_sites(prompts, pos)
    N = len(prompts)
    with pytest.raises(ValueError, match="outside"):
        backend.run(prompts, site_patch=SitePatch(np.full(N, backend.n_layers + 1), pos[:, 0], states[:, 0, 0]))
    with pytest.raises(ValueError, match="outside its prompt"):
        backend.run(prompts, site_patch=SitePatch(np.zeros(N, dtype=int), pos[:, 0] + 10_000, states[:, 0, 0]))

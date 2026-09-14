"""The damage check (docs/DECISIONS.md D24): KL from the unsteered next-token distribution at the query token."""

import numpy as np
import pytest

from directions.config import ModelConfig, PromptConfig
from directions.model import Intervention, ModelBackend
from directions.prompts import zero_shot_prompt
from directions.tasks import build_task


@pytest.fixture(scope="module")
def backend():
    return ModelBackend(ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=4), run_seed=3)


@pytest.fixture(scope="module")
def prompts():
    task = build_task("antonym")
    return [zero_shot_prompt(PromptConfig(), it) for it in task.items[:6]]


def test_kl_against_own_distribution_is_zero_and_matches_numpy(backend, prompts):
    base = backend.run(prompts, capture_logprobs=True)
    assert base.query_logprobs.shape == (6, backend.tokenizer.vocab_size) and base.query_logprobs.dtype == np.float32
    assert np.allclose(np.exp(base.query_logprobs.astype(np.float64)).sum(axis=1), 1.0, atol=1e-4)  # a distribution
    same = backend.run(prompts, reference_logprobs=base.query_logprobs)
    assert np.allclose(same.kl_from_reference, 0.0, atol=1e-6) and not same.argmax_changed.any()
    assert same.metrics_dict()["argmax_change_rate"] == 0.0 and "kl_mean" in same.metrics_dict()
    assert "kl_mean" not in base.metrics_dict()  # no reference, no damage fields
    # a steered run: KL >= 0, equal to the NumPy KL of the two captured distributions, batched across 2 batches
    v = np.random.default_rng(0).standard_normal(backend.hidden_size)
    v /= np.linalg.norm(v)
    steered = backend.run(prompts, interventions=[Intervention(1, v, 3.0)], capture_logprobs=True, reference_logprobs=base.query_logprobs)
    p = np.exp(base.query_logprobs.astype(np.float64))
    kl = (p * (base.query_logprobs.astype(np.float64) - steered.query_logprobs.astype(np.float64))).sum(axis=1)
    assert np.all(steered.kl_from_reference >= -1e-6) and np.allclose(steered.kl_from_reference, kl, atol=1e-4)
    assert steered.kl_from_reference.max() > 1e-3
    assert np.array_equal(steered.argmax_changed, steered.query_logprobs.argmax(1) != base.query_logprobs.argmax(1))
    # the reference may be given as the backend's device tensor, once, for several runs
    ref = backend.reference_tensor(base.query_logprobs)
    again = backend.run(prompts, interventions=[Intervention(1, v, 3.0)], reference_logprobs=ref)
    assert np.array_equal(again.kl_from_reference, steered.kl_from_reference)
    with pytest.raises(ValueError):
        backend.run(prompts[:3], reference_logprobs=base.query_logprobs)

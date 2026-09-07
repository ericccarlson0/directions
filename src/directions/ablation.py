"""Causal validation of important blocks (exploratory, secondary to the geometry).

For a block ``l`` with large conversion or new-subspace creation:

* necessity: run the steered pass and subtract the steering-induced block
  contribution ``b_l(x)`` at ``resid[l+1]`` (the block's output);
* sufficiency: inject ``+b_l(x)`` at ``resid[l+1]`` into an unsteered pass.

Report the fraction of the steering-induced behavioural improvement lost and
reproduced, using the decision metric (target log-probability per token).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import BlockAblationConfig
from .layerwise import LayerwiseProfile
from .model import ForwardResult, Intervention, ModelBackend
from .prompts import Prompt
from .stats import paired_bootstrap_test


def select_blocks(profile: LayerwiseProfile, cfg: BlockAblationConfig) -> list[int]:
    """Blocks with the largest median conversion (or uncentred new-subspace fraction)."""
    ls = profile.intervention_layer
    if cfg.criterion == "conversion":
        curve = profile.metric_curve("conversion")
    else:
        curve = profile.new_subspace_uncentered
    order = [int(i) for i in np.argsort(-np.nan_to_num(curve, nan=-np.inf)) if i >= ls and not np.isnan(curve[i])]
    return sorted(order[: cfg.n_blocks])


def block_ablation(
    backend: ModelBackend,
    prompts: list[Prompt],
    base: ForwardResult,
    steered: ForwardResult,
    delta: np.ndarray,
    layer: int,
    v: np.ndarray,
    alpha: float,
    blocks: list[int],
    rng: np.random.Generator,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Necessity/sufficiency of each block's steering-induced contribution."""
    steer_gain = float(np.mean(steered.logprob_per_token - base.logprob_per_token))
    out: dict[str, Any] = {"steering_improvement": steer_gain, "blocks": {}}
    for l in blocks:
        b_l = (delta[l + 1] - delta[l]).astype(np.float32)  # (n, d) per-example block response
        nec = backend.run(
            prompts,
            interventions=[Intervention(layer, v, alpha), Intervention(l + 1, b_l, -1.0)],
        )
        suf = backend.run(prompts, interventions=[Intervention(l + 1, b_l, 1.0)])
        nec_gain = float(np.mean(nec.logprob_per_token - base.logprob_per_token))
        suf_gain = float(np.mean(suf.logprob_per_token - base.logprob_per_token))
        out["blocks"][str(l)] = {
            "block": l,
            "necessity": {
                "improvement_without_block": nec_gain,
                "fraction_lost": None if steer_gain == 0 else float(1.0 - nec_gain / steer_gain),
                "test_vs_steered": paired_bootstrap_test(
                    steered.logprob_per_token, nec.logprob_per_token, rng, n_boot=n_boot
                ).__dict__,
                "metrics": nec.metrics_dict(),
            },
            "sufficiency": {
                "improvement_from_block_alone": suf_gain,
                "fraction_reproduced": None if steer_gain == 0 else float(suf_gain / steer_gain),
                "test_vs_baseline": paired_bootstrap_test(
                    suf.logprob_per_token, base.logprob_per_token, rng, n_boot=n_boot
                ).__dict__,
                "metrics": suf.metrics_dict(),
            },
        }
    return out

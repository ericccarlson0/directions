"""Canonical function-vector extraction (Todd et al., 2024; docs/DECISIONS.md D21).

A function vector is the sum, over a small set of attention heads, of each head's mean output at the
final query token of few-shot prompts, mapped into the residual stream through the head's slice of the
attention output projection. Heads are ranked by their average indirect effect: the change of the
decision metric when the head's mean output is patched into deranged-label prompts at the query token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .geometry import normalize, stability
from .model import ForwardResult, HeadPatch, ModelBackend
from .prompts import Prompt


@dataclass
class FunctionVector:
    vector: np.ndarray  # (d,) float64, the raw sum over the selected heads (pooled mean outputs)
    direction: np.ndarray  # unit-norm ``vector``
    natural_norm: float  # ||vector||: the strength the canonical construction would inject
    heads: list[dict[str, Any]]  # selected heads with their indirect effects, in rank order
    seed_directions: np.ndarray  # (n_seeds, d) unit vectors from each seed's mean outputs
    seed_norms: list[float]
    stability: float  # min pairwise |cos| across seeds
    cos_pooled_vs_seeds: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "natural_norm": self.natural_norm,
            "heads": self.heads,
            "seed_norms": self.seed_norms,
            "stability": self.stability,
            "cos_pooled_vs_seeds": self.cos_pooled_vs_seeds,
        }


def mean_head_outputs(result: ForwardResult) -> np.ndarray:
    """Mean per-head output at the query token over the prompts of ``result``: ``(L, n_heads, head_dim)`` float64."""
    if result.head_outputs is None:
        raise ValueError("head outputs were not captured")
    return result.head_outputs.astype(np.float64).mean(axis=1)


def head_effects(
    backend: ModelBackend,
    prompts: list[Prompt],
    baseline_logprob_per_token: np.ndarray,
    head_means: np.ndarray,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Average indirect effect of every head: ``(L, n_heads)`` mean change of log p per target token on
    ``prompts`` (deranged-label prompts) when the head's output at the query token is replaced by its mean
    output ``head_means[layer, head]`` from positive prompts.

    Several heads are patched per forward pass by repeating the prompt list once per head with a
    per-example patch; ``tests/test_function_vector.py`` checks this against one run per head.
    """
    L, H, _ = head_means.shape
    n = len(prompts)
    if baseline_logprob_per_token.shape != (n,):
        raise ValueError("baseline must have one value per prompt")
    per_pass = max(1, backend.cfg.batch_size // n)
    aie = np.zeros((L, H), dtype=np.float64)
    done = 0
    for layer in range(L):
        for start in range(0, H, per_pass):
            chunk = list(range(start, min(H, start + per_pass)))
            k = len(chunk)
            heads = np.repeat(np.array(chunk, dtype=np.int64), n)
            values = head_means[layer][heads].astype(np.float32)  # (k*n, head_dim)
            r = backend.run(prompts * k, head_patches=[HeadPatch(layer, heads, values)])
            diffs = (r.logprob_per_token - np.tile(baseline_logprob_per_token, k)).reshape(k, n).mean(axis=1)
            aie[layer, chunk] = diffs
            done += k
            if progress is not None:
                progress(done, L * H)
    return aie


def select_heads(effects: dict[str, np.ndarray], n_heads: int, mode: str) -> dict[str, list[tuple[int, int, float, float]]]:
    """Top heads per task: ``(layer, head, ranking_effect, task_effect)`` in rank order.

    ``universal``: one ranking from the mean effect over tasks (the paper's construction), the same set
    for every task; ``per_task``: each task ranked by its own effects.
    """
    if not effects:
        return {}
    if mode == "universal":
        mean = np.mean(np.stack(list(effects.values())), axis=0)
        rank = _top(mean, n_heads)
        return {t: [(l, h, float(mean[l, h]), float(e[l, h])) for l, h in rank] for t, e in effects.items()}
    if mode == "per_task":
        return {t: [(l, h, float(e[l, h]), float(e[l, h])) for l, h in _top(e, n_heads)] for t, e in effects.items()}
    raise ValueError(f"unknown head selection mode {mode!r}")


def _top(effects: np.ndarray, n: int) -> list[tuple[int, int]]:
    flat = np.argsort(-effects, axis=None, kind="stable")[:n]
    return [tuple(int(i) for i in np.unravel_index(f, effects.shape)) for f in flat]  # type: ignore[misc]


def compose(backend: ModelBackend, head_means: np.ndarray, heads: list[tuple[int, int]]) -> np.ndarray:
    """The function vector for ``head_means`` ``(L, n_heads, head_dim)``: the sum of the selected heads'
    mean outputs mapped into the residual stream. float64, shape (d,)."""
    v = np.zeros(backend.hidden_size, dtype=np.float64)
    for layer, head in heads:
        v += backend.head_to_residual(layer, head, head_means[layer, head])
    return v


def build_function_vector(
    backend: ModelBackend,
    seed_head_means: np.ndarray,
    selection: list[tuple[int, int, float, float]],
) -> FunctionVector:
    """Pooled and per-seed function vectors from ``seed_head_means`` ``(n_seeds, L, n_heads, head_dim)``."""
    heads = [(l, h) for l, h, _, _ in selection]
    pooled = compose(backend, seed_head_means.mean(axis=0), heads)
    per_seed = np.stack([compose(backend, m, heads) for m in seed_head_means])
    seed_dirs = normalize(per_seed, axis=1)
    direction = normalize(pooled)
    return FunctionVector(
        vector=pooled,
        direction=direction,
        natural_norm=float(np.linalg.norm(pooled)),
        heads=[{"layer": l, "head": h, "ranking_effect": r, "task_effect": e} for l, h, r, e in selection],
        seed_directions=seed_dirs,
        seed_norms=[float(np.linalg.norm(v)) for v in per_seed],
        stability=stability(seed_dirs) if len(seed_dirs) > 1 else 1.0,
        cos_pooled_vs_seeds=[float(abs(direction @ d)) for d in seed_dirs],
    )

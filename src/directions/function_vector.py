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


def effect_values(result: ForwardResult, metric: str) -> np.ndarray:
    """Per-prompt values of the indirect-effect metric.

    ``target_probability`` is the teacher-forced probability of the whole target (the product of the
    per-token probabilities), which equals the first-token probability for one-token targets and
    stays informative when the target starts with a token that carries no answer (a lone space).
    """
    if metric == "target_probability":
        return np.exp(result.logprob_sum)
    if metric == "first_token_probability":
        assert result.first_token_logprob is not None
        return np.exp(result.first_token_logprob)
    if metric == "logprob_per_token":
        return result.logprob_per_token
    raise ValueError(f"unknown indirect-effect metric {metric!r}")


def head_effects(
    backend: ModelBackend,
    prompts: list[Prompt],
    baseline: np.ndarray,
    head_means: np.ndarray,
    metric: str = "target_probability",
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Average indirect effect of every head: ``(L, n_heads)`` mean change of ``metric`` on ``prompts``
    (deranged-label prompts; ``baseline`` holds their unpatched per-prompt values) when the head's output at
    the query token is replaced by its mean output ``head_means[layer, head]`` from positive prompts.

    ``target_probability`` is the recovered probability of the correct answer (the paper's metric, taken
    over the whole target rather than its first token; see :func:`effect_values`). Several heads are
    patched per forward pass by repeating the prompt list once per head with a per-example patch;
    ``tests/test_function_vector.py`` checks this against one run per head.
    """
    L, H, _ = head_means.shape
    n = len(prompts)
    if baseline.shape != (n,):
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
            diffs = (effect_values(r, metric) - np.tile(baseline, k)).reshape(k, n).mean(axis=1)
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


# --------------------------------------------------------------------------- #
# Head count and head support (docs/DECISIONS.md D26)
# --------------------------------------------------------------------------- #


def joint_head_effect(
    backend: ModelBackend,
    prompts: list[Prompt],
    baseline: np.ndarray,
    head_means: np.ndarray,
    heads: list[tuple[int, int]],
    metric: str = "target_probability",
) -> np.ndarray:
    """Per-prompt change of ``metric`` when all of ``heads`` are patched at once (each with its mean output
    ``head_means[layer, head]``) into ``prompts`` at the query token: shape ``(n,)``."""
    patches = [HeadPatch(layer, head, head_means[layer, head].astype(np.float32)) for layer, head in heads]
    r = backend.run(prompts, head_patches=patches)
    return effect_values(r, metric) - baseline


def random_head_sets(rng: np.random.Generator, n_layers: int, n_heads: int, k: int, n_sets: int) -> list[list[tuple[int, int]]]:
    """``n_sets`` sets of ``k`` distinct heads drawn uniformly from the ``n_layers x n_heads`` grid."""
    total = n_layers * n_heads
    out = []
    for _ in range(n_sets):
        flat = rng.choice(total, size=min(k, total), replace=False)
        out.append([(int(f // n_heads), int(f % n_heads)) for f in sorted(flat)])
    return out


def choose_head_count(
    effects_by_k: dict[int, np.ndarray], rng: np.random.Generator, alpha: float = 0.05, n_boot: int = 2000
) -> dict[str, Any]:
    """The smallest head count whose joint effect is not significantly below the largest one.

    ``effects_by_k[k]`` holds the per-prompt joint effect of the top-``k`` heads (prompts pooled over tasks;
    the same prompts for every ``k``). The largest mean effect is found, then every smaller ``k`` is
    tested against it with the paired bootstrap; the smallest ``k`` whose deficit is not significant
    (``p > alpha``) is chosen. Returns the choice with the per-``k`` means and p-values.
    """
    from .stats import paired_bootstrap_test

    ks = sorted(effects_by_k)
    means = {k: float(np.mean(effects_by_k[k])) for k in ks}
    k_best = max(ks, key=lambda k: means[k])
    p_vs_best: dict[int, float | None] = {}
    chosen = k_best
    for k in ks:
        if k == k_best:
            p_vs_best[k] = None
            break
        t = paired_bootstrap_test(effects_by_k[k_best], effects_by_k[k], rng, n_boot=n_boot)
        p_vs_best[k] = t.p_value  # small: the top-k_best set is reliably better than the top-k set
        if t.p_value > alpha:
            chosen = k
            break
    return {"candidates": ks, "mean_effect_by_k": means, "k_best": k_best, "p_vs_best_by_k": p_vs_best,
            "chosen": chosen, "alpha": alpha}


def head_support_test(
    backend: ModelBackend,
    prompts: list[Prompt],
    baseline: np.ndarray,
    head_means: np.ndarray,
    heads: list[tuple[int, int]],
    rng: np.random.Generator,
    n_null: int = 16,
    metric: str = "target_probability",
    n_boot: int = 2000,
    alpha: float = 0.05,
    real_effect: np.ndarray | None = None,
    positive_mean: float | None = None,
    min_restored: float = 0.0,
) -> dict[str, Any]:
    """Does the selected head set carry the task? The joint patched effect of ``heads`` on the deranged
    prompts must be positive (paired bootstrap), exceed that of ``n_null`` random sets of the same size
    (paired excess test) and, when ``positive_mean`` (the metric on the positive prompts) is given, restore at
    least ``min_restored`` of the gap between the deranged baseline and the positive prompts. Returns the
    tests, the null's per-set means, the restored fraction and the verdict."""
    from .stats import paired_bootstrap_test, paired_excess_test

    L, H, _ = head_means.shape
    real = joint_head_effect(backend, prompts, baseline, head_means, heads, metric) if real_effect is None else real_effect
    null = np.stack([joint_head_effect(backend, prompts, baseline, head_means, s, metric)
                     for s in random_head_sets(rng, L, H, len(heads), n_null)])
    test = paired_bootstrap_test(real, np.zeros_like(real), rng, n_boot=n_boot)
    excess = paired_excess_test(real, null, rng, n_boot=n_boot)
    baseline_mean = float(np.mean(baseline))
    restored = None
    if positive_mean is not None and positive_mean > baseline_mean:
        restored = float(np.mean(real) / (positive_mean - baseline_mean))
    return {
        "n_heads": len(heads),
        "n_prompts": int(real.shape[0]),
        "mean_effect": float(np.mean(real)),
        "baseline_mean": baseline_mean,
        "positive_mean": positive_mean,
        "restored_fraction": restored,
        "min_restored": min_restored,
        "test": test.__dict__,
        "null_mean_effects": [float(x) for x in null.mean(axis=1)],
        "excess_test": excess.__dict__,
        "alpha": alpha,
        "supported": bool(test.p_value <= alpha and test.mean_diff > 0 and excess.p_value <= alpha
                          and (restored is None or restored >= min_restored)),
    }

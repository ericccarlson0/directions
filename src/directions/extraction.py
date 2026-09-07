"""Control-direction extraction from paired (positive vs permuted) prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import ExtractionConfig, PromptConfig
from .geometry import pca_direction, stability
from .model import ModelBackend
from .prompts import few_shot_prompt, paired_prompts
from .seeds import rng_for
from .tasks import Item


@dataclass
class SeedExtraction:
    seed_index: int
    seed: int
    differences: np.ndarray  # (n, L+1, d) float32: h(p+) - h(p-) at every read point
    metrics: dict[str, Any]  # behavioural metrics of positive / permuted prompts


@dataclass
class LayerDirection:
    layer: int
    direction: np.ndarray  # pooled PC1, unit norm (float64)
    seed_directions: np.ndarray  # (n_seeds, d)
    explained_variance_ratio: list[float]  # per seed
    cos_with_mean: list[float]  # per seed
    stability: float  # min pairwise |cos| across seeds
    pooled_explained_variance_ratio: float
    pooled_cos_with_mean: float
    mean_difference_norm: float
    cos_pooled_vs_seeds: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cos_with_mean": self.cos_with_mean,
            "stability": self.stability,
            "pooled_explained_variance_ratio": self.pooled_explained_variance_ratio,
            "pooled_cos_with_mean": self.pooled_cos_with_mean,
            "mean_difference_norm": self.mean_difference_norm,
            "cos_pooled_vs_seeds": self.cos_pooled_vs_seeds,
        }


def candidate_layers(n_layers: int, fractions: list[float]) -> list[int]:
    """Intervention layers at the given depth fractions (deduplicated, ascending)."""
    layers = sorted({min(n_layers - 1, max(0, int(round(f * n_layers)))) for f in fractions})
    return layers


def extract_differences(
    backend: ModelBackend,
    prompt_cfg: PromptConfig,
    pool: list[Item],
    run_seed: int,
    task_name: str,
    n_seeds: int,
) -> list[SeedExtraction]:
    """For each seed, resample demonstrations/derangements and capture h(p+) - h(p-)."""
    out: list[SeedExtraction] = []
    for s in range(n_seeds):
        seed = rng_for(run_seed, "extraction", task_name, s).integers(0, 2**31 - 1)
        rng = np.random.default_rng(seed)
        pos, neg = [], []
        for q in pool:
            p, n = paired_prompts(prompt_cfg, pool, q, rng)
            pos.append(p)
            neg.append(n)
        rp = backend.run(pos, capture=True)
        rn = backend.run(neg, capture=True)
        assert rp.residuals is not None and rn.residuals is not None
        diffs = np.transpose(rp.residuals - rn.residuals, (1, 0, 2))  # (n, L+1, d)
        out.append(
            SeedExtraction(
                seed_index=s,
                seed=int(seed),
                differences=diffs,
                metrics={"positive": rp.metrics_dict(), "permuted": rn.metrics_dict()},
            )
        )
    return out


def directions_from_differences(
    seeds: list[SeedExtraction], layers: list[int], cfg: ExtractionConfig
) -> dict[int, LayerDirection]:
    """PC1 per seed and per layer, cross-seed stability, and the pooled direction."""
    out: dict[int, LayerDirection] = {}
    for l in layers:
        per_seed = [pca_direction(s.differences[:, l, :], center=cfg.center) for s in seeds]
        seed_dirs = np.stack([p.direction for p in per_seed])
        pooled_mat = np.concatenate([s.differences[:, l, :] for s in seeds], axis=0)
        pooled = pca_direction(pooled_mat, center=cfg.center)
        out[l] = LayerDirection(
            layer=l,
            direction=pooled.direction,
            seed_directions=seed_dirs,
            explained_variance_ratio=[p.explained_variance_ratio for p in per_seed],
            cos_with_mean=[p.cos_with_mean for p in per_seed],
            stability=stability(seed_dirs),
            pooled_explained_variance_ratio=pooled.explained_variance_ratio,
            pooled_cos_with_mean=pooled.cos_with_mean,
            mean_difference_norm=float(np.linalg.norm(pooled_mat.astype(np.float64).mean(axis=0))),
            cos_pooled_vs_seeds=[float(abs(pooled.direction @ d)) for d in seed_dirs],
        )
    return out


# --------------------------------------------------------------------------- #
# Demonstration-variation null directions (docs/DECISIONS.md D15)
# --------------------------------------------------------------------------- #


def extract_demo_variation(
    backend: ModelBackend,
    prompt_cfg: PromptConfig,
    pool: list[Item],
    run_seed: int,
    task_name: str,
    n: int,
) -> list[np.ndarray]:
    """``n`` difference matrices ``h(p_i^{+,a}) - h(p_i^{+,b})`` between two *correct*
    demonstration samples for the same query: the same kind of prompt-difference
    at the query token as the extraction contrast, but without any task contrast.

    Returns a list of arrays shaped (n_items, L+1, d).
    """
    out: list[np.ndarray] = []
    for k in range(n):
        rng = np.random.default_rng(rng_for(run_seed, "demo_variation", task_name, k).integers(0, 2**31 - 1))
        a = [few_shot_prompt(prompt_cfg, pool, q, rng) for q in pool]
        b = [few_shot_prompt(prompt_cfg, pool, q, rng) for q in pool]
        ra = backend.run(a, capture=True)
        rb = backend.run(b, capture=True)
        assert ra.residuals is not None and rb.residuals is not None
        out.append(np.transpose(ra.residuals - rb.residuals, (1, 0, 2)))
    return out


def demo_variation_directions(
    diffs: list[np.ndarray], layers: list[int], cfg: ExtractionConfig
) -> dict[int, list[np.ndarray]]:
    """PC1 (same extraction rule as the control) of each demo-variation matrix, per layer."""
    return {l: [pca_direction(d[:, l, :], center=cfg.center).direction for d in diffs] for l in layers}

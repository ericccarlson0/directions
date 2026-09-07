"""Control-direction extraction from paired activation differences.

For query ``i`` we build a positive prompt (correct demonstrations) and a
permuted prompt (identical demonstration *inputs*, deranged demonstration
*outputs*), read the residual stream at the final query token, and take

    d_i = h_l(p_i^+) - h_l(p_i^-)

The control direction ``v_{t,l}`` is the first principal component of
``{d_i}``, normalised to unit norm. Independent extraction seeds resample the
demonstrations and the derangement, giving a cross-seed stability measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mathx
from .config import Config
from .model import LanguageModel
from .prompts import build_pairs
from .tasks import TaskItem


@dataclass(frozen=True)
class DirectionSpec:
    """A concrete residual-stream intervention: ``alpha * vector`` at ``layer``."""

    layer: int
    vector: np.ndarray
    alpha: float
    label: str = "control"

    def scaled(self, alpha: float) -> "DirectionSpec":
        return DirectionSpec(self.layer, self.vector, alpha, self.label)


@dataclass
class LayerExtraction:
    layer: int
    seed_directions: np.ndarray  # (n_seeds, d_model), unit norm, sign-aligned
    consensus: np.ndarray  # (d_model,), unit norm
    stability_min: float
    stability_mean: float
    pairwise_abs_cosine: list[float]
    explained_variance_ratio: list[float]
    mean_diff_cosine: list[float]  # cos(PC1_s, mean_i d_i) per seed
    mean_diff_norm: list[float]
    consensus_seed_cosine: list[float]

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "stability_min": self.stability_min,
            "stability_mean": self.stability_mean,
            "pairwise_abs_cosine": self.pairwise_abs_cosine,
            "explained_variance_ratio": self.explained_variance_ratio,
            "mean_diff_cosine": self.mean_diff_cosine,
            "mean_diff_norm": self.mean_diff_norm,
            "consensus_seed_cosine": self.consensus_seed_cosine,
        }


@dataclass
class TaskExtraction:
    task: str
    layers: dict[int, LayerExtraction] = field(default_factory=dict)
    baseline_norms: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "layers": {str(k): v.to_dict() for k, v in sorted(self.layers.items())},
            "baseline_residual_norm_median": self.baseline_norms,
        }


def extraction_rng(base_seed: int, task: str, seed_index: int) -> np.random.Generator:
    """Reproducible per-(run, task, extraction-seed) generator."""
    ss = np.random.SeedSequence(
        entropy=base_seed, spawn_key=(mathx.stable_key("extraction", task), seed_index)
    )
    return np.random.default_rng(ss)


def extract_directions(
    lm: LanguageModel,
    cfg: Config,
    task_name: str,
    pool: list[TaskItem],
    queries: list[TaskItem],
    candidate_layers: list[int],
) -> TaskExtraction:
    """Extract a control direction at every candidate layer, for every seed.

    A single pair of forward passes per seed yields the residual stream at
    *all* layers, so candidate layers cost nothing extra.
    """
    n_seeds = cfg.extraction.n_seeds
    diffs: dict[int, list[np.ndarray]] = {l: [] for l in candidate_layers}
    per_seed: dict[int, list[np.ndarray]] = {l: [] for l in candidate_layers}
    evr: dict[int, list[float]] = {l: [] for l in candidate_layers}
    mean_cos: dict[int, list[float]] = {l: [] for l in candidate_layers}
    mean_norm: dict[int, list[float]] = {l: [] for l in candidate_layers}

    for s in range(n_seeds):
        rng = extraction_rng(cfg.run.seed, task_name, s)
        pairs = build_pairs(rng, cfg.data.template, pool, queries, cfg.data.n_shot_extraction)
        pos = [p for p, _ in pairs]
        neg = [n for _, n in pairs]
        h_pos = lm.capture(pos, batch_size=cfg.compute.batch_size, max_seq_len=cfg.compute.max_seq_len)
        h_neg = lm.capture(neg, batch_size=cfg.compute.batch_size, max_seq_len=cfg.compute.max_seq_len)
        for l in candidate_layers:
            D = mathx.as_f64(h_pos[l] - h_neg[l])  # (N, d)
            mu = D.mean(axis=0)
            v, ratio = mathx.pc1(D, center=cfg.extraction.center_pca)
            v = mathx.orient_like(v, mu)  # PCA sign is arbitrary; fix it to the mean
            diffs[l].append(D)
            per_seed[l].append(v)
            evr[l].append(ratio)
            mean_cos[l].append(mathx.cosine(v, mu))
            mean_norm[l].append(float(np.linalg.norm(mu)))

    out = TaskExtraction(task=task_name)
    for l in candidate_layers:
        V = np.stack(per_seed[l])
        pairwise = [
            mathx.abs_cosine(V[i], V[j]) for i in range(n_seeds) for j in range(i + 1, n_seeds)
        ]
        pooled = np.concatenate(diffs[l], axis=0)
        consensus, _ = mathx.pc1(pooled, center=cfg.extraction.center_pca)
        consensus = mathx.orient_like(consensus, pooled.mean(axis=0))
        out.layers[l] = LayerExtraction(
            layer=l,
            seed_directions=V,
            consensus=consensus,
            stability_min=float(np.min(pairwise)),
            stability_mean=float(np.mean(pairwise)),
            pairwise_abs_cosine=[float(x) for x in pairwise],
            explained_variance_ratio=[float(x) for x in evr[l]],
            mean_diff_cosine=[float(x) for x in mean_cos[l]],
            mean_diff_norm=[float(x) for x in mean_norm[l]],
            consensus_seed_cosine=[mathx.abs_cosine(consensus, v) for v in V],
        )
    return out


def median_residual_norms(residuals: np.ndarray) -> np.ndarray:
    """Median ``||h_l(x)||`` over examples, per layer. Shape ``(n_layers + 1,)``."""
    return np.median(np.linalg.norm(mathx.as_f64(residuals), axis=-1), axis=-1)


def alpha_from_rho(rho: float, median_norm: float) -> float:
    r"""Convert a relative strength to an absolute one.

    ``rho = ||alpha v|| / median_x ||h_l(x)||`` with ``||v|| = 1``, so
    ``alpha = rho * median_norm``.
    """
    return float(rho) * float(median_norm)

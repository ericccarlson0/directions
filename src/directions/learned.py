"""Learned single vector: the best one direction can do at a layer (docs/DECISIONS.md D31).

The head-mean function vector (D21) is one candidate direction. A learned vector is fitted directly: with the
model frozen, one vector ``v`` added to the residual at a candidate layer at the query token of the zero-shot
prompts is optimised to raise the summed (scored) log-probability of the target over the extraction pool. The
gradient with respect to ``v`` is the activation gradient at that read point of the steered forward pass, so
each step is one forward and one backward over the pool (``ModelBackend.gradients_with_scores``); the
optimiser is Adam on the vector, in float64 on the host, with the vector projected back to a fixed norm after
every step. The norm is the median residual norm at the layer (``radius``), so the fitted vector's natural
strength is one residual norm and the calibration grid's ``rho`` scales it as it scales the function vector.

Several seeds (random unit initialisations) give the stability of the solution (min pairwise cosine, signed:
the sign is meaningful), and their normalised mean is the pooled direction. The same procedure with the
targets permuted across items (a derangement) gives the matched null of the construction: a vector fitted
with the same budget to answers that do not belong to the inputs (the ``demo_variation`` control kind under
this control).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import LearnedVectorConfig, PromptConfig
from .extraction import LayerDirection
from .geometry import normalize, random_unit_vector
from .model import Intervention, ModelBackend
from .prompts import Prompt, zero_shot_prompt
from .seeds import rng_for
from .tasks import Item, derangement


@dataclass
class FitResult:
    vector: np.ndarray  # (d,) float64, norm == radius
    radius: float
    losses: list[float]  # mean loss (−summed scored log p per example) before step 0, every ``log_every`` steps, and after the last
    loss_steps: list[int]
    n_steps: int


def _loss(scores: np.ndarray) -> float:
    return float(-np.mean(scores))


def fit_vector(backend: ModelBackend, prompts: list[Prompt], layer: int, radius: float, rng: np.random.Generator,
               cfg: LearnedVectorConfig, init: np.ndarray | None = None) -> FitResult:
    """Projected Adam on one vector at ``layer`` over ``prompts`` (D31). Deterministic given ``rng`` and the
    backend's forward pass."""
    d = backend.hidden_size
    v = radius * (normalize(np.asarray(init, dtype=np.float64)) if init is not None else random_unit_vector(rng, d))
    lr = cfg.lr_fraction * radius
    b1, b2, eps = cfg.beta1, cfg.beta2, 1e-8
    m = np.zeros(d)
    s = np.zeros(d)
    losses: list[float] = []
    steps: list[int] = []
    n = len(prompts)
    for t in range(cfg.n_steps + 1):
        grads, scores = backend.gradients_with_scores(prompts, interventions=[Intervention(layer, v, 1.0)], read_points=[layer])
        if t % cfg.log_every == 0 or t == cfg.n_steps:
            losses.append(_loss(scores))
            steps.append(t)
        if t == cfg.n_steps:
            break
        g = -grads[0].astype(np.float64).sum(axis=0) / n  # d loss / d v: loss = −mean over examples of the summed scored log p
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** (t + 1))
        s_hat = s / (1 - b2 ** (t + 1))
        v = v - lr * m_hat / (np.sqrt(s_hat) + eps)
        v = radius * normalize(v)
    return FitResult(vector=v, radius=float(radius), losses=losses, loss_steps=steps, n_steps=cfg.n_steps)


def learned_directions(backend: ModelBackend, prompt_cfg: PromptConfig, pool: list[Item], layers: list[int],
                       radii: dict[int, float], run_seed: int, task_name: str, cfg: LearnedVectorConfig, n_seeds: int
                       ) -> tuple[dict[int, LayerDirection], dict[int, list[FitResult]]]:
    """One learned vector per candidate layer, from ``n_seeds`` random initialisations; the pooled direction
    is the normalised mean of the seed vectors and the stability their min pairwise (signed) cosine."""
    prompts = [zero_shot_prompt(prompt_cfg, x) for x in pool]
    directions: dict[int, LayerDirection] = {}
    fits: dict[int, list[FitResult]] = {}
    for layer in layers:
        fits[layer] = [fit_vector(backend, prompts, layer, radii[layer], rng_for(run_seed, "learned_vector", task_name, layer, i), cfg)
                       for i in range(n_seeds)]
        units = np.stack([normalize(f.vector) for f in fits[layer]])
        pooled = normalize(units.mean(axis=0))
        cos = units @ units.T
        stability = float(min(cos[i, j] for i in range(n_seeds) for j in range(i + 1, n_seeds))) if n_seeds > 1 else 1.0
        directions[layer] = LayerDirection(
            layer=layer, direction=pooled, seed_directions=units,
            explained_variance_ratio=[], cos_with_mean=[float(u @ pooled) for u in units],
            stability=stability, pooled_explained_variance_ratio=float("nan"), pooled_cos_with_mean=float("nan"),
            mean_difference_norm=float(radii[layer]), cos_pooled_vs_seeds=[float(u @ pooled) for u in units],
            kind="learned_vector",
        )
    return directions, fits


def permuted_target_vectors(backend: ModelBackend, prompt_cfg: PromptConfig, pool: list[Item], layer: int, radius: float,
                            run_seed: int, task_name: str, cfg: LearnedVectorConfig, n: int) -> list[np.ndarray]:
    """``n`` unit vectors fitted at ``layer`` with the same budget to the pool's targets permuted across items
    (a derangement per control): the matched null of the learned construction."""
    out = []
    for i in range(n):
        rng = rng_for(run_seed, "learned_permuted", task_name, layer, i)
        perm = derangement(len(pool), rng)
        items = [Item(x.input, pool[j].output) for x, j in zip(pool, perm)]
        prompts = [zero_shot_prompt(prompt_cfg, x) for x in items]
        out.append(normalize(fit_vector(backend, prompts, layer, radius, rng, cfg).vector))
    return out


def fit_summary(fits: dict[int, list[FitResult]]) -> dict[str, Any]:
    return {str(l): {"radius": fs[0].radius, "n_steps": fs[0].n_steps, "loss_steps": fs[0].loss_steps,
                     "losses": [f.losses for f in fs], "final_loss": [f.losses[-1] for f in fs],
                     "initial_loss": [f.losses[0] for f in fs]}
            for l, fs in fits.items()}

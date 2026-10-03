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

from .config import LearnedVectorConfig, LearnedVectorWDConfig, PromptConfig
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
    """One learned vector per candidate layer from ``n_seeds`` random initialisations. The direction carried
    forward is the first seed's fit (one deterministic fit); the other seeds measure how unique the solution is,
    the stability being the min pairwise (signed) cosine between the seeds' vectors. With d free parameters
    and 64 prompts the solution set can be large (D31, amended), so the stability is reported, not gated, and
    the mean of dissimilar solutions is not used as the control."""
    prompts = [zero_shot_prompt(prompt_cfg, x) for x in pool]
    directions: dict[int, LayerDirection] = {}
    fits: dict[int, list[FitResult]] = {}
    for layer in layers:
        fits[layer] = [fit_vector(backend, prompts, layer, radii[layer], rng_for(run_seed, "learned_vector", task_name, layer, i), cfg)
                       for i in range(n_seeds)]
        units = np.stack([normalize(f.vector) for f in fits[layer]])
        carried = units[0]
        cos = units @ units.T
        stability = float(min(cos[i, j] for i in range(n_seeds) for j in range(i + 1, n_seeds))) if n_seeds > 1 else 1.0
        directions[layer] = LayerDirection(
            layer=layer, direction=carried, seed_directions=units,
            explained_variance_ratio=[], cos_with_mean=[float(u @ carried) for u in units],
            stability=stability, pooled_explained_variance_ratio=float("nan"), pooled_cos_with_mean=float("nan"),
            mean_difference_norm=float(radii[layer]), cos_pooled_vs_seeds=[float(u @ carried) for u in units],
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


# --------------------------------------------------------------------------- #
# The weight-decayed learned vector (docs/DECISIONS.md D43)
# --------------------------------------------------------------------------- #
#
# A second learned construction beside D31's, not a replacement: the D31 fit starts from a random direction at
# the full median residual norm and keeps the norm fixed, so its solution set is large (seed cosines 0.1-0.2) and
# the norm is imposed, not found. Here the norm is free and penalised: the objective is the D31 loss plus
# (weight_decay / 2) * (||v|| / radius)^2, the radius (median residual norm at the layer) only setting the scale
# of the penalty and of the step. The first seed starts at zero, so the carried fit is the one gradient descent
# reaches from no intervention at all; the other seeds start from random directions at ``init_fraction`` of the
# radius and say whether different starts reach the same stationary point (the stability). The penalty is in the
# objective (an L2 penalty, coupled), so its stationary point does not depend on the optimiser. The optimiser is
# gradient descent with momentum (rotation-equivariant: the fit does not depend on the residual's coordinate basis,
# and the penalty shrinks every direction the task does not use at the same known rate); Adam is kept as an option,
# because the CPU probe of D43 found that with it random starts keep most of their start (Adam's per-coordinate
# scaling drowns the penalty wherever the task gradient is large). The stationarity is reported: the norm of the objective's gradient at the
# end over the norm of the penalty's gradient (0 at an exact stationary point, where the two balance).


@dataclass
class RegularisedFit:
    vector: np.ndarray  # (d,) float64, its norm found by the fit
    radius: float  # the scale of the penalty and of the step: the median residual norm at the layer
    weight_decay: float
    losses: list[float]  # the task loss (D31's: −mean summed scored log p) at the logged steps
    objectives: list[float]  # the task loss plus the penalty, at the logged steps
    norms: list[float]  # ||v|| / radius at the logged steps
    loss_steps: list[int]
    n_steps: int
    stationarity: float  # ||grad of the objective|| / ||grad of the penalty|| at the final vector (nan at v = 0)
    drift: float  # ||v_final - v at 90 % of the steps|| / ||v_final||: how far the last tenth of the fit still moved it
    init: np.ndarray  # (d,) the starting vector (zero for the carried fit)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.vector))


def regularised_objective(task_loss: float, v: np.ndarray, radius: float, weight_decay: float) -> float:
    return float(task_loss + 0.5 * weight_decay * float(v @ v) / radius ** 2)


def cosine_schedule(start: float, n_steps: int) -> np.ndarray:
    """``start`` cosine-decayed to zero over ``n_steps``."""
    t = np.arange(n_steps)
    return start * 0.5 * (1.0 + np.cos(np.pi * t / max(1, n_steps)))


def regularised_step_schedule(step_fraction: float, radius: float, d: int, n_steps: int) -> np.ndarray:
    """The Adam step size per coordinate at every step: ``step_fraction * radius / sqrt(d)`` (an update moving
    every coordinate by the step moves the vector by about ``step_fraction * radius``), cosine-decayed to zero
    over ``n_steps`` so that the iterates settle on a stationary point."""
    return cosine_schedule(step_fraction * radius / np.sqrt(d), n_steps)


def adam_regularised(grad_fn: Any, v0: np.ndarray, radius: float, weight_decay: float, n_steps: int, lrs: np.ndarray,
                     beta1: float = 0.9, beta2: float = 0.999, log_every: int = 10
                     ) -> tuple[np.ndarray, list[int], list[float], list[float], list[float], float, float]:
    """Adam on ``task_loss(v) + (weight_decay / 2) * ||v||^2 / radius^2`` from ``v0``. ``grad_fn(v)`` returns the
    task loss and its gradient with respect to ``v``. Returns the final vector, the logged steps, task losses,
    objectives and norms (over the radius), the stationarity at the final vector and the drift over the last
    tenth of the steps. Separated from the model so that it can be checked on a problem with a known minimiser.

    Two convergence readouts because neither suffices alone: the stationarity (gradient norm of the objective over
    the penalty's) is dominated by the stiffest directions, where an iterate a step size away from the minimum
    already has a large gradient; the drift says whether the vector itself had stopped moving."""
    v = np.asarray(v0, dtype=np.float64).copy()
    m = np.zeros_like(v)
    s = np.zeros_like(v)
    eps = 1e-12  # float64 arithmetic; only guards a coordinate whose gradient has always been zero
    steps: list[int] = []
    losses: list[float] = []
    objectives: list[float] = []
    norms: list[float] = []
    stationarity = float("nan")
    t_late = int(0.9 * n_steps)
    v_late = v.copy()
    for t in range(n_steps + 1):
        if t == t_late:
            v_late = v.copy()
        loss, g_task = grad_fn(v)
        g_pen = weight_decay * v / radius ** 2
        g = g_task + g_pen
        if t % log_every == 0 or t == n_steps:
            steps.append(t)
            losses.append(float(loss))
            objectives.append(regularised_objective(loss, v, radius, weight_decay))
            norms.append(float(np.linalg.norm(v) / radius))
        if t == n_steps:
            pen = float(np.linalg.norm(g_pen))
            stationarity = float(np.linalg.norm(g) / pen) if pen > 0 else float("nan")
            break
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** (t + 1))
        s_hat = s / (1 - beta2 ** (t + 1))
        v = v - lrs[t] * m_hat / (np.sqrt(s_hat) + eps)
    nv = float(np.linalg.norm(v))
    drift = float(np.linalg.norm(v - v_late) / nv) if nv > 0 else float("nan")
    return v, steps, losses, objectives, norms, stationarity, drift


def sgd_regularised(grad_fn: Any, v0: np.ndarray, radius: float, weight_decay: float, n_steps: int, lrs: np.ndarray,
                    momentum: float = 0.9, max_step_fraction: float = 0.05, log_every: int = 10
                    ) -> tuple[np.ndarray, list[int], list[float], list[float], list[float], float, float]:
    """Gradient descent with heavy-ball momentum on ``task_loss(v) + (weight_decay / 2) * ||v||^2 / radius^2`` from
    ``v0``: ``u <- momentum * u + grad``, ``v <- v - lrs[t] * radius^2 * u``, the step's length capped at
    ``max_step_fraction * radius`` (which keeps the fixed points and the rotation equivariance). ``lrs`` is in units
    of 1 / (nats per example), so that ``lrs[t] * weight_decay`` is the rate per step at which the penalty shrinks a
    direction the task does not use. Returns what :func:`adam_regularised` returns."""
    v = np.asarray(v0, dtype=np.float64).copy()
    u = np.zeros_like(v)
    steps: list[int] = []
    losses: list[float] = []
    objectives: list[float] = []
    norms: list[float] = []
    stationarity = float("nan")
    t_late = int(0.9 * n_steps)
    v_late = v.copy()
    cap = max_step_fraction * radius
    for t in range(n_steps + 1):
        if t == t_late:
            v_late = v.copy()
        loss, g_task = grad_fn(v)
        g_pen = weight_decay * v / radius ** 2
        g = g_task + g_pen
        if t % log_every == 0 or t == n_steps:
            steps.append(t)
            losses.append(float(loss))
            objectives.append(regularised_objective(loss, v, radius, weight_decay))
            norms.append(float(np.linalg.norm(v) / radius))
        if t == n_steps:
            pen = float(np.linalg.norm(g_pen))
            stationarity = float(np.linalg.norm(g) / pen) if pen > 0 else float("nan")
            break
        u = momentum * u + g
        step = lrs[t] * radius ** 2 * u
        n = float(np.linalg.norm(step))
        if n > cap:
            step *= cap / n
        v = v - step
    nv = float(np.linalg.norm(v))
    drift = float(np.linalg.norm(v - v_late) / nv) if nv > 0 else float("nan")
    return v, steps, losses, objectives, norms, stationarity, drift


def fit_vector_wd(backend: ModelBackend, prompts: list[Prompt], layer: int, radius: float, init: np.ndarray,
                  cfg: LearnedVectorWDConfig) -> RegularisedFit:
    """The weight-decayed fit at ``layer`` over ``prompts`` from ``init`` (D43). Deterministic given ``init`` and
    the backend's forward pass."""
    n = len(prompts)

    def grad_fn(v: np.ndarray) -> tuple[float, np.ndarray]:
        grads, scores = backend.gradients_with_scores(prompts, interventions=[Intervention(layer, v, 1.0)], read_points=[layer])
        return _loss(scores), -grads[0].astype(np.float64).sum(axis=0) / n

    d = backend.hidden_size
    v0 = np.asarray(init, dtype=np.float64)
    if cfg.optimizer == "sgd":
        lrs = cosine_schedule(cfg.lr, cfg.n_steps)
        v, steps, losses, objectives, norms, stationarity, drift = sgd_regularised(
            grad_fn, v0, radius, cfg.weight_decay, cfg.n_steps, lrs, momentum=cfg.momentum,
            max_step_fraction=cfg.max_step_fraction, log_every=cfg.log_every)
    else:
        lrs = regularised_step_schedule(cfg.step_fraction, radius, d, cfg.n_steps)
        v, steps, losses, objectives, norms, stationarity, drift = adam_regularised(
            grad_fn, v0, radius, cfg.weight_decay, cfg.n_steps, lrs, beta1=cfg.beta1, beta2=cfg.beta2, log_every=cfg.log_every)
    return RegularisedFit(vector=v, radius=float(radius), weight_decay=cfg.weight_decay, losses=losses, objectives=objectives,
                          norms=norms, loss_steps=steps, n_steps=cfg.n_steps, stationarity=stationarity, drift=drift,
                          init=np.asarray(init, dtype=np.float64))


def wd_init(seed_index: int, rng: np.random.Generator, d: int, radius: float, init_fraction: float) -> np.ndarray:
    """The first seed starts at zero (the carried fit); the others at a random direction of ``init_fraction``
    times the radius."""
    if seed_index == 0:
        return np.zeros(d)
    return init_fraction * radius * random_unit_vector(rng, d)


def learned_directions_wd(backend: ModelBackend, prompt_cfg: PromptConfig, pool: list[Item], layers: list[int],
                          radii: dict[int, float], run_seed: int, task_name: str, cfg: LearnedVectorWDConfig, n_seeds: int
                          ) -> tuple[dict[int, LayerDirection], dict[int, list[RegularisedFit]]]:
    """One weight-decayed vector per candidate layer (D43). The direction carried forward is the zero-start fit,
    its natural norm the norm the fit found (``mean_difference_norm``, the unit of ``rho`` under the natural
    strength unit); the random starts give the stability (min pairwise signed cosine over all seeds' fits)."""
    prompts = [zero_shot_prompt(prompt_cfg, x) for x in pool]
    d = backend.hidden_size
    directions: dict[int, LayerDirection] = {}
    fits: dict[int, list[RegularisedFit]] = {}
    for layer in layers:
        fits[layer] = [fit_vector_wd(backend, prompts, layer, radii[layer],
                                     wd_init(i, rng_for(run_seed, "learned_vector_wd", task_name, layer, i), d, radii[layer],
                                             cfg.init_fraction), cfg)
                       for i in range(n_seeds)]
        if not fits[layer][0].norm > 0:
            raise ValueError(f"{task_name}: the weight-decayed fit at layer {layer} stayed at zero (weight_decay too large)")
        units = np.stack([normalize(f.vector) for f in fits[layer]])
        carried = units[0]
        cos = units @ units.T
        stability = float(min(cos[i, j] for i in range(n_seeds) for j in range(i + 1, n_seeds))) if n_seeds > 1 else 1.0
        directions[layer] = LayerDirection(
            layer=layer, direction=carried, seed_directions=units,
            explained_variance_ratio=[], cos_with_mean=[float(u @ carried) for u in units],
            stability=stability, pooled_explained_variance_ratio=float("nan"), pooled_cos_with_mean=float("nan"),
            mean_difference_norm=fits[layer][0].norm, cos_pooled_vs_seeds=[float(u @ carried) for u in units],
            kind="learned_vector_wd",
        )
    return directions, fits


def permuted_target_vectors_wd(backend: ModelBackend, prompt_cfg: PromptConfig, pool: list[Item], layer: int, radius: float,
                               run_seed: int, task_name: str, cfg: LearnedVectorWDConfig, n: int) -> list[np.ndarray]:
    """The matched null of the weight-decayed construction: ``n`` unit vectors fitted from zero with the same
    budget and penalty to the pool's targets permuted across items (a derangement per control). A fit that stays
    at zero (nothing to gain on deranged targets) is replaced by a random direction from the same generator."""
    out = []
    for i in range(n):
        rng = rng_for(run_seed, "learned_wd_permuted", task_name, layer, i)
        perm = derangement(len(pool), rng)
        items = [Item(x.input, pool[j].output) for x, j in zip(pool, perm)]
        prompts = [zero_shot_prompt(prompt_cfg, x) for x in items]
        fit = fit_vector_wd(backend, prompts, layer, radius, np.zeros(backend.hidden_size), cfg)
        out.append(normalize(fit.vector) if fit.norm > 0 else random_unit_vector(rng, backend.hidden_size))
    return out


def fit_summary_wd(fits: dict[int, list[RegularisedFit]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for l, fs in fits.items():
        out[str(l)] = {
            "radius": fs[0].radius, "weight_decay": fs[0].weight_decay, "n_steps": fs[0].n_steps, "loss_steps": fs[0].loss_steps,
            "losses": [f.losses for f in fs], "objectives": [f.objectives for f in fs], "norms": [f.norms for f in fs],
            "final_loss": [f.losses[-1] for f in fs], "initial_loss": [f.losses[0] for f in fs],
            "fitted_norm": [f.norm for f in fs], "fitted_norm_over_radius": [f.norm / f.radius for f in fs],
            "stationarity": [f.stationarity for f in fs], "drift": [f.drift for f in fs],
            "init_norm_over_radius": [float(np.linalg.norm(f.init) / f.radius) for f in fs],
            # how much of a random start survives in the fit (nan for the zero start)
            "cos_with_init": [float(normalize(f.vector) @ normalize(f.init)) if np.linalg.norm(f.init) > 0 and f.norm > 0
                              else float("nan") for f in fs],
        }
    return out

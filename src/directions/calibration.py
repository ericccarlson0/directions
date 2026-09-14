"""Intervention calibration: layer x strength sweep on the calibration pool.

A grid point injects the unit direction at a candidate layer with absolute
strength ``alpha = rho * unit(layer)``, where the unit is the median residual
norm at the layer (``strength_unit: layer_norm``) or the direction's own
natural norm (``natural``; docs/DECISIONS.md D22). A point is *reliable* when

* its one-sided paired bootstrap p <= ``bootstrap_alpha``;
* its mean improvement of the decision metric (target log-probability per
  token) >= ``min_improvement`` (nats per token);
* its improvement exceeds that of ``n_random_screen`` matched random
  directions (same layer, same norm) on the calibration pool: by default the
  paired excess test of docs/DECISIONS.md D18 (``screen_test: paired_excess``,
  p <= ``random_screen_max_p``), or the iteration-2 rank rule (``rank``).

Selection (docs/EXPERIMENT.md): the layer is the earliest candidate layer with
a reliable point (``layer_rule: earliest``) or the one whose selected point
improves most (``best``); within it, the reliable strength nearest
``reference_rho`` in log distance, or the weakest reliable strength when no
reference is set. The grid is extended geometrically when the best point sits
at its upper edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .config import CalibrationConfig, EvaluationConfig, PromptConfig
from .extraction import LayerDirection
from .geometry import matched_random_controls
from .model import ForwardResult, Intervention, ModelBackend
from .prompts import Prompt, zero_shot_prompt
from .seeds import rng_for
from .stats import PairedTest, compare_to_null, paired_bootstrap_test, paired_excess_test
from .tasks import Item


@dataclass
class GridPoint:
    layer: int
    rho: float  # in the configured strength unit
    alpha: float  # absolute injected norm
    rho_layer_norm: float  # alpha / median ||h_l||, the D1-D21 unit (for comparison across protocols)
    extended: bool
    metrics: dict[str, float]
    test: PairedTest
    passes_bootstrap: bool
    passes_min_improvement: bool
    screen: dict[str, Any] | None = None  # random-control screen result, if run
    passes_screen: bool | None = None
    reliable: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = {
            "layer": self.layer,
            "rho": self.rho,
            "alpha": self.alpha,
            "rho_layer_norm": self.rho_layer_norm,
            "extended": self.extended,
            "metrics": self.metrics,
            "test": self.test.__dict__,
            "passes_bootstrap": self.passes_bootstrap,
            "passes_min_improvement": self.passes_min_improvement,
            "screen": self.screen,
            "passes_screen": self.passes_screen,
            "reliable": self.reliable,
        }
        return d

    def selection(self) -> Selection:
        return Selection(self.layer, self.rho, self.alpha, self.rho_layer_norm)


@dataclass
class Selection:
    layer: int
    rho: float
    alpha: float
    rho_layer_norm: float | None = None


@dataclass
class CalibrationResult:
    baseline_metrics: dict[str, float]
    layer_norms: dict[int, float]  # median ||h_l(x)|| over the calibration pool
    strength_unit: str
    strength_units: dict[int, float]  # what rho multiplies at each layer
    grid: list[GridPoint]
    selected: Selection | None
    weakest: Selection | None  # weakest reliable strength at the selected layer (exploratory unless selected)
    strongest: Selection | None  # strongest reliable strength at the selected layer (exploratory)
    middle: Selection | None  # a reliable strength between weakest and strongest (exploratory)
    reference_rho: float | None = None
    layer_rule: str = "earliest"
    reason: str | None = None  # why nothing was selected
    per_example: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        def sel(s: Selection | None) -> dict[str, Any] | None:
            return None if s is None else s.__dict__

        return {
            "baseline_metrics": self.baseline_metrics,
            "layer_norms": {str(k): v for k, v in self.layer_norms.items()},
            "strength_unit": self.strength_unit,
            "strength_units": {str(k): v for k, v in self.strength_units.items()},
            "reference_rho": self.reference_rho,
            "layer_rule": self.layer_rule,
            "grid": [g.as_dict() for g in self.grid],
            "selected": sel(self.selected),
            "weakest": sel(self.weakest),
            "strongest": sel(self.strongest),
            "middle": sel(self.middle),
            "reason": self.reason,
        }


def median_layer_norms(base: ForwardResult, layers: list[int]) -> dict[int, float]:
    assert base.residuals is not None
    return {l: float(np.median(np.linalg.norm(base.residuals[l].astype(np.float64), axis=1))) for l in layers}


def random_screen(
    backend: ModelBackend,
    prompts: list[Prompt],
    base: ForwardResult,
    layer: int,
    alpha: float,
    v: np.ndarray,
    real_mean_diff: float,
    real_diff: np.ndarray,
    controls: list[tuple[str, np.ndarray]],
    cfg: CalibrationConfig,
    rng: np.random.Generator,
    reference: Any = None,
    real_kl: np.ndarray | None = None,
    kl_rng: np.random.Generator | None = None,
    real_collateral_kl: np.ndarray | None = None,
    neutral_probe: Callable[[int, np.ndarray, float], np.ndarray] | None = None,
    real_neutral_kl: np.ndarray | None = None,
) -> dict[str, Any]:
    """Improvement of each matched random direction at ``(layer, alpha)``, and the
    rank comparison plus the paired excess test of the real direction against them.
    With ``reference`` (the baseline's query-token distribution) the controls' damage is
    recorded too and the real direction's per-example KL ``real_kl`` is tested against it."""
    diffs = []
    per_example = []
    kls = []
    ckls = []
    nkls = []
    for kind, u in controls:
        r = backend.run(prompts, interventions=[Intervention(layer, u, alpha)], reference_logprobs=reference)
        t = paired_bootstrap_test(r.logprob_per_token, base.logprob_per_token, rng, n_boot=cfg.n_boot)
        metrics = r.metrics_dict()
        if neutral_probe is not None:
            nkl = neutral_probe(layer, u, alpha)
            nkls.append(nkl)
            metrics["neutral_kl_mean"], metrics["neutral_kl_median"] = float(np.mean(nkl)), float(np.median(nkl))
        diffs.append({"kind": kind, "mean_diff": t.mean_diff, "p_value": t.p_value, "metrics": metrics})
        per_example.append(r.logprob_per_token - base.logprob_per_token)
        if r.kl_from_reference is not None:
            kls.append(r.kl_from_reference)
            ckls.append(r.collateral_kl)
    comp = compare_to_null(real_mean_diff, np.array([d["mean_diff"] for d in diffs]))
    excess = paired_excess_test(real_diff, np.stack(per_example), rng, n_boot=cfg.n_boot)
    out = {"controls": diffs, "comparison": comp.__dict__, "excess_test": excess.__dict__}
    if real_kl is not None and kls:
        # damage excess (D24): positive means the real direction disturbs the next-token distribution more
        # than the matched random directions of the same norm do (raw, and with the answer token removed)
        out["kl_excess_test"] = paired_excess_test(real_kl, np.stack(kls), kl_rng or rng, n_boot=cfg.n_boot).__dict__
        if real_collateral_kl is not None:
            out["collateral_kl_excess_test"] = paired_excess_test(real_collateral_kl, np.stack(ckls), kl_rng or rng,
                                                                  n_boot=cfg.n_boot).__dict__
    if real_neutral_kl is not None and nkls:
        out["neutral_kl_excess_test"] = paired_excess_test(real_neutral_kl, np.stack(nkls), kl_rng or rng, n_boot=cfg.n_boot).__dict__
    return out


def pick_strength(reliable: list[GridPoint], reference_rho: float | None) -> GridPoint:
    """The reliable point to select at a layer: the weakest, or the one nearest ``reference_rho`` in
    log distance (ties go to the weaker point)."""
    if not reliable:
        raise ValueError("no reliable grid point")
    if reference_rho is None:
        return min(reliable, key=lambda g: g.rho)
    return min(reliable, key=lambda g: (abs(np.log(g.rho) - np.log(reference_rho)), g.rho))


def select_from_grid(
    grid: list[GridPoint], cfg: CalibrationConfig
) -> tuple[Selection | None, Selection | None, Selection | None, Selection | None]:
    """``(selected, weakest, strongest, middle)`` from the reliable points of ``grid``.

    The layer follows ``cfg.layer_rule`` and the strength ``cfg.reference_rho`` (:func:`pick_strength`);
    weakest/strongest are the reliable extremes at the selected layer and middle the reliable point
    nearest their geometric mean strictly between them (when at least three points are reliable).
    """
    by_layer: dict[int, list[GridPoint]] = {}
    for g in grid:
        if g.reliable:
            by_layer.setdefault(g.layer, []).append(g)
    if not by_layer:
        return None, None, None, None
    picks = {l: pick_strength(pts, cfg.reference_rho) for l, pts in sorted(by_layer.items())}
    if cfg.layer_rule == "best":
        layer = max(sorted(picks), key=lambda l: picks[l].test.mean_diff)  # ties: the earliest
    else:
        layer = min(picks)
    pts = sorted(by_layer[layer], key=lambda g: g.rho)
    selected, weakest, strongest = picks[layer], pts[0], pts[-1]
    middle: GridPoint | None = None
    if len(pts) >= 3:
        target = float(np.sqrt(weakest.rho * strongest.rho))
        inner = [g for g in pts if weakest.rho < g.rho < strongest.rho]
        if inner:
            middle = min(inner, key=lambda g: abs(np.log(g.rho) - np.log(target)))
    return selected.selection(), weakest.selection(), strongest.selection(), None if middle is None else middle.selection()


def calibrate(
    backend: ModelBackend,
    prompt_cfg: PromptConfig,
    cfg: CalibrationConfig,
    eval_cfg: EvaluationConfig,
    directions: dict[int, LayerDirection],
    pool: list[Item],
    run_seed: int,
    task_name: str,
    neutral_probe: Callable[[int, np.ndarray, float], np.ndarray] | None = None,
) -> CalibrationResult:
    """``neutral_probe(layer, direction, alpha)`` returns the per-prompt KL on the neutral prose set (D24)."""
    prompts = [zero_shot_prompt(prompt_cfg, q) for q in pool]
    base = backend.run(prompts, capture=True, capture_logprobs=True)
    reference = backend.reference_tensor(base.query_logprobs)  # the damage check's baseline distribution (D24)
    layers = sorted(directions)
    norms = median_layer_norms(base, layers)
    if cfg.strength_unit == "natural":
        units = {l: float(directions[l].mean_difference_norm) for l in layers}
        if any(not (u > 0) for u in units.values()):
            raise ValueError(f"calibration.strength_unit 'natural' needs a positive natural norm at every layer, got {units}")
    else:
        units = dict(norms)
    rng = rng_for(run_seed, "calibration", task_name)
    kl_rng = rng_for(run_seed, "calibration_damage", task_name)  # its own stream: the damage tests never move the others' draws

    grid: list[GridPoint] = []
    first_reliable_layer: int | None = None

    for layer in layers:
        v = directions[layer].direction
        controls: list[tuple[str, np.ndarray]] | None = None
        rhos = list(cfg.rho_grid)
        layer_points: list[GridPoint] = []
        i = 0
        while i < len(rhos):
            rho = rhos[i]
            alpha = rho * units[layer]
            r = backend.run(prompts, interventions=[Intervention(layer, v, alpha)], reference_logprobs=reference)
            t = paired_bootstrap_test(r.logprob_per_token, base.logprob_per_token, rng, n_boot=cfg.n_boot)
            metrics = r.metrics_dict()
            real_nkl = None
            if neutral_probe is not None:
                real_nkl = neutral_probe(layer, v, alpha)
                metrics["neutral_kl_mean"], metrics["neutral_kl_median"] = float(np.mean(real_nkl)), float(np.median(real_nkl))
            gp = GridPoint(
                layer=layer,
                rho=float(rho),
                alpha=float(alpha),
                rho_layer_norm=float(alpha / norms[layer]),
                extended=i >= len(cfg.rho_grid),
                metrics=metrics,
                test=t,
                passes_bootstrap=bool(t.p_value <= cfg.bootstrap_alpha),
                passes_min_improvement=bool(t.mean_diff >= cfg.min_improvement),
            )
            # Screen only while this layer can still be (or is) the selected layer.
            need_screen = gp.passes_bootstrap and gp.passes_min_improvement and (
                cfg.layer_rule == "best" or first_reliable_layer is None or first_reliable_layer == layer
            )
            if need_screen:
                if controls is None:
                    controls = matched_random_controls(
                        rng_for(run_seed, "calibration_screen", task_name, layer),
                        v,
                        cfg.n_random_screen,
                        tuple(cfg.screen_kinds),
                    )
                gp.screen = random_screen(backend, prompts, base, layer, alpha, v, t.mean_diff,
                                          r.logprob_per_token - base.logprob_per_token, controls, cfg, rng,
                                          reference=reference, real_kl=r.kl_from_reference, kl_rng=kl_rng,
                                          real_collateral_kl=r.collateral_kl, neutral_probe=neutral_probe,
                                          real_neutral_kl=real_nkl)
                if cfg.screen_test == "paired_excess":
                    gp.passes_screen = bool(gp.screen["excess_test"]["p_value"] <= cfg.random_screen_max_p)
                else:
                    gp.passes_screen = bool(gp.screen["comparison"]["p_upper"] <= cfg.random_screen_max_p)
                gp.reliable = gp.passes_screen
                if gp.reliable and first_reliable_layer is None:
                    first_reliable_layer = layer
            layer_points.append(gp)
            i += 1
            # Extend the grid if the best point sits at the upper edge.
            if i == len(rhos) and cfg.extend_grid:
                best = int(np.argmax([p.test.mean_diff for p in layer_points]))
                nxt = rhos[-1] * cfg.extension_factor
                if best == len(rhos) - 1 and nxt <= cfg.max_rho * (1 + 1e-9):
                    rhos.append(float(nxt))
        grid.extend(layer_points)

    selected, weakest, strongest, middle = select_from_grid(grid, cfg)
    reason = None
    if selected is None:
        n_boot_pass = sum(g.passes_bootstrap and g.passes_min_improvement for g in grid)
        reason = (
            "no grid point passed the bootstrap + minimum-improvement criteria"
            if n_boot_pass == 0
            else f"{n_boot_pass} grid point(s) passed the bootstrap criteria but none passed the random-control screen"
        )
    return CalibrationResult(
        baseline_metrics=base.metrics_dict(),
        layer_norms=norms,
        strength_unit=cfg.strength_unit,
        strength_units=units,
        grid=grid,
        selected=selected,
        weakest=weakest,
        strongest=strongest,
        middle=middle,
        reference_rho=cfg.reference_rho,
        layer_rule=cfg.layer_rule,
        reason=reason,
        per_example={"baseline_logprob_per_token": base.logprob_per_token.tolist()},
    )

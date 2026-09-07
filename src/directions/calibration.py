"""Intervention calibration: layer x strength sweep on the calibration pool.

Selection rule (docs/EXPERIMENT.md): the earliest candidate layer and, within
it, the smallest relative strength that *reliably* improves the decision metric
(target log-probability per token), where reliable means

* one-sided paired bootstrap p <= ``bootstrap_alpha``;
* mean improvement >= ``min_improvement`` (nats per token);
* empirical p <= ``random_screen_max_p`` against ``n_random_screen`` matched
  random directions (same layer, same norm) on the calibration pool.

The grid is extended geometrically when the best point sits at its upper edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import CalibrationConfig, EvaluationConfig, PromptConfig
from .extraction import LayerDirection
from .geometry import matched_random_controls
from .model import ForwardResult, Intervention, ModelBackend
from .prompts import Prompt, zero_shot_prompt
from .seeds import rng_for
from .stats import PairedTest, compare_to_null, paired_bootstrap_test
from .tasks import Item


@dataclass
class GridPoint:
    layer: int
    rho: float
    alpha: float
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


@dataclass
class Selection:
    layer: int
    rho: float
    alpha: float


@dataclass
class CalibrationResult:
    baseline_metrics: dict[str, float]
    layer_norms: dict[int, float]  # median ||h_l(x)|| over the calibration pool
    grid: list[GridPoint]
    selected: Selection | None
    strongest: Selection | None  # strongest reliable strength at the selected layer (exploratory)
    middle: Selection | None  # a reliable strength between weakest and strongest (exploratory)
    reason: str | None = None  # why nothing was selected
    per_example: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_metrics": self.baseline_metrics,
            "layer_norms": {str(k): v for k, v in self.layer_norms.items()},
            "grid": [g.as_dict() for g in self.grid],
            "selected": None if self.selected is None else self.selected.__dict__,
            "strongest": None if self.strongest is None else self.strongest.__dict__,
            "middle": None if self.middle is None else self.middle.__dict__,
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
    controls: list[tuple[str, np.ndarray]],
    cfg: CalibrationConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Mean improvement of each matched random direction at ``(layer, alpha)``."""
    diffs = []
    for kind, u in controls:
        r = backend.run(prompts, interventions=[Intervention(layer, u, alpha)])
        t = paired_bootstrap_test(r.logprob_per_token, base.logprob_per_token, rng, n_boot=cfg.n_boot)
        diffs.append({"kind": kind, "mean_diff": t.mean_diff, "p_value": t.p_value, "metrics": r.metrics_dict()})
    comp = compare_to_null(real_mean_diff, np.array([d["mean_diff"] for d in diffs]))
    return {"controls": diffs, "comparison": comp.__dict__}


def calibrate(
    backend: ModelBackend,
    prompt_cfg: PromptConfig,
    cfg: CalibrationConfig,
    eval_cfg: EvaluationConfig,
    directions: dict[int, LayerDirection],
    pool: list[Item],
    run_seed: int,
    task_name: str,
) -> CalibrationResult:
    prompts = [zero_shot_prompt(prompt_cfg, q) for q in pool]
    base = backend.run(prompts, capture=True)
    layers = sorted(directions)
    norms = median_layer_norms(base, layers)
    rng = rng_for(run_seed, "calibration", task_name)

    grid: list[GridPoint] = []
    selected: Selection | None = None
    strongest: Selection | None = None
    reliable_rhos: list[float] = []

    for layer in layers:
        v = directions[layer].direction
        controls: list[tuple[str, np.ndarray]] | None = None
        rhos = list(cfg.rho_grid)
        layer_points: list[GridPoint] = []
        i = 0
        while i < len(rhos):
            rho = rhos[i]
            alpha = rho * norms[layer]
            r = backend.run(prompts, interventions=[Intervention(layer, v, alpha)])
            t = paired_bootstrap_test(r.logprob_per_token, base.logprob_per_token, rng, n_boot=cfg.n_boot)
            gp = GridPoint(
                layer=layer,
                rho=float(rho),
                alpha=float(alpha),
                extended=i >= len(cfg.rho_grid),
                metrics=r.metrics_dict(),
                test=t,
                passes_bootstrap=bool(t.p_value <= cfg.bootstrap_alpha),
                passes_min_improvement=bool(t.mean_diff >= cfg.min_improvement),
            )
            # Screen only while this layer can still be (or is) the selected layer.
            need_screen = gp.passes_bootstrap and gp.passes_min_improvement and (
                selected is None or selected.layer == layer
            )
            if need_screen:
                if controls is None:
                    controls = matched_random_controls(
                        rng_for(run_seed, "calibration_screen", task_name, layer),
                        v,
                        cfg.n_random_screen,
                        tuple(cfg.screen_kinds),
                    )
                gp.screen = random_screen(backend, prompts, base, layer, alpha, v, t.mean_diff, controls, cfg, rng)
                gp.passes_screen = bool(gp.screen["comparison"]["p_upper"] <= cfg.random_screen_max_p)
                gp.reliable = gp.passes_screen
                if gp.reliable:
                    if selected is None:
                        selected = Selection(layer, gp.rho, gp.alpha)
                    if selected.layer == layer:
                        reliable_rhos.append(gp.rho)
                        strongest = Selection(layer, gp.rho, gp.alpha)
            layer_points.append(gp)
            i += 1
            # Extend the grid if the best point sits at the upper edge.
            if i == len(rhos) and cfg.extend_grid:
                best = int(np.argmax([p.test.mean_diff for p in layer_points]))
                nxt = rhos[-1] * cfg.extension_factor
                if best == len(rhos) - 1 and nxt <= cfg.max_rho * (1 + 1e-9):
                    rhos.append(float(nxt))
        grid.extend(layer_points)

    middle: Selection | None = None
    if selected is not None and strongest is not None and len(reliable_rhos) >= 3:
        target = float(np.sqrt(selected.rho * strongest.rho))
        inner = [r for r in reliable_rhos if selected.rho < r < strongest.rho]
        if inner:
            r_mid = min(inner, key=lambda r: abs(np.log(r) - np.log(target)))
            middle = Selection(selected.layer, r_mid, r_mid * norms[selected.layer])

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
        grid=grid,
        selected=selected,
        strongest=strongest,
        middle=middle,
        reason=reason,
        per_example={"baseline_logprob_per_token": base.logprob_per_token.tolist()},
    )

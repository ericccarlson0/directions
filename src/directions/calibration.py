"""Intervention layer and strength calibration.

Sweeps candidate intervention layers (fractions of model depth) against
candidate relative strengths ``rho`` on the *calibration* split, which is
disjoint from both the extraction split and the final evaluation split. The
selection rule follows ``docs/EXPERIMENT.md``: the earliest layer and, within
it, the smallest strength that improves the behavioural metric by at least
``intervention.min_improvement`` over the unsteered baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mathx
from .config import Config
from .controls import build_random_controls, control_rng
from .extraction import DirectionSpec, TaskExtraction, alpha_from_rho
from .model import BehaviorResult, LanguageModel
from .prompts import Prompt


@dataclass
class SweepPoint:
    layer: int
    rho: float
    alpha: float
    metric: float
    improvement: float
    p_value: float
    reliable: bool
    summary: dict
    control_screen: dict | None = None

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "rho": self.rho,
            "alpha": self.alpha,
            "metric": self.metric,
            "improvement": self.improvement,
            "paired_p_value": self.p_value,
            "reliable": self.reliable,
            "behavior": self.summary,
            "control_screen": self.control_screen,
        }


@dataclass
class CalibrationResult:
    task: str
    metric_name: str
    baseline: dict
    baseline_metric: float
    points: list[SweepPoint] = field(default_factory=list)
    selected: SweepPoint | None = None
    selection_rule: str = (
        "earliest layer, then smallest rho, whose improvement clears "
        "intervention.min_improvement AND a paired one-sided bootstrap test at "
        "intervention.max_selection_p"
    )
    rejected_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "metric": self.metric_name,
            "baseline_behavior": self.baseline,
            "baseline_metric": self.baseline_metric,
            "selection_rule": self.selection_rule,
            "selected": self.selected.to_dict() if self.selected else None,
            "rejected_reason": self.rejected_reason,
            "grid": [p.to_dict() for p in self.points],
        }


def _screen(
    lm: LanguageModel,
    cfg: Config,
    task_name: str,
    extraction: TaskExtraction,
    prompts: list[Prompt],
    base_metric: float,
    point: SweepPoint,
) -> dict:
    """Compare one calibration grid point against matched random controls.

    Runs on the calibration split with its own control seed, so the final
    held-out control comparison stays an independent test.
    """
    n = cfg.intervention.control_screen_n
    metric = cfg.intervention.selection_metric
    if n == 0:
        return {"passed": True, "n_controls": 0, "p_value": float("nan"), "z": float("nan"),
                "note": "control screen disabled"}
    direction = DirectionSpec(
        layer=point.layer,
        vector=extraction.layers[point.layer].consensus.astype(np.float32),
        alpha=point.alpha,
    )
    rng = control_rng(cfg.run.seed + 9973, task_name, point.layer)
    improvements = []
    for spec in build_random_controls(rng, direction, n, cfg.controls.kinds):
        out = lm.score(
            prompts, intervention=spec,
            batch_size=cfg.compute.batch_size, max_seq_len=cfg.compute.max_seq_len,
        )
        improvements.append(out.metric(metric) - base_metric)
    arr = np.asarray(improvements)
    p_value = mathx.empirical_p_value(point.improvement, arr)
    return {
        "passed": bool(p_value <= cfg.intervention.control_screen_max_p),
        "n_controls": n,
        "p_value": p_value,
        "z": mathx.z_against_null(point.improvement, arr),
        "control_improvement_mean": float(arr.mean()),
        "control_improvement_max": float(arr.max()),
        "threshold": cfg.intervention.control_screen_max_p,
    }


def sweep(
    lm: LanguageModel,
    cfg: Config,
    task_name: str,
    extraction: TaskExtraction,
    prompts: list[Prompt],
    baseline: BehaviorResult,
    median_norms: np.ndarray,
    candidate_layers: list[int],
) -> CalibrationResult:
    metric = cfg.intervention.selection_metric
    base_metric = baseline.metric(metric)
    base_per_example = baseline.per_example(metric)
    res = CalibrationResult(
        task=task_name,
        metric_name=metric,
        baseline=baseline.summary(),
        baseline_metric=base_metric,
    )
    for layer in sorted(candidate_layers):
        v = extraction.layers[layer].consensus
        for rho in sorted(cfg.intervention.strengths):
            alpha = alpha_from_rho(rho, median_norms[layer])
            spec = DirectionSpec(layer=layer, vector=v.astype(np.float32), alpha=alpha)
            out = lm.score(
                prompts,
                intervention=spec,
                batch_size=cfg.compute.batch_size,
                max_seq_len=cfg.compute.max_seq_len,
            )
            m = out.metric(metric)
            diffs = out.per_example(metric) - base_per_example
            p_value = mathx.paired_bootstrap_p(
                diffs, n_boot=cfg.bootstrap.n_boot, seed=cfg.run.seed + layer
            )
            reliable = bool(
                m - base_metric >= cfg.intervention.min_improvement
                and p_value <= cfg.intervention.max_selection_p
            )
            res.points.append(
                SweepPoint(layer, float(rho), alpha, m, m - base_metric, p_value, reliable,
                           out.summary())
            )
    # "earliest layer, then smallest rho" among points that clear both the
    # effect-size threshold and the paired reliability test, and that also beat
    # a matched random-control screen at the same layer and norm.
    hits = sorted((p for p in res.points if p.reliable), key=lambda p: (p.layer, p.rho))
    for point in hits:
        point.control_screen = _screen(
            lm, cfg, task_name, extraction, prompts, base_metric, point
        )
        if point.control_screen["passed"]:
            res.selected = point
            break
    if res.selected is None and hits:
        best = max(hits, key=lambda p: p.control_screen["z"] if p.control_screen else -np.inf)
        res.rejected_reason = (
            f"{len(hits)} grid point(s) improved the metric but none beat the matched "
            f"random-control screen (max_p={cfg.intervention.control_screen_max_p}); best was "
            f"layer {best.layer}, rho {best.rho} with screen p="
            f"{best.control_screen['p_value']:.3f}"
        )
    elif res.selected is None:
        best = max(res.points, key=lambda p: p.improvement) if res.points else None
        res.rejected_reason = (
            "no (layer, rho) reliably improved the metric "
            f"(min_improvement={cfg.intervention.min_improvement}, "
            f"max_selection_p={cfg.intervention.max_selection_p}); best improvement was "
            f"{best.improvement:.4f} (p={best.p_value:.3f}) at layer {best.layer}, rho {best.rho}"
            if best
            else "empty calibration grid"
        )
    return res

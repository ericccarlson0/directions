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

from .config import Config
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
    summary: dict

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "rho": self.rho,
            "alpha": self.alpha,
            "metric": self.metric,
            "improvement": self.improvement,
            "behavior": self.summary,
        }


@dataclass
class CalibrationResult:
    task: str
    metric_name: str
    baseline: dict
    baseline_metric: float
    points: list[SweepPoint] = field(default_factory=list)
    selected: SweepPoint | None = None
    selection_rule: str = "earliest layer, then smallest rho, meeting min_improvement"
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
            res.points.append(
                SweepPoint(layer, float(rho), alpha, m, m - base_metric, out.summary())
            )
    hits = [p for p in res.points if p.improvement >= cfg.intervention.min_improvement]
    if hits:
        res.selected = min(hits, key=lambda p: (p.layer, p.rho))
    else:
        best = max(res.points, key=lambda p: p.improvement) if res.points else None
        res.rejected_reason = (
            "no (layer, rho) reached intervention.min_improvement="
            f"{cfg.intervention.min_improvement}; best improvement was "
            f"{best.improvement:.4f} at layer {best.layer}, rho {best.rho}"
            if best
            else "empty calibration grid"
        )
    return res

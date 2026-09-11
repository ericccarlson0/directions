"""Calibration selection rules (docs/DECISIONS.md D22) and the run profiler."""

import numpy as np
import pytest

from directions.calibration import GridPoint, pick_strength, select_from_grid
from directions.config import CalibrationConfig
from directions.profiling import Profiler
from directions.stats import PairedTest


def _gp(layer: int, rho: float, mean_diff: float, reliable: bool = True, unit: float = 10.0) -> GridPoint:
    return GridPoint(layer=layer, rho=rho, alpha=rho * unit, rho_layer_norm=rho * unit / 20.0, extended=False, metrics={},
                     test=PairedTest(mean_diff=mean_diff, median_diff=mean_diff, p_value=0.0, ci_low=0.0, ci_high=0.0, n=1,
                                     frac_improved=1.0),
                     passes_bootstrap=True, passes_min_improvement=True, passes_screen=reliable, reliable=reliable)


def test_pick_strength_weakest_and_nearest_reference():
    pts = [_gp(6, r, m) for r, m in ((0.1, 0.1), (0.5, 0.4), (1.0, 0.7), (2.0, 0.9))]
    assert pick_strength(pts, None).rho == 0.1
    assert pick_strength(pts, 1.0).rho == 1.0
    assert pick_strength(pts, 0.8).rho == 1.0  # log distance: 0.8 is nearer 1.0 than 0.5
    assert pick_strength(pts, 0.7).rho == 0.5  # |log 0.7 - log 0.5| = 0.34 < |log 0.7 - log 1| = 0.36
    assert pick_strength(pts, 100.0).rho == 2.0  # beyond the grid: the strongest reliable point
    # an exact tie in log distance (0.5 between 0.25 and 1.0) goes to the weaker point
    assert pick_strength([_gp(6, 0.25, 0.2), _gp(6, 1.0, 0.7)], 0.5).rho == 0.25
    with pytest.raises(ValueError):
        pick_strength([], 1.0)


def test_select_from_grid_layer_and_strength_rules():
    grid = [
        _gp(6, 0.1, 0.05, reliable=False), _gp(6, 0.5, 0.4), _gp(6, 1.0, 0.7), _gp(6, 2.0, 0.6),
        _gp(8, 0.1, 0.2), _gp(8, 0.5, 0.9), _gp(8, 1.0, 1.2), _gp(8, 2.0, 1.0),
        _gp(11, 0.1, 0.1, reliable=False), _gp(11, 0.5, 0.2, reliable=False),
    ]
    # the D1-D21 rule: earliest layer, weakest reliable strength
    sel, weakest, strongest, middle = select_from_grid(grid, CalibrationConfig())
    assert (sel.layer, sel.rho) == (6, 0.5) and weakest.rho == 0.5 and strongest.rho == 2.0 and middle.rho == 1.0
    assert sel.alpha == pytest.approx(5.0) and sel.rho_layer_norm == pytest.approx(0.25)
    # D22: the reliable strength nearest the reference, at the earliest reliable layer
    sel, weakest, strongest, middle = select_from_grid(grid, CalibrationConfig(reference_rho=1.0))
    assert (sel.layer, sel.rho) == (6, 1.0) and (weakest.rho, strongest.rho, middle.rho) == (0.5, 2.0, 1.0)
    # the best-layer rule compares the layers' selected points
    sel, weakest, strongest, middle = select_from_grid(grid, CalibrationConfig(reference_rho=1.0, layer_rule="best"))
    assert (sel.layer, sel.rho) == (8, 1.0) and weakest.rho == 0.1 and strongest.rho == 2.0
    # no middle with fewer than three reliable points; nothing selected without reliable points
    sel, weakest, strongest, middle = select_from_grid([_gp(6, 0.5, 0.4), _gp(6, 1.0, 0.7)], CalibrationConfig(reference_rho=1.0))
    assert sel.rho == 1.0 and weakest.rho == 0.5 and strongest.rho == 1.0 and middle is None
    assert select_from_grid([_gp(6, 0.5, 0.4, reliable=False)], CalibrationConfig()) == (None, None, None, None)


class _Backend:
    device = None

    def __init__(self):
        self.counters = {"forward_examples": 0, "forward_batches": 0, "gradient_examples": 0, "gradient_batches": 0}


def test_profiler_sections_accumulate_and_count():
    b = _Backend()
    prof = Profiler(b)
    with prof.section("total"):
        with prof.section("stage", "task_a"):
            b.counters["forward_examples"] += 10
            b.counters["forward_batches"] += 1
        for _ in range(2):
            with prof.section("stage", "task_b"):
                b.counters["forward_examples"] += 3
                b.counters["gradient_examples"] += 2
    t = prof.timings()
    assert list(t) == ["stage:task_a", "stage:task_b", "total"]  # keyed on first exit, nested before outer
    rep = prof.report()
    a, bb = rep["sections"]["stage:task_a"], rep["sections"]["stage:task_b"]
    assert (a["forward_examples"], a["forward_batches"], a["calls"]) == (10, 1, 1)
    assert (bb["forward_examples"], bb["gradient_examples"], bb["calls"]) == (6, 4, 2)
    assert rep["sections"]["total"]["forward_examples"] == 16 and rep["totals"]["forward_examples"] == 16
    assert rep["totals"]["max_rss_mb"] > 0 and "peak_gpu_mb" not in a and "gpu_max_allocated_mb" not in rep["totals"]
    assert all(v >= 0 for v in t.values()) and "stage:task_a" in prof.table(min_seconds=0.0)

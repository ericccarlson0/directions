"""scripts/das_summary.py: the D48 summary on a toy run's output."""

import importlib.util
from pathlib import Path

import numpy as np

from directions.config import load_das_config
from directions.das import run_das

_SPEC = importlib.util.spec_from_file_location("das_summary", Path(__file__).resolve().parents[1] / "scripts" / "das_summary.py")
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def test_summary_reads_a_toy_run(tmp_path):
    cfg = load_das_config("configs/smoke_toy_das.yaml")
    cfg.output_dir, cfg.figures, cfg.determinism_check = str(tmp_path), False, False
    run_das(cfg, run_id="das_toy")
    assert summary.main([str(tmp_path), "--out", str(tmp_path / "s.json")]) == 0
    import json

    s = json.loads((tmp_path / "s.json").read_text())["das_toy"]
    assert set(s["sites"]) == {str(x) for x in cfg.sites} and s["control_site"] == cfg.control_site
    row = s["sites"][str(cfg.sites[-1])]
    assert set(row["variables"]) == set(cfg.variables) and row["count"] == len([v for v in row["held"] if v != "a"])
    assert set(row["overlap"]) == {"S1-S2", "S1-Y", "S2-Y"}
    for o in row["overlap"].values():
        assert 0.0 <= o["share"] <= 1.0 + 1e-6 and 0.0 < o["chance"] <= 1.0


def test_overlap_of_known_subspaces():
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.standard_normal((32, 32)))[0].T  # orthonormal rows
    inside = summary.overlap(Q[:2], Q[:6])  # a 2-dim subspace inside a 6-dim one
    apart = summary.overlap(Q[:2], Q[10:16])  # orthogonal subspaces
    assert abs(inside["share"] - 1.0) < 1e-9 and abs(inside["max_cos"] - 1.0) < 1e-9 and inside["chance"] == 6 / 32
    assert apart["share"] < 1e-9 and apart["max_cos"] < 1e-9

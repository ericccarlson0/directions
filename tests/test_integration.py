"""Inexpensive integration path: the full pipeline on a tiny random model (CPU)."""

import json
from pathlib import Path

import numpy as np
import pytest

from directions.cli import main
from directions.config import load_config
from directions.pipeline import run_pipeline

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    cfg = load_config(CONFIGS / "smoke_toy.yaml")
    cfg.output_dir = str(tmp_path_factory.mktemp("results"))
    root = run_pipeline(cfg, "pilot", config_path=str(CONFIGS / "smoke_toy.yaml"))
    return root


def test_smoke_pilot_outputs(smoke_run):
    root = smoke_run
    for rel in ("metadata.json", "config.resolved.yaml", "rejections.jsonl", "log.txt", "core/summary.json", "core/cross_task.json"):
        assert (root / rel).exists(), rel
    meta = json.loads((root / "metadata.json").read_text())
    for key in ("git", "environment", "seed", "seeds", "model", "config", "candidate_layers", "timings_seconds"):
        assert key in meta, key
    assert meta["model"]["n_layers"] == 4
    for task in ("antonym", "arithmetic", "number_to_words"):
        d = root / "core" / "tasks" / task
        for rel in ("splits.json", "qualification.json", "extraction.json", "directions.npz", "calibration.json",
                    "evaluation.json", "layerwise.json", "layerwise_arrays.npz"):
            assert (d / rel).exists(), f"{task}/{rel}"
        e = root / "exploratory" / "tasks" / task
        assert (e / "strength_robustness.json").exists() and (e / "block_ablation.json").exists()
        lw = json.loads((d / "layerwise.json").read_text())
        real = lw["real"]
        ls = real["intervention_layer"]
        assert real["pre_intervention_max_abs_delta"] == 0.0
        # undefined at/below the intervention layer, defined afterwards
        assert real["d_eff"][ls] is None and all(x is None for x in real["d_eff"][:ls])
        assert all(x is not None for x in real["d_eff"][ls + 1 :])
        assert real["new_subspace"][ls] is None and real["new_subspace_uncentered"][ls] is not None
        # alignment is 1 at the intervention layer by construction
        assert real["summaries"]["alignment"]["median"][ls] == pytest.approx(1.0, abs=1e-4)
        assert sorted({c["kind"] for c in lw["controls"]}) == ["covariance", "demo_variation", "isotropic", "orthogonal", "other_task"]
        assert len(lw["controls"]) == 7 and set(lw["null_summaries"]) >= {"primary", "isotropic"}
        assert set(lw["comparison"]["primary"]["metrics"]) >= {"log_gain", "d_eff", "new_subspace", "alignment"}
        assert set(lw["comparison"]["by_kind"]) == {"covariance", "demo_variation", "isotropic", "orthogonal", "other_task"}
        assert lw["comparison"]["primary"]["n_random"] == 3
        assert isinstance(lw["signature"]["labels"], list)
        arrays = np.load(d / "layerwise_arrays.npz")
        assert arrays["log_gain"].shape == (4, 6)
        assert (root / "figures" / f"{task}_structured_nulls.png").exists()
        cal = json.loads((d / "calibration.json").read_text())
        # only candidate layers passing the stability filter are calibrated
        assert set(cal["layer_norms"]) <= {"1", "2"} and len(cal["layer_norms"]) >= 1
        assert len(cal["grid"]) >= 3 * len(cal["layer_norms"])
    figs = list((root / "figures").glob("*.png"))
    assert any(f.name.startswith("heatmap_") for f in figs)
    assert any(f.name == "antonym_log_gain.png" for f in figs)
    # gates are recorded even though they are not enforced in smoke mode
    q = json.loads((root / "core" / "tasks" / "antonym" / "qualification.json").read_text())
    assert set(q["gates"]) == {"fewshot", "stability", "calibration", "steering", "random_controls"}
    assert q["enforce"] is False
    rejections = [json.loads(l) for l in (root / "rejections.jsonl").read_text().splitlines()]
    assert not any(r["stage"] == "error" for r in rejections), rejections


def test_smoke_pilot_is_reproducible(smoke_run, tmp_path):
    cfg = load_config(CONFIGS / "smoke_toy.yaml")
    cfg.output_dir = str(tmp_path)
    second = run_pipeline(cfg, "pilot")
    assert main(["compare", str(smoke_run), str(second)]) == 0


def test_validate_command_stops_after_steering(tmp_path):
    rc = main(["validate", "--config", str(CONFIGS / "smoke_toy.yaml"), "--output-dir", str(tmp_path), "--run-id", "v"])
    assert rc == 0
    root = tmp_path / "v"
    assert (root / "core" / "tasks" / "antonym" / "evaluation.json").exists()
    assert not (root / "core" / "tasks" / "antonym" / "layerwise.json").exists()
    assert not (root / "core" / "cross_task.json").exists()


def test_enforced_gates_reject_random_model(tmp_path):
    cfg = load_config(CONFIGS / "smoke_toy.yaml")
    cfg.output_dir = str(tmp_path)
    cfg.qualification.enforce = True
    cfg.tasks = cfg.tasks[:1]
    root = run_pipeline(cfg, "pilot")
    q = json.loads((root / "core" / "tasks" / "antonym" / "qualification.json").read_text())
    assert q["qualified"] is False
    rejections = [json.loads(l) for l in (root / "rejections.jsonl").read_text().splitlines()]
    assert any(r["task"] == "antonym" and r["stage"] != "item_filter" for r in rejections)
    summary = json.loads((root / "core" / "summary.json").read_text())
    assert summary["n_qualified"] == 0

"""Inexpensive end-to-end integration path.

Runs the complete pilot -- qualification, extraction, calibration, steering,
matched controls, layerwise measurement, exploratory analyses, figures and
serialization -- on the offline `tiny_random` backend. No download, no network,
no GPU. This is the same code path the real runs take; only the model differs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import warnings

from directions.config import load_config
from directions.pipeline import run_pilot, run_validation

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def pilot_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("pilot")
    cfg = load_config(CONFIG, {"run": {"output_root": str(out)}, "figures": {"dpi": 60}})
    summary = run_pilot(cfg)
    return summary, Path(summary["run_directory"])


def test_pilot_qualifies_its_tasks(pilot_run):
    summary, _ = pilot_run
    assert summary["tasks_qualified"] == summary["tasks_requested"]
    assert summary["layerwise_completed"] == sorted(summary["tasks_requested"])
    assert not any(v["error"] for v in summary["task_status"].values())


def test_run_directory_has_the_required_provenance(pilot_run):
    _, root = pilot_run
    for name in ("config.resolved.yaml", "metadata.json", "summary.json", "log.txt"):
        assert (root / name).exists(), name
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["git"]["commit"]
    assert meta["model"]["n_layers"] > 0
    assert meta["model"]["name_or_path"]
    assert meta["environment"]["packages"]["torch"]
    assert meta["seeds"]["run_seed"] == 0
    assert meta["config_fingerprint"] == json.loads((root / "summary.json").read_text())[
        "config_fingerprint"
    ]


def test_core_and_exploratory_outputs_are_separated(pilot_run):
    summary, root = pilot_run
    for task in summary["tasks_qualified"]:
        assert (root / "core" / f"validation__{task}.json").exists()
        assert (root / "core" / f"layerwise__{task}.json").exists()
        assert (root / "core" / f"layerwise__{task}.npz").exists()
        assert (root / "exploratory" / f"{task}.json").exists()
    # no exploratory content leaks into core/
    for path in (root / "core").glob("*.json"):
        blob = json.loads(path.read_text())
        assert "profile" not in blob and "block_ablation" not in blob


def test_layerwise_output_contains_every_required_quantity(pilot_run):
    summary, root = pilot_run
    task = summary["tasks_qualified"][0]
    blob = json.loads((root / "core" / f"layerwise__{task}.json").read_text())
    m = blob["measurement"]
    n_layers = m["n_layers"]
    for key in ("S_median", "d_eff", "d90", "N", "N_uncentered", "control_alignment"):
        assert len(m[key]) == n_layers + 1, key
    assert len(m["log_G_median"]) == n_layers + 1
    l0 = m["intervention_layer"]
    # upstream of the intervention everything is null; at and after it, defined
    assert all(v is None for v in m["S_median"][:l0])
    assert all(v is not None for v in m["S_median"][l0:])
    # the perturbation at the intervention layer is exactly the control direction
    assert m["control_alignment"][l0] == pytest.approx(1.0, abs=1e-4)
    assert blob["random_control_null"]["S"]["n_controls"] == 4
    assert set(blob["real_vs_null"]) >= {"S", "log_G", "d_eff", "N"}


def test_npz_arrays_are_consistent_with_the_json(pilot_run):
    summary, root = pilot_run
    task = summary["tasks_qualified"][0]
    arrays = np.load(root / "core" / f"layerwise__{task}.npz")
    blob = json.loads((root / "core" / f"layerwise__{task}.json").read_text())
    m = blob["measurement"]
    assert arrays["S"].shape == (m["n_layers"] + 1, m["n_examples"])
    assert np.isclose(np.linalg.norm(arrays["direction"]), 1.0, atol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        got = np.nanmedian(arrays["S"], axis=1)
    want = np.array([np.nan if v is None else v for v in m["S_median"]])
    assert np.allclose(got[~np.isnan(want)], want[~np.isnan(want)])


def test_validation_records_every_qualification_stage(pilot_run):
    summary, root = pilot_run
    task = summary["tasks_qualified"][0]
    blob = json.loads((root / "core" / f"validation__{task}.json").read_text())
    assert blob["qualified"] is True
    assert set(blob["stages"]) == {
        "fewshot_qualification",
        "direction_stability",
        "intervention_calibration",
        "heldout_steering",
        "random_control_comparison",
    }
    assert all(s["passed"] for s in blob["stages"].values())
    assert blob["stages"]["intervention_calibration"]["grid"]
    assert len(blob["stages"]["random_control_comparison"]["controls"]) == 4
    # splits recorded and disjoint
    ids = blob["splits"]
    assert not (set(ids["extraction"]) & set(ids["evaluation"]))
    assert not (set(ids["calibration"]) & set(ids["evaluation"]))


def test_exploratory_outputs_exist(pilot_run):
    summary, root = pilot_run
    task = summary["tasks_qualified"][0]
    blob = json.loads((root / "exploratory" / f"{task}.json").read_text())
    assert "profile" in blob and "block_ablation" in blob
    assert blob["profile"]["thresholds"]
    assert len(blob["block_ablation"]["blocks"]) <= 2


def test_figures_are_generated(pilot_run):
    summary, root = pilot_run
    names = {p.name for p in (root / "figures").glob("*.png")}
    assert "extraction_stability.png" in names
    assert {"heatmap_log_G.png", "heatmap_d_eff.png", "heatmap_N.png"} <= names
    for task in summary["tasks_qualified"]:
        for suffix in ("1_magnitude", "2_amplification", "3_dimensionality",
                       "4_new_subspace", "6_real_vs_random", "0_calibration"):
            assert f"{task}__{suffix}.png" in names
    assert all((root / "figures" / n).stat().st_size > 1000 for n in names)


def test_pilot_is_reproducible(tmp_path):
    """Two runs of the same config must agree on every reported number."""
    def once():
        cfg = load_config(
            CONFIG,
            {"run": {"output_root": str(tmp_path)}, "figures": {"enabled": False},
             "data": {"tasks": ["antonym"]}},
        )
        summary = run_pilot(cfg)
        blob = json.loads(
            (Path(summary["run_directory"]) / "core" / "layerwise__antonym.json").read_text()
        )
        return summary, blob

    a_summary, a = once()
    b_summary, b = once()
    assert a_summary["config_fingerprint"] == b_summary["config_fingerprint"]
    assert a_summary["task_status"] == b_summary["task_status"]
    assert a["measurement"]["S_median"] == b["measurement"]["S_median"]
    assert a["measurement"]["N"] == b["measurement"]["N"]
    assert a["random_control_null"]["d_eff"] == b["random_control_null"]["d_eff"]


def test_validate_command_stops_before_the_layerwise_stage(tmp_path):
    cfg = load_config(
        CONFIG,
        {"run": {"output_root": str(tmp_path)}, "data": {"tasks": ["antonym"]},
         "figures": {"enabled": False}},
    )
    summary = run_validation(cfg)
    root = Path(summary["run_directory"])
    assert summary["layerwise_completed"] == []
    assert (root / "core" / "validation__antonym.json").exists()
    assert not list((root / "core").glob("layerwise__*"))


def test_rejections_are_recorded_when_a_task_fails(tmp_path):
    """A task that cannot pass ICL qualification is rejected, not crashed on."""
    cfg = load_config(
        CONFIG,
        {"run": {"output_root": str(tmp_path)}, "data": {"tasks": ["antonym"]},
         "figures": {"enabled": False},
         "qualification": {"min_fewshot_accuracy": 1.5}},
    )
    summary = run_validation(cfg)
    root = Path(summary["run_directory"])
    assert summary["tasks_qualified"] == []
    assert summary["task_status"]["antonym"]["failed_stage"] == "fewshot_qualification"
    lines = [json.loads(l) for l in (root / "rejections.jsonl").read_text().splitlines()]
    assert lines and lines[0]["stage"] == "fewshot_qualification"
    assert "detail" in lines[0]


def test_unknown_task_is_recorded_as_an_error_not_a_crash(tmp_path):
    cfg = load_config(
        CONFIG,
        {"run": {"output_root": str(tmp_path)}, "data": {"tasks": ["not_a_task"]},
         "figures": {"enabled": False}},
    )
    summary = run_validation(cfg)
    assert summary["tasks_qualified"] == []
    assert summary["task_status"]["not_a_task"]["failed_stage"] == "exception"
    assert "KeyError" in summary["task_status"]["not_a_task"]["error"]

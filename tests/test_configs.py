from pathlib import Path

import pytest

from directions.config import config_from_dict, config_to_dict, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_all_configs_load():
    for p in sorted(CONFIGS.glob("*.yaml")):
        cfg = load_config(p)
        assert cfg.tasks, p


def test_qwen_configs_differ_only_in_model_name():
    a = config_to_dict(load_config(CONFIGS / "pilot_qwen3_0.6b.yaml"))
    b = config_to_dict(load_config(CONFIGS / "pilot_qwen3_1.7b.yaml"))
    assert a["model"]["name"] == "Qwen/Qwen3-0.6B-Base"
    assert b["model"]["name"] == "Qwen/Qwen3-1.7B-Base"
    a["model"]["name"] = b["model"]["name"]
    assert a == b


def test_pilot_configs_match_preregistered_sizes():
    cfg = load_config(CONFIGS / "pilot_qwen3_0.6b.yaml")
    assert (cfg.data.n_extraction, cfg.data.n_calibration, cfg.data.n_evaluation) == (64, 64, 192)
    assert len(cfg.tasks) == 10
    assert cfg.extraction.n_seeds == 3 and not cfg.extraction.center
    assert sum(cfg.evaluation.controls[k] for k in cfg.evaluation.gate_kinds) >= 64 and cfg.calibration.n_random_screen >= 8
    assert set(cfg.evaluation.controls) == {"isotropic", "orthogonal", "covariance", "other_task", "demo_variation"}
    assert cfg.qualification.enforce
    assert cfg.model.dtype == "bfloat16"


def test_config_validation_errors():
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym"}], "model": {"backend": "onnx"}})
    with pytest.raises(ValueError):
        config_from_dict({"tasks": []})
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym"}], "calibration": {"rho_grid": [1.0, 0.5]}})
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym"}], "unknown_key": 1})
    cfg = config_from_dict({"tasks": [{"name": "antonym"}, {"name": "arithmetic", "params": {"operand": 7}}]})
    assert cfg.tasks[1].params == {"operand": 7}

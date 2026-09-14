from pathlib import Path

import pytest

from directions.config import config_from_dict, config_to_dict, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_all_configs_load():
    for p in sorted(CONFIGS.glob("*.yaml")):
        cfg = load_config(p)
        assert cfg.tasks, p


QWEN_CONFIGS = {
    "pilot_qwen3_0.6b.yaml": "Qwen/Qwen3-0.6B-Base",
    "pilot_qwen3_1.7b.yaml": "Qwen/Qwen3-1.7B-Base",
    "pilot_qwen3_4b.yaml": "Qwen/Qwen3-4B-Base",
    "pilot_qwen3_8b.yaml": "Qwen/Qwen3-8B-Base",
}


def test_qwen_configs_differ_only_in_model_name():
    a = config_to_dict(load_config(CONFIGS / "pilot_qwen3_0.6b.yaml"))
    for fname, model_name in QWEN_CONFIGS.items():
        b = config_to_dict(load_config(CONFIGS / fname))
        assert b["model"]["name"] == model_name, fname
        b["model"]["name"] = a["model"]["name"]
        assert a == b, fname


def test_pilot_configs_match_preregistered_sizes():
    cfg = load_config(CONFIGS / "pilot_qwen3_0.6b.yaml")
    assert (cfg.data.n_extraction, cfg.data.n_calibration, cfg.data.n_evaluation) == (64, 64, 192)
    assert len(cfg.tasks) == 10
    names = {t.name for t in cfg.tasks}
    assert {"last_antonym", "arithmetic_words"} <= names and not ({"add_two", "en_fr"} & names)  # D19
    assert cfg.extraction.n_seeds == 3 and not cfg.extraction.center
    assert sum(cfg.evaluation.controls[k] for k in cfg.evaluation.gate_kinds) >= 32 and cfg.calibration.n_random_screen >= 8  # D20
    assert set(cfg.evaluation.controls) == {"isotropic", "orthogonal", "covariance", "other_task", "demo_variation"}
    assert cfg.qualification.enforce
    assert cfg.model.dtype == "bfloat16"
    # D18: paired-excess gate and screen against all three random-direction kinds
    assert cfg.qualification.gate_test == "paired_excess" and cfg.calibration.screen_test == "paired_excess"
    assert set(cfg.qualification.gate_control_kinds) == {"isotropic", "orthogonal", "covariance"}
    assert sum(cfg.evaluation.controls[k] for k in cfg.qualification.gate_control_kinds) >= 48  # D20: 16 per kind
    assert cfg.model.batch_size == 128  # D20
    # D21: the canonical function vector is the control (10 universal heads); PC1 is still extracted
    assert cfg.extraction.control == "function_vector"
    assert (cfg.extraction.function_vector.n_heads, cfg.extraction.function_vector.head_selection) == (None, "universal")  # D26
    assert cfg.extraction.function_vector.head_count_candidates == [1, 2, 4, 8, 16, 32] and cfg.extraction.function_vector.head_support_null == 16
    assert cfg.extraction.function_vector.head_support_min_restored == 0.1
    assert cfg.extraction.function_vector.aie_seeds == 1 and cfg.extraction.function_vector.aie_metric == "target_probability"
    assert cfg.qualification.random_control_max_p <= 0.05 and cfg.calibration.random_screen_max_p <= 0.05


def test_config_validation_errors():
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym"}], "model": {"backend": "onnx"}})
    with pytest.raises(ValueError):
        config_from_dict({"tasks": []})
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym"}], "calibration": {"rho_grid": [1.0, 0.5]}})
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym"}], "unknown_key": 1})
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym"}], "qualification": {"gate_test": "median"}})
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym"}], "qualification": {"gate_control_kinds": ["other_task"]}})
    with pytest.raises(ValueError):  # a gate kind with no controls configured
        config_from_dict({"tasks": [{"name": "antonym"}], "evaluation": {"controls": {"isotropic": 8}}})
    legacy = config_from_dict({"tasks": [{"name": "antonym"}], "qualification": {"gate_test": "rank"},
                               "calibration": {"screen_test": "rank"}})
    assert legacy.qualification.gate_test == "rank"
    cfg = config_from_dict({"tasks": [{"name": "antonym"}, {"name": "arithmetic", "params": {"operand": 7}}]})
    assert cfg.tasks[1].params == {"operand": 7}

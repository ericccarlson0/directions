from pathlib import Path

import pytest

from directions.config import config_from_dict, config_to_dict, load_config, load_trajectories_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_all_configs_load():
    for p in sorted(CONFIGS.glob("*.yaml")):
        if "trajectories" in p.name:  # the comparison's own config (D32): no model, no tasks
            cfg = load_trajectories_config(p)
            assert cfg.n_isotropic >= 1 and cfg.layers == "selected", p
            continue
        if p.name == "source.yaml":  # the source test's own config (D38)
            from directions.config import load_source_config

            cfg = load_source_config(p)
            assert cfg.n_controls >= 1 and cfg.variants, p
            continue
        cfg = load_config(p)
        assert cfg.tasks, p


QWEN_CONFIGS = {
    "pilot_qwen3_0.6b.yaml": "Qwen/Qwen3-0.6B-Base",
    "pilot_qwen3_1.7b.yaml": "Qwen/Qwen3-1.7B-Base",
    "pilot_qwen3_4b.yaml": "Qwen/Qwen3-4B-Base",
    "pilot_qwen3_8b.yaml": "Qwen/Qwen3-8B-Base",
    "pilot_olmo3_7b.yaml": "allenai/Olmo-3-1025-7B",  # D35: two further families, the same protocol
    "pilot_gemma4_12b.yaml": "google/gemma-4-12B",
    "pilot_qwen3_8b_post.yaml": "Qwen/Qwen3-8B",  # D36: the post-trained checkpoint with a fitted Jacobian lens
}


def _normalise_model(a: dict, b: dict, fname: str) -> None:
    """The OLMo 3 configs run at batch 32 (D35: batch 128 does not fit the 24 GB tier); the batch enters no
    statistic. Everything else must match."""
    b["model"]["name"] = a["model"]["name"]
    if "olmo3" in fname:
        assert b["model"]["batch_size"] == 32, fname
        b["model"]["batch_size"] = a["model"]["batch_size"]


def test_qwen_configs_differ_only_in_model_name():
    a = config_to_dict(load_config(CONFIGS / "pilot_qwen3_0.6b.yaml"))
    for fname, model_name in QWEN_CONFIGS.items():
        b = config_to_dict(load_config(CONFIGS / fname))
        assert b["model"]["name"] == model_name, fname
        _normalise_model(a, b, fname)
        assert a == b, fname


def test_pilot_configs_match_preregistered_sizes():
    cfg = load_config(CONFIGS / "pilot_qwen3_0.6b.yaml")
    assert (cfg.data.n_extraction, cfg.data.n_calibration, cfg.data.n_evaluation) == (64, 64, 192)
    assert len(cfg.tasks) == 10
    names = {t.name for t in cfg.tasks}
    assert {"last_antonym", "arithmetic_words"} <= names and not ({"add_two", "en_fr"} & names)  # D19
    assert cfg.extraction.n_seeds == 3 and not cfg.extraction.center
    assert sum(cfg.evaluation.controls[k] for k in cfg.evaluation.gate_kinds) >= 32 and cfg.calibration.n_random_screen >= 8  # D20
    assert set(cfg.evaluation.controls) == {"isotropic", "orthogonal", "covariance", "other_task", "demo_variation", "common"}  # D28
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
    assert cfg.extraction.function_vector.head_count_candidates == [1, 2, 4, 8, 16, 32, 64] and cfg.extraction.function_vector.head_count_min_gain == 0.1 and cfg.extraction.function_vector.head_support_null == 16
    assert cfg.extraction.function_vector.head_support_min_restored == 0.1
    assert cfg.extraction.function_vector.aie_seeds == 1 and cfg.extraction.function_vector.aie_metric == "target_probability"
    assert cfg.qualification.random_control_max_p <= 0.05 and cfg.calibration.random_screen_max_p <= 0.05
    # D27: no injection past half depth; the layer is the one whose selected strength improves most
    assert cfg.extraction.candidate_depth_fractions == [0.2, 0.3, 0.4, 0.5] and cfg.calibration.layer_rule == "best"
    assert cfg.evaluation.controls["common"] == 1


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


def test_task_labels_and_target_scoring():
    # one registry task twice needs distinct labels; the label is the task's identity in the run
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "arithmetic", "params": {"operand": 1}}, {"name": "arithmetic", "params": {"operand": 2}}]})
    cfg = config_from_dict({"tasks": [{"name": "arithmetic", "label": "add_1", "params": {"operand": 1}},
                                      {"name": "arithmetic", "label": "add_2", "params": {"operand": 2}, "target_scoring": "changed_tokens"}]})
    assert [t.key for t in cfg.tasks] == ["add_1", "add_2"] and cfg.tasks[0].name == "arithmetic"
    assert cfg.tasks[0].target_scoring is None and cfg.tasks[1].target_scoring == "changed_tokens"
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym", "target_scoring": "last_token"}]})
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym"}], "prompt": {"target_scoring": "last_token"}})
    with pytest.raises(ValueError):
        config_from_dict({"tasks": [{"name": "antonym", "label": "a/b"}]})


LEARNED_CONFIGS = {
    "learned_qwen3_0.6b.yaml": "Qwen/Qwen3-0.6B-Base",
    "learned_qwen3_1.7b.yaml": "Qwen/Qwen3-1.7B-Base",
    "learned_qwen3_4b.yaml": "Qwen/Qwen3-4B-Base",
    "learned_qwen3_8b.yaml": "Qwen/Qwen3-8B-Base",
    "learned_olmo3_7b.yaml": "allenai/Olmo-3-1025-7B",
    "learned_gemma4_12b.yaml": "google/gemma-4-12B",
    "learned_qwen3_8b_post.yaml": "Qwen/Qwen3-8B",
}


def test_learned_configs_are_the_pilot_with_the_learned_control():
    """The learned-vector configs (D31) differ from the pilot only in the run name, the control (and its
    settings) and the arithmetic task's label and scoring (D30)."""
    pilot = config_to_dict(load_config(CONFIGS / "pilot_qwen3_0.6b.yaml"))
    a = config_to_dict(load_config(CONFIGS / "learned_qwen3_0.6b.yaml"))
    for fname, model_name in LEARNED_CONFIGS.items():
        b = config_to_dict(load_config(CONFIGS / fname))
        assert b["model"]["name"] == model_name, fname
        _normalise_model(a, b, fname)
        assert a == b, fname
    assert a["extraction"]["control"] == "learned_vector" and a["extraction"]["learned_vector"]["n_steps"] == 100
    ext_a, ext_p = dict(a["extraction"]), dict(pilot["extraction"])
    for k in ("control", "learned_vector"):
        ext_a.pop(k), ext_p.pop(k)
    assert ext_a == ext_p
    tasks_a = [t for t in a["tasks"] if t["name"] != "arithmetic"]
    tasks_p = [t for t in pilot["tasks"] if t["name"] != "arithmetic"]
    assert tasks_a == tasks_p
    arith = next(t for t in a["tasks"] if t["name"] == "arithmetic")
    assert arith["label"] == "add_3" and arith["target_scoring"] == "changed_tokens" and arith["params"]["operand"] == 3
    for key in ("model", "prompt", "data", "qualification", "calibration", "evaluation", "commitment", "exploratory",
                "analysis", "figures", "seed"):
        assert a[key] == pilot[key], key


ARITH_CONFIGS = {
    "arith_qwen3_0.6b.yaml": "Qwen/Qwen3-0.6B-Base",
    "arith_qwen3_1.7b.yaml": "Qwen/Qwen3-1.7B-Base",
    "arith_qwen3_4b.yaml": "Qwen/Qwen3-4B-Base",
    "arith_qwen3_8b.yaml": "Qwen/Qwen3-8B-Base",
}


def test_arith_configs_are_the_pilot_with_the_add_k_family():
    """The arithmetic configs differ from the pilot only in the tasks (D30) and the run name."""
    pilot = config_to_dict(load_config(CONFIGS / "pilot_qwen3_0.6b.yaml"))
    a = config_to_dict(load_config(CONFIGS / "arith_qwen3_0.6b.yaml"))
    for fname, model_name in ARITH_CONFIGS.items():
        b = config_to_dict(load_config(CONFIGS / fname))
        assert b["model"]["name"] == model_name, fname
        b["model"]["name"] = a["model"]["name"]
        assert a == b, fname
    labels = [t["label"] for t in a["tasks"]]
    assert labels == [f"add_{k}" for k in (1, 2, 3, 5, 10)] + [f"add_{k}_words" for k in (1, 2, 3, 5, 10)]
    assert all(t["name"] in ("arithmetic", "arithmetic_words") for t in a["tasks"])
    assert all(t["target_scoring"] == "changed_tokens" for t in a["tasks"] if t["name"] == "arithmetic")
    assert [t["params"]["operand"] for t in a["tasks"]] == [1, 2, 3, 5, 10] * 2
    for key in ("model", "prompt", "data", "qualification", "extraction", "calibration", "evaluation", "commitment",
                "exploratory", "analysis", "figures", "seed"):
        assert a[key] == pilot[key], key

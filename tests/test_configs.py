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
        if p.name in ("mixing.yaml", "mixing_within.yaml"):  # the geometry test's own configs (D40, D44)
            from directions.config import load_mixing_config

            cfg = load_mixing_config(p)
            assert cfg.pairs and {pp.family for pp in cfg.pairs} == {"kth_word", "add_k", "lexical"}, p
            continue
        if p.name in ("serial.yaml", "smoke_toy_serial.yaml"):  # the serial injection test's own config (D41)
            from directions.config import load_serial_config

            cfg = load_serial_config(p)
            assert cfg.compositions and all(len(c.steps) == 2 for c in cfg.compositions), p
            continue
        if p.name.startswith("das_") or p.name == "smoke_toy_das.yaml":  # DAS at the answer token (D48)
            from directions.config import load_das_config

            cfg = load_das_config(p)
            assert cfg.sites and (cfg.control_site in cfg.sites or (cfg.control_site is None and "_b.yaml" in p.name)), p
            continue
        if p.name.startswith("posset_"):  # the position-set check (D52)
            from directions.config import load_posset_config

            cfg = load_posset_config(p)
            assert cfg.read_points and cfg.control in cfg.read_points, p
            continue
        if "varsplit" in p.name:  # splitting the block where the sums appear (D47)
            from directions.config import load_varsplit_config

            cfg = load_varsplit_config(p)
            assert cfg.split_blocks and cfg.head_blocks, p
            continue
        if "varprobe" in p.name:  # the conditional probe and logit lens (D46)
            from directions.config import load_varprobe_config

            cfg = load_varprobe_config(p)
            assert cfg.expressions and cfg.n_examples >= 2 * cfg.n_folds, p
            continue
        if "varmap" in p.name:  # the variable map's own configs (D45)
            from directions.config import load_varmap_config

            cfg = load_varmap_config(p)
            assert cfg.expressions and cfg.n_bases >= 2, p
            continue
        if p.name in ("staging.yaml", "smoke_toy_staging.yaml"):  # the composition test re-run's own config (D42)
            from directions.config import load_staging_config

            cfg = load_staging_config(p)
            assert cfg.compositions and all(c.references and c.intermediates for c in cfg.compositions), p
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


KTH_CONFIGS = {
    "kth_qwen3_0.6b.yaml": "Qwen/Qwen3-0.6B-Base",
    "kth_qwen3_1.7b.yaml": "Qwen/Qwen3-1.7B-Base",
    "kth_qwen3_4b.yaml": "Qwen/Qwen3-4B-Base",
    "kth_qwen3_8b.yaml": "Qwen/Qwen3-8B-Base",
    "kth_olmo3_7b.yaml": "allenai/Olmo-3-1025-7B",
    "kth_gemma4_12b.yaml": "google/gemma-4-12B",
}


def _kth_tasks(cfg: dict) -> None:
    assert [t["label"] for t in cfg["tasks"]] == [f"kth_{k}" for k in range(1, 4)]
    assert all(t["name"] == "kth_word" for t in cfg["tasks"])
    assert [t["params"] for t in cfg["tasks"]] == [{"n_words": 3, "k": k, "n_items": 500, "items_seed": 20260908} for k in range(1, 4)]


def test_kth_configs_are_the_pilot_and_the_learned_config_with_the_kth_word_family():
    """The k-th word configs (D39) differ from the pilot (head mean) and the learned config only in the tasks and
    the run name; the files of a family differ only in `model.name`."""
    for base, prefix in (("pilot_qwen3_0.6b.yaml", "kth_"), ("learned_qwen3_0.6b.yaml", "kth_learned_")):
        ref = config_to_dict(load_config(CONFIGS / base))
        a = config_to_dict(load_config(CONFIGS / (prefix + "qwen3_0.6b.yaml")))
        for fname, model_name in KTH_CONFIGS.items():
            b = config_to_dict(load_config(CONFIGS / (prefix + fname[len("kth_"):])))
            assert b["model"]["name"] == model_name, fname
            _normalise_model(a, b, fname)
            assert a == b, fname
        _kth_tasks(a)
        for key in ("model", "prompt", "data", "qualification", "extraction", "calibration", "evaluation", "commitment",
                    "exploratory", "analysis", "figures", "seed"):
            assert a[key] == ref[key], (base, key)


ARITH_LEARNED_CONFIGS = {
    "arith_learned_qwen3_0.6b.yaml": "Qwen/Qwen3-0.6B-Base",
    "arith_learned_qwen3_1.7b.yaml": "Qwen/Qwen3-1.7B-Base",
    "arith_learned_qwen3_4b.yaml": "Qwen/Qwen3-4B-Base",
    "arith_learned_qwen3_8b.yaml": "Qwen/Qwen3-8B-Base",
}


def test_arith_learned_configs_are_the_learned_config_with_the_add_k_family():
    """The learned add-k configs (D40) differ from the learned config only in the tasks and the run name."""
    ref = config_to_dict(load_config(CONFIGS / "learned_qwen3_0.6b.yaml"))
    a = config_to_dict(load_config(CONFIGS / "arith_learned_qwen3_0.6b.yaml"))
    for fname, model_name in ARITH_LEARNED_CONFIGS.items():
        b = config_to_dict(load_config(CONFIGS / fname))
        assert b["model"]["name"] == model_name, fname
        _normalise_model(a, b, fname)
        assert a == b, fname
    assert [t["label"] for t in a["tasks"]] == [f"add_{k}" for k in (1, 2, 3, 5, 10)]
    assert all(t["name"] == "arithmetic" and t["target_scoring"] == "changed_tokens" for t in a["tasks"])
    assert [t["params"]["operand"] for t in a["tasks"]] == [1, 2, 3, 5, 10]
    for key in ("model", "prompt", "data", "qualification", "extraction", "calibration", "evaluation", "commitment",
                "exploratory", "analysis", "figures", "seed"):
        assert a[key] == ref[key], key


COMP_CONFIGS = {
    "comp_qwen3_0.6b.yaml": "Qwen/Qwen3-0.6B-Base",
    "comp_qwen3_1.7b.yaml": "Qwen/Qwen3-1.7B-Base",
    "comp_qwen3_4b.yaml": "Qwen/Qwen3-4B-Base",
    "comp_qwen3_8b.yaml": "Qwen/Qwen3-8B-Base",
    "comp_olmo3_7b.yaml": "allenai/Olmo-3-1025-7B",
    "comp_gemma4_12b.yaml": "google/gemma-4-12B",
}


def test_comp_configs_are_the_pilot_and_the_learned_config_with_the_composition_family():
    """The composition configs (D41) differ from the pilot (head mean) and the learned config only in the tasks and
    the run name; the files of a family differ only in `model.name`; every composed task's references in the
    landmark config are tasks of the family, in step order."""
    from directions.config import load_trajectories_config
    from directions.tasks import build_task

    refs = load_trajectories_config(CONFIGS / "trajectories_composition.yaml").patch.references
    for base, prefix in (("pilot_qwen3_0.6b.yaml", "comp_"), ("learned_qwen3_0.6b.yaml", "comp_learned_")):
        ref = config_to_dict(load_config(CONFIGS / base))
        a = config_to_dict(load_config(CONFIGS / (prefix + "qwen3_0.6b.yaml")))
        for fname, model_name in COMP_CONFIGS.items():
            b = config_to_dict(load_config(CONFIGS / (prefix + fname[len("comp_"):])))
            assert b["model"]["name"] == model_name, fname
            _normalise_model(a, b, fname)
            assert a == b, fname
        keys = [t["label"] or t["name"] for t in a["tasks"]]
        assert keys == ["antonym", "uppercase", "plural", "kth_3", "upper_antonym", "upper_plural", "last_antonym", "upper_last",
                        "upper_last_antonym"]
        assert set(refs) == {"upper_antonym", "upper_plural", "last_antonym", "upper_last", "upper_last_antonym"}
        for task, components in refs.items():
            assert all(c in keys for c in components) and task in keys
            spec = next(t for t in a["tasks"] if (t["label"] or t["name"]) == task)
            steps = spec["params"]["steps"] if spec["name"] == "compose" else ["last_antonym"]
            n_steps = sum(2 if s == "last_antonym" else 1 for s in steps)  # last_antonym is itself two steps
            assert len(components) == n_steps  # one reference per step, the innermost first
            build_task(spec["name"], spec["params"])
        for key in ("model", "prompt", "data", "qualification", "extraction", "calibration", "evaluation", "commitment",
                    "exploratory", "analysis", "figures", "seed"):
            assert a[key] == ref[key], (base, key)


LEARNED_WD_CONFIGS = {
    "learned_wd_qwen3_0.6b.yaml": "Qwen/Qwen3-0.6B-Base",
    "learned_wd_qwen3_1.7b.yaml": "Qwen/Qwen3-1.7B-Base",
    "learned_wd_qwen3_4b.yaml": "Qwen/Qwen3-4B-Base",
    "learned_wd_qwen3_8b.yaml": "Qwen/Qwen3-8B-Base",
}


def test_learned_wd_configs_are_the_learned_config_with_the_weight_decayed_control():
    """The weight-decayed configs (D43) differ from the D31 learned config of the same model only in the run name,
    the control and its settings, so that the two learned constructions compare on the same protocol."""
    for fname, model_name in LEARNED_WD_CONFIGS.items():
        ref = config_to_dict(load_config(CONFIGS / fname.replace("learned_wd_", "learned_")))
        a = config_to_dict(load_config(CONFIGS / fname))
        assert a["model"]["name"] == model_name, fname
        assert a["name"] == "learned_wd" and ref["name"] == "learned"
        assert a["extraction"]["control"] == "learned_vector_wd"
        wd = a["extraction"]["learned_vector_wd"]
        assert wd["optimizer"] == "sgd" and wd["weight_decay"] == 10.0 and wd["n_steps"] == 200 and wd["lr"] == 0.0005
        for x in (a, ref):
            x.pop("name")
            x["extraction"].pop("control")
            x["extraction"].pop("learned_vector_wd")
        assert a == ref, fname


def test_family_learned_wd_configs_are_the_family_learned_configs_with_the_weight_decayed_control():
    """The k-th word and add-k weight-decayed configs (D43, D44) differ from the D31 family configs of the same model
    only in the run name, the control and its settings, which are the lexical weight-decayed config's."""
    for fam in ("kth", "arith"):
        for m in ("0.6b", "1.7b", "4b", "8b"):
            ref = config_to_dict(load_config(CONFIGS / f"{fam}_learned_qwen3_{m}.yaml"))
            a = config_to_dict(load_config(CONFIGS / f"{fam}_learned_wd_qwen3_{m}.yaml"))
            lex = config_to_dict(load_config(CONFIGS / f"learned_wd_qwen3_{m}.yaml"))
            assert a["name"] == f"{fam}_learned_wd" and a["extraction"]["control"] == "learned_vector_wd"
            assert a["extraction"]["learned_vector_wd"] == lex["extraction"]["learned_vector_wd"]
            for x in (a, ref):
                x.pop("name")
                x["extraction"].pop("control")
                x["extraction"].pop("learned_vector_wd")
            assert a == ref, (fam, m)


def test_varmap_configs_differ_only_in_model_name():
    """The variable map's family (D45): the Qwen3 files differ only in `model.name`, and carry D45's settings."""
    from dataclasses import asdict

    from directions.config import load_varmap_config

    cfgs = {p.name: asdict(load_varmap_config(p)) for p in sorted(CONFIGS.glob("varmap_qwen3_*.yaml"))}
    assert set(cfgs) == {"varmap_qwen3_0.6b.yaml", "varmap_qwen3_1.7b.yaml", "varmap_qwen3_4b.yaml", "varmap_qwen3_8b.yaml", "varmap_qwen3_14b.yaml"}
    ref = next(iter(cfgs.values()))
    for name, c in cfgs.items():
        a = dict(c, model=dict(c["model"], name=None))
        b = dict(ref, model=dict(ref["model"], name=None))
        assert a == b, name
    assert ref["expressions"] == ["two_lane", "one_lane"] and ref["n_bases"] == 64 and ref["n_shots"] == 8
    assert ref["present_flip"] == 0.5 and ref["value_change"] == 0.25 and ref["read_points"] is None
    assert ref["model"]["device"] == "cuda" and ref["seed"] == 20260916


def test_varprobe_configs_differ_only_in_model_name_and_marks():
    """The conditional probe's family (D46): the Qwen3 files differ only in `model.name` and the figure marks."""
    from dataclasses import asdict

    from directions.config import load_varprobe_config

    cfgs = {p.name: asdict(load_varprobe_config(p)) for p in sorted(CONFIGS.glob("varprobe_qwen3_*.yaml"))}
    assert set(cfgs) == {"varprobe_qwen3_0.6b.yaml", "varprobe_qwen3_1.7b.yaml", "varprobe_qwen3_4b.yaml", "varprobe_qwen3_8b.yaml", "varprobe_qwen3_14b.yaml"}
    ref = next(iter(cfgs.values()))
    for name, c in cfgs.items():
        assert dict(c, model=dict(c["model"], name=None), marks=None) == dict(ref, model=dict(ref["model"], name=None), marks=None), name
    assert ref["n_examples"] == 2048 and ref["n_folds"] == 5 and ref["n_perm"] == 200 and ref["alpha"] == 0.05
    assert ref["lambdas"] == [0.1, 1.0, 10.0, 100.0] and ref["gate"] and ref["positive_control"]
    assert ref["model"]["device"] == "cuda" and ref["seed"] == 20260916


def test_varsplit_configs_differ_only_in_model_tasks_and_blocks():
    """The block split's family (D47): the Qwen3 files differ only in `model.name`, the task list and the blocks
    around the D45 gather (G-1..G+3 probed, G..G+2 swapped)."""
    from dataclasses import asdict

    from directions.config import load_varsplit_config

    cfgs = {p.name: asdict(load_varsplit_config(p)) for p in sorted(CONFIGS.glob("varsplit_qwen3_*.yaml"))}
    assert set(cfgs) == {"varsplit_qwen3_0.6b.yaml", "varsplit_qwen3_1.7b.yaml", "varsplit_qwen3_4b.yaml", "varsplit_qwen3_8b.yaml", "varsplit_qwen3_14b.yaml"}
    varying = ("expressions", "gather", "split_blocks", "head_blocks")
    ref = next(iter(cfgs.values()))
    for name, c in cfgs.items():
        G = c["gather"]
        assert c["split_blocks"] == list(range(G - 1, G + 4)) and c["head_blocks"] == [G, G + 1, G + 2], name
        a = dict(c, model=dict(c["model"], name=None), **{k: None for k in varying})
        b = dict(ref, model=dict(ref["model"], name=None), **{k: None for k in varying})
        assert a == b, name
    assert cfgs["varsplit_qwen3_0.6b.yaml"]["expressions"] == ["one_lane"]
    assert cfgs["varsplit_qwen3_4b.yaml"]["gather"] == cfgs["varsplit_qwen3_8b.yaml"]["gather"] == 24
    assert cfgs["varsplit_qwen3_14b.yaml"]["gather"] == 30
    assert ref["n_examples"] == 2048 and ref["n_bases"] == 64 and ref["n_perm"] == 200 and ref["model"]["device"] == "cuda"


def test_das_configs_differ_only_in_model_and_sites():
    """DAS (D48): the Qwen3 files differ only in `model.name`, the sites and the control site, which D48 fixes."""
    from dataclasses import asdict

    from directions.config import load_das_config

    cfgs = {p.name: asdict(load_das_config(p)) for p in sorted(CONFIGS.glob("das_qwen3_*.yaml"))}
    assert set(cfgs) == {"das_qwen3_1.7b.yaml", "das_qwen3_4b.yaml"}
    assert cfgs["das_qwen3_1.7b.yaml"]["sites"] == [19, 21, 22, 23, 24, 28] and cfgs["das_qwen3_1.7b.yaml"]["control_site"] == 19
    assert cfgs["das_qwen3_4b.yaml"]["sites"] == [22, 24, 25, 26, 36] and cfgs["das_qwen3_4b.yaml"]["control_site"] == 22
    ref = cfgs["das_qwen3_1.7b.yaml"]
    for name, c in cfgs.items():
        strip = dict(model=dict(c["model"], name=None), sites=None, control_site=None)
        assert dict(c, **strip) == dict(ref, **dict(model=dict(ref["model"], name=None), sites=None, control_site=None)), name
    assert ref["variables"] == ["S1", "S2", "Y", "a"] and ref["ks"] == [1, 2, 4, 8, 16] and ref["null_k"] == 16
    assert (ref["n_train"], ref["n_val"], ref["n_test"]) == (1536, 256, 512) and ref["gate"] and ref["model"]["device"] == "cuda"
    assert (ref["cause_lower"], ref["isolate_lower"], ref["k_tolerance"]) == (0.5, 0.75, 0.05)


def test_das_8b_config_is_the_d49_design():
    """DAS on the 8B (D49): fixed relative to the D45 gather (G = 24) before any 8B DAS data; otherwise D48's settings."""
    from dataclasses import asdict

    from directions.config import load_das_config

    c = asdict(load_das_config(CONFIGS / "das_d49_qwen3_8b.yaml"))
    ref = asdict(load_das_config(CONFIGS / "das_qwen3_4b.yaml"))
    assert c["model"]["name"] == "Qwen/Qwen3-8B-Base" and c["sites"] == [21, 24, 25, 26] and c["control_site"] == 21
    assert c["ks"] == [8, 16, 32] and c["null_k"] == 32 and c["epochs"] == 20
    fixed = ("seed", "variables", "n_train", "n_val", "n_test", "n_shots", "n_demo_pool", "batch_size", "lr", "lambdas",
             "n_boot", "alpha", "k_tolerance", "cause_lower", "isolate_lower", "gate", "determinism_check")
    assert {k: c[k] for k in fixed} == {k: ref[k] for k in fixed}


def test_das_14b_config_is_the_d49_design_at_the_14b_gather():
    """DAS on the 14B (D50): D49's design at the 14B's gather (G = 30) with the control by D50's rule."""
    from dataclasses import asdict

    from directions.config import load_das_config

    c = asdict(load_das_config(CONFIGS / "das_d50_qwen3_14b.yaml"))
    ref = asdict(load_das_config(CONFIGS / "das_d49_qwen3_8b.yaml"))
    assert c["model"]["name"] == "Qwen/Qwen3-14B-Base" and c["sites"] == [28, 30, 31, 32] and c["control_site"] == 28
    strip = dict(name=None, model=None, sites=None, control_site=None)
    assert dict(c, **strip) == dict(ref, **strip)


def test_three_bracket_map_and_probe_configs_differ_from_two_brackets_only_in_the_task():
    """D51: the three-bracket map and probe files equal the two-bracket ones but for expressions (and the probe's marks)."""
    from dataclasses import asdict

    from directions.config import load_varmap_config, load_varprobe_config

    for fam, load in (("varmap", load_varmap_config), ("varprobe", load_varprobe_config)):
        three = {p.name: asdict(load(p)) for p in sorted(CONFIGS.glob(f"{fam}_three_qwen3_*.yaml"))}
        assert set(three) == {f"{fam}_three_qwen3_{s}.yaml" for s in ("1.7b", "4b", "8b", "14b")}
        for name, c in three.items():
            two = asdict(load(CONFIGS / name.replace("_three", "")))
            # the 8B runs at batch 64: 128 ran out of memory on the 24 GB card with the longer prompts (D51)
            batch = 64 if "_8b" in name else two["model"]["batch_size"]
            assert c["expressions"] == ["three_lane"] and c["model"] == dict(two["model"], batch_size=batch), name
            assert dict(c, expressions=None, marks=None, model=None) == dict(two, expressions=None, marks=None, model=None), name


def test_three_bracket_split_and_das_configs_follow_d51():
    """D51 job 2: the block split around the three-bracket gather; DAS with D51's variables and grid, otherwise D49's."""
    from dataclasses import asdict

    from directions.config import load_das_config, load_varsplit_config

    for p in sorted(CONFIGS.glob("varsplit_three_qwen3_*.yaml")):
        c = asdict(load_varsplit_config(p))
        two = asdict(load_varsplit_config(CONFIGS / p.name.replace("_three", "")))
        G = c["gather"]
        assert c["expressions"] == ["three_lane"] and c["split_blocks"] == list(range(G - 1, G + 4)), p.name
        assert c["head_blocks"] == [G, G + 1, G + 2], p.name
        varying = dict(expressions=None, gather=None, split_blocks=None, head_blocks=None, model=None)
        assert dict(c, **varying) == dict(two, **varying), p.name
        batch = 64 if "_8b" in p.name else two["model"]["batch_size"]  # D51: the 8B at batch 64 on the 24 GB card
        assert c["model"] == dict(two["model"], batch_size=batch), p.name
    ref = asdict(load_das_config(CONFIGS / "das_d49_qwen3_8b.yaml"))
    for p in sorted(CONFIGS.glob("das_d51_qwen3_*.yaml")):
        c = asdict(load_das_config(p))
        assert c["task"] == "three_lane" and c["variables"] == ["S1", "S2", "S3", "T", "U", "Y", "a"], p.name
        assert c["ks"] == [16, 32, 64] and c["null_k"] == 64 and c["epochs"] == 20, p.name
        assert c["control_site"] == c["sites"][0] and len(c["sites"]) in (4, 5), p.name
        strip = dict(name=None, model=None, task=None, sites=None, control_site=None, variables=None, ks=None, null_k=None)
        assert dict(c, **strip) == dict(ref, **strip), p.name


def test_position_set_configs_follow_d52():
    """D52: read points from the D51 control to F, marks at the control, G3 and F, the model of the three-bracket map;
    the files differ only in the model, the read points, the marks and the control."""
    from dataclasses import asdict

    import yaml

    from directions.config import load_das_config, load_posset_config

    files = sorted(CONFIGS.glob("posset_qwen3_*.yaml"))
    assert [p.name for p in files] == ["posset_qwen3_14b.yaml", "posset_qwen3_4b.yaml", "posset_qwen3_8b.yaml"]
    ref = None
    for p in files:
        size = p.stem.split("_")[-1]
        c = asdict(load_posset_config(p))
        das = load_das_config(CONFIGS / f"das_d51_qwen3_{size}.yaml")
        G3, F = min(s for s in das.sites if s != das.control_site), max(das.sites)
        assert c["read_points"] == list(range(das.control_site, F + 1)), p.name
        assert c["marks"] == {"control": das.control_site, "G3": G3, "F": F} and c["control"] == das.control_site, p.name
        vm = yaml.safe_load((CONFIGS / f"varmap_three_qwen3_{size}.yaml").read_text())
        assert c["model"]["name"] == vm["model"]["name"] and c["model"]["batch_size"] == vm["model"]["batch_size"], p.name
        assert c["n_bases"] == 256 and c["sources"] == ["S1", "S2", "S3", "all"] and c["min_gain"] == 0.05
        varying = dict(model=None, read_points=None, marks=None, control=None)
        if ref is None:
            ref = dict(c, **varying)
        assert dict(c, **varying) == ref, p.name


def test_joint_das_configs_follow_d53():
    """D53: D51's DAS configs but for the sites (D52's rule, split in two parts with the control in the first), the
    positions (the non-digit set) and the variables (a dropped)."""
    from dataclasses import asdict

    from directions.config import load_das_config

    rule = {"4b": [21, 23, 24, 25, 26], "8b": [21, 23, 25, 26, 27], "14b": [27, 30, 31, 32, 34]}  # D52's outcome
    for size, sites in rule.items():
        d51 = asdict(load_das_config(CONFIGS / f"das_d51_qwen3_{size}.yaml"))
        a = asdict(load_das_config(CONFIGS / f"das_d53_qwen3_{size}_a.yaml"))
        b = asdict(load_das_config(CONFIGS / f"das_d53_qwen3_{size}_b.yaml"))
        assert sorted(a["sites"] + b["sites"]) == sites and not set(a["sites"]) & set(b["sites"]), size
        assert a["control_site"] == d51["control_site"] == sites[0] and b["control_site"] is None, size
        for c in (a, b):
            assert c["positions"] == "non_digit" and c["variables"] == ["S1", "S2", "S3", "T", "U", "Y"], size
            varying = dict(name=None, sites=None, control_site=None, positions=None, variables=None)
            assert dict(c, **varying) == dict(d51, **varying), size

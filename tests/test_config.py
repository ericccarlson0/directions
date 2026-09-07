from __future__ import annotations

from pathlib import Path

import pytest

from directions.config import Config, build, deep_merge, load_config, validate

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_are_valid():
    validate(Config())


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        build(Config, {"run": {"sed": 1}})
    with pytest.raises(ValueError, match="unknown config key"):
        build(Config, {"nope": {}})


def test_scientific_notation_string_is_rejected():
    """YAML 1.1 parses `-1.0e9` as a string; that must not reach a comparison."""
    with pytest.raises(TypeError, match="must be float"):
        build(Config, {"qualification": {"min_target_logprob": "-1.0e9"}})


def test_int_is_widened_to_float():
    cfg = build(Config, {"qualification": {"min_fewshot_accuracy": 1}})
    assert isinstance(cfg.qualification.min_fewshot_accuracy, float)


def test_deep_merge_is_recursive():
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}}


@pytest.mark.parametrize(
    "name", ["base.yaml", "smoke.yaml", "qwen3_0.6b.yaml", "qwen3_1.7b.yaml"]
)
def test_shipped_configs_load(name):
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.run.name
    assert len(cfg.fingerprint()) == 10


def test_extends_chain_overrides_only_named_keys():
    base = load_config(CONFIG_DIR / "qwen3_0.6b.yaml")
    big = load_config(CONFIG_DIR / "qwen3_1.7b.yaml")
    assert big.model.name_or_path == "Qwen/Qwen3-1.7B-Base"
    assert big.data.tasks == base.data.tasks
    assert big.extraction.candidate_layer_fracs == base.extraction.candidate_layer_fracs


def test_fingerprint_changes_with_content():
    a = build(Config, {"run": {"seed": 0}})
    b = build(Config, {"run": {"seed": 1}})
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == build(Config, {"run": {"seed": 0}}).fingerprint()


@pytest.mark.parametrize(
    "payload",
    [
        {"model": {"backend": "vllm"}},
        {"extraction": {"n_seeds": 1}},
        {"extraction": {"candidate_layer_fracs": []}},
        {"extraction": {"candidate_layer_fracs": [1.5]}},
        {"intervention": {"strengths": [-0.1]}},
        {"intervention": {"selection_metric": "bleu"}},
        {"layerwise": {"variance_fraction": 1.0}},
        {"controls": {"kinds": ["adversarial"]}},
        {"controls": {"n_random": 0}},
        {"data": {"n_eval": 1}},
        {"run": {"device": "tpu"}},
    ],
)
def test_validation_rejects_bad_values(payload):
    with pytest.raises(ValueError):
        validate(build(Config, payload))


def test_every_declared_metric_passes_validation():
    from directions.model import BehaviorResult

    for name in BehaviorResult.METRICS:
        validate(build(Config, {"intervention": {"selection_metric": name}}))


def test_selection_p_bounds():
    with pytest.raises(ValueError, match="max_selection_p"):
        validate(build(Config, {"intervention": {"max_selection_p": 0.0}}))
    with pytest.raises(ValueError, match="max_selection_p"):
        validate(build(Config, {"intervention": {"max_selection_p": 1.5}}))

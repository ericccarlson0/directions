from __future__ import annotations

import pytest

from directions.config import Config, build


@pytest.fixture(scope="session")
def tiny_cfg() -> Config:
    """A minimal config on the offline `tiny_random` backend."""
    return build(
        Config,
        {
            "run": {"name": "test", "device": "cpu", "dtype": "float32", "seed": 0},
            "model": {
                "backend": "tiny_random",
                "name_or_path": "tiny_random_qwen3",
                "tiny": {"num_hidden_layers": 4, "hidden_size": 64, "intermediate_size": 128},
            },
            "data": {
                "tasks": ["antonym"],
                "n_extraction": 6,
                "n_calibration": 4,
                "n_eval": 6,
                "n_shot_extraction": 3,
                "n_shot_qualification": 3,
                "n_shot_eval": 0,
            },
            "extraction": {"n_seeds": 2, "candidate_layer_fracs": [0.25, 0.5]},
            "intervention": {"strengths": [0.05, 0.2], "min_improvement": -1.0},
            "qualification": {
                "min_fewshot_accuracy": -1.0,
                "min_target_logprob": -1000000000.0,
                "min_direction_stability": -1.0,
                "min_steering_improvement": -1.0,
                "max_control_p_value": 1.0,
            },
            "controls": {"n_random": 2},
            "bootstrap": {"n_boot": 50},
            "exploratory": {"block_ablation": {"top_k_layers": 1}},
            "compute": {"batch_size": 3, "max_seq_len": 512, "layerwise_batch_size": 3},
        },
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg):
    from directions.model import LanguageModel

    return LanguageModel.from_config(tiny_cfg)

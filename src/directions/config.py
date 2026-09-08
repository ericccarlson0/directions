"""Experiment configuration.

Every scientifically meaningful parameter is a field here, with its default, and
is overridden from a version-controlled YAML file. The fully resolved config is
written into each run directory.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ToyModelConfig:
    hidden_size: int = 32
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 64


@dataclass
class ModelConfig:
    backend: str = "hf"  # "hf" (Hugging Face Transformers) or "toy" (random tiny model, tests only)
    name: str = "Qwen/Qwen3-0.6B-Base"
    dtype: str = "bfloat16"  # model execution dtype; captured activations are float32
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    batch_size: int = 32
    toy: ToyModelConfig = field(default_factory=ToyModelConfig)


@dataclass
class PromptConfig:
    instruction: str = ""  # optional prefix placed before the first demonstration/query
    demo_template: str = "Q: {input}\nA: {output}"
    query_template: str = "Q: {input}\nA:"
    target_template: str = " {output}"
    separator: str = "\n\n"
    n_shots: int = 8


@dataclass
class DataConfig:
    n_extraction: int = 64
    n_calibration: int = 64
    n_evaluation: int = 64
    max_target_tokens: int = 4
    allow_reduced_splits: bool = False  # shrink pools evenly when a task has too few items


@dataclass
class TaskConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    max_target_tokens: int | None = None  # per-task override of data.max_target_tokens


@dataclass
class QualificationConfig:
    enforce: bool = True  # False = smoke mode: gates are computed and logged but never block
    min_fewshot_accuracy: float = 0.5
    min_fewshot_logprob_per_token: float | None = None
    min_stability: float = 0.8  # min pairwise |cos| across extraction seeds at the candidate layer
    steering_alpha: float = 0.05  # one-sided paired bootstrap p threshold on the evaluation pool
    # Gate against the matched random controls (docs/DECISIONS.md D18):
    #   "paired_excess": hierarchical paired bootstrap that the real direction's mean improvement
    #                    exceeds the mean improvement of the gate controls (p <= random_control_max_p)
    #   "rank":          iteration-2 rule, empirical p of the real mean among the control means
    gate_test: str = "paired_excess"
    gate_control_kinds: list[str] = field(default_factory=lambda: ["isotropic", "orthogonal", "covariance"])
    random_control_max_p: float = 0.05  # threshold on the gate test's p-value


@dataclass
class ExtractionConfig:
    n_seeds: int = 3
    center: bool = False  # mean-center differences before PCA (preregistered: off)
    candidate_depth_fractions: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6])


@dataclass
class CalibrationConfig:
    rho_grid: list[float] = field(default_factory=lambda: [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    extend_grid: bool = True  # extend when the best point sits at the upper grid edge
    extension_factor: float = 1.25
    max_rho: float = 2.0
    bootstrap_alpha: float = 0.05
    min_improvement: float = 0.05  # nats per target token over the unsteered baseline
    n_random_screen: int = 16  # matched random directions screened at a candidate point
    screen_kinds: list[str] = field(default_factory=lambda: ["isotropic", "orthogonal"])  # alternating
    screen_test: str = "paired_excess"  # "paired_excess" (D18) or "rank" (iteration-2 rule)
    random_screen_max_p: float = 0.05  # threshold on the screen test's p-value
    n_boot: int = 2000


CONTROL_KINDS = ("isotropic", "orthogonal", "covariance", "other_task", "demo_variation")


@dataclass
class EvaluationConfig:
    """Matched controls on the evaluation pool, by kind (see docs/DECISIONS.md D15).

    isotropic       uniform random unit vector
    orthogonal      uniform random unit vector orthogonal to the control direction
    covariance      random direction drawn from the residual-stream covariance at the layer
    other_task      the control direction of another task at the same layer (up to n)
    demo_variation  PC1 of differences between two correct-demonstration prompts (no task contrast)
    Every control is injected at the same layer, token and absolute norm as the real direction.
    """

    controls: dict[str, int] = field(default_factory=lambda: {"isotropic": 32, "orthogonal": 32, "covariance": 32})
    gate_kinds: list[str] = field(default_factory=lambda: ["isotropic", "orthogonal"])  # the preregistered null
    n_boot: int = 1000
    ci_alpha: float = 0.05
    variance_fraction: float = 0.9  # for d90 and the top-subspace projector P_l


@dataclass
class BlockAblationConfig:
    enabled: bool = True
    n_blocks: int = 2
    criterion: str = "conversion"  # "conversion" | "new_subspace"
    rank_by: str = "value"  # "value": largest median metric; "z": largest z vs matched random controls


@dataclass
class ExploratoryConfig:
    strength_robustness: bool = True
    block_ablation: BlockAblationConfig = field(default_factory=BlockAblationConfig)


@dataclass
class AnalysisConfig:
    """Thresholds for the automatic qualitative profile labels (see docs/PROJECT.md)."""

    conserved_alignment_min: float = 0.7  # median A_l at the final layer
    conserved_abs_cumlog_gain_max: float = 0.5
    dominant_block_share_min: float = 0.4  # one block carries this share of conversion mass
    cascade_entropy_ratio_min: float = 0.85
    delayed_centre_of_mass_min: float = 0.6  # normalised depth (0 = intervention layer, 1 = last block)
    expansion_ratio_min: float = 1.5  # d_eff(final) / d_eff(first defined downstream layer)
    new_direction_z_min: float = 2.0  # mean z of N_l^unc vs random controls


@dataclass
class FiguresConfig:
    enabled: bool = True
    dpi: int = 120


@dataclass
class Config:
    name: str = "pilot"
    seed: int = 0
    output_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tasks: list[TaskConfig] = field(default_factory=list)
    qualification: QualificationConfig = field(default_factory=QualificationConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    exploratory: ExploratoryConfig = field(default_factory=ExploratoryConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    figures: FiguresConfig = field(default_factory=FiguresConfig)


# --------------------------------------------------------------------------- #
# (De)serialisation
# --------------------------------------------------------------------------- #


def _from_dict(cls: type, data: Any, path: str = "") -> Any:
    if not is_dataclass(cls):
        return data
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"config section '{path or cls.__name__}' must be a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys in '{path or 'root'}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        sub = f"{path}.{name}" if path else name
        ftype = hints.get(name)
        if isinstance(ftype, type) and is_dataclass(ftype):
            kwargs[name] = _from_dict(ftype, value, sub)
        elif name == "tasks":
            kwargs[name] = [_from_dict(TaskConfig, t, f"{sub}[{i}]") for i, t in enumerate(value or [])]
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> Config:
    cfg = _from_dict(Config, data)
    validate_config(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(data)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def validate_config(cfg: Config) -> None:
    if cfg.model.backend not in ("hf", "toy"):
        raise ValueError(f"model.backend must be 'hf' or 'toy', got {cfg.model.backend!r}")
    if not cfg.tasks:
        raise ValueError("at least one task is required")
    names = [t.name for t in cfg.tasks]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate task names: {names}")
    if cfg.extraction.n_seeds < 1:
        raise ValueError("extraction.n_seeds must be >= 1")
    if not cfg.extraction.candidate_depth_fractions:
        raise ValueError("extraction.candidate_depth_fractions must be non-empty")
    if any(not (0 <= f < 1) for f in cfg.extraction.candidate_depth_fractions):
        raise ValueError("candidate_depth_fractions must lie in [0, 1)")
    if not cfg.calibration.rho_grid or any(r <= 0 for r in cfg.calibration.rho_grid):
        raise ValueError("calibration.rho_grid must be non-empty and positive")
    if sorted(cfg.calibration.rho_grid) != list(cfg.calibration.rho_grid):
        raise ValueError("calibration.rho_grid must be ascending")
    if cfg.calibration.extension_factor <= 1.0:
        raise ValueError("calibration.extension_factor must be > 1")
    for k, n in cfg.evaluation.controls.items():
        if k not in CONTROL_KINDS:
            raise ValueError(f"unknown control kind {k!r}; choose from {CONTROL_KINDS}")
        if int(n) < 0:
            raise ValueError(f"evaluation.controls[{k!r}] must be >= 0")
    if not cfg.evaluation.gate_kinds:
        raise ValueError("evaluation.gate_kinds must be non-empty")
    for k in cfg.evaluation.gate_kinds:
        if k not in ("isotropic", "orthogonal", "covariance"):
            raise ValueError("evaluation.gate_kinds must be random-direction kinds (isotropic/orthogonal/covariance)")
        if cfg.evaluation.controls.get(k, 0) < 1:
            raise ValueError(f"gate kind {k!r} needs at least one control in evaluation.controls")
    for k in cfg.calibration.screen_kinds:
        if k not in ("isotropic", "orthogonal"):
            raise ValueError("calibration.screen_kinds must be isotropic/orthogonal")
    if cfg.qualification.gate_test not in ("paired_excess", "rank"):
        raise ValueError("qualification.gate_test must be 'paired_excess' or 'rank'")
    if cfg.calibration.screen_test not in ("paired_excess", "rank"):
        raise ValueError("calibration.screen_test must be 'paired_excess' or 'rank'")
    if not cfg.qualification.gate_control_kinds:
        raise ValueError("qualification.gate_control_kinds must be non-empty")
    for k in cfg.qualification.gate_control_kinds:
        if k not in ("isotropic", "orthogonal", "covariance"):
            raise ValueError("qualification.gate_control_kinds must be random-direction kinds (isotropic/orthogonal/covariance)")
        if cfg.evaluation.controls.get(k, 0) < 1:
            raise ValueError(f"gate control kind {k!r} needs at least one control in evaluation.controls")
    if cfg.exploratory.block_ablation.criterion not in ("conversion", "new_subspace"):
        raise ValueError("exploratory.block_ablation.criterion must be 'conversion' or 'new_subspace'")
    if cfg.exploratory.block_ablation.rank_by not in ("value", "z"):
        raise ValueError("exploratory.block_ablation.rank_by must be 'value' or 'z'")
    if cfg.prompt.n_shots < 1:
        raise ValueError("prompt.n_shots must be >= 1")
    for tmpl, needed in (
        (cfg.prompt.demo_template, ("{input}", "{output}")),
        (cfg.prompt.query_template, ("{input}",)),
        (cfg.prompt.target_template, ("{output}",)),
    ):
        for n in needed:
            if n not in tmpl:
                raise ValueError(f"prompt template {tmpl!r} must contain {n}")

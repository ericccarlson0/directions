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
    # How the target's log-probability is scored (docs/DECISIONS.md D30): "all" tokens, or only the
    # "changed_tokens", those that differ from the input's token at the same right-aligned position (for
    # digit-by-digit numbers this drops the digits the operation leaves untouched; identity items keep all)
    target_scoring: str = "all"


@dataclass
class DataConfig:
    n_extraction: int = 64
    n_calibration: int = 64
    n_evaluation: int = 64
    max_target_tokens: int = 4
    allow_reduced_splits: bool = False  # shrink pools evenly when a task has too few items


@dataclass
class TaskConfig:
    name: str  # registry name (directions.tasks.TASKS)
    params: dict[str, Any] = field(default_factory=dict)
    max_target_tokens: int | None = None  # per-task override of data.max_target_tokens
    label: str | None = None  # the task's identity in the run (directories, controls, seeds); default: name.
    # Lets one registry task appear several times with different params, e.g. arithmetic with operands 1..10.
    target_scoring: str | None = None  # per-task override of prompt.target_scoring

    @property
    def key(self) -> str:
        return self.label or self.name


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
class FunctionVectorConfig:
    """Canonical function-vector extraction after Todd et al. (2024); docs/DECISIONS.md D21.

    n_heads         attention heads summed into the function vector: a fixed count (the paper uses 10), or null
                    to choose it by the joint-effect sweep over head_count_candidates (D26: the smallest count
                    whose joint patched effect is not significantly below the best count's)
    head_count_candidates  head counts tried by the sweep (in rank order of the universal ranking)
    head_support_null      random head sets of the chosen size against which the selected set is tested (the
                    head_support gate: the set's joint effect must be positive and exceed the random sets')
    head_support_alpha     p threshold of both tests
    head_count_min_gain  a larger candidate replaces a smaller one only if its pooled effect exceeds the
                    smaller one's by this fraction of it *and* significantly (D26 amendment: with hundreds of
                    pooled prompts a 5 % gain is significant, and the sweep would otherwise always take the
                    largest candidate)
    head_support_min_restored  the set must also restore this fraction of the gap between the deranged prompts'
                    answer probability and the positive prompts' (scale-free; 0 disables)
    head_selection  "universal": one head set for all tasks, ranked by the mean indirect effect over the
                    tasks that reached extraction (the paper's construction); "per_task": each task's own top heads
    aie_seeds       the indirect effect of a head is measured on the deranged-label prompts of this many
                    extraction seeds (their positive prompts supply the mean head outputs of every seed)
    aie_metric      what the indirect effect is measured in: "target_probability" (the recovered probability of
                    the correct answer, teacher-forced over the whole target; bounded and comparable across
                    tasks), "first_token_probability" (the paper's first-token version, which is degenerate when
                    a target starts with a token that carries no answer, e.g. a lone space before digits) or
                    "logprob_per_token" (the pipeline's decision metric; its scale differs by task)
    """

    n_heads: int | None = None
    head_count_candidates: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    head_count_min_gain: float = 0.1
    head_support_null: int = 16
    head_support_alpha: float = 0.05
    head_support_min_restored: float = 0.1
    head_selection: str = "universal"
    aie_seeds: int = 1
    aie_metric: str = "target_probability"


@dataclass
class ExtractionConfig:
    n_seeds: int = 3
    center: bool = False  # mean-center differences before PCA (preregistered: off)
    candidate_depth_fractions: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6])
    # The control direction carried into calibration, the gates and the layerwise measurement:
    #   "pca"              PC1 of the paired few-shot-minus-deranged differences at the layer (D1-D19 protocol)
    #   "function_vector"  the canonical function vector (D21); the PCA direction is still extracted and reported
    control: str = "pca"
    function_vector: FunctionVectorConfig = field(default_factory=FunctionVectorConfig)


@dataclass
class CalibrationConfig:
    """Layer x strength sweep on the calibration pool (docs/EXPERIMENT.md; docs/DECISIONS.md D22).

    strength_unit   what rho multiplies: "layer_norm", the median residual norm at the layer (alpha = rho *
                    median ||h_l||, the D1-D21 protocol), or "natural", the direction's own natural norm
                    (||FV|| for the function vector, the mean-difference norm for PC1), so rho = 1 is the
                    canonical unscaled injection of Todd et al. (2024)
    reference_rho   select, at the chosen layer, the reliable strength nearest this rho (in log distance;
                    ties to the weaker one); None selects the weakest reliable strength (the D1-D21 rule)
    layer_rule      "earliest": the earliest candidate layer with a reliable strength (preregistered);
                    "best": the layer whose selected strength improves the decision metric most
    """

    strength_unit: str = "layer_norm"
    rho_grid: list[float] = field(default_factory=lambda: [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    reference_rho: float | None = None
    layer_rule: str = "earliest"
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


CONTROL_KINDS = ("isotropic", "orthogonal", "covariance", "other_task", "demo_variation", "common")


@dataclass
class EvaluationConfig:
    """Matched controls on the evaluation pool, by kind (see docs/DECISIONS.md D15).

    isotropic       uniform random unit vector
    orthogonal      uniform random unit vector orthogonal to the control direction
    covariance      random direction drawn from the residual-stream covariance at the layer
    other_task      the control direction of another task at the same layer (up to n)
    demo_variation  PC1 of differences between two correct-demonstration prompts (no task contrast)
    common          the leave-one-out common direction: the normalised mean of the *other* tasks' unit control
                    directions (D28; at most 1)
    Every control is injected at the same layer, token and absolute norm as the real direction.
    """

    controls: dict[str, int] = field(default_factory=lambda: {"isotropic": 32, "orthogonal": 32, "covariance": 32})
    gate_kinds: list[str] = field(default_factory=lambda: ["isotropic", "orthogonal"])  # the preregistered null
    n_boot: int = 1000
    ci_alpha: float = 0.05
    variance_fraction: float = 0.9  # for d90 and the top-subspace projector P_l
    # Damage on neutral prose (D24): the intervention applied at the last token of this many task-free sentences
    # (directions.neutral), the KL from their unsteered next-token distribution recorded for every condition; 0 disables
    neutral_prompts: int = 48


@dataclass
class CommitmentConfig:
    """Depth of commitment (docs/DECISIONS.md D29): at every read point after the injection, the perturbation is
    edited along the injected direction (removed, or kept alone) and the surviving held-out effect measured
    against the same edit along random directions.

    n_prompts   evaluation prompts used (the first n of the pool; 0 = all); the base and steered captures are
                re-run on exactly these prompts so the edits are exact
    n_controls  random unit directions the edits are matched against
    handover_shares  shares of the full effect (0-1) at which the hand-over depths are read off the curves:
                the first read point from which removing the direction leaves at least that share, and the
                last read point up to which the direction alone carries at least that share
    """

    enabled: bool = True
    n_prompts: int = 96
    n_controls: int = 8
    alpha: float = 0.05
    n_boot: int = 1000
    handover_shares: list[float] = field(default_factory=lambda: [0.5, 0.9])


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
    # Repeat one steered forward pass, one gradient pass and one profile in the run and require bit
    # identity (docs/DECISIONS.md D23); recorded in metadata.json/determinism_check.
    determinism_check: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tasks: list[TaskConfig] = field(default_factory=list)
    qualification: QualificationConfig = field(default_factory=QualificationConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    commitment: CommitmentConfig = field(default_factory=CommitmentConfig)
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
    if cfg.extraction.n_seeds < 1:
        raise ValueError("extraction.n_seeds must be >= 1")
    if not cfg.extraction.candidate_depth_fractions:
        raise ValueError("extraction.candidate_depth_fractions must be non-empty")
    if any(not (0 <= f < 1) for f in cfg.extraction.candidate_depth_fractions):
        raise ValueError("candidate_depth_fractions must lie in [0, 1)")
    if cfg.extraction.control not in ("pca", "function_vector"):
        raise ValueError("extraction.control must be 'pca' or 'function_vector'")
    fv = cfg.extraction.function_vector
    if fv.n_heads is not None and fv.n_heads < 1:
        raise ValueError("extraction.function_vector.n_heads must be >= 1 or null")
    if not fv.head_count_candidates or any(k < 1 for k in fv.head_count_candidates) or \
            sorted(fv.head_count_candidates) != list(fv.head_count_candidates):
        raise ValueError("extraction.function_vector.head_count_candidates must be positive and ascending")
    if fv.head_support_null < 1:
        raise ValueError("extraction.function_vector.head_support_null must be >= 1")
    if not (0 < fv.head_support_alpha < 1):
        raise ValueError("extraction.function_vector.head_support_alpha must lie in (0, 1)")
    if fv.head_count_min_gain < 0:
        raise ValueError("extraction.function_vector.head_count_min_gain must be >= 0")
    if cfg.commitment.n_prompts < 0 or cfg.commitment.n_controls < 1 or not (0 < cfg.commitment.alpha < 1) or cfg.commitment.n_boot < 1:
        raise ValueError("commitment: n_prompts >= 0, n_controls >= 1, 0 < alpha < 1, n_boot >= 1")
    shares = list(cfg.commitment.handover_shares)
    if not shares or any(not (0 < s < 1) for s in shares) or len(set(shares)) != len(shares):
        raise ValueError("commitment.handover_shares: a non-empty list of distinct shares in (0, 1)")
    if not (0 <= fv.head_support_min_restored < 1):
        raise ValueError("extraction.function_vector.head_support_min_restored must lie in [0, 1)")
    if fv.head_selection not in ("universal", "per_task"):
        raise ValueError("extraction.function_vector.head_selection must be 'universal' or 'per_task'")
    if not (1 <= fv.aie_seeds <= cfg.extraction.n_seeds):
        raise ValueError("extraction.function_vector.aie_seeds must be between 1 and extraction.n_seeds")
    if fv.aie_metric not in ("target_probability", "first_token_probability", "logprob_per_token"):
        raise ValueError("extraction.function_vector.aie_metric must be 'target_probability', 'first_token_probability' "
                         "or 'logprob_per_token'")
    if not cfg.calibration.rho_grid or any(r <= 0 for r in cfg.calibration.rho_grid):
        raise ValueError("calibration.rho_grid must be non-empty and positive")
    if sorted(cfg.calibration.rho_grid) != list(cfg.calibration.rho_grid):
        raise ValueError("calibration.rho_grid must be ascending")
    if cfg.calibration.extension_factor <= 1.0:
        raise ValueError("calibration.extension_factor must be > 1")
    if cfg.calibration.strength_unit not in ("layer_norm", "natural"):
        raise ValueError("calibration.strength_unit must be 'layer_norm' or 'natural'")
    if cfg.calibration.reference_rho is not None and cfg.calibration.reference_rho <= 0:
        raise ValueError("calibration.reference_rho must be positive (or null for the weakest reliable strength)")
    if cfg.calibration.layer_rule not in ("earliest", "best"):
        raise ValueError("calibration.layer_rule must be 'earliest' or 'best'")
    for k, n in cfg.evaluation.controls.items():
        if k not in CONTROL_KINDS:
            raise ValueError(f"unknown control kind {k!r}; choose from {CONTROL_KINDS}")
        if int(n) < 0:
            raise ValueError(f"evaluation.controls[{k!r}] must be >= 0")
    if cfg.evaluation.neutral_prompts < 0:
        raise ValueError("evaluation.neutral_prompts must be >= 0")
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
    scorings = ("all", "changed_tokens")
    if cfg.prompt.target_scoring not in scorings:
        raise ValueError(f"prompt.target_scoring must be one of {scorings}")
    keys = [t.key for t in cfg.tasks]
    if len(set(keys)) != len(keys):
        raise ValueError(f"task labels must be unique (a registry task used twice needs a `label`): {keys}")
    for t in cfg.tasks:
        if t.target_scoring is not None and t.target_scoring not in scorings:
            raise ValueError(f"tasks[{t.key}].target_scoring must be one of {scorings}")
        if t.label is not None and (not t.label or "/" in t.label or t.label != t.label.strip()):
            raise ValueError(f"tasks[{t.name}].label must be a plain name usable as a directory: {t.label!r}")
    for tmpl, needed in (
        (cfg.prompt.demo_template, ("{input}", "{output}")),
        (cfg.prompt.query_template, ("{input}",)),
        (cfg.prompt.target_template, ("{output}",)),
    ):
        for n in needed:
            if n not in tmpl:
                raise ValueError(f"prompt template {tmpl!r} must contain {n}")

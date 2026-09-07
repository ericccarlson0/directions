"""Typed, version-controllable experiment configuration.

Every scientifically meaningful parameter lives here (and therefore in a YAML
file under ``configs/``); nothing that changes a number should be hard-coded in
the pipeline. Unknown keys are a hard error so that a typo in a config never
silently reverts a setting to its default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    name: str = "run"
    output_root: str = "results"
    seed: int = 0
    device: str = "auto"  # auto | cuda | cpu
    dtype: str = "bfloat16"  # model execution dtype; stats are always float64
    notes: str = ""


@dataclass
class TinyModelConfig:
    """Architecture for the dependency-free integration-test backend."""

    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    max_position_embeddings: int = 512
    init_seed: int = 1234


@dataclass
class ModelConfig:
    backend: str = "hf"  # hf | tiny_random
    name_or_path: str = "Qwen/Qwen3-0.6B-Base"
    revision: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str = "eager"
    tiny: TinyModelConfig = field(default_factory=TinyModelConfig)


@dataclass
class PromptTemplate:
    demo: str = "Q: {input}\nA: {output}"
    query: str = "Q: {input}\nA:"
    separator: str = "\n\n"
    target_prefix: str = " "


@dataclass
class DataConfig:
    tasks: list[str] = field(
        default_factory=lambda: ["antonym", "plural", "past_tense", "en_fr", "arithmetic_add"]
    )
    n_extraction: int = 64
    n_calibration: int = 48
    n_eval: int = 64
    n_shot_extraction: int = 10
    n_shot_eval: int = 0
    n_shot_qualification: int = 10
    max_target_tokens: int = 6
    template: PromptTemplate = field(default_factory=PromptTemplate)


@dataclass
class ExtractionConfig:
    n_seeds: int = 3
    center_pca: bool = False
    candidate_layer_fracs: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6])


@dataclass
class InterventionConfig:
    strengths: list[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    )
    token: str = "last_prompt"
    # accuracy | target_logprob | target_logprob_per_token | logit_margin
    selection_metric: str = "target_logprob_per_token"
    min_improvement: float = 0.30
    # "reliably improves" (docs/EXPERIMENT.md) is enforced as a paired one-sided
    # bootstrap test over calibration examples, not by the threshold alone.
    max_selection_p: float = 0.05
    # docs/EXPERIMENT.md requires the steering effect to beat matched random
    # controls. Screening candidate grid points against a small matched-control
    # set on the *calibration* split keeps the "earliest layer, smallest rho"
    # rule from locking onto a point that is real-looking but not separated from
    # random directions. See docs/DECISIONS.md (D12).
    control_screen_n: int = 8
    control_screen_max_p: float = 0.15


@dataclass
class QualificationConfig:
    min_fewshot_accuracy: float = 0.4
    min_target_logprob: float = -6.0
    min_direction_stability: float = 0.5
    min_steering_improvement: float = 0.30
    max_control_p_value: float = 0.1


@dataclass
class ControlsConfig:
    n_random: int = 16
    kinds: list[str] = field(default_factory=lambda: ["isotropic", "orthogonal"])


@dataclass
class LayerwiseConfig:
    variance_fraction: float = 0.9
    measure_token: str = "intervened"  # intervened | last_prompt
    # The preregistered operating point is the *smallest* reliable strength
    # (docs/EXPERIMENT.md). Repeating the layerwise measurement at the strongest
    # reliable strength at the same layer shows whether the amplification
    # profile is an artefact of the injection magnitude. Exploratory.
    also_measure_strongest: bool = True


@dataclass
class BootstrapConfig:
    n_boot: int = 2000
    alpha: float = 0.05


@dataclass
class BlockAblationConfig:
    enabled: bool = True
    top_k_layers: int = 3


@dataclass
class ExploratoryConfig:
    profile_labels: bool = True
    block_ablation: BlockAblationConfig = field(default_factory=BlockAblationConfig)


@dataclass
class FiguresConfig:
    enabled: bool = True
    dpi: int = 150
    format: str = "png"


@dataclass
class ComputeConfig:
    batch_size: int = 16
    max_seq_len: int = 1024
    layerwise_batch_size: int = 8


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    qualification: QualificationConfig = field(default_factory=QualificationConfig)
    controls: ControlsConfig = field(default_factory=ControlsConfig)
    layerwise: LayerwiseConfig = field(default_factory=LayerwiseConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    exploratory: ExploratoryConfig = field(default_factory=ExploratoryConfig)
    figures: FiguresConfig = field(default_factory=FiguresConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)

    # -- convenience -----------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Stable short hash of the fully resolved config, for run directory ids."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:10]


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def _is_dataclass_type(tp) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _unwrap_optional(tp):
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def build(cls, data: dict, path: str = ""):
    """Recursively construct a dataclass from a plain dict, rejecting unknown keys."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"config section {path or '<root>'!r} must be a mapping, got {type(data)}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        where = path or "<root>"
        raise ValueError(
            f"unknown config key(s) {sorted(unknown)} in section {where!r}; "
            f"known keys: {sorted(fields)}"
        )
    kwargs = {}
    for name, value in data.items():
        ftype = _unwrap_optional(fields[name].type)
        if isinstance(ftype, str):  # postponed annotations
            ftype = _unwrap_optional(typing.get_type_hints(cls).get(name, ftype))
        sub = f"{path}.{name}" if path else name
        kwargs[name] = (
            build(ftype, value, sub) if _is_dataclass_type(ftype) else _coerce(ftype, value, sub)
        )
    return cls(**kwargs)


def _coerce(ftype, value, path: str):
    """Coerce and type-check a scalar/list field.

    YAML 1.1 parses e.g. ``-1.0e9`` as a *string*; silently accepting that would
    turn a threshold into an un-comparable value deep inside a run, so scalar
    types are checked (and int->float widened) at load time.
    """
    if ftype is float and isinstance(value, (int, bool)) and not isinstance(value, bool):
        return float(value)
    if ftype in (int, float, str, bool) and value is not None:
        if isinstance(value, bool) != (ftype is bool) or not isinstance(value, ftype):
            raise TypeError(
                f"config key {path!r} must be {ftype.__name__}, got {type(value).__name__} "
                f"({value!r}); quote-free YAML scientific notation such as 1.0e9 parses as a "
                f"string -- write 1.0e+9 or the plain number instead"
            )
    origin = typing.get_origin(ftype)
    if origin is list and value is not None:
        if not isinstance(value, list):
            raise TypeError(f"config key {path!r} must be a list, got {type(value).__name__}")
        (item_type,) = typing.get_args(ftype) or (None,)
        if item_type in (int, float, str, bool):
            return [_coerce(item_type, v, f"{path}[]") for v in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load a YAML config, following an optional ``extends:`` chain.

    ``extends`` is resolved relative to the including file, so configs can share
    a common base (e.g. ``qwen3_1.7b.yaml`` extending ``qwen3_0.6b.yaml``) and
    differ only in the parameters that actually change.
    """
    path = Path(path)
    raw = _load_with_extends(path, seen=set())
    if overrides:
        raw = deep_merge(raw, overrides)
    cfg = build(Config, raw)
    validate(cfg)
    return cfg


def _load_with_extends(path: Path, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular 'extends' chain at {path}")
    seen.add(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return deep_merge(_load_with_extends(path.parent / parent, seen), raw)


def validate(cfg: Config) -> None:
    """Cheap sanity checks that catch scientifically meaningful misconfiguration."""
    if cfg.model.backend not in {"hf", "tiny_random"}:
        raise ValueError(f"unknown model backend {cfg.model.backend!r}")
    if cfg.run.device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"unknown device {cfg.run.device!r}")
    if cfg.extraction.n_seeds < 2:
        raise ValueError("extraction.n_seeds must be >= 2 to measure cross-seed stability")
    if not cfg.extraction.candidate_layer_fracs:
        raise ValueError("extraction.candidate_layer_fracs must be non-empty")
    if any(not 0.0 <= f <= 1.0 for f in cfg.extraction.candidate_layer_fracs):
        raise ValueError("extraction.candidate_layer_fracs must lie in [0, 1]")
    if not cfg.intervention.strengths:
        raise ValueError("intervention.strengths must be non-empty")
    if any(s <= 0 for s in cfg.intervention.strengths):
        raise ValueError("intervention.strengths must be positive")
    valid_metrics = {"accuracy", "target_logprob", "target_logprob_per_token", "logit_margin"}
    if cfg.intervention.selection_metric not in valid_metrics:
        raise ValueError(
            f"unknown selection metric {cfg.intervention.selection_metric!r}; "
            f"expected one of {sorted(valid_metrics)}"
        )
    if not 0.0 < cfg.intervention.max_selection_p <= 1.0:
        raise ValueError("intervention.max_selection_p must lie in (0, 1]")
    if cfg.intervention.control_screen_n < 0:
        raise ValueError("intervention.control_screen_n must be >= 0")
    if not 0.0 < cfg.intervention.control_screen_max_p <= 1.0:
        raise ValueError("intervention.control_screen_max_p must lie in (0, 1]")
    if not 0.0 < cfg.layerwise.variance_fraction < 1.0:
        raise ValueError("layerwise.variance_fraction must lie in (0, 1)")
    if cfg.layerwise.measure_token not in {"intervened", "last_prompt"}:
        raise ValueError(f"unknown layerwise.measure_token {cfg.layerwise.measure_token!r}")
    unknown_kinds = set(cfg.controls.kinds) - {"isotropic", "orthogonal"}
    if unknown_kinds:
        raise ValueError(f"unknown control kind(s) {sorted(unknown_kinds)}")
    if cfg.controls.n_random < 1:
        raise ValueError("controls.n_random must be >= 1")
    if min(cfg.data.n_extraction, cfg.data.n_calibration, cfg.data.n_eval) < 2:
        raise ValueError("each data split needs at least 2 examples")

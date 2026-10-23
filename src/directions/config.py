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
    family: str = "qwen3"  # "qwen3" | "olmo3" | "gemma4": the architecture the random tiny model is built with (tests)
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
class LearnedVectorConfig:
    """Learned single vector (docs/DECISIONS.md D31): projected Adam on one vector per candidate layer, fitted
    on the extraction pool's zero-shot prompts to raise the summed scored log-probability of the target, with
    the model frozen; the vector is kept at the median residual norm of the layer (its natural strength).

    n_steps      optimiser steps per fit (each one forward and one backward over the pool)
    lr_fraction  Adam step size as a fraction of the radius (the vector's norm)
    log_every    record the loss every this many steps (plus before the first and after the last)
    """

    n_steps: int = 100
    lr_fraction: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    log_every: int = 10


@dataclass
class LearnedVectorWDConfig:
    """The weight-decayed learned vector (docs/DECISIONS.md D43): a second learned construction beside D31's, with
    a free norm and an L2 penalty. Per candidate layer, the D31 loss (−mean summed scored log p of the target over
    the extraction pool's zero-shot prompts, the model frozen) plus ``(weight_decay / 2) * (||v|| / radius)^2``,
    the radius being the median residual norm at the layer (the scale of the penalty and of the step only); the
    first seed starts at zero (the carried fit), the others at a random direction of ``init_fraction`` times the
    radius.

    optimizer          "sgd": gradient descent with heavy-ball momentum (rotation-equivariant; the D43 choice);
                       "adam": Adam with a per-coordinate step (kept to reproduce the D43 probe)
    n_steps            optimiser steps per fit (each one forward and one backward over the pool)
    lr                 sgd: step = lr * radius^2 * (momentum buffer), cosine-decayed to zero; lr * weight_decay is
                       the rate per step at which the penalty shrinks a direction the task does not use
    momentum           sgd: the heavy-ball coefficient
    max_step_fraction  sgd: a step's length is capped at this fraction of the radius
    step_fraction      adam: step per coordinate = step_fraction * radius / sqrt(d), cosine-decayed to zero
    weight_decay       the penalty's coefficient, in units of the loss (nats per example) at ||v|| = radius:
                       the penalty there is weight_decay / 2
    init_fraction      the norm of the random starts of the seeds after the first, over the radius
    log_every          record the loss, the objective and the norm every this many steps (plus first and last)
    """

    optimizer: str = "sgd"
    n_steps: int = 200
    lr: float = 5e-4
    momentum: float = 0.9
    max_step_fraction: float = 0.05
    step_fraction: float = 0.02
    weight_decay: float = 10.0
    init_fraction: float = 0.5
    beta1: float = 0.9
    beta2: float = 0.999
    log_every: int = 10


@dataclass
class ExtractionConfig:
    n_seeds: int = 3
    center: bool = False  # mean-center differences before PCA (preregistered: off)
    candidate_depth_fractions: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6])
    # The control direction carried into calibration, the gates and the layerwise measurement:
    #   "pca"              PC1 of the paired few-shot-minus-deranged differences at the layer (D1-D19 protocol)
    #   "function_vector"  the canonical function vector (D21); the PCA direction is still extracted and reported
    #   "learned_vector"   one vector per candidate layer fitted by gradient descent with the model frozen (D31);
    #                      the PCA direction is still extracted and reported, there is no head-support gate
    #   "learned_vector_wd" one vector per candidate layer fitted from zero with a free norm and an L2 penalty
    #                      (weight decay; D43); otherwise as "learned_vector"
    control: str = "pca"
    function_vector: FunctionVectorConfig = field(default_factory=FunctionVectorConfig)
    learned_vector: LearnedVectorConfig = field(default_factory=LearnedVectorConfig)
    learned_vector_wd: LearnedVectorWDConfig = field(default_factory=LearnedVectorWDConfig)


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
    # D35: a read point is readable when the random edits keep their premise (random removal retains the effect
    # within this tolerance of 1, random keeping alone within it of 0); the hand-over depths skip the others
    readability_tolerance: float = 0.1
    # D35: also run the norm-matched random edits (the real edit's size per prompt along the random directions),
    # which catch a model that is sensitive to the edit's size whatever its direction (Gemma 4's deep half)
    matched_controls: bool = True
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


@dataclass
class PatchConfig:
    """The patch test (docs/DECISIONS.md D33): is the component a construction's trajectory shares with the
    natural one what carries the effect? At read points at fixed fractions of the downstream depth, the
    steered perturbation is edited on the query token before the remaining blocks run: the component along
    the natural difference of that prompt removed, or kept alone; and, in the unsteered run, the natural
    difference itself patched in. Each edit is matched against the same edit along random per-example
    directions (paired excess test), the depth-of-commitment mechanics (D29) with a per-example reference.

    constructions    which constructions to test (at the canonical strength)
    layers           "primary": at the primary injection layer only; "all": at every compared layer (the
                     neighbouring candidate layers included, D33 amended)
    depth_fractions  read points as fractions of the downstream depth (0 = the injection layer, 1 = the end)
    n_controls       random per-example directions matched to each edit
    reference        the natural trajectory the shared component is taken along: "icl" (demos minus none)
                     or "task" (demos minus deranged demos)
    references       the composition test (docs/DECISIONS.md D41): per task (by its key in the runs), the keys of
                     its component tasks in step order. Each component's natural difference is captured on the
                     composed task's own held-out prompts (the component's demonstrations before the composed
                     query), and at every read point of the patch grid the composed control's perturbation is
                     read against it: its cosine with the component's difference minus its cosine with the
                     composed task's own, paired over prompts and against the matched isotropic control, plus
                     (``reference_edits``) the keep and remove edits along the component's difference
    reference_edits  run the keep/remove edits along each reference (the causal reading; costs 2 edits x
                     (1 + n_controls) passes per reference and read point)
    """

    enabled: bool = True
    constructions: list[str] = field(default_factory=lambda: ["learned", "fv"])
    layers: str = "primary"
    depth_fractions: list[float] = field(default_factory=lambda: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    n_controls: int = 3
    reference: str = "icl"
    references: dict[str, list[str]] = field(default_factory=dict)
    reference_edits: bool = True


@dataclass
class SubspaceConfig:
    """The causal dimensionality of the shared component (docs/DECISIONS.md D34): how many directions across
    prompts carry the effect. At read points of the primary layer, a k-dimensional subspace is fitted to the
    natural differences of prompts that are not evaluated (the pools named in ``pools``, uncentred principal
    components), and on the held-out prompts the steered perturbation is kept only within that subspace,
    or that subspace is removed, or the prompt's natural difference projected on it is patched into the
    unsteered run; the effect retained is recorded per k against random k-dimensional subspaces per
    example, against the top-k subspace of the unsteered residuals (the background) and against a subspace
    fitted on the other tasks' natural differences (leave-one-task-out: a shared in-context subspace or the
    task's own).

    constructions    which constructions to test (primary layer, canonical strength)
    k_grid           subspace ranks
    depth_fractions  read points as fractions of the downstream depth
    at_handover      also at the hand-over read point: the first patch-grid read point at which keeping the
                     prompt's own direction retains ``handover_share`` of the effect (needs the patch test)
    handover_share   that share, also the share defining k90 (the smallest k reaching it)
    n_controls       random per-example k-dimensional subspaces matched to each edit
    pool_spectrum_all_read_points  also record the pool's uncentred spectrum (explained fractions, participation
                     ratio, top component's cosine with the mean) at every read point (the landmark test, D37)
    other_tasks      fit the leave-one-task-out subspace as well
    background       fit the unsteered residuals' top-k subspace as well
    pools            the runs' prompt pools the subspaces are fitted on (disjoint from the evaluation pool)
    """

    enabled: bool = True
    constructions: list[str] = field(default_factory=lambda: ["learned"])
    k_grid: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    depth_fractions: list[float] = field(default_factory=lambda: [0.5, 1.0])
    at_handover: bool = True
    handover_share: float = 0.9
    n_controls: int = 2
    pool_spectrum_all_read_points: bool = False
    other_tasks: bool = True
    background: bool = True
    pools: list[str] = field(default_factory=lambda: ["extraction", "calibration"])


@dataclass
class TrajectoriesConfig:
    """The all-to-all trajectory comparison (`directions trajectories`; docs/DECISIONS.md D32).

    It reads two finished runs of one model (the head-mean and the learned-vector run, which share seed,
    splits and held-out prompts) and re-captures the held-out residuals under the natural few-shot context
    and under each construction injected at common layers. Everything about the model, the prompts and the
    splits comes from the runs; only the comparison's own parameters live here.

    layers          "selected": per task, the union of the two runs' selected layers (the learned run's is
                    the primary); or an explicit list of candidate layers (calibrated in both runs)
    neighbour_layers  with "selected": also the n nearest candidate layers of the learned run below and above
                    the primary layer (D33 amended; 0 = none). At a neighbouring layer each construction
                    enters at the strength its run's calibration gives it there (the reliable grid point
                    nearest the reference rho, else the natural norm); PC1 is calibrated as everywhere.
    n_isotropic     matched random directions per construction and layer, at the construction's norm: the
                    floor of every alignment
    second_demo_sample  a second few-shot sample per query: the cosine between two natural trajectories is
                    the ceiling of the alignment a construction can reach
    remove_answer_direction  also compare the trajectories with the answer's unembedding direction (the
                    final norm's scale times the first target token's unembedding row) projected out, so that
                    two trajectories that merely raise the same answer do not read as one computation
    remove_generic_response  also compare them with the generic response projected out: the mean trajectory of
                    the isotropic controls (per example and read point), what any perturbation of these norms
                    does downstream, which alone aligns every trajectory with every other late in the stack
    pc1_rho_grid    PC1 has no calibration in the two runs: its strength at a layer is the grid point (in
                    units of the median residual norm, the D1 unit) with the largest mean improvement of the
                    per-token log-probability on the calibration pool
    strength_factors  multiples of each construction's canonical strength at which the comparison is repeated
                    (D33); the first must be 1.0 (the canonical strength, the primary result); every other
                    factor gets its own steered passes, floors, generic response and coherence, plus the
                    cosine between the same construction's trajectories at the two strengths
    generic_diagnostics  record what the generic response is: its energy in its top coordinates and in the
                    unsteered residual's largest coordinates, its cosine with the residual mean, its logit
                    lens (the tokens it promotes at the last read point), and its cosine across strengths
    patch           the patch test (PatchConfig)
    subspace        the causal dimensionality of the shared component (SubspaceConfig, D34)
    """

    name: str = "trajectories"
    seed: int = 20260907
    output_dir: str = "results"
    layers: str | list[int] = "selected"
    neighbour_layers: int = 0
    n_isotropic: int = 4
    second_demo_sample: bool = True
    remove_answer_direction: bool = True
    remove_generic_response: bool = True
    pc1_rho_grid: list[float] = field(default_factory=lambda: [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    strength_factors: list[float] = field(default_factory=lambda: [1.0, 0.5])
    generic_diagnostics: bool = True
    diagnostic_top_k: list[int] = field(default_factory=lambda: [1, 4, 16, 64])
    logit_lens_top: int = 10
    patch: PatchConfig = field(default_factory=PatchConfig)
    subspace: SubspaceConfig = field(default_factory=SubspaceConfig)
    n_boot: int = 2000
    ci_alpha: float = 0.05
    determinism_check: bool = True
    device: str | None = None  # override the runs' model device (tests); None keeps it
    figures: bool = True


def load_trajectories_config(path: str | Path) -> TrajectoriesConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_dict(TrajectoriesConfig, data)
    if cfg.n_isotropic < 1:
        raise ValueError("n_isotropic must be at least 1")
    if isinstance(cfg.layers, str) and cfg.layers != "selected":
        raise ValueError("layers must be 'selected' or a list of layers")
    if not isinstance(cfg.layers, str) and (not cfg.layers or any(int(l) != l or l < 0 for l in cfg.layers)):
        raise ValueError("layers must be a non-empty list of non-negative integers")
    if cfg.neighbour_layers < 0:
        raise ValueError("neighbour_layers must be non-negative")
    if not cfg.pc1_rho_grid or any(r <= 0 for r in cfg.pc1_rho_grid):
        raise ValueError("pc1_rho_grid must be positive")
    if not cfg.strength_factors or cfg.strength_factors[0] != 1.0 or any(f <= 0 for f in cfg.strength_factors) \
            or len(set(cfg.strength_factors)) != len(cfg.strength_factors):
        raise ValueError("strength_factors must start with 1.0 and be positive and distinct")
    if any(k < 1 for k in cfg.diagnostic_top_k):
        raise ValueError("diagnostic_top_k must be positive")
    p = cfg.patch
    if p.reference not in ("icl", "task"):
        raise ValueError("patch.reference must be 'icl' or 'task'")
    if any(c not in ("pca", "fv", "learned") for c in p.constructions):
        raise ValueError("patch.constructions must be among pca, fv, learned")
    if p.layers not in ("primary", "all"):
        raise ValueError("patch.layers must be 'primary' or 'all'")
    if any(not (0 < f <= 1) for f in p.depth_fractions) or p.n_controls < 1:
        raise ValueError("patch.depth_fractions must lie in (0, 1] and patch.n_controls be at least 1")
    for task, refs in (p.references or {}).items():
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) for r in refs):
            raise ValueError(f"patch.references[{task!r}] must be a non-empty list of task keys")
        if task in refs or len(set(refs)) != len(refs):
            raise ValueError(f"patch.references[{task!r}] must name other tasks, each once")
    q = cfg.subspace
    if any(c not in ("pca", "fv", "learned") for c in q.constructions):
        raise ValueError("subspace.constructions must be among pca, fv, learned")
    if not q.k_grid or any(int(k) != k or k < 1 for k in q.k_grid) or len(set(q.k_grid)) != len(q.k_grid):
        raise ValueError("subspace.k_grid must be distinct positive integers")
    if any(not (0 < f <= 1) for f in q.depth_fractions) or q.n_controls < 1 or not (0 < q.handover_share <= 1):
        raise ValueError("subspace.depth_fractions must lie in (0, 1], n_controls be at least 1, handover_share in (0, 1]")
    if q.at_handover and q.enabled and not p.enabled:
        raise ValueError("subspace.at_handover needs the patch test (patch.enabled)")
    if not q.pools or any(s not in ("extraction", "calibration") for s in q.pools):
        raise ValueError("subspace.pools must be a non-empty subset of extraction, calibration")
    return cfg


@dataclass
class SourceConfig:
    """The source test (docs/DECISIONS.md D38): which sublayer writes the increments that turn the injected
    direction into the natural one. On the held-out prompts of every task, the learned vector at the primary
    layer and its canonical strength: the attention and MLP writes of every block at the query token are
    captured in the unsteered zero-shot run, the steered run and the demonstration run, and each block's
    steered and natural increments are split into their attention and MLP parts and their components along the
    prompt's natural difference at the read point they land on. Over the window from the injection layer to
    the hand-over read point (from the landmark run, D37) the aligned write is shared between the two
    sublayers; then each sublayer's aligned component (and its whole increment) is removed from the steered
    run over that window, against random per-example directions of the same norms (D29 mechanics).

    n_controls        random directions injected at the learned vector's norm (the generic response's split)
    n_edit_controls   random per-example directions matched to each removal
    handover_share    the share defining the hand-over read point in the landmark run's patch grid
    variants          removals: "along" (the component along the natural difference), "full" (the whole increment)
    per_block         also remove each block's aligned component alone, one block at a time
    """

    name: str = "source"
    seed: int = 20260907
    output_dir: str = "results"
    n_controls: int = 4
    n_edit_controls: int = 2
    handover_share: float = 0.9
    variants: list[str] = field(default_factory=lambda: ["along", "full"])
    per_block: bool = True
    n_boot: int = 1000
    ci_alpha: float = 0.05
    determinism_check: bool = True
    device: str | None = None


def load_source_config(path: str | Path) -> SourceConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_dict(SourceConfig, data)
    if cfg.n_controls < 1 or cfg.n_edit_controls < 1:
        raise ValueError("n_controls and n_edit_controls must be at least 1")
    if not (0 < cfg.handover_share <= 1):
        raise ValueError("handover_share must lie in (0, 1]")
    if not cfg.variants or any(v not in ("along", "full") for v in cfg.variants):
        raise ValueError("variants must be a non-empty subset of along, full")
    return cfg


@dataclass
class MixingPair:
    """One pair of end labels of a family for the geometry test (docs/DECISIONS.md D40).

    family         the family's name; the command line maps it to a learned run and a head-mean run
    labels         the two end labels (task keys in the runs), a and b
    intermediates  the labels read as intermediates (their masses carry the parameter reading)
    readout        "list_words" (the k-th word family: the listed words' masses), "operands" (add-k: the masses of
                   n + k over the family's operands) or "own_targets" (two tasks with different inputs: each task's
                   own held-out effect along the path)
    operands       the operands of an "operands" readout (a label add_<k> reads as the operand k)
    constructions  "fv" (the head mean), "learned" and/or "within" (the within-label path of the learned vectors:
                   each end label's first fit mixed with its other starts' fits, docs/DECISIONS.md D44)
    """

    family: str
    labels: list[str] = field(default_factory=list)
    intermediates: list[str] = field(default_factory=list)
    readout: str = "list_words"
    operands: list[int] = field(default_factory=list)
    constructions: list[str] = field(default_factory=lambda: ["fv", "learned"])


@dataclass
class MixingConfig:
    """The geometry test (docs/DECISIONS.md D40): a control mixed between two labels of a family, v(t) = alpha *
    normalise((1 - t) u_a + t u_b), injected at label a's selected layer and strength on label a's held-out
    zero-shot prompts and read as the masses of the family's candidate continuations at every weight t, against the
    dilution null (each endpoint mixed with random directions along the same path).

    weights           the mixing weights t (0 and 1 are the endpoints)
    n_null            random unit directions per side for the dilution curves
    interior          the weights within which an intermediate label's maximum counts as interior
    determinism_check one steered pass repeated, bit identity required (D23)
    within_retained_min  a within-label path (D44) is *connected* when at t = 0.5 it beats the dilution null and
                      keeps at least this share of the endpoints' gain over the unsteered readout
    within_null_ceiling  a within-label path is *uninformative* when the dilution null itself keeps at least this share
                      of the gain at t = 0.5 (no path can beat it; D44 amendment)
    within_near_identical_cos  a within-label path whose two fits have at least this cosine is flagged near-identical:
                      its blend is nearly the same vector, so a kept effect is weak evidence (D44 amendment)
    """

    name: str = "mixing"
    seed: int = 20260907
    output_dir: str = "results"
    weights: list[float] = field(default_factory=lambda: [i / 10 for i in range(11)])
    n_null: int = 4
    n_prompts: int = 96
    n_boot: int = 1000
    ci_alpha: float = 0.05
    interior: list[float] = field(default_factory=lambda: [0.2, 0.8])
    determinism_check: bool = True
    within_retained_min: float = 0.5
    within_null_ceiling: float = 0.9
    within_near_identical_cos: float = 0.85
    device: str | None = None
    pairs: list[MixingPair] = field(default_factory=list)


def load_mixing_config(path: str | Path) -> MixingConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    pairs = data.pop("pairs", None) or []
    cfg = _from_dict(MixingConfig, data)
    cfg.pairs = [_from_dict(MixingPair, p, f"pairs[{i}]") for i, p in enumerate(pairs)]
    if not cfg.pairs:
        raise ValueError("the mixing config needs at least one pair")
    if not cfg.weights or any(not (0 <= w <= 1) for w in cfg.weights) or 0.0 not in cfg.weights or 1.0 not in cfg.weights:
        raise ValueError("weights must lie in [0, 1] and include 0 and 1")
    if cfg.n_null < 1:
        raise ValueError("n_null must be at least 1")
    if len(cfg.interior) != 2 or not (0 < cfg.interior[0] < cfg.interior[1] < 1):
        raise ValueError("interior must be [low, high] with 0 < low < high < 1")
    for p in cfg.pairs:
        if len(p.labels) != 2 or p.labels[0] == p.labels[1]:
            raise ValueError(f"pair {p.family}: labels must be two distinct task keys")
        if p.readout not in ("list_words", "operands", "own_targets"):
            raise ValueError(f"pair {p.family}: unknown readout {p.readout!r}")
        if p.readout == "operands" and not p.operands:
            raise ValueError(f"pair {p.family}: an operands readout needs operands")
        if any(c not in ("fv", "learned", "within") for c in p.constructions) or not p.constructions:
            raise ValueError(f"pair {p.family}: constructions must be a non-empty subset of fv, learned, within")
        if "within" in p.constructions and 0.5 not in cfg.weights:
            raise ValueError("the within-label path is read at t = 0.5, which must be among the weights")
    if not (0 < cfg.within_retained_min <= 1):
        raise ValueError("within_retained_min must lie in (0, 1]")
    if not (0 < cfg.within_null_ceiling <= 1) or not (0 < cfg.within_near_identical_cos < 1):
        raise ValueError("within_null_ceiling must lie in (0, 1] and within_near_identical_cos in (0, 1)")
    return cfg


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
    if cfg.extraction.control not in ("pca", "function_vector", "learned_vector", "learned_vector_wd"):
        raise ValueError("extraction.control must be 'pca', 'function_vector', 'learned_vector' or 'learned_vector_wd'")
    wd = cfg.extraction.learned_vector_wd
    if wd.optimizer not in ("sgd", "adam") or wd.n_steps < 1 or not wd.lr > 0 or not (0 <= wd.momentum < 1) \
            or not wd.max_step_fraction > 0 or not (0 < wd.step_fraction < 1) or not wd.weight_decay > 0 \
            or not (0 < wd.init_fraction <= 1) or not (0 <= wd.beta1 < 1) or not (0 <= wd.beta2 < 1) or wd.log_every < 1:
        raise ValueError("extraction.learned_vector_wd: optimizer sgd or adam, n_steps >= 1, lr > 0, 0 <= momentum < 1, "
                         "max_step_fraction > 0, 0 < step_fraction < 1, weight_decay > 0, 0 < init_fraction <= 1, "
                         "0 <= beta1, beta2 < 1, log_every >= 1")
    lv = cfg.extraction.learned_vector
    if lv.n_steps < 1 or not (0 < lv.lr_fraction < 1) or not (0 <= lv.beta1 < 1) or not (0 <= lv.beta2 < 1) or lv.log_every < 1:
        raise ValueError("extraction.learned_vector: n_steps >= 1, 0 < lr_fraction < 1, 0 <= beta1, beta2 < 1, log_every >= 1")
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


@dataclass
class SerialComposition:
    """One composition for the serial injection test (docs/DECISIONS.md D41).

    task    the composed task's key in the runs (its held-out prompts and target are what is read)
    steps   the two component tasks' keys, the first step first (a three-step composition names its two-step
            prefix, itself a task of the family, as the first step)
    """

    task: str
    steps: list[str] = field(default_factory=list)


@dataclass
class SerialConfig:
    """The serial injection test (docs/DECISIONS.md D41): the first component's control at its own selected layer
    and strength, the second component's at a grid of deeper layers, read as the composed task's held-out effect
    (the change in the composed target's log-probability per token) against the second control replaced by random
    unit directions at the same layer and strength; beside it each control alone, the two added at the first
    layer (superposition) and the composed task's own control.

    second_layer_fractions  the second injection's layers as fractions of the stack for the head mean (one vector
                            regardless of layer); the learned vector is injected at its own candidate layers only
    n_null                  random unit directions replacing the second control at each layer
    n_prompts               the first n held-out prompts of the composed task
    determinism_check       one serial pass repeated, bit identity required (D23)
    """

    name: str = "serial"
    seed: int = 20260907
    output_dir: str = "results"
    second_layer_fractions: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    n_null: int = 4
    n_prompts: int = 96
    n_boot: int = 1000
    ci_alpha: float = 0.05
    constructions: list[str] = field(default_factory=lambda: ["fv", "learned"])
    determinism_check: bool = True
    device: str | None = None
    compositions: list[SerialComposition] = field(default_factory=list)


def load_serial_config(path: str | Path) -> SerialConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    comps = data.pop("compositions", None) or []
    cfg = _from_dict(SerialConfig, data)
    cfg.compositions = [_from_dict(SerialComposition, c, f"compositions[{i}]") for i, c in enumerate(comps)]
    if not cfg.compositions:
        raise ValueError("the serial config needs at least one composition")
    if not cfg.second_layer_fractions or any(not (0 < f < 1) for f in cfg.second_layer_fractions):
        raise ValueError("second_layer_fractions must lie in (0, 1)")
    if cfg.n_null < 1 or cfg.n_prompts < 1:
        raise ValueError("n_null and n_prompts must be at least 1")
    if any(c not in ("fv", "learned") for c in cfg.constructions) or not cfg.constructions:
        raise ValueError("constructions must be a non-empty subset of fv, learned")
    for c in cfg.compositions:
        if len(c.steps) != 2 or c.steps[0] == c.steps[1] or c.task in c.steps:
            raise ValueError(f"composition {c.task}: steps must be two distinct component task keys")
    return cfg


@dataclass
class StagingIntermediate:
    """An intermediate of a composition (docs/DECISIONS.md D42): the item's input carried through ``steps`` in
    order. A step is a registry word task (its item list as a lookup), ``uppercase`` or ``last_word`` (the last
    word of a comma-separated list)."""

    label: str
    steps: list[str] = field(default_factory=list)


@dataclass
class StagingComposition:
    """One composition for the staging test (docs/DECISIONS.md D42).

    task           the composed task's key in the runs
    references     the component tasks' keys in step order (the first is step one, read as the intermediate's
                   reference; the first two make the exploratory serial construction)
    intermediates  the intermediates the lens and the masking readouts look for, step one's first
    """

    task: str
    references: list[str] = field(default_factory=list)
    intermediates: list[StagingIntermediate] = field(default_factory=list)


@dataclass
class StagingConfig:
    """The composition test with readouts that can see an intermediate (docs/DECISIONS.md D42): component-specific
    references (each natural difference residualised against the other), the logit lens on the intermediate's and
    the final's first tokens, and layer-wise context masking of the model's own few-shot computation.

    n_prompts               the first n held-out prompts of the composed task (None: all)
    n_isotropic             matched isotropic controls of the composed control (random directions at its layer
                            and strength)
    masking                 run the layer-wise context masking readout
    removal_edits           the exploratory removal edit along the step-one-specific direction at every read point
    n_edit_controls         random directions each removal edit is matched against
    serial_offset_fraction  the exploratory serial construction's second layer, this fraction of the stack
                            past the first
    determinism_check       one steered capture and one masked pass repeated, bit identity required (D23)
    """

    name: str = "staging"
    seed: int = 20260907
    output_dir: str = "results"
    n_prompts: int | None = None
    n_boot: int = 1000
    alpha: float = 0.05
    constructions: list[str] = field(default_factory=lambda: ["fv", "learned"])
    n_isotropic: int = 1
    masking: bool = True
    removal_edits: bool = True
    n_edit_controls: int = 2
    serial_offset_fraction: float = 0.2
    determinism_check: bool = True
    device: str | None = None
    compositions: list[StagingComposition] = field(default_factory=list)


def load_staging_config(path: str | Path) -> StagingConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    comps = data.pop("compositions", None) or []
    cfg = _from_dict(StagingConfig, data)
    cfg.compositions = []
    for i, c in enumerate(comps):
        c = dict(c)
        inter = c.pop("intermediates", None) or []
        comp = _from_dict(StagingComposition, c, f"compositions[{i}]")
        comp.intermediates = [_from_dict(StagingIntermediate, x, f"compositions[{i}].intermediates[{j}]") for j, x in enumerate(inter)]
        cfg.compositions.append(comp)
    if not cfg.compositions:
        raise ValueError("the staging config needs at least one composition")
    if any(c not in ("fv", "learned") for c in cfg.constructions) or not cfg.constructions:
        raise ValueError("constructions must be a non-empty subset of fv, learned")
    if cfg.n_isotropic < 1 or cfg.n_edit_controls < 1 or (cfg.n_prompts is not None and cfg.n_prompts < 2):
        raise ValueError("n_isotropic and n_edit_controls must be at least 1, n_prompts at least 2")
    if not 0 < cfg.serial_offset_fraction < 1:
        raise ValueError("serial_offset_fraction must lie in (0, 1)")
    for c in cfg.compositions:
        if not c.references or c.task in c.references or len(set(c.references)) != len(c.references):
            raise ValueError(f"composition {c.task}: references must be distinct component task keys")
        if not c.intermediates or any(not x.steps for x in c.intermediates):
            raise ValueError(f"composition {c.task}: at least one intermediate, each with at least one step")
    return cfg


@dataclass
class VarmapConfig:
    """The variable map (docs/DECISIONS.md D45): whole-vector swaps between base and source runs of the arithmetic
    expression tasks at every (read point x query token) site, read as the answer the swapped run gives.

    expressions      "two_lane" ((a+b) - (c+d) = ), "one_lane" ((a+b) - c = , the twin) and/or "three_lane"
                     ((a+b) - (c+d) - (e+f) = , D51)
    n_bases          base expressions per task; one source per base and source type
    n_shots          demonstrations per prompt, from a pool disjoint from the bases (shared by a base and its sources)
    n_demo_pool      the demonstration pool's size
    read_points      the read points swapped at (None: all, 0..L)
    present_flip     a variable is present at a site when its flip rate's lower bound reaches this (judgement, D45)
    value_change     ... and held as a value when the same-value swap's change rate's upper bound stays at or below this
    min_accuracy     few-shot accuracy on the bases below which a task is skipped
    gate             drop the bases whose base or any source the model answers wrongly
    identity_check   swap the run's own states back at three sites; bit identity required
    determinism_check  one swap batch repeated; bit identity required (D23)
    """

    name: str = "varmap"
    seed: int = 20260916
    output_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    expressions: list[str] = field(default_factory=lambda: ["two_lane", "one_lane"])
    n_bases: int = 64
    n_shots: int = 8
    n_demo_pool: int = 96
    read_points: list[int] | None = None
    n_boot: int = 1000
    alpha: float = 0.05
    present_flip: float = 0.5
    value_change: float = 0.25
    min_accuracy: float = 0.5
    gate: bool = True
    identity_check: bool = True
    determinism_check: bool = True
    figures: bool = True


def load_varmap_config(path: str | Path) -> VarmapConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_dict(VarmapConfig, data)
    if not cfg.expressions or any(t not in ("two_lane", "one_lane", "three_lane", "max", "maxmin") for t in cfg.expressions):
        raise ValueError("expressions must be a non-empty subset of two_lane, one_lane, three_lane, max, maxmin")
    if "maxmin" in cfg.expressions and cfg.n_shots % 2:
        raise ValueError("maxmin needs an even n_shots: half of the demonstrations of each operator (D57)")
    if cfg.n_bases < 2 or cfg.n_shots < 0 or cfg.n_demo_pool < cfg.n_shots:
        raise ValueError("n_bases must be at least 2 and n_demo_pool at least n_shots")
    if not (0 < cfg.present_flip <= 1 and 0 <= cfg.value_change < 1 and 0 < cfg.alpha < 1):
        raise ValueError("present_flip in (0, 1], value_change in [0, 1), alpha in (0, 1)")
    return cfg


@dataclass
class PossetConfig:
    """The position-set check (docs/DECISIONS.md D52): D45's whole-state swaps, with a set of query tokens swapped
    together at one read point, read against the answer token alone.

    expressions      the expression tasks (two_lane, three_lane)
    n_bases          base expressions; one source per base and source type, drawn as in D45
    sources          the source types (S1, S2, S3 and all on three brackets)
    read_points      the read points swapped at (D52: the D51 control to F)
    marks            read points to mark on the figure, by label (annotation only)
    n_boot           bootstrap resamples of the rates (percentile intervals)
    n_boot_gain      bootstrap resamples of the paired gains over the answer token
    alpha            family-wise error of the gain test, over the read points and sources (Bonferroni)
    min_gain         the gain's point estimate the rule asks for
    jointly_enough   answer + = stands for the non-digit set when its gain is at least this share of the set's
    max_sites        at most this many read points for the joint DAS (the largest gains)
    control          the D51 control read point (joins the joint DAS's sites)
    """

    name: str = "posset"
    seed: int = 20260916
    output_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    expressions: list[str] = field(default_factory=lambda: ["three_lane"])
    n_bases: int = 256
    n_shots: int = 8
    n_demo_pool: int = 96
    sources: list[str] = field(default_factory=lambda: ["S1", "S2", "S3", "all"])
    read_points: list[int] = field(default_factory=list)
    marks: dict[str, int] = field(default_factory=dict)
    n_boot: int = 1000
    n_boot_gain: int = 10000
    alpha: float = 0.05
    min_gain: float = 0.05
    jointly_enough: float = 0.8
    max_sites: int = 4
    control: int | None = None
    min_accuracy: float = 0.5
    gate: bool = True
    identity_check: bool = True
    determinism_check: bool = True
    figures: bool = True


def load_posset_config(path: str | Path) -> PossetConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_dict(PossetConfig, data)
    if not cfg.expressions or any(t not in ("two_lane", "three_lane") for t in cfg.expressions):
        raise ValueError("expressions must be a non-empty subset of two_lane, three_lane")
    allowed = {"two_lane": {"S1", "S2", "both"}, "three_lane": {"S1", "S2", "S3", "all"}}
    for t in cfg.expressions:
        if not cfg.sources or not set(cfg.sources) <= allowed[t]:
            raise ValueError(f"sources for {t} must be a non-empty subset of {sorted(allowed[t])}")
    if not cfg.read_points or any(m < 0 for m in cfg.read_points) or len(set(cfg.read_points)) != len(cfg.read_points):
        raise ValueError("read_points must be distinct non-negative read points")
    if cfg.n_bases < 2 or cfg.n_shots < 0 or cfg.n_demo_pool < cfg.n_shots:
        raise ValueError("n_bases must be at least 2 and n_demo_pool at least n_shots")
    if not (0 < cfg.alpha < 1 and 0 <= cfg.min_gain < 1 and 0 < cfg.jointly_enough <= 1 and cfg.max_sites >= 1):
        raise ValueError("alpha in (0, 1), min_gain in [0, 1), jointly_enough in (0, 1], max_sites >= 1")
    return cfg


@dataclass
class VarprobeConfig:
    """The conditional probe and the logit lens at the answer token (docs/DECISIONS.md D46), on the D45 expression
    tasks: does the answer token carry the answer's decomposition (S1 given Y, a digit given its bracket's sum), and
    from which read point does the model's own unembedding read the answer or a sub-result off it.

    n_examples       expressions per task (capped at those outside the demonstration pool); answered ones enter
    n_folds          outer cross-validation folds of the probe
    lambdas          the ridge grid (the kernel's mean diagonal is 1), chosen per training fold by leave-one-out accuracy
    n_perm           permutations of the target within the conditioning classes (probe) or shuffles (lens)
    alpha            family-wise error over the read points of one target, model and task
    positive_control a given S1 at the a token itself, read point L/2 (must be carried)
    marks            read points to mark on the figures, by label (annotation only)
    """

    name: str = "varprobe"
    seed: int = 20260916
    output_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    expressions: list[str] = field(default_factory=lambda: ["two_lane", "one_lane"])
    n_examples: int = 2048
    n_shots: int = 8
    n_demo_pool: int = 96
    n_folds: int = 5
    lambdas: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    n_perm: int = 200
    alpha: float = 0.05
    min_accuracy: float = 0.5
    gate: bool = True
    positive_control: bool = True
    determinism_check: bool = True
    figures: bool = True
    marks: dict[str, int] = field(default_factory=dict)


@dataclass
class VarsplitConfig:
    """Splitting the block where the sums appear (docs/DECISIONS.md D47): the D46 probe at the states before a block,
    after its attention and after its MLP (and on the two writes alone), and single attention heads of the answer
    token swapped between the D45 bases and sources.

    split_blocks     blocks probed (block b maps read point b-1 to b): D47 sets G-1..G+3 around the D45 gather G
    head_blocks      blocks whose heads are swapped one at a time: G..G+2
    gather           the D45 gather read point (figure annotation)
    share_gain_min   a block's attention share is read only where its accuracy gain is at least this
    carry_lower      a head carries a variable when its transfer's lower bound reaches this (judgement, D47)
    not_carry_upper  ... and not a variable when that transfer's upper bound stays below this
    digit_change     a carrier holds a digit of its bracket when the same-sum swap's change-rate lower bound reaches this
    """

    name: str = "varsplit"
    seed: int = 20260916
    output_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    expressions: list[str] = field(default_factory=lambda: ["two_lane", "one_lane"])
    n_examples: int = 2048
    n_bases: int = 64
    n_shots: int = 8
    n_demo_pool: int = 96
    split_blocks: list[int] = field(default_factory=list)
    head_blocks: list[int] = field(default_factory=list)
    gather: int | None = None
    n_folds: int = 5
    lambdas: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    n_perm: int = 200
    alpha: float = 0.05
    n_boot: int = 1000
    min_accuracy: float = 0.5
    gate: bool = True
    share_gain_min: float = 0.1
    carry_lower: float = 0.1
    not_carry_upper: float = 0.05
    digit_change: float = 0.25
    parts: list[str] = field(default_factory=lambda: ["probe", "heads"])  # Part 1 (probe) and/or Part 2 (heads)
    determinism_check: bool = True
    figures: bool = True


def load_varsplit_config(path: str | Path) -> VarsplitConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_dict(VarsplitConfig, data)
    if not cfg.expressions or any(t not in ("two_lane", "one_lane", "three_lane", "max") for t in cfg.expressions):
        raise ValueError("expressions must be a non-empty subset of two_lane, one_lane, three_lane, max")
    if not cfg.split_blocks or not cfg.head_blocks or min(cfg.split_blocks + cfg.head_blocks) < 1:
        raise ValueError("split_blocks and head_blocks must be non-empty lists of blocks >= 1")
    if not cfg.parts or any(x not in ("probe", "heads") for x in cfg.parts):
        raise ValueError("parts must be a non-empty subset of probe, heads")
    if list(cfg.split_blocks) != list(range(cfg.split_blocks[0], cfg.split_blocks[-1] + 1)):
        raise ValueError("split_blocks must be consecutive")
    if cfg.n_examples < 2 * cfg.n_folds or cfg.n_bases < 2 or cfg.n_demo_pool < cfg.n_shots:
        raise ValueError("n_examples must be at least twice n_folds, n_bases at least 2, n_demo_pool at least n_shots")
    if not cfg.lambdas or any(x <= 0 for x in cfg.lambdas) or list(cfg.lambdas) != sorted(cfg.lambdas):
        raise ValueError("lambdas must be positive and increasing")
    return cfg


@dataclass
class DasConfig:
    """Distributed alignment search at the answer token, scored by Cause and Isolate (docs/DECISIONS.md D48), on the
    two-lane expression task.

    sites            read points of the answer token fitted at (the control site among them)
    control_site     where the operands are not causally present (D45): nothing may be held there; None in a part of
                     a run split by sites whose other part holds the control (D53)
    task             two_lane, or three_lane (D51)
    positions        the swapped tokens: "answer" (the answer token, D48), or a D52 set swapped jointly, "answer_eq" or
                     "non_digit" (one subspace of their concatenated states, D53)
    shared_basis     with a set: one subspace of the residual stream swapped at each of its tokens instead (D54)
    variables        S1, S2, Y and the digit a (the control variable); three brackets also S3, T = S1 - S2, U = S2 + S3
    ks               subspace dimensions; null_k the shuffled-target null's
    n_train/val/test bases per split (disjoint expressions); each with one Cause source, test bases with Isolate sources
    batch_size       bases per training/evaluation batch on the single-token path (the model's batch_size is for captures)
    k_tolerance      k* = the smallest k whose validation score is within this of the best
    cause_lower, isolate_lower   a variable is held when Cause's and Isolate's lower bounds reach these (D48)
    objective        "cause": each subspace trained on Cause alone, selected by validation Cause (D48-D55);
                     "cause_isolate": trained also to keep the base answer on its variable's Isolate sources, and
                     selected by the mean of validation Cause and Isolate (D56)
    preflight        before the training, every step on one batch with throwaway copies of the fits (results unchanged;
                     docs/INFRA.md, item 8)
    """

    name: str = "das"
    seed: int = 20260916
    output_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    task: str = "two_lane"        # or "three_lane" (D51), "max" (D55)
    sites: list[int] = field(default_factory=list)
    control_site: int | None = 0
    positions: str = "answer"     # or a D52 set swapped jointly (D53)
    shared_basis: bool = False    # one subspace shared by the set's tokens (D54)
    variables: list[str] = field(default_factory=lambda: ["S1", "S2", "Y", "a"])
    ks: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    null_k: int = 16
    n_train: int = 1536
    n_val: int = 256
    n_test: int = 512
    n_shots: int = 8
    n_demo_pool: int = 96
    batch_size: int = 32
    epochs: int = 6
    lr: float = 0.05
    lambdas: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    n_boot: int = 1000
    alpha: float = 0.05
    k_tolerance: float = 0.05
    cause_lower: float = 0.5
    isolate_lower: float = 0.75
    objective: str = "cause"      # or "cause_isolate" (D56)
    gate: bool = True             # keep only answered bases and sources (off for the toy smoke)
    determinism_check: bool = True
    preflight: bool = True        # every step on one batch with throwaway fits before the training (results unchanged)
    figures: bool = True


def load_das_config(path: str | Path) -> DasConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_dict(DasConfig, data)
    if not cfg.sites or list(cfg.sites) != sorted(set(cfg.sites)) or (cfg.control_site is not None and cfg.control_site not in cfg.sites):
        raise ValueError("sites must be increasing, distinct and include control_site")
    if cfg.positions not in ("answer", "answer_eq", "non_digit"):
        raise ValueError("positions must be answer, answer_eq or non_digit")
    if cfg.shared_basis and cfg.positions == "answer":
        raise ValueError("shared_basis needs a set of tokens (positions answer_eq or non_digit)")
    if cfg.objective not in ("cause", "cause_isolate"):
        raise ValueError("objective must be cause or cause_isolate")
    allowed = {"two_lane": ("S1", "S2", "Y", "a"), "three_lane": ("S1", "S2", "S3", "T", "U", "Y", "a"),
               "max": ("S1", "S2", "C", "Y", "a")}
    if cfg.task not in allowed:
        raise ValueError("task must be two_lane, three_lane or max")
    if not cfg.variables or any(v not in allowed[cfg.task] for v in cfg.variables):
        raise ValueError(f"variables must be a non-empty subset of {', '.join(allowed[cfg.task])}")
    if not cfg.ks or any(k < 1 for k in cfg.ks) or list(cfg.ks) != sorted(set(cfg.ks)) or cfg.null_k < 1:
        raise ValueError("ks must be increasing positive integers; null_k positive")
    if min(cfg.n_train, cfg.n_val, cfg.n_test) < 2 or cfg.batch_size < 1 or cfg.epochs < 1 or cfg.lr <= 0:
        raise ValueError("n_train, n_val, n_test >= 2; batch_size, epochs >= 1; lr > 0")
    return cfg


def load_varprobe_config(path: str | Path) -> VarprobeConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_dict(VarprobeConfig, data)
    if not cfg.expressions or any(t not in ("two_lane", "one_lane", "three_lane", "max", "maxmin") for t in cfg.expressions):
        raise ValueError("expressions must be a non-empty subset of two_lane, one_lane, three_lane, max, maxmin")
    if "maxmin" in cfg.expressions and cfg.n_shots % 2:
        raise ValueError("maxmin needs an even n_shots: half of the demonstrations of each operator (D57)")
    if cfg.n_examples < 2 * cfg.n_folds or cfg.n_folds < 2 or cfg.n_demo_pool < cfg.n_shots:
        raise ValueError("n_examples must be at least twice n_folds (>= 2), and n_demo_pool at least n_shots")
    if not cfg.lambdas or any(x <= 0 for x in cfg.lambdas) or list(cfg.lambdas) != sorted(cfg.lambdas):
        raise ValueError("lambdas must be positive and increasing")
    if cfg.n_perm < 1 or not 0 < cfg.alpha < 1:
        raise ValueError("n_perm >= 1 and alpha in (0, 1)")
    return cfg


@dataclass
class OpsteerConfig:
    """The operator's natural direction at the operator token, added to max runs and mapped (docs/DECISIONS.md D58).

    n_dir            max expressions (each with its min twin) whose operator-token states give the direction
    n_bases          bases of the switch curve and the steered map (min expressions shown as max), disjoint from them
    n_shots          demonstrations per prompt, half of each operator (D57), from a pool of n_demo_pool
    inject_read_point  the read point of the steered map's direction (D58: 6)
    max_per_type     at most this many bases per swap type enter the steered map (in draw order)
    n_null           random directions of the direction's norm per read point (the switch curve's null)
    read_points      the steered map's swap read points (None: all, 0..L); the switch curve covers 0..L
    present_flip, value_change, n_boot, alpha  as in the variable map (D45)
    gate             keep only the bases the model answers (D58's rules); off only for the toy smoke run
    identity_check   a steered run's own states swapped back at three sites change nothing (bit identity)
    determinism_check  one swap batch repeated; bit identity required (D23)
    """

    name: str = "opsteer"
    seed: int = 20260916
    output_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    n_dir: int = 256
    n_bases: int = 256
    n_shots: int = 8
    n_demo_pool: int = 96
    inject_read_point: int = 6
    max_per_type: int = 64
    n_null: int = 2
    read_points: list[int] | None = None
    n_boot: int = 1000
    alpha: float = 0.05
    present_flip: float = 0.5
    value_change: float = 0.25
    gate: bool = True
    identity_check: bool = True
    determinism_check: bool = True
    figures: bool = True


def load_opsteer_config(path: str | Path) -> OpsteerConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_dict(OpsteerConfig, data)
    if cfg.n_shots % 2 or cfg.n_shots < 2 or cfg.n_demo_pool < cfg.n_shots:
        raise ValueError("n_shots must be even and positive (half of each operator, D57), n_demo_pool at least n_shots")
    if cfg.n_dir < 2 or cfg.n_bases < 2 or cfg.max_per_type < 2 or cfg.n_null < 0 or cfg.inject_read_point < 0:
        raise ValueError("n_dir, n_bases and max_per_type must be at least 2, n_null and inject_read_point non-negative")
    if not (0 < cfg.present_flip <= 1 and 0 <= cfg.value_change < 1 and 0 < cfg.alpha < 1):
        raise ValueError("present_flip in (0, 1], value_change in [0, 1), alpha in (0, 1)")
    return cfg


@dataclass
class OpsetConfig:
    """Where the operator sits between its token and the answer token: token-set swaps (docs/DECISIONS.md D59).

    arms             "natural" (D57's operator swap: a base and its twin), "added" (D58's steered runs, the source the
                     same prompt without the direction) and/or "sums" (D60: the natural bases with D45's both source)
    n_bases          bases per arm (natural: drawn as D57's; added: D58's draws), kept where both runs are answered
    n_dir, n_shots, n_demo_pool, inject_read_point  as in D57 and D58 (the added arm recomputes D58's direction)
    read_points      the read points of the sweep (D59: 8-32)
    n_boot, alpha    bootstrap intervals of the flip rates
    sufficient       a set is sufficient where its flip rate's lower bound reaches this, not where the upper bound
                     stays below it (D59: 0.5)
    gate             keep only the bases the model answers (off only for the toy smoke run)
    identity_check   a run's own states swapped back at every query token change nothing (bit identity)
    determinism_check  one swap batch repeated; bit identity required (D23)
    """

    name: str = "opset"
    seed: int = 20260916
    output_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    arms: list[str] = field(default_factory=lambda: ["natural", "added"])
    n_bases: int = 256
    n_dir: int = 256
    n_shots: int = 8
    n_demo_pool: int = 96
    inject_read_point: int = 6
    read_points: list[int] = field(default_factory=lambda: list(range(8, 33)))
    n_boot: int = 1000
    alpha: float = 0.05
    sufficient: float = 0.5
    gate: bool = True
    identity_check: bool = True
    determinism_check: bool = True
    figures: bool = True


def load_opset_config(path: str | Path) -> OpsetConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_dict(OpsetConfig, data)
    if not cfg.arms or any(a not in ("natural", "added", "sums") for a in cfg.arms) or len(set(cfg.arms)) != len(cfg.arms):
        raise ValueError("arms must be a non-empty subset of natural, added, sums")
    if cfg.n_shots % 2 or cfg.n_shots < 2 or cfg.n_demo_pool < cfg.n_shots:
        raise ValueError("n_shots must be even and positive (half of each operator, D57), n_demo_pool at least n_shots")
    if cfg.n_bases < 2 or cfg.n_dir < 2 or not cfg.read_points or min(cfg.read_points) < 0:
        raise ValueError("n_bases and n_dir must be at least 2, read_points non-empty and non-negative")
    if "added" in cfg.arms and min(cfg.read_points) < cfg.inject_read_point:
        # at the injection read point itself the swap follows the added vector, so it removes it (bit-exact, tested)
        raise ValueError("the added arm's swaps must not come before the injection read point")
    if not (0 < cfg.sufficient <= 1 and 0 < cfg.alpha < 1):
        raise ValueError("sufficient in (0, 1], alpha in (0, 1)")
    return cfg

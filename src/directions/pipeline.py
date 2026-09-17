"""End-to-end pilot orchestration.

Phase A, for every task: (1) items -> filter -> three disjoint pools;
(2) few-shot qualification; (3) control-direction extraction at every candidate
layer with cross-seed stability (plus demonstration-variation null directions).
With ``extraction.control: function_vector`` (D21) stage 3 also records the
per-head outputs and their indirect effects, and a phase between A and B ranks
the heads across tasks and builds every task's canonical function vector, which
then replaces the PCA direction as the control (the PCA direction stays as the
per-layer readout direction and a reported comparison).

Phase B, for every task that reached stage 3: (4) calibration on the
calibration pool; (5) held-out steering on the evaluation pool against matched
controls of every configured kind; (6) layerwise measurement (real and every
control) and cross-task analysis; (7) exploratory strength robustness and block
ablation; (8) figures. ``validate`` stops after stage 5.

Phase A runs first for all tasks so that ``other_task`` controls (another
task's direction at the same layer) exist when phase B needs them.
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .ablation import block_ablation, select_blocks
from .analysis import cross_task_table, profile_signature
from .calibration import CalibrationResult, Selection, calibrate, median_layer_norms
from .commitment import depth_of_commitment
from .config import Config, PromptConfig, TaskConfig, config_to_dict
from .extraction import (
    LayerDirection,
    candidate_layers,
    demo_variation_directions,
    directions_from_differences,
    extract_demo_variation,
    extract_differences,
    permuted_head_means,
)
from .function_vector import (
    FunctionVector,
    build_function_vector,
    choose_head_count,
    compose,
    effect_values,
    head_effects,
    head_support_test,
    joint_head_effect,
    select_heads,
)
from .geometry import (
    covariance_matched_unit_vector,
    normalize,
    random_orthogonal_unit_vector,
    random_unit_vector,
    set_linalg_device,
)
from .learned import FitResult, fit_summary, learned_directions, permuted_target_vectors
from .layerwise import LayerwiseProfile, compare_by_kind, compute_profile, metric_curves, null_summary, profile_arrays
from .model import ForwardResult, Intervention, ModelBackend, environment_metadata, set_torch_determinism
from .neutral import neutral_prompts
from .profiling import Profiler
from .prompts import Prompt, few_shot_prompt, score_reference, zero_shot_prompt
from .runinfo import RejectionLog, git_info, make_run_dir, setup_logging, write_json, write_resolved_config
from .seeds import derive_seed, rng_for
from .stats import compare_to_null, paired_bootstrap_test, paired_excess_test, spearman
from .tasks import Splits, build_task, filter_items, make_splits


@dataclass
class ControlResult:
    kind: str
    label: str
    mean_diff: float
    p_value: float
    metrics: dict[str, float]
    profile: LayerwiseProfile | None = None
    per_example_diff: np.ndarray | None = None  # steered - baseline log p per token, per evaluation example
    per_example_kl: np.ndarray | None = None  # KL(baseline || steered) at the query token, per evaluation example (D24)
    per_example_collateral_kl: np.ndarray | None = None  # the same with the answer token removed (D24)
    per_example_neutral_kl: np.ndarray | None = None  # KL on the neutral prose prompts, per prompt (D24)


@dataclass
class TaskState:
    name: str
    splits: Splits
    prompt_cfg: PromptConfig | None = None  # the run's prompt configuration with this task's target scoring (D30)
    registry_name: str = ""  # the registry task this one was built from (the config's `name`; `label` is `name` here)
    directions: dict[int, LayerDirection] = field(default_factory=dict)  # the control at each candidate layer
    pca_directions: dict[int, LayerDirection] = field(default_factory=dict)  # PC1 at each candidate layer (always)
    all_layer_directions: np.ndarray | None = None  # (L+1, d) pooled PC1 at every read point (D17 readouts)
    # Function-vector inputs and result (D21), when extraction.control == "function_vector":
    seed_head_means: np.ndarray | None = None  # (n_seeds, L, n_heads, head_dim) mean head outputs, positive prompts
    head_effects: np.ndarray | None = None  # (L, n_heads) average indirect effect on deranged-label prompts
    head_effects_n_prompts: int = 0
    permuted_head_means: list[np.ndarray] = field(default_factory=list)  # demo_variation null inputs
    fv: FunctionVector | None = None
    # Learned-vector inputs and result (D31), when extraction.control == "learned_vector":
    learned_fits: dict[int, list[FitResult]] = field(default_factory=dict)  # per candidate layer, per seed
    learned_radii: dict[int, float] = field(default_factory=dict)  # the fitted norm per candidate layer (median residual norm)
    aie_prompts: list[Prompt] = field(default_factory=list)  # the deranged-label prompts the indirect effects use
    aie_baseline: np.ndarray | None = None  # their unpatched metric values
    aie_positive_mean: float | None = None  # the metric's mean on the positive prompts (the head-support ceiling)
    gradients: np.ndarray | None = None  # (L+1, n_eval, d) d log p(target) / d resid at the query token (D17)
    demo_variation: dict[int, list[np.ndarray]] = field(default_factory=dict)
    stable_layers: list[int] = field(default_factory=list)
    ready: bool = False  # reached the end of phase A without a blocking rejection
    calibration: CalibrationResult | None = None
    selection: Selection | None = None
    fallback_selection: bool = False
    eval_prompts: list[Prompt] = field(default_factory=list)
    base: ForwardResult | None = None
    steered: ForwardResult | None = None
    controls: list[ControlResult] = field(default_factory=list)
    qualified: bool = False
    qualification: dict[str, Any] = field(default_factory=dict)
    profile: LayerwiseProfile | None = None
    comparison: dict[str, Any] | None = None
    signature: dict[str, Any] | None = None


class Pipeline:
    def __init__(self, cfg: Config, command: str, config_path: str | None = None, run_id: str | None = None) -> None:
        self.cfg = cfg
        self.command = command
        self.root = make_run_dir(cfg, command, run_id)
        self.log = setup_logging(self.root / "log.txt")
        self.rejections = RejectionLog(self.root / "rejections.jsonl")
        self.prof = Profiler()
        self.stop_after = "steering" if command == "validate" else "figures"
        write_resolved_config(self.root / "config.resolved.yaml", cfg)
        self.metadata: dict[str, Any] = {
            "run_id": self.root.name,
            "command": command,
            "argv": sys.argv,
            "config_path": config_path,
            "directions_version": __version__,
            "git": git_info(Path(__file__).resolve().parents[2]),
            "environment": environment_metadata(),
            "seed": cfg.seed,
            "config": config_to_dict(cfg),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json(self.root / "metadata.json", self.metadata)

    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        cfg = self.cfg
        set_torch_determinism(derive_seed(cfg.seed, "torch"))
        try:
            with self.prof.section("total"):
                self._run()
        finally:
            set_linalg_device(None)
        self.metadata["timings_seconds"] = self.prof.timings()
        self.metadata["profile"] = self.prof.report()
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("profile (sections >= 1 s):\n%s", self.prof.table())
        self.log.info("done in %.1fs -> %s", self.metadata["timings_seconds"]["total"], self.root)
        return self.root

    def _run(self) -> None:
        cfg = self.cfg
        with self.prof.section("model_load"):
            self.backend = ModelBackend(cfg.model, run_seed=cfg.seed)
        self.prof.backend = self.backend  # forward/gradient counters and GPU memory from here on
        # The profile decompositions run on the model's device (docs/DECISIONS.md D22: GPU SVDs).
        set_linalg_device(self.backend.device if self.backend.device.type == "cuda" else None)
        self.metadata["model"] = self.backend.metadata()
        self.metadata["linalg_device"] = str(self.backend.device) if self.backend.device.type == "cuda" else "numpy"
        # the neutral prose set for the damage measure (D24): its unsteered distributions, once per run
        self.neutral = neutral_prompts(cfg.evaluation.neutral_prompts) if cfg.evaluation.neutral_prompts > 0 else []
        self.neutral_ref = None
        if self.neutral:
            with self.prof.section("neutral_baseline"):
                self.neutral_ref = self.backend.reference_tensor(self.backend.run(self.neutral, capture_logprobs=True).query_logprobs)
        self.metadata["neutral_prompts"] = len(self.neutral)
        self.layers = candidate_layers(self.backend.n_layers, cfg.extraction.candidate_depth_fractions)
        self.metadata["candidate_layers"] = self.layers
        self.metadata["control"] = cfg.extraction.control
        self.metadata["seeds"] = self._seed_table()
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("run %s: model %s (%d layers, d=%d), candidate layers %s",
                      self.root.name, self.metadata["model"]["name"], self.backend.n_layers,
                      self.backend.hidden_size, self.layers)

        states: dict[str, TaskState] = {}
        for tcfg in cfg.tasks:
            with self.prof.section("prepare", tcfg.key):
                try:
                    states[tcfg.key] = self._prepare_task(tcfg)
                except Exception as e:
                    self.log.error("task %s failed in phase A: %s", tcfg.key, e)
                    self.rejections.add("error", tcfg.key, f"{type(e).__name__}: {e}", {"traceback": traceback.format_exc()})
        if cfg.extraction.control == "function_vector":
            with self.prof.section("function_vectors"):
                self._build_function_vectors(states)
        for name, st in states.items():
            with self.prof.section("measure", name):
                if st.ready and self.stop_after not in ("fewshot", "extraction"):
                    try:
                        self._measure_task(st, states)
                    except Exception as e:
                        self.log.error("task %s failed in phase B: %s", name, e)
                        self.rejections.add("error", name, f"{type(e).__name__}: {e}", {"traceback": traceback.format_exc()})
                self._finish_task(st)

        summary = self._summarise(states)
        if self.stop_after == "figures" and cfg.figures.enabled:
            with self.prof.section("figures"):
                try:
                    from .figures import make_all_figures

                    make_all_figures(self.root, states, cfg)
                except Exception as e:
                    self.log.error("figure generation failed: %s", e)
                    self.rejections.add("error", "*", f"figures: {type(e).__name__}: {e}", {"traceback": traceback.format_exc()})
        write_json(self.root / "core" / "summary.json", summary)

    def _seed_table(self) -> dict[str, Any]:
        s = self.cfg.seed
        table: dict[str, Any] = {"run_seed": s, "torch": derive_seed(s, "torch"), "head_count": derive_seed(s, "head_count")}
        n_dv = int(self.cfg.evaluation.controls.get("demo_variation", 0))
        for t in self.cfg.tasks:
            table[t.name] = {
                "split": derive_seed(s, "split", t.name),
                "fewshot_eval": derive_seed(s, "fewshot_eval", t.name),
                "extraction": [derive_seed(s, "extraction", t.name, i) for i in range(self.cfg.extraction.n_seeds)],
                "demo_variation": [derive_seed(s, "demo_variation", t.name, k) for k in range(n_dv)],
                "calibration": derive_seed(s, "calibration", t.name),
                "calibration_screen": {str(l): derive_seed(s, "calibration_screen", t.name, l) for l in self.layers},
                "calibration_damage": derive_seed(s, "calibration_damage", t.name),
                "damage_bootstrap": derive_seed(s, "damage_bootstrap", t.name),
                "head_support": derive_seed(s, "head_support", t.name),
                "decomposition_bootstrap": derive_seed(s, "decomposition_bootstrap", t.name),
                "random_controls": derive_seed(s, "random_controls", t.name),
                "covariance_controls": derive_seed(s, "covariance_controls", t.name),
                "bootstrap": derive_seed(s, "bootstrap", t.name),
                "profile_bootstrap": derive_seed(s, "profile_bootstrap", t.name),
            }
        return table

    # ------------------------------------------------------------------ #

    def _task_dir(self, name: str, exploratory: bool = False) -> Path:
        d = self.root / ("exploratory" if exploratory else "core") / "tasks" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- phase A ------------------------------------------------------ #

    def _prepare_task(self, tcfg: TaskConfig) -> TaskState:
        cfg = self.cfg
        log = self.log
        name = tcfg.key  # the task's identity in the run; tcfg.name is the registry task it is built from
        out = self._task_dir(name)
        q: dict[str, Any] = {"task": name, "registry_task": tcfg.name, "gates": {}}
        # the prompt configuration of this task: the run's, with the task's own target scoring (D30)
        prompt_cfg = replace(cfg.prompt, target_scoring=tcfg.target_scoring or cfg.prompt.target_scoring)

        # 1. data --------------------------------------------------------
        with self.prof.section("data", name):
            task = build_task(tcfg.name, tcfg.params)
            max_tok = tcfg.max_target_tokens if tcfg.max_target_tokens is not None else cfg.data.max_target_tokens
            kept, rejected = filter_items(task, self.backend.target_token_count, prompt_cfg.target_template, max_tok)
            for r in rejected:
                self.rejections.add(r.stage, r.task, r.reason, r.details)
            splits = make_splits(kept, cfg.seed, name, cfg.data.n_extraction, cfg.data.n_calibration,
                                 cfg.data.n_evaluation, allow_reduced=cfg.data.allow_reduced_splits)
        if splits.reduced:
            self.rejections.add("data", name, "pools_reduced", {"n_items": len(kept), "sizes": {
                "extraction": len(splits.extraction), "calibration": len(splits.calibration), "evaluation": len(splits.evaluation)}})
        st = TaskState(name=name, splits=splits, prompt_cfg=prompt_cfg, registry_name=tcfg.name)
        st.qualification = q
        # How the formatted targets tokenise: the first token of a multi-token target may carry no answer
        # (a lone space before digits), which matters for first-token metrics (docs/DECISIONS.md D22).
        tok = self._target_tokenisation(kept, prompt_cfg)
        write_json(out / "splits.json", {"n_items_total": len(task.items), "n_items_kept": len(kept),
                                         "n_items_rejected": len(rejected), "max_target_tokens": max_tok,
                                         "registry_task": tcfg.name, "params": task.params,
                                         "target_scoring": prompt_cfg.target_scoring, "target_tokenisation": tok,
                                         **splits.as_dict()})
        log.info("[%s] %d items kept (%d rejected); pools %d/%d/%d; target tokens: mean %.2f, first-token share of "
                 "single-space %.2f, scored %.2f (%s)", name, len(kept), len(rejected), len(splits.extraction),
                 len(splits.calibration), len(splits.evaluation), tok["n_tokens_mean"], tok["first_token_is_space_fraction"],
                 tok["n_scored_mean"], prompt_cfg.target_scoring)

        # 2. few-shot qualification -------------------------------------
        with self.prof.section("fewshot", name):
            rng = rng_for(cfg.seed, "fewshot_eval", name)
            fs_prompts = [few_shot_prompt(prompt_cfg, splits.evaluation, x, rng) for x in splits.evaluation]
            fs = self.backend.run(fs_prompts)
            zs_prompts = [zero_shot_prompt(prompt_cfg, x) for x in splits.evaluation]
            st.eval_prompts = zs_prompts
            st.base = self.backend.run(zs_prompts, capture=True, capture_logprobs=True)
        q["fewshot"] = fs.metrics_dict()
        q["zeroshot_baseline"] = st.base.metrics_dict()
        q["example_prompt"] = {"fewshot": fs_prompts[0].prompt, "zeroshot": zs_prompts[0].prompt, "target": zs_prompts[0].target}
        gate = fs.metrics_dict()["accuracy"] >= cfg.qualification.min_fewshot_accuracy
        if cfg.qualification.min_fewshot_logprob_per_token is not None:
            gate = gate and fs.metrics_dict()["logprob_per_token_mean"] >= cfg.qualification.min_fewshot_logprob_per_token
        q["gates"]["fewshot"] = bool(gate)
        log.info("[%s] few-shot acc %.3f (lp/tok %.3f); zero-shot acc %.3f (lp/tok %.3f)", name,
                 q["fewshot"]["accuracy"], q["fewshot"]["logprob_per_token_mean"],
                 q["zeroshot_baseline"]["accuracy"], q["zeroshot_baseline"]["logprob_per_token_mean"])
        if not gate:
            self.rejections.add("fewshot", name, "fewshot_below_threshold",
                                {"fewshot": q["fewshot"], "min_accuracy": cfg.qualification.min_fewshot_accuracy})
            if cfg.qualification.enforce:
                return st
        if self.stop_after == "fewshot":
            return st

        # 3. extraction --------------------------------------------------
        fv_mode = cfg.extraction.control == "function_vector"
        learned_mode = cfg.extraction.control == "learned_vector"
        with self.prof.section("extraction", name):
            seeds = extract_differences(self.backend, prompt_cfg, splits.extraction, cfg.seed, name, cfg.extraction.n_seeds,
                                        capture_heads=fv_mode)
            st.pca_directions = directions_from_differences(seeds, self.layers, cfg.extraction)
            st.directions = st.pca_directions
            all_dirs = directions_from_differences(seeds, list(range(self.backend.n_layers + 1)), cfg.extraction)
            st.all_layer_directions = np.stack([all_dirs[l].direction for l in range(self.backend.n_layers + 1)])
        ext = {
            "control": cfg.extraction.control,
            "seeds": [{"seed_index": s.seed_index, "seed": s.seed, "metrics": s.metrics} for s in seeds],
            "layers": {str(l): d.summary() for l, d in st.pca_directions.items()},
            "all_layer_stability": {str(l): d.stability for l, d in all_dirs.items()},
            "all_layer_pooled_evr": {str(l): d.pooled_explained_variance_ratio for l, d in all_dirs.items()},
        }
        n_dv = int(cfg.evaluation.controls.get("demo_variation", 0))
        if fv_mode:
            # Function-vector inputs (D21): mean head outputs per seed, and each head's indirect effect
            # on the deranged-label prompts of the first `aie_seeds` seeds (the same demonstrations).
            fvc = cfg.extraction.function_vector
            k = fvc.aie_seeds
            st.seed_head_means = np.stack([s.head_means for s in seeds])
            neg = [p for s in seeds[:k] for p in s.permuted_prompts]
            base = np.concatenate([effect_values(s.permuted_result, fvc.aie_metric) for s in seeds[:k]])
            with self.prof.section("head_effects", name):
                st.head_effects = head_effects(self.backend, neg, base, st.seed_head_means.mean(axis=0), metric=fvc.aie_metric)
            st.head_effects_n_prompts = len(neg)
            st.aie_prompts, st.aie_baseline = neg, base
            st.aie_positive_mean = float(np.mean(np.concatenate([effect_values(s.positive_result, fvc.aie_metric) for s in seeds[:k]])))
            top = np.dstack(np.unravel_index(np.argsort(-st.head_effects, axis=None)[:5], st.head_effects.shape))[0]
            log.info("[%s] head effects (%s) over %d deranged prompts in %.0fs: baseline mean %.4f, max effect %.4f, "
                     "top (layer, head) %s", name, fvc.aie_metric, len(neg), self.prof.sections[f"head_effects:{name}"]["seconds"],
                     float(base.mean()), float(st.head_effects.max()), [(int(l), int(h)) for l, h in top])
            ext["function_vector"] = {"aie_seeds": k, "aie_n_prompts": len(neg), "aie_metric": fvc.aie_metric,
                                      "aie_baseline_mean": float(base.mean()),
                                      "aie_baseline_first_token_probability_mean":
                                      float(np.mean(np.concatenate([np.exp(s.permuted_result.first_token_logprob) for s in seeds[:k]])))}
            if n_dv > 0:
                with self.prof.section("demo_variation", name):
                    st.permuted_head_means = permuted_head_means(self.backend, prompt_cfg, splits.extraction, cfg.seed, name, n_dv)
                ext["demo_variation"] = {"n": n_dv, "construction": "function vector of deranged-label prompts (D21)"}
        elif learned_mode:
            # D31: one vector per candidate layer, fitted on the extraction pool's zero-shot prompts at the median
            # residual norm of the layer; the demo-variation null (vectors fitted to permuted targets) is built at
            # the selected layer when the controls are (phase B), so it costs one layer, not four.
            lvc = cfg.extraction.learned_vector
            with self.prof.section("learned_vector", name):
                base_ext = self.backend.run([zero_shot_prompt(prompt_cfg, x) for x in splits.extraction], capture=True)
                st.learned_radii = median_layer_norms(base_ext, self.layers)
                base_loss = float(-np.mean(base_ext.logprob_sum))
                st.directions, st.learned_fits = learned_directions(self.backend, prompt_cfg, splits.extraction, self.layers,
                                                                    st.learned_radii, cfg.seed, name, lvc, cfg.extraction.n_seeds)
            ext["learned_vector"] = {
                "n_seeds": cfg.extraction.n_seeds, "n_steps": lvc.n_steps, "lr_fraction": lvc.lr_fraction,
                "n_prompts": len(splits.extraction), "base_loss": base_loss, "layers": fit_summary(st.learned_fits),
                "stability": {str(l): st.directions[l].stability for l in self.layers},
                "cos_with_pca": {str(l): float(st.directions[l].direction @ st.pca_directions[l].direction) for l in self.layers},
                "demo_variation": {"n": n_dv, "construction": "vectors fitted to permuted targets at the selected layer (D31)"},
            }
            log.info("[%s] learned vector (%d steps, %d seeds, %.0fs): loss %.3f -> %s per candidate layer; stability %s; "
                     "cos with PC1 %s", name, lvc.n_steps, cfg.extraction.n_seeds, self.prof.sections[f"learned_vector:{name}"]["seconds"],
                     base_loss, {l: round(float(np.mean([f.losses[-1] for f in fs])), 3) for l, fs in st.learned_fits.items()},
                     {l: round(st.directions[l].stability, 3) for l in self.layers},
                     {l: round(float(st.directions[l].direction @ st.pca_directions[l].direction), 3) for l in self.layers})
        elif n_dv > 0:
            with self.prof.section("demo_variation", name):
                dv = extract_demo_variation(self.backend, prompt_cfg, splits.extraction, cfg.seed, name, n_dv)
            st.demo_variation = demo_variation_directions(dv, self.layers, cfg.extraction)
            ext["demo_variation"] = {
                "n": n_dv,
                "cos_with_control": {str(l): [float(abs(u @ st.directions[l].direction)) for u in us]
                                     for l, us in st.demo_variation.items()},
            }
        write_json(out / "extraction.json", ext)
        self._write_directions(st)
        pca_stability = {str(l): st.pca_directions[l].stability for l in self.layers}
        log.info("[%s] PC1 stability per candidate layer: %s", name, {l: round(st.pca_directions[l].stability, 3) for l in self.layers})
        if fv_mode:
            # The control's stability gate is applied to the function vector once the heads are chosen.
            q["pca_stability"] = pca_stability
            st.ready = True
            return st
        if learned_mode:
            # D31 (amended): the learned vector is one deterministic fit; its cross-seed stability says how unique
            # the solution is and is reported, not gated (the held-out gates and the permuted-target null judge it).
            q["stability"] = {str(l): st.directions[l].stability for l in self.layers}
            q["pca_stability"] = pca_stability
            st.stable_layers = list(self.layers)
            q["stable_layers"] = st.stable_layers
            st.ready = True
            return st
        st.stable_layers = [l for l in self.layers if st.directions[l].stability >= cfg.qualification.min_stability]
        q["stability"] = {str(l): st.directions[l].stability for l in self.layers}  # the control's own stability
        q["pca_stability"] = pca_stability
        q["stable_layers"] = st.stable_layers
        q["gates"]["stability"] = bool(st.stable_layers)
        if not st.stable_layers:
            self.rejections.add("stability", name, "no_stable_candidate_layer",
                                {"stability": q["stability"], "min_stability": cfg.qualification.min_stability})
            if cfg.qualification.enforce:
                return st
            st.stable_layers = list(self.layers)
        st.ready = True
        return st

    def _neutral_probe(self, layer: int, u: np.ndarray, alpha: float) -> np.ndarray:
        """Per-prompt KL from the unsteered distribution on the neutral prose set with ``alpha * u`` added at ``layer``."""
        r = self.backend.run(self.neutral, interventions=[Intervention(layer, u, alpha)], reference_logprobs=self.neutral_ref)
        assert r.kl_from_reference is not None
        return r.kl_from_reference

    def _target_tokenisation(self, items: list[Any], prompt_cfg: PromptConfig) -> dict[str, Any]:
        """How the kept items' formatted targets tokenise (counts, what the first token is, how many tokens
        the task's target scoring counts)."""
        n_tokens = []
        n_scored = []
        first_space = 0
        examples: list[list[str]] = []
        for it in items:
            pieces = self.backend.target_tokens(prompt_cfg.target_template.format(output=it.output))
            n_tokens.append(len(pieces))
            n_scored.append(self.backend.scored_token_count(prompt_cfg.target_template.format(output=it.output),
                                                            score_reference(prompt_cfg, it)))
            first_space += pieces[0].strip() == ""
            if len(examples) < 3:
                examples.append(pieces)
        n = max(1, len(items))
        return {
            "n_tokens_mean": float(np.mean(n_tokens)) if n_tokens else 0.0,
            "n_scored_mean": float(np.mean(n_scored)) if n_scored else 0.0,
            "n_tokens_histogram": {str(k): int(v) for k, v in sorted(zip(*np.unique(n_tokens, return_counts=True)))} if n_tokens else {},
            "first_token_is_space_fraction": first_space / n,
            "examples": examples,
        }

    def _write_directions(self, st: TaskState) -> None:
        arrays: dict[str, np.ndarray] = {
            "layers": np.array(self.layers),
            "pooled": np.stack([st.pca_directions[l].direction for l in self.layers]),
            "per_seed": np.stack([st.pca_directions[l].seed_directions for l in self.layers]),
            "all_layers": st.all_layer_directions,
        }
        if st.learned_fits:
            arrays.update({
                "learned": np.stack([st.directions[l].direction for l in self.layers]),
                "learned_per_seed": np.stack([st.directions[l].seed_directions for l in self.layers]),
                "learned_radii": np.array([st.learned_radii[l] for l in self.layers]),
            })
        if st.fv is not None:
            arrays.update({
                "fv": st.fv.vector,
                "fv_direction": st.fv.direction,
                "fv_per_seed": st.fv.seed_directions,
                "fv_heads": np.array([[h["layer"], h["head"]] for h in st.fv.heads], dtype=np.int64),
                "head_effects": st.head_effects,
            })
        np.savez_compressed(self._task_dir(st.name) / "directions.npz", **arrays)

    # ---- phase A2: function vectors (D21) ------------------------------ #

    def _build_function_vectors(self, states: dict[str, TaskState]) -> None:
        """Rank heads across the tasks that reached extraction, build each task's function vector and
        make it the control at every candidate layer; apply the stability gate to it."""
        cfg = self.cfg
        fvc = cfg.extraction.function_vector
        effects = {n: st.head_effects for n, st in states.items() if st.head_effects is not None}
        n_total = 0 if not effects else int(np.prod(next(iter(effects.values())).shape))
        candidates = [k for k in fvc.head_count_candidates if k <= n_total] or ([n_total] if n_total else [])
        k_max = fvc.n_heads if fvc.n_heads is not None else (max(candidates) if candidates else 1)
        ranking = select_heads(effects, k_max, fvc.head_selection)  # the top-k_max heads per task, in rank order
        head_count: dict[str, Any] = {"fixed": fvc.n_heads is not None, "candidates": candidates}
        joint_by_task: dict[str, dict[int, np.ndarray]] = {}
        if fvc.n_heads is None and ranking:
            # D26: the joint patched effect of the top-k sets, per task; the count is chosen on the prompts
            # pooled over tasks (universal ranking) or per task (per_task ranking).
            with self.prof.section("head_count"):
                for name, sel in ranking.items():
                    st = states[name]
                    assert st.seed_head_means is not None and st.aie_baseline is not None
                    joint_by_task[name] = {k: joint_head_effect(self.backend, st.aie_prompts, st.aie_baseline, st.seed_head_means.mean(axis=0),
                                                                [(l, h) for l, h, _, _ in sel[:k]], fvc.aie_metric) for k in candidates}
            rng = rng_for(cfg.seed, "head_count")
            if fvc.head_selection == "universal":
                pooled = {k: np.concatenate([joint_by_task[n][k] for n in sorted(joint_by_task)]) for k in candidates}
                choice = choose_head_count(pooled, rng, alpha=fvc.head_support_alpha, n_boot=cfg.calibration.n_boot,
                                           min_gain=fvc.head_count_min_gain)
                chosen = {n: choice["chosen"] for n in ranking}
                head_count.update(choice)
            else:
                per_task = {n: choose_head_count(joint_by_task[n], rng, alpha=fvc.head_support_alpha, n_boot=cfg.calibration.n_boot,
                                                 min_gain=fvc.head_count_min_gain)
                            for n in sorted(joint_by_task)}
                chosen = {n: c["chosen"] for n, c in per_task.items()}
                head_count["per_task"] = per_task
            head_count["mean_effect_by_k_per_task"] = {n: {str(k): float(np.mean(v)) for k, v in d.items()} for n, d in joint_by_task.items()}
            self.log.info("head count (D26): pooled joint effect by k %s; chosen k = %s",
                          {k: round(v, 4) for k, v in head_count.get("mean_effect_by_k", {}).items()}, sorted(set(chosen.values())))
        else:
            chosen = {n: k_max for n in ranking}
        selection = {n: sel[: chosen[n]] for n, sel in ranking.items()}
        self.metadata["function_vector"] = {
            "head_selection": fvc.head_selection,
            "n_heads": None if not chosen else (next(iter(chosen.values())) if len(set(chosen.values())) == 1 else chosen),
            "head_count": {k: v for k, v in head_count.items() if k != "mean_effect_by_k_per_task"},
            "tasks_ranked": sorted(effects),
            "universal_heads": None if fvc.head_selection != "universal" or not selection else
            [[l, h, r] for l, h, r, _ in next(iter(selection.values()))],
        }
        if fvc.head_selection == "universal" and selection:
            self.log.info("universal function-vector heads (layer, head, mean effect over %d tasks): %s", len(effects),
                          [(l, h, round(r, 4)) for l, h, r, _ in next(iter(selection.values()))])
        for name, sel in selection.items():
            st = states[name]
            q = st.qualification
            assert st.seed_head_means is not None and st.head_effects is not None and st.aie_baseline is not None
            # D26: does this set of heads carry this task? (its joint effect, against random sets of the same size)
            with self.prof.section("head_support", name):
                support = head_support_test(
                    self.backend, st.aie_prompts, st.aie_baseline, st.seed_head_means.mean(axis=0), [(l, h) for l, h, _, _ in sel],
                    rng_for(cfg.seed, "head_support", name), n_null=fvc.head_support_null, metric=fvc.aie_metric,
                    n_boot=cfg.calibration.n_boot, alpha=fvc.head_support_alpha,
                    real_effect=joint_by_task.get(name, {}).get(len(sel)),
                    positive_mean=st.aie_positive_mean, min_restored=fvc.head_support_min_restored)
            q["head_support"] = support
            q["gates"]["head_support"] = support["supported"]
            self.log.info("[%s] head support: %d heads move the deranged prompts by %+.4f (p=%.3f), restoring %s of the gap to the "
                          "positive prompts (%.3f -> %.3f); random sets %+.4f; excess p=%.3f -> %s",
                          name, len(sel), support["mean_effect"], support["test"]["p_value"],
                          "n/a" if support["restored_fraction"] is None else f"{support['restored_fraction']:.1%}",
                          support["baseline_mean"], support["positive_mean"] or float("nan"), float(np.mean(support["null_mean_effects"])),
                          support["excess_test"]["p_value"], "supported" if support["supported"] else "NOT supported")
            st.fv = build_function_vector(self.backend, st.seed_head_means, sel)
            heads = [(h["layer"], h["head"]) for h in st.fv.heads]
            st.directions = {
                l: LayerDirection(
                    layer=l, direction=st.fv.direction, seed_directions=st.fv.seed_directions,
                    explained_variance_ratio=[], cos_with_mean=[], stability=st.fv.stability,
                    pooled_explained_variance_ratio=float("nan"), pooled_cos_with_mean=float("nan"),
                    mean_difference_norm=st.fv.natural_norm, cos_pooled_vs_seeds=st.fv.cos_pooled_vs_seeds,
                    kind="function_vector",
                )
                for l in self.layers
            }
            perm = [normalize(compose(self.backend, m, heads)) for m in st.permuted_head_means]
            st.demo_variation = {l: perm for l in self.layers}
            cos_pca = {str(l): float(st.fv.direction @ st.pca_directions[l].direction) for l in self.layers}
            assert st.all_layer_directions is not None
            cos_pca_all = [float(st.fv.direction @ u) for u in normalize(st.all_layer_directions, axis=1)]
            info = {
                "head_selection": fvc.head_selection,
                "n_heads": len(sel),
                "head_count": {**{k: v for k, v in head_count.items() if k != "mean_effect_by_k_per_task"},
                               "mean_effect_by_k": head_count.get("mean_effect_by_k_per_task", {}).get(name)},
                "head_support": support,
                **st.fv.summary(),
                "head_effects": st.head_effects.tolist(),
                "head_effects_n_prompts": st.head_effects_n_prompts,
                "cos_with_pca": cos_pca,  # descriptive: the function vector against PC1 at each candidate layer
                "cos_with_pca_all_layers": cos_pca_all,
                "demo_variation_cos_with_control": [float(abs(u @ st.fv.direction)) for u in perm],
            }
            write_json(self._task_dir(name) / "function_vector.json", info)
            self._write_directions(st)
            q["function_vector"] = {k: info[k] for k in ("natural_norm", "heads", "stability", "cos_with_pca")}
            stable = st.fv.stability >= cfg.qualification.min_stability
            st.stable_layers = list(self.layers) if stable else []
            q["stability"] = {str(l): st.fv.stability for l in self.layers}
            q["stable_layers"] = st.stable_layers
            q["gates"]["stability"] = stable
            self.log.info("[%s] function vector: norm %.3f, cross-seed stability %.3f, cos with PC1 per candidate layer %s",
                          name, st.fv.natural_norm, st.fv.stability, {l: round(c, 3) for l, c in cos_pca.items()})
            if not stable:
                self.rejections.add("stability", name, "function_vector_unstable",
                                    {"stability": st.fv.stability, "min_stability": cfg.qualification.min_stability})
                if cfg.qualification.enforce:
                    st.ready = False
                else:
                    st.stable_layers = list(self.layers)
            if not support["supported"]:
                self.rejections.add("head_support", name, "no_head_support",
                                    {"mean_effect": support["mean_effect"], "p": support["test"]["p_value"],
                                     "excess_p": support["excess_test"]["p_value"], "n_heads": len(sel),
                                     "restored_fraction": support["restored_fraction"], "min_restored": support["min_restored"]})
                if cfg.qualification.enforce:
                    st.ready = False

    # ---- phase B ------------------------------------------------------ #

    def _build_controls(self, st: TaskState, states: dict[str, TaskState], layer: int, v: np.ndarray
                        ) -> list[tuple[str, str, np.ndarray]]:
        """Unit-norm control directions of every configured kind (see docs/DECISIONS.md D15)."""
        cfg = self.cfg
        counts = {k: int(n) for k, n in cfg.evaluation.controls.items()}
        out: list[tuple[str, str, np.ndarray]] = []
        rng = rng_for(cfg.seed, "random_controls", st.name)
        for i in range(counts.get("isotropic", 0)):
            out.append(("isotropic", f"isotropic_{i}", random_unit_vector(rng, v.shape[0])))
        for i in range(counts.get("orthogonal", 0)):
            out.append(("orthogonal", f"orthogonal_{i}", random_orthogonal_unit_vector(rng, v)))
        if counts.get("covariance", 0):
            assert st.base is not None and st.base.residuals is not None
            crng = rng_for(cfg.seed, "covariance_controls", st.name)
            H = st.base.residuals[layer]
            for i in range(counts["covariance"]):
                out.append(("covariance", f"covariance_{i}", covariance_matched_unit_vector(crng, H)))
        if counts.get("other_task", 0):
            others = sorted(n for n, s2 in states.items() if n != st.name and layer in s2.directions)
            for n in others[: counts["other_task"]]:
                out.append(("other_task", f"other_task:{n}", states[n].directions[layer].direction))
        if counts.get("demo_variation", 0):
            if st.learned_fits and layer not in st.demo_variation:
                assert st.prompt_cfg is not None
                with self.prof.section("demo_variation", st.name):
                    st.demo_variation[layer] = permuted_target_vectors(
                        self.backend, st.prompt_cfg, st.splits.extraction, layer, st.learned_radii[layer], cfg.seed, st.name,
                        cfg.extraction.learned_vector, counts["demo_variation"])
            for i, u in enumerate(st.demo_variation.get(layer, [])[: counts["demo_variation"]]):
                out.append(("demo_variation", f"demo_variation_{i}", u))
        if counts.get("common", 0):
            c, others = self._common_direction(st, states, layer)
            if c is not None:
                out.append(("common", "common", c))
        return out

    def _common_direction(self, st: TaskState, states: dict[str, TaskState], layer: int) -> tuple[np.ndarray | None, list[str]]:
        """The leave-one-out common direction at ``layer`` (D28): the normalised mean of the other ready tasks'
        unit control directions there. ``(None, [])`` when no other task reached phase B."""
        others = sorted(n for n, s2 in states.items() if n != st.name and s2.ready and layer in s2.directions)
        if not others:
            return None, []
        mean = np.mean([normalize(states[n].directions[layer].direction) for n in others], axis=0)
        if not np.linalg.norm(mean) > 0:
            return None, others
        return normalize(mean), others

    def _measure_task(self, st: TaskState, states: dict[str, TaskState]) -> None:
        cfg = self.cfg
        log = self.log
        name = st.name
        out = self._task_dir(name)
        q = st.qualification
        splits = st.splits
        zs_prompts = st.eval_prompts
        assert st.base is not None and st.base.residuals is not None

        # 4. calibration -------------------------------------------------
        with self.prof.section("calibration", name):
            cal = calibrate(self.backend, st.prompt_cfg, cfg.calibration, cfg.evaluation,
                            {l: st.directions[l] for l in st.stable_layers}, splits.calibration, cfg.seed, name,
                            neutral_probe=self._neutral_probe if self.neutral else None)
        st.calibration = cal
        write_json(out / "calibration.json", cal.as_dict())
        q["gates"]["calibration"] = cal.selected is not None
        if cal.selected is None:
            self.rejections.add("calibration", name, "no_reliable_intervention", {"reason": cal.reason})
            if cfg.qualification.enforce:
                return
            best = max(cal.grid, key=lambda g: g.test.mean_diff)
            st.selection = best.selection()
            st.fallback_selection = True
            log.warning("[%s] gates not enforced: falling back to best grid point layer %d rho %.3g", name, best.layer, best.rho)
        else:
            st.selection = cal.selected
        q["selection"] = {**st.selection.__dict__, "fallback": st.fallback_selection, "strength_unit": cal.strength_unit,
                          "reference_rho": cal.reference_rho, "layer_rule": cal.layer_rule,
                          "weakest_rho": None if cal.weakest is None else cal.weakest.rho,
                          "strongest_rho": None if cal.strongest is None else cal.strongest.rho}
        log.info("[%s] selected layer %d, rho %.3g in %s units (alpha %.3g, %.3g x median residual norm; reliable rho %s-%s)%s",
                 name, st.selection.layer, st.selection.rho, cal.strength_unit, st.selection.alpha,
                 st.selection.rho_layer_norm or float("nan"), q["selection"]["weakest_rho"], q["selection"]["strongest_rho"],
                 " [FALLBACK]" if st.fallback_selection else "")
        if st.fv is not None:
            # The strength the canonical (unscaled) injection would have, in units of the median residual norm,
            # and the selected strength in units of the natural norm (1 = the canonical injection).
            q["function_vector"]["natural_rho"] = {str(l): st.fv.natural_norm / n for l, n in cal.layer_norms.items()}
            q["function_vector"]["alpha_over_natural_norm"] = st.selection.alpha / st.fv.natural_norm
            q["function_vector"]["cos_with_pca_at_selected_layer"] = q["function_vector"]["cos_with_pca"][str(st.selection.layer)]
            log.info("[%s] function vector: natural rho at the selected layer %.3g; selected alpha / natural norm %.3g; "
                     "cos with PC1 there %.3f", name, q["function_vector"]["natural_rho"][str(st.selection.layer)],
                     q["function_vector"]["alpha_over_natural_norm"], q["function_vector"]["cos_with_pca_at_selected_layer"])

        if st.learned_fits:
            ld = st.directions[st.selection.layer]
            q["learned_vector"] = {
                "natural_norm": ld.mean_difference_norm,
                "natural_rho": {str(l): st.directions[l].mean_difference_norm / n for l, n in cal.layer_norms.items()},
                "alpha_over_natural_norm": st.selection.alpha / ld.mean_difference_norm,
                "cos_with_pca_at_selected_layer": float(ld.direction @ st.pca_directions[st.selection.layer].direction),
                "stability": ld.stability,
                "final_loss": [f.losses[-1] for f in st.learned_fits[st.selection.layer]],
            }
            log.info("[%s] learned vector: selected alpha / fitted norm %.3g; cos with PC1 there %.3f; stability %.3f",
                     name, q["learned_vector"]["alpha_over_natural_norm"], q["learned_vector"]["cos_with_pca_at_selected_layer"],
                     ld.stability)

        # 5. held-out steering vs matched controls of every kind ----------
        sel = st.selection
        v = st.directions[sel.layer].direction
        do_layerwise = self.stop_after in ("layerwise", "exploratory", "figures")
        with self.prof.section("controls_setup", name):
            # the damage check's reference (D24): the baseline's next-token distribution at the query token
            ref = self.backend.reference_tensor(st.base.query_logprobs)
            st.steered = self.backend.run(zs_prompts, interventions=[Intervention(sel.layer, v, sel.alpha)], capture=True,
                                          reference_logprobs=ref)
            brng = rng_for(cfg.seed, "bootstrap", name)
            krng = rng_for(cfg.seed, "damage_bootstrap", name)
            steered_nkl = self._neutral_probe(sel.layer, v, sel.alpha) if self.neutral else None
            prng = rng_for(cfg.seed, "profile_bootstrap", name)
            if do_layerwise:
                # Direction-specific readouts (D17): the baseline target-log-prob gradient at every
                # read point, computed once per task and shared by the real and every control profile.
                st.gradients = self.backend.gradients(zs_prompts)
            readout_kw = {"task_directions": st.all_layer_directions, "gradients": st.gradients}
            test = paired_bootstrap_test(st.steered.logprob_per_token, st.base.logprob_per_token, brng, n_boot=cfg.calibration.n_boot)
        if cfg.determinism_check and "determinism_check" not in self.metadata:
            with self.prof.section("determinism_check", name):
                self._determinism_check(st, v, do_layerwise, readout_kw, ref)
        for kind, label, u in self._build_controls(st, states, sel.layer, v):
            with self.prof.section("controls_forward", name):
                r = self.backend.run(zs_prompts, interventions=[Intervention(sel.layer, u, sel.alpha)], capture=do_layerwise,
                                     reference_logprobs=ref)
                rt = paired_bootstrap_test(r.logprob_per_token, st.base.logprob_per_token, brng, n_boot=cfg.calibration.n_boot)
            prof = None
            if do_layerwise:
                assert r.residuals is not None
                with self.prof.section("controls_profiles", name):
                    prof = compute_profile(st.base.residuals, r.residuals, u, sel.layer, cfg.evaluation, prng, with_ci=False,
                                           **readout_kw)
            metrics = r.metrics_dict()
            nkl = None
            if self.neutral:
                with self.prof.section("controls_neutral", name):
                    nkl = self._neutral_probe(sel.layer, u, sel.alpha)
                metrics["neutral_kl_mean"], metrics["neutral_kl_median"] = float(np.mean(nkl)), float(np.median(nkl))
            st.controls.append(ControlResult(kind, label, rt.mean_diff, rt.p_value, metrics, prof,
                                             r.logprob_per_token - st.base.logprob_per_token, r.kl_from_reference,
                                             r.collateral_kl, nkl))
        sec = self.prof.sections
        log.info("[%s] %d controls: steered run + gradients %.0fs, control forwards %.0fs, control profiles %.0fs", name,
                 len(st.controls), sec[f"controls_setup:{name}"]["seconds"], sec.get(f"controls_forward:{name}", {}).get("seconds", 0.0),
                 sec.get(f"controls_profiles:{name}", {}).get("seconds", 0.0))
        gate_kinds = list(cfg.evaluation.gate_kinds)  # primary null of the layerwise metrics
        behav_gate_kinds = list(cfg.qualification.gate_control_kinds)  # behavioural gate (D18)
        real_diff = st.steered.logprob_per_token - st.base.logprob_per_token
        gate_null = np.array([c.mean_diff for c in st.controls if c.kind in behav_gate_kinds])
        null = compare_to_null(test.mean_diff, gate_null)
        excess = paired_excess_test(real_diff, np.stack([c.per_example_diff for c in st.controls if c.kind in behav_gate_kinds]),
                                    brng, n_boot=cfg.calibration.n_boot)
        by_kind_behav = {}
        by_kind_excess = {}
        for kind in sorted({c.kind for c in st.controls}):
            vals = np.array([c.mean_diff for c in st.controls if c.kind == kind])
            by_kind_behav[kind] = compare_to_null(test.mean_diff, vals).__dict__
            mat = np.stack([c.per_example_diff for c in st.controls if c.kind == kind])
            by_kind_excess[kind] = paired_excess_test(real_diff, mat, brng, n_boot=cfg.calibration.n_boot).__dict__
        # Damage (D24): the real direction's disturbance of the next-token distribution against the controls'
        # at the same norm (positive excess = more damage than random); its own bootstrap stream.
        real_kl, real_ckl = st.steered.kl_from_reference, st.steered.collateral_kl
        gate_ctl = [c for c in st.controls if c.kind in behav_gate_kinds]
        damage = {
            "steered": {k: st.steered.metrics_dict()[k] for k in ("kl_mean", "kl_median", "collateral_kl_mean", "collateral_kl_median",
                                                                  "argmax_change_rate")},
            "gate_excess_test": paired_excess_test(real_kl, np.stack([c.per_example_kl for c in gate_ctl]),
                                                   krng, n_boot=cfg.calibration.n_boot).__dict__,
            "collateral_gate_excess_test": paired_excess_test(real_ckl, np.stack([c.per_example_collateral_kl for c in gate_ctl]),
                                                              krng, n_boot=cfg.calibration.n_boot).__dict__,
            "by_kind": {kind: {"kl_mean": float(np.mean([c.metrics["kl_mean"] for c in st.controls if c.kind == kind])),
                               "collateral_kl_mean": float(np.mean([c.metrics["collateral_kl_mean"] for c in st.controls if c.kind == kind])),
                               "excess_test": paired_excess_test(real_kl, np.stack([c.per_example_kl for c in st.controls if c.kind == kind]),
                                                                 krng, n_boot=cfg.calibration.n_boot).__dict__,
                               "collateral_excess_test": paired_excess_test(
                                   real_ckl, np.stack([c.per_example_collateral_kl for c in st.controls if c.kind == kind]),
                                   krng, n_boot=cfg.calibration.n_boot).__dict__}
                        for kind in sorted({c.kind for c in st.controls})},
        }
        if steered_nkl is not None:
            # damage on neutral prose (D24): the direction at the selected layer and norm, on task-free sentences
            damage["neutral"] = {
                "n_prompts": int(steered_nkl.shape[0]),
                "steered": {"kl_mean": float(np.mean(steered_nkl)), "kl_median": float(np.median(steered_nkl))},
                "gate_excess_test": paired_excess_test(steered_nkl, np.stack([c.per_example_neutral_kl for c in gate_ctl]),
                                                       krng, n_boot=cfg.calibration.n_boot).__dict__,
                "by_kind": {kind: {"kl_mean": float(np.mean([c.metrics["neutral_kl_mean"] for c in st.controls if c.kind == kind])),
                                   "excess_test": paired_excess_test(
                                       steered_nkl, np.stack([c.per_example_neutral_kl for c in st.controls if c.kind == kind]),
                                       krng, n_boot=cfg.calibration.n_boot).__dict__}
                            for kind in sorted({c.kind for c in st.controls})},
                "per_prompt": {"steered": steered_nkl.tolist(), "controls": {c.label: c.per_example_neutral_kl.tolist() for c in st.controls}},
            }
        ev = {
            "selection": q["selection"],
            "control": cfg.extraction.control,
            "function_vector": q.get("function_vector"),
            "baseline": st.base.metrics_dict(),
            "steered": st.steered.metrics_dict(),
            "steering_test": test.__dict__,
            "gate_kinds": gate_kinds,
            "gate_test": cfg.qualification.gate_test,
            "gate_control_kinds": behav_gate_kinds,
            "controls": [{"kind": c.kind, "label": c.label, "mean_diff": c.mean_diff, "p_value": c.p_value,
                          "metrics": c.metrics} for c in st.controls],
            "random_comparison": null.__dict__,
            "excess_test": excess.__dict__,
            "by_kind_comparison": by_kind_behav,
            "by_kind_excess": by_kind_excess,
            "damage": damage,
            "per_example": {
                "baseline_logprob_per_token": st.base.logprob_per_token.tolist(),
                "steered_logprob_per_token": st.steered.logprob_per_token.tolist(),
                "baseline_exact_match": st.base.exact_match.tolist(),
                "steered_exact_match": st.steered.exact_match.tolist(),
                "controls_logprob_per_token_diff": {c.label: c.per_example_diff.tolist() for c in st.controls},
                "steered_kl": real_kl.tolist(),
                "steered_collateral_kl": real_ckl.tolist(),
                "controls_kl": {c.label: c.per_example_kl.tolist() for c in st.controls},
                "controls_collateral_kl": {c.label: c.per_example_collateral_kl.tolist() for c in st.controls},
            },
        }
        write_json(out / "evaluation.json", ev)
        q["steering"] = {"test": test.__dict__, "random_comparison": null.__dict__, "excess_test": excess.__dict__,
                         "by_kind": by_kind_behav, "by_kind_excess": by_kind_excess}
        q["damage"] = {**damage["steered"], "gate_excess_mean": damage["gate_excess_test"]["excess_mean"],
                       "gate_excess_p": damage["gate_excess_test"]["p_value"],
                       "collateral_gate_excess_mean": damage["collateral_gate_excess_test"]["excess_mean"],
                       "collateral_gate_excess_p": damage["collateral_gate_excess_test"]["p_value"]}
        if "neutral" in damage:
            q["damage"].update({"neutral_kl_mean": damage["neutral"]["steered"]["kl_mean"],
                                "neutral_gate_excess_mean": damage["neutral"]["gate_excess_test"]["excess_mean"],
                                "neutral_gate_excess_p": damage["neutral"]["gate_excess_test"]["p_value"]})
        log.info("[%s] damage: KL(base||steered) at the query token mean %.3f, collateral (answer token removed) %.3f (argmax "
                 "changed for %.0f%%); on neutral prose %s; neutral excess over the gate controls %s", name, q["damage"]["kl_mean"],
                 q["damage"]["collateral_kl_mean"], 100 * q["damage"]["argmax_change_rate"],
                 "n/a" if "neutral_kl_mean" not in q["damage"] else "%.3f" % q["damage"]["neutral_kl_mean"],
                 "n/a" if "neutral_kl_mean" not in q["damage"] else
                 "%+.3f (p=%.3f)" % (q["damage"]["neutral_gate_excess_mean"], q["damage"]["neutral_gate_excess_p"]))
        q["gates"]["steering"] = bool(test.p_value <= cfg.qualification.steering_alpha and test.mean_diff > 0)
        if cfg.qualification.gate_test == "paired_excess":
            q["gates"]["random_controls"] = bool(excess.p_value <= cfg.qualification.random_control_max_p)
        else:
            q["gates"]["random_controls"] = bool(null.p_upper <= cfg.qualification.random_control_max_p)
        log.info("[%s] held-out steering: mean d(lp/tok)=%.4f p=%.4f; vs %d gate controls (%s): excess %.4f p=%.4f "
                 "[rank z=%.2f p=%.3f]; excess (mean, p) by kind %s",
                 name, test.mean_diff, test.p_value, excess.n_controls, "+".join(behav_gate_kinds), excess.excess_mean,
                 excess.p_value, null.z, null.p_upper,
                 {k: (round(c["excess_mean"], 4), round(c["p_value"], 4)) for k, c in by_kind_excess.items()})
        if not q["gates"]["steering"]:
            self.rejections.add("steering", name, "no_reliable_heldout_steering", {"test": test.__dict__})
        if not q["gates"]["random_controls"]:
            self.rejections.add("random_controls", name, "steering_not_above_random_controls",
                                {"gate_test": cfg.qualification.gate_test, "excess_test": excess.__dict__, "comparison": null.__dict__})
        st.qualified = all(q["gates"].values())
        if not st.qualified and cfg.qualification.enforce:
            return
        if not do_layerwise:
            return

        # 6. layerwise ---------------------------------------------------
        assert st.steered.residuals is not None
        with self.prof.section("layerwise", name):
            st.profile = compute_profile(st.base.residuals, st.steered.residuals, v, sel.layer, cfg.evaluation, prng, **readout_kw)
            by_kind: dict[str, list[LayerwiseProfile]] = {}
            for c in st.controls:
                if c.profile is not None:
                    by_kind.setdefault(c.kind, []).append(c.profile)
            st.comparison = compare_by_kind(st.profile, by_kind, gate_kinds)
            st.signature = profile_signature(st.profile, st.comparison["primary"], cfg.analysis,
                                             by_kind=st.comparison["by_kind"])
            pooled = [p for k in gate_kinds for p in by_kind.get(k, [])]
            write_json(out / "layerwise.json", {
                "real": st.profile.as_dict(),
                "gate_kinds": gate_kinds,
                "controls": [{"kind": c.kind, "label": c.label, "curves": metric_curves(c.profile)}
                             for c in st.controls if c.profile is not None],
                "null_summaries": {"primary": null_summary(pooled), **{k: null_summary(ps) for k, ps in by_kind.items()}},
                "comparison": st.comparison,
                "signature": st.signature,
            })
            np.savez_compressed(out / "layerwise_arrays.npz", **profile_arrays(st.profile))
        zk = {k: round(d.get("new_subspace_uncentered_z_mean") or float("nan"), 2) for k, d in st.signature["by_kind"].items()}
        log.info("[%s] profile: cum log G %.3f, final alignment %.3f, d_eff %.1f->%.1f, labels %s; N_unc z by kind %s", name,
                 st.signature["cumulative_log_gain"], st.signature["alignment_final"] or float("nan"),
                 st.signature["d_eff_first"] or float("nan"), st.signature["d_eff_final"] or float("nan"),
                 st.signature["labels"], zk)
        rz = {k: (round(d.get("task_alignment_z_mean") or float("nan"), 2), round(d.get("gradient_alignment_z_mean") or float("nan"), 2))
              for k, d in st.signature["by_kind"].items()}
        log.info("[%s] readouts: task-alignment downstream %.3f (final %.3f); gradient-alignment at l* %.3f, downstream %.3f; "
                 "(task, gradient) alignment z by kind %s", name,
                 st.signature["task_alignment_downstream_mean"] or float("nan"), st.signature["task_alignment_final"] or float("nan"),
                 st.signature["gradient_alignment_at_intervention"] or float("nan"),
                 st.signature["gradient_alignment_downstream_mean"] or float("nan"), rz)
        if self.stop_after == "layerwise":
            return

        # 6b. depth of commitment (D29): where the injected direction stops being needed as a direction ---
        if cfg.commitment.enabled:
            try:
                with self.prof.section("commitment", name):
                    self._commitment(st, v)
            except Exception as e:
                log.error("[%s] commitment stage failed: %s", name, e)
                self.rejections.add("error", name, f"commitment: {type(e).__name__}: {e}", {"traceback": traceback.format_exc()})

        # 7. exploratory -------------------------------------------------
        try:
            with self.prof.section("exploratory", name):
                self._exploratory(st, v, prng)
        except Exception as e:
            log.error("[%s] exploratory stage failed: %s", name, e)
            self.rejections.add("error", name, f"exploratory: {type(e).__name__}: {e}", {"traceback": traceback.format_exc()})

        # 8. decomposition into the common and the task-specific part (D28) ---
        if st.fv is not None or st.learned_fits:
            try:
                with self.prof.section("decomposition", name):
                    self._decomposition(st, states, readout_kw, ref, gate_kinds)
            except Exception as e:
                log.error("[%s] decomposition stage failed: %s", name, e)
                self.rejections.add("error", name, f"decomposition: {type(e).__name__}: {e}", {"traceback": traceback.format_exc()})

    def _commitment(self, st: TaskState, v: np.ndarray) -> None:
        """Edit the perturbation along the injected direction at every later read point and measure what
        survives, against the same edits along random directions (D29). Writes ``commitment.json``."""
        cfg = self.cfg
        sel = st.selection
        assert sel is not None and st.base is not None and st.steered is not None
        n = cfg.commitment.n_prompts or len(st.eval_prompts)
        prompts = st.eval_prompts[:n]
        if n >= len(st.eval_prompts):
            base, steered = st.base, st.steered
        else:
            # the edits must land on the captured residuals exactly, so the captures are taken on this very prompt list
            base = self.backend.run(prompts, capture=True)
            steered = self.backend.run(prompts, interventions=[Intervention(sel.layer, v, sel.alpha)], capture=True)
        res = depth_of_commitment(self.backend, prompts, base, steered, sel.layer, v, sel.alpha, st.all_layer_directions,
                                  cfg.commitment, rng_for(cfg.seed, "commitment_controls", st.name),
                                  rng_for(cfg.seed, "commitment_bootstrap", st.name))
        res["selection"] = sel.__dict__
        write_json(self._task_dir(st.name) / "commitment.json", res)
        st.qualification["commitment"] = res["summary"]
        s = res["summary"]["injected"]
        fmt = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
        self.log.info("[%s] commitment: handed over (removal leaves >= 50 %% / 90 %% from) read point %s (%s of the downstream "
                      "depth) / %s (%s); carried by the direction alone (>= 50 %%) until %s (%s); retained after removal at "
                      "the end %s, carried alone at the end %s; needed against random edits until %s (commitment layer %s); "
                      "full effect %+.3f on %d prompts",
                      st.name, s.get("handed_over_50"), fmt(s.get("handed_over_50_fraction")), s.get("handed_over_90"),
                      fmt(s.get("handed_over_90_fraction")), s.get("carried_alone_until_50"), fmt(s.get("carried_alone_until_50_fraction")),
                      fmt(s["retained_after_removal_final"]), fmt(s["carried_by_direction_final"]),
                      s["needed_until"], s["commitment_layer"], res["full_effect"], res["n_prompts"])

    def _decomposition(self, st: TaskState, states: dict[str, TaskState], readout_kw: dict[str, Any], reference: Any,
                       gate_kinds: list[str]) -> None:
        """Split the unit function vector into its projection on the leave-one-out common direction and the
        residual, inject each at its natural share of the selected strength (they sum to the real injection),
        and measure both like the real direction: held-out effect, damage, layerwise profile against the same
        control profiles, signature. Writes ``decomposition.json``."""
        cfg = self.cfg
        sel = st.selection
        assert sel is not None and st.base is not None and st.steered is not None
        assert st.base.residuals is not None and st.profile is not None
        control = st.directions[sel.layer].direction  # the unit control direction at l* (function vector or learned vector)
        c, others = self._common_direction(st, states, sel.layer)
        out: dict[str, Any] = {"selection": sel.__dict__, "common_direction": None, "parts": {}}
        if c is None:
            out["reason"] = "no other task reached phase B; no common direction"
            write_json(self._task_dir(st.name) / "decomposition.json", out)
            return
        drng = rng_for(cfg.seed, "decomposition_bootstrap", st.name)
        cos = float(control @ c)
        parts = {"common": cos * c, "residual": control - cos * c}  # unit-vector coordinates; they sum to the FV
        out["common_direction"] = {"cos_with_fv": cos, "other_tasks": others, "n_other_tasks": len(others),
                                   "norm_share_common": abs(cos), "norm_share_residual": float(np.sqrt(max(0.0, 1 - cos**2)))}
        by_kind: dict[str, list[LayerwiseProfile]] = {}
        for ctl in st.controls:
            if ctl.profile is not None:
                by_kind.setdefault(ctl.kind, []).append(ctl.profile)
        gate_mat = np.stack([ctl.per_example_diff for ctl in st.controls if ctl.kind in cfg.qualification.gate_control_kinds])
        real_diff = st.steered.logprob_per_token - st.base.logprob_per_token
        metrics = ("log_gain", "d_eff", "new_subspace", "new_subspace_uncentered", "conversion", "magnitude",
                   "task_alignment", "gradient_alignment", "gradient_projection", "first_order_increment")
        effects: dict[str, float] = {}
        for label, part in parts.items():
            norm = float(np.linalg.norm(part))
            if not norm > 0:
                out["parts"][label] = {"norm_share": 0.0, "skipped": "zero vector"}
                effects[label] = 0.0
                continue
            u = part / norm
            alpha = sel.alpha * norm  # the part's natural size under the selected strength
            r = self.backend.run(st.eval_prompts, interventions=[Intervention(sel.layer, u, alpha)], capture=True,
                                 reference_logprobs=reference)
            assert r.residuals is not None
            diff = r.logprob_per_token - st.base.logprob_per_token
            test = paired_bootstrap_test(r.logprob_per_token, st.base.logprob_per_token, drng, n_boot=cfg.calibration.n_boot)
            excess = paired_excess_test(diff, gate_mat, drng, n_boot=cfg.calibration.n_boot)
            prof = compute_profile(st.base.residuals, r.residuals, u, sel.layer, cfg.evaluation, drng, **readout_kw)
            comparison = compare_by_kind(prof, by_kind, gate_kinds)
            sig = profile_signature(prof, comparison["primary"], cfg.analysis, by_kind=comparison["by_kind"])
            effects[label] = test.mean_diff
            part_metrics = r.metrics_dict()
            if self.neutral:
                nkl = self._neutral_probe(sel.layer, u, alpha)
                part_metrics["neutral_kl_mean"], part_metrics["neutral_kl_median"] = float(np.mean(nkl)), float(np.median(nkl))
            out["parts"][label] = {
                "norm_share": norm, "alpha": alpha, "cos_with_fv": float(u @ control),
                "metrics": part_metrics, "steering_test": test.__dict__, "excess_vs_gate_controls": excess.__dict__,
                "excess_vs_fv": paired_excess_test(diff, real_diff[None, :], drng, n_boot=cfg.calibration.n_boot).__dict__,
                "signature": sig, "curves": metric_curves(prof),
                "rank_correlations_with_fv": {m: spearman(st.profile.metric_curve(m), prof.metric_curve(m)) for m in metrics},
            }
        out["fv"] = {"mean_diff": float(np.mean(real_diff)), "metrics": st.steered.metrics_dict()}
        out["additivity"] = {"fv_minus_sum_of_parts": float(np.mean(real_diff)) - effects.get("common", 0.0) - effects.get("residual", 0.0)}
        write_json(self._task_dir(st.name) / "decomposition.json", out)
        st.qualification["decomposition"] = {
            "cos_with_common": cos, "n_other_tasks": len(others),
            "common_part_mean_diff": effects.get("common"), "residual_part_mean_diff": effects.get("residual"),
            "common_part_labels": out["parts"].get("common", {}).get("signature", {}).get("labels"),
            "residual_part_labels": out["parts"].get("residual", {}).get("signature", {}).get("labels"),
        }
        self.log.info("[%s] decomposition: cos(FV, common of %d tasks) %.3f; held-out d(lp/tok): FV %+.3f, common part %+.3f, "
                      "residual part %+.3f", st.name, len(others), cos, float(np.mean(real_diff)), effects.get("common", 0.0),
                      effects.get("residual", 0.0))

    def _determinism_check(self, st: TaskState, v: np.ndarray, do_layerwise: bool, readout_kw: dict[str, Any],
                           reference: Any = None) -> None:
        """Repeat the steered pass, the gradient pass and one profile of this task; require bit identity (D23)."""
        from .determinism import compare_arrays, determinism_report, forward_arrays, profile_check_arrays

        cfg = self.cfg
        sel = st.selection
        assert sel is not None and st.steered is not None and st.base is not None and st.base.residuals is not None
        again = self.backend.run(st.eval_prompts, interventions=[Intervention(sel.layer, v, sel.alpha)], capture=True,
                                 reference_logprobs=reference)
        forward = compare_arrays(forward_arrays(st.steered), forward_arrays(again))
        gradients = profile = None
        if do_layerwise:
            assert st.gradients is not None and st.steered.residuals is not None
            gradients = compare_arrays({"gradients": st.gradients}, {"gradients": self.backend.gradients(st.eval_prompts)})
            # the profile twice on the same residuals (medians only: the bootstrap draws are seeded separately)
            first = compute_profile(st.base.residuals, st.steered.residuals, v, sel.layer, cfg.evaluation,
                                    np.random.default_rng(0), with_ci=False, **readout_kw)
            second = compute_profile(st.base.residuals, st.steered.residuals, v, sel.layer, cfg.evaluation,
                                     np.random.default_rng(0), with_ci=False, **readout_kw)
            profile = compare_arrays(profile_check_arrays(first), profile_check_arrays(second))
        report = determinism_report(forward, gradients, profile, st.name)
        self.metadata["determinism_check"] = report
        write_json(self.root / "metadata.json", self.metadata)
        if report["identical"]:
            self.log.info("[%s] determinism check: steered pass, gradients and profile repeat bit-identically", st.name)
        else:
            failed = {part: {k: a["max_abs_diff"] for k, a in report[part]["arrays"].items() if not a["identical"]}
                      for part in ("forward", "gradients", "profile") if report[part] is not None and not report[part]["identical"]}
            self.log.error("[%s] determinism check FAILED: %s", st.name, failed)
            self.rejections.add("determinism", st.name, "repeat_not_bit_identical", {"max_abs_diff_by_array": failed})

    def _finish_task(self, st: TaskState) -> None:
        if st.base is not None:
            st.base.query_logprobs = None  # (N, V) per task; not needed after the measurement
        q = st.qualification
        q["qualified"] = bool(st.qualified)
        q["enforce"] = self.cfg.qualification.enforce
        write_json(self._task_dir(st.name) / "qualification.json", q)

    # ------------------------------------------------------------------ #

    def _exploratory(self, st: TaskState, v: np.ndarray, prng: np.random.Generator) -> None:
        cfg = self.cfg
        assert st.calibration is not None and st.selection is not None and st.profile is not None
        assert st.base is not None and st.steered is not None and st.base.residuals is not None
        assert st.steered.residuals is not None and st.comparison is not None
        out = self._task_dir(st.name, exploratory=True)
        sel = st.selection

        if cfg.exploratory.strength_robustness:
            # The profile at the other reliable strengths of the selected layer (weakest/middle/strongest),
            # against the selected one; a strength equal to the selected one is skipped (recorded as null).
            rob: dict[str, Any] = {"selected": sel.__dict__, "profiles": {}, "rank_correlations": {}}
            with self.prof.section("strength_robustness", st.name):
                for label, alt in (("weakest", st.calibration.weakest), ("middle", st.calibration.middle),
                                   ("strongest", st.calibration.strongest)):
                    if alt is None or alt.rho == sel.rho:
                        rob["profiles"][label] = None
                        continue
                    r = self.backend.run(st.eval_prompts, interventions=[Intervention(alt.layer, v, alt.alpha)], capture=True,
                                         reference_logprobs=self.backend.reference_tensor(st.base.query_logprobs))
                    assert r.residuals is not None
                    p = compute_profile(st.base.residuals, r.residuals, v, alt.layer, cfg.evaluation, prng,
                                        task_directions=st.all_layer_directions, gradients=st.gradients)
                    metrics = r.metrics_dict()
                    if self.neutral:
                        nkl = self._neutral_probe(alt.layer, v, alt.alpha)
                        metrics["neutral_kl_mean"], metrics["neutral_kl_median"] = float(np.mean(nkl)), float(np.median(nkl))
                    rob["profiles"][label] = {"selection": alt.__dict__, "metrics": metrics, "profile": p.as_dict()}
                    rob["rank_correlations"][label] = {
                        m: spearman(st.profile.metric_curve(m), p.metric_curve(m))
                        for m in ("log_gain", "d_eff", "new_subspace", "new_subspace_uncentered", "conversion", "magnitude",
                                  "task_alignment", "gradient_alignment")
                    }
            write_json(out / "strength_robustness.json", rob)

        if cfg.exploratory.block_ablation.enabled:
            with self.prof.section("block_ablation", st.name):
                blocks = select_blocks(st.profile, cfg.exploratory.block_ablation, st.comparison["primary"])
                delta = st.steered.residuals.astype(np.float64) - st.base.residuals.astype(np.float64)
                abl = block_ablation(self.backend, st.eval_prompts, st.base, st.steered, delta, sel.layer, v, sel.alpha,
                                     blocks, prng, n_boot=cfg.evaluation.n_boot)
            abl["criterion"] = cfg.exploratory.block_ablation.criterion
            abl["rank_by"] = cfg.exploratory.block_ablation.rank_by
            write_json(out / "block_ablation.json", abl)
            self.log.info("[%s] block ablation of %s: %s", st.name, blocks,
                          {b: (round(x["necessity"]["fraction_lost"] or float("nan"), 3),
                               round(x["sufficiency"]["fraction_reproduced"] or float("nan"), 3))
                           for b, x in abl["blocks"].items()})

    # ------------------------------------------------------------------ #

    def _summarise(self, states: dict[str, TaskState]) -> dict[str, Any]:
        sigs = {n: s.signature for n, s in states.items() if s.signature is not None}
        summary: dict[str, Any] = {
            "run_id": self.root.name,
            "command": self.command,
            "seed": self.cfg.seed,
            "model": self.metadata.get("model", {}).get("name"),
            "control": self.cfg.extraction.control,
            "tasks": {
                n: {
                    "qualified": s.qualified,
                    "gates": s.qualification.get("gates", {}),
                    "selection": s.qualification.get("selection"),
                    "labels": None if s.signature is None else s.signature["labels"],
                    "damage": s.qualification.get("damage"),
                    "decomposition": s.qualification.get("decomposition"),
                    "commitment": s.qualification.get("commitment"),
                    "learned_vector": None if not s.learned_fits else s.qualification.get("learned_vector"),
                    "function_vector": None if s.fv is None else {
                        "natural_rho_at_selection": None if s.selection is None else
                        s.qualification["function_vector"]["natural_rho"][str(s.selection.layer)],
                        "cos_with_pca_at_selected_layer": s.qualification["function_vector"].get("cos_with_pca_at_selected_layer"),
                        "stability": s.fv.stability,
                    },
                }
                for n, s in states.items()
            },
            "n_qualified": sum(s.qualified for s in states.values()),
            "n_rejections": len(self.rejections.entries),
            "deterministic": None if "determinism_check" not in self.metadata else self.metadata["determinism_check"]["identical"],
        }
        if sigs:
            table = cross_task_table(sigs)
            write_json(self.root / "core" / "cross_task.json", {"signatures": sigs, "table": table})
            summary["cross_task"] = table
        return summary


def run_pipeline(cfg: Config, command: str, config_path: str | None = None, run_id: str | None = None) -> Path:
    return Pipeline(cfg, command, config_path=config_path, run_id=run_id).run()

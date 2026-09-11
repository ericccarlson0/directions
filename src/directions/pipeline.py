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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .ablation import block_ablation, select_blocks
from .analysis import cross_task_table, profile_signature
from .calibration import CalibrationResult, Selection, calibrate
from .config import Config, config_to_dict
from .extraction import (
    LayerDirection,
    candidate_layers,
    demo_variation_directions,
    directions_from_differences,
    extract_demo_variation,
    extract_differences,
    permuted_head_means,
)
from .function_vector import FunctionVector, build_function_vector, compose, head_effects, select_heads
from .geometry import covariance_matched_unit_vector, normalize, random_orthogonal_unit_vector, random_unit_vector
from .layerwise import LayerwiseProfile, compare_by_kind, compute_profile, metric_curves, null_summary, profile_arrays
from .model import ForwardResult, Intervention, ModelBackend, environment_metadata, set_torch_determinism
from .prompts import Prompt, few_shot_prompt, zero_shot_prompt
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


@dataclass
class TaskState:
    name: str
    splits: Splits
    directions: dict[int, LayerDirection] = field(default_factory=dict)  # the control at each candidate layer
    pca_directions: dict[int, LayerDirection] = field(default_factory=dict)  # PC1 at each candidate layer (always)
    all_layer_directions: np.ndarray | None = None  # (L+1, d) pooled PC1 at every read point (D17 readouts)
    # Function-vector inputs and result (D21), when extraction.control == "function_vector":
    seed_head_means: np.ndarray | None = None  # (n_seeds, L, n_heads, head_dim) mean head outputs, positive prompts
    head_effects: np.ndarray | None = None  # (L, n_heads) average indirect effect on deranged-label prompts
    head_effects_n_prompts: int = 0
    permuted_head_means: list[np.ndarray] = field(default_factory=list)  # demo_variation null inputs
    fv: FunctionVector | None = None
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
        self.timings: dict[str, float] = {}
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
        t0 = time.time()
        self.backend = ModelBackend(cfg.model, run_seed=cfg.seed)
        self.timings["model_load"] = time.time() - t0
        self.metadata["model"] = self.backend.metadata()
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
            t1 = time.time()
            try:
                states[tcfg.name] = self._prepare_task(tcfg.name, tcfg.params, tcfg.max_target_tokens)
            except Exception as e:
                self.log.error("task %s failed in phase A: %s", tcfg.name, e)
                self.rejections.add("error", tcfg.name, f"{type(e).__name__}: {e}", {"traceback": traceback.format_exc()})
            self.timings[f"prepare:{tcfg.name}"] = time.time() - t1
        if cfg.extraction.control == "function_vector":
            t1 = time.time()
            self._build_function_vectors(states)
            self.timings["function_vectors"] = time.time() - t1
        for name, st in states.items():
            t1 = time.time()
            if st.ready and self.stop_after not in ("fewshot", "extraction"):
                try:
                    self._measure_task(st, states)
                except Exception as e:
                    self.log.error("task %s failed in phase B: %s", name, e)
                    self.rejections.add("error", name, f"{type(e).__name__}: {e}", {"traceback": traceback.format_exc()})
            self._finish_task(st)
            self.timings[f"measure:{name}"] = time.time() - t1

        summary = self._summarise(states)
        if self.stop_after == "figures" and cfg.figures.enabled:
            t2 = time.time()
            try:
                from .figures import make_all_figures

                make_all_figures(self.root, states, cfg)
            except Exception as e:
                self.log.error("figure generation failed: %s", e)
                self.rejections.add("error", "*", f"figures: {type(e).__name__}: {e}", {"traceback": traceback.format_exc()})
            self.timings["figures"] = time.time() - t2
        self.timings["total"] = time.time() - t0
        self.metadata["timings_seconds"] = self.timings
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        write_json(self.root / "core" / "summary.json", summary)
        self.log.info("done in %.1fs -> %s", self.timings["total"], self.root)
        return self.root

    def _seed_table(self) -> dict[str, Any]:
        s = self.cfg.seed
        table: dict[str, Any] = {"run_seed": s, "torch": derive_seed(s, "torch")}
        n_dv = int(self.cfg.evaluation.controls.get("demo_variation", 0))
        for t in self.cfg.tasks:
            table[t.name] = {
                "split": derive_seed(s, "split", t.name),
                "fewshot_eval": derive_seed(s, "fewshot_eval", t.name),
                "extraction": [derive_seed(s, "extraction", t.name, i) for i in range(self.cfg.extraction.n_seeds)],
                "demo_variation": [derive_seed(s, "demo_variation", t.name, k) for k in range(n_dv)],
                "calibration": derive_seed(s, "calibration", t.name),
                "calibration_screen": {str(l): derive_seed(s, "calibration_screen", t.name, l) for l in self.layers},
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

    def _prepare_task(self, name: str, params: dict[str, Any], max_target_tokens: int | None) -> TaskState:
        cfg = self.cfg
        log = self.log
        out = self._task_dir(name)
        q: dict[str, Any] = {"task": name, "gates": {}}

        # 1. data --------------------------------------------------------
        task = build_task(name, params)
        max_tok = max_target_tokens if max_target_tokens is not None else cfg.data.max_target_tokens
        kept, rejected = filter_items(task, self.backend.target_token_count, cfg.prompt.target_template, max_tok)
        for r in rejected:
            self.rejections.add(r.stage, r.task, r.reason, r.details)
        splits = make_splits(kept, cfg.seed, name, cfg.data.n_extraction, cfg.data.n_calibration,
                             cfg.data.n_evaluation, allow_reduced=cfg.data.allow_reduced_splits)
        if splits.reduced:
            self.rejections.add("data", name, "pools_reduced", {"n_items": len(kept), "sizes": {
                "extraction": len(splits.extraction), "calibration": len(splits.calibration), "evaluation": len(splits.evaluation)}})
        st = TaskState(name=name, splits=splits)
        st.qualification = q
        write_json(out / "splits.json", {"n_items_total": len(task.items), "n_items_kept": len(kept),
                                         "n_items_rejected": len(rejected), "max_target_tokens": max_tok,
                                         "params": task.params, **splits.as_dict()})
        log.info("[%s] %d items kept (%d rejected); pools %d/%d/%d", name, len(kept), len(rejected),
                 len(splits.extraction), len(splits.calibration), len(splits.evaluation))

        # 2. few-shot qualification -------------------------------------
        rng = rng_for(cfg.seed, "fewshot_eval", name)
        fs_prompts = [few_shot_prompt(cfg.prompt, splits.evaluation, x, rng) for x in splits.evaluation]
        fs = self.backend.run(fs_prompts)
        zs_prompts = [zero_shot_prompt(cfg.prompt, x) for x in splits.evaluation]
        st.eval_prompts = zs_prompts
        st.base = self.backend.run(zs_prompts, capture=True)
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
        seeds = extract_differences(self.backend, cfg.prompt, splits.extraction, cfg.seed, name, cfg.extraction.n_seeds,
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
            if fvc.aie_metric == "first_token_probability":
                base = np.exp(np.concatenate([s.permuted_first_token_logprob for s in seeds[:k]]))
            else:
                base = np.concatenate([s.permuted_logprob_per_token for s in seeds[:k]])
            t2 = time.time()
            st.head_effects = head_effects(self.backend, neg, base, st.seed_head_means.mean(axis=0), metric=fvc.aie_metric)
            st.head_effects_n_prompts = len(neg)
            self.timings[f"head_effects:{name}"] = time.time() - t2
            top = np.dstack(np.unravel_index(np.argsort(-st.head_effects, axis=None)[:5], st.head_effects.shape))[0]
            log.info("[%s] head effects (%s) over %d deranged prompts in %.0fs: baseline mean %.4f, max effect %.4f, "
                     "top (layer, head) %s", name, fvc.aie_metric, len(neg), self.timings[f"head_effects:{name}"],
                     float(base.mean()), float(st.head_effects.max()), [(int(l), int(h)) for l, h in top])
            ext["function_vector"] = {"aie_seeds": k, "aie_n_prompts": len(neg), "aie_metric": fvc.aie_metric,
                                      "aie_baseline_mean": float(base.mean())}
            if n_dv > 0:
                st.permuted_head_means = permuted_head_means(self.backend, cfg.prompt, splits.extraction, cfg.seed, name, n_dv)
                ext["demo_variation"] = {"n": n_dv, "construction": "function vector of deranged-label prompts (D21)"}
        elif n_dv > 0:
            dv = extract_demo_variation(self.backend, cfg.prompt, splits.extraction, cfg.seed, name, n_dv)
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
        st.stable_layers = [l for l in self.layers if st.directions[l].stability >= cfg.qualification.min_stability]
        q["stability"] = pca_stability
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

    def _write_directions(self, st: TaskState) -> None:
        arrays: dict[str, np.ndarray] = {
            "layers": np.array(self.layers),
            "pooled": np.stack([st.pca_directions[l].direction for l in self.layers]),
            "per_seed": np.stack([st.pca_directions[l].seed_directions for l in self.layers]),
            "all_layers": st.all_layer_directions,
        }
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
        selection = select_heads(effects, fvc.n_heads, fvc.head_selection)
        self.metadata["function_vector"] = {
            "head_selection": fvc.head_selection,
            "n_heads": fvc.n_heads,
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
            assert st.seed_head_means is not None and st.head_effects is not None
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
                "n_heads": fvc.n_heads,
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
            for i, u in enumerate(st.demo_variation.get(layer, [])[: counts["demo_variation"]]):
                out.append(("demo_variation", f"demo_variation_{i}", u))
        return out

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
        cal = calibrate(self.backend, cfg.prompt, cfg.calibration, cfg.evaluation,
                        {l: st.directions[l] for l in st.stable_layers}, splits.calibration, cfg.seed, name)
        st.calibration = cal
        write_json(out / "calibration.json", cal.as_dict())
        q["gates"]["calibration"] = cal.selected is not None
        if cal.selected is None:
            self.rejections.add("calibration", name, "no_reliable_intervention", {"reason": cal.reason})
            if cfg.qualification.enforce:
                return
            best = max(cal.grid, key=lambda g: g.test.mean_diff)
            st.selection = Selection(best.layer, best.rho, best.alpha)
            st.fallback_selection = True
            log.warning("[%s] gates not enforced: falling back to best grid point layer %d rho %.3g", name, best.layer, best.rho)
        else:
            st.selection = cal.selected
        q["selection"] = {**st.selection.__dict__, "fallback": st.fallback_selection}
        log.info("[%s] selected layer %d, rho %.3g (alpha %.3g)%s", name, st.selection.layer, st.selection.rho,
                 st.selection.alpha, " [FALLBACK]" if st.fallback_selection else "")
        if st.fv is not None:
            # The strength the canonical (unscaled) injection would have, in the calibration's units.
            q["function_vector"]["natural_rho"] = {str(l): st.fv.natural_norm / n for l, n in cal.layer_norms.items()}
            q["function_vector"]["cos_with_pca_at_selected_layer"] = q["function_vector"]["cos_with_pca"][str(st.selection.layer)]
            log.info("[%s] function vector's natural rho at the selected layer %.3g (selected rho %.3g); cos with PC1 there %.3f",
                     name, q["function_vector"]["natural_rho"][str(st.selection.layer)], st.selection.rho,
                     q["function_vector"]["cos_with_pca_at_selected_layer"])

        # 5. held-out steering vs matched controls of every kind ----------
        sel = st.selection
        v = st.directions[sel.layer].direction
        do_layerwise = self.stop_after in ("layerwise", "exploratory", "figures")
        t_stage = time.time()
        st.steered = self.backend.run(zs_prompts, interventions=[Intervention(sel.layer, v, sel.alpha)], capture=True)
        brng = rng_for(cfg.seed, "bootstrap", name)
        prng = rng_for(cfg.seed, "profile_bootstrap", name)
        if do_layerwise:
            # Direction-specific readouts (D17): the baseline target-log-prob gradient at every
            # read point, computed once per task and shared by the real and every control profile.
            st.gradients = self.backend.gradients(zs_prompts)
        readout_kw = {"task_directions": st.all_layer_directions, "gradients": st.gradients}
        test = paired_bootstrap_test(st.steered.logprob_per_token, st.base.logprob_per_token, brng, n_boot=cfg.calibration.n_boot)
        t_setup = time.time() - t_stage
        t_forward = t_profile = 0.0
        for kind, label, u in self._build_controls(st, states, sel.layer, v):
            t3 = time.time()
            r = self.backend.run(zs_prompts, interventions=[Intervention(sel.layer, u, sel.alpha)], capture=do_layerwise)
            rt = paired_bootstrap_test(r.logprob_per_token, st.base.logprob_per_token, brng, n_boot=cfg.calibration.n_boot)
            t_forward += time.time() - t3
            prof = None
            if do_layerwise:
                assert r.residuals is not None
                t3 = time.time()
                prof = compute_profile(st.base.residuals, r.residuals, u, sel.layer, cfg.evaluation, prng, with_ci=False,
                                       **readout_kw)
                t_profile += time.time() - t3
            st.controls.append(ControlResult(kind, label, rt.mean_diff, rt.p_value, r.metrics_dict(), prof,
                                             r.logprob_per_token - st.base.logprob_per_token))
        self.timings[f"controls_setup:{name}"] = t_setup
        self.timings[f"controls_forward:{name}"] = t_forward
        self.timings[f"controls_profiles:{name}"] = t_profile
        log.info("[%s] %d controls: steered run + gradients %.0fs, control forwards %.0fs, control profiles %.0fs", name,
                 len(st.controls), t_setup, t_forward, t_profile)
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
            "per_example": {
                "baseline_logprob_per_token": st.base.logprob_per_token.tolist(),
                "steered_logprob_per_token": st.steered.logprob_per_token.tolist(),
                "baseline_exact_match": st.base.exact_match.tolist(),
                "steered_exact_match": st.steered.exact_match.tolist(),
                "controls_logprob_per_token_diff": {c.label: c.per_example_diff.tolist() for c in st.controls},
            },
        }
        write_json(out / "evaluation.json", ev)
        q["steering"] = {"test": test.__dict__, "random_comparison": null.__dict__, "excess_test": excess.__dict__,
                         "by_kind": by_kind_behav, "by_kind_excess": by_kind_excess}
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

        # 7. exploratory -------------------------------------------------
        try:
            self._exploratory(st, v, prng)
        except Exception as e:
            log.error("[%s] exploratory stage failed: %s", name, e)
            self.rejections.add("error", name, f"exploratory: {type(e).__name__}: {e}", {"traceback": traceback.format_exc()})

    def _finish_task(self, st: TaskState) -> None:
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
            rob: dict[str, Any] = {"weakest": sel.__dict__, "profiles": {}, "rank_correlations": {}}
            for label, alt in (("strongest", st.calibration.strongest), ("middle", st.calibration.middle)):
                if alt is None or alt.rho == sel.rho:
                    rob["profiles"][label] = None
                    continue
                r = self.backend.run(st.eval_prompts, interventions=[Intervention(alt.layer, v, alt.alpha)], capture=True)
                assert r.residuals is not None
                p = compute_profile(st.base.residuals, r.residuals, v, alt.layer, cfg.evaluation, prng,
                                    task_directions=st.all_layer_directions, gradients=st.gradients)
                rob["profiles"][label] = {"selection": alt.__dict__, "metrics": r.metrics_dict(), "profile": p.as_dict()}
                rob["rank_correlations"][label] = {
                    m: spearman(st.profile.metric_curve(m), p.metric_curve(m))
                    for m in ("log_gain", "d_eff", "new_subspace", "new_subspace_uncentered", "conversion", "magnitude",
                              "task_alignment", "gradient_alignment")
                }
            write_json(out / "strength_robustness.json", rob)

        if cfg.exploratory.block_ablation.enabled:
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
        }
        if sigs:
            table = cross_task_table(sigs)
            write_json(self.root / "core" / "cross_task.json", {"signatures": sigs, "table": table})
            summary["cross_task"] = table
        return summary


def run_pipeline(cfg: Config, command: str, config_path: str | None = None, run_id: str | None = None) -> Path:
    return Pipeline(cfg, command, config_path=config_path, run_id=run_id).run()

"""End-to-end pilot orchestration.

Stages per task (``validate`` stops after stage 5; ``pilot`` runs all):

1. items -> filter -> three disjoint pools
2. few-shot qualification (accuracy on the evaluation pool)
3. control-direction extraction at every candidate layer, cross-seed stability
4. intervention calibration (layer x strength) on the calibration pool
5. held-out steering on the evaluation pool vs matched random controls
6. layerwise measurement (real + random controls), cross-task analysis
7. exploratory: strength robustness, causal block ablation
8. figures
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .ablation import block_ablation, select_blocks
from .analysis import cross_task_table, profile_signature
from .calibration import CalibrationResult, Selection, calibrate
from .config import Config, config_to_dict
from .extraction import LayerDirection, candidate_layers, directions_from_differences, extract_differences
from .geometry import matched_random_controls
from .layerwise import LayerwiseProfile, compare_to_random, compute_profile, profile_arrays
from .model import ForwardResult, Intervention, ModelBackend, environment_metadata, set_torch_determinism
from .prompts import Prompt, few_shot_prompt, zero_shot_prompt
from .runinfo import RejectionLog, git_info, make_run_dir, setup_logging, write_json, write_resolved_config
from .seeds import derive_seed, rng_for
from .stats import compare_to_null, paired_bootstrap_test, spearman
from .tasks import Splits, build_task, filter_items, make_splits

STAGES = ("data", "fewshot", "extraction", "calibration", "steering", "layerwise", "exploratory", "figures")


@dataclass
class TaskState:
    name: str
    splits: Splits
    directions: dict[int, LayerDirection] = field(default_factory=dict)
    stable_layers: list[int] = field(default_factory=list)
    calibration: CalibrationResult | None = None
    selection: Selection | None = None
    fallback_selection: bool = False
    eval_prompts: list[Prompt] = field(default_factory=list)
    base: ForwardResult | None = None
    steered: ForwardResult | None = None
    randoms: list[tuple[str, np.ndarray, ForwardResult]] = field(default_factory=list)
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
        self.metadata["seeds"] = self._seed_table()
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("run %s: model %s (%d layers, d=%d), candidate layers %s",
                      self.root.name, self.metadata["model"]["name"], self.backend.n_layers,
                      self.backend.hidden_size, self.layers)

        states: dict[str, TaskState] = {}
        for tcfg in cfg.tasks:
            t1 = time.time()
            try:
                st = self._run_task(tcfg.name, tcfg.params)
                states[tcfg.name] = st
            except Exception as e:  # keep going with other tasks, but record the failure
                self.log.error("task %s failed: %s", tcfg.name, e)
                self.rejections.add("error", tcfg.name, f"{type(e).__name__}: {e}", {"traceback": traceback.format_exc()})
            self.timings[f"task:{tcfg.name}"] = time.time() - t1

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
        for t in self.cfg.tasks:
            table[t.name] = {
                "split": derive_seed(s, "split", t.name),
                "fewshot_eval": derive_seed(s, "fewshot_eval", t.name),
                "extraction": [derive_seed(s, "extraction", t.name, i) for i in range(self.cfg.extraction.n_seeds)],
                "calibration": derive_seed(s, "calibration", t.name),
                "calibration_screen": {str(l): derive_seed(s, "calibration_screen", t.name, l) for l in self.layers},
                "random_controls": derive_seed(s, "random_controls", t.name),
                "bootstrap": derive_seed(s, "bootstrap", t.name),
            }
        return table

    # ------------------------------------------------------------------ #

    def _task_dir(self, name: str, exploratory: bool = False) -> Path:
        d = self.root / ("exploratory" if exploratory else "core") / "tasks" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _run_task(self, name: str, params: dict[str, Any]) -> TaskState:
        cfg = self.cfg
        log = self.log
        out = self._task_dir(name)
        q: dict[str, Any] = {"task": name, "gates": {}}

        # 1. data --------------------------------------------------------
        task = build_task(name, params)
        kept, rejected = filter_items(task, self.backend.target_token_count, cfg.prompt.target_template, cfg.data.max_target_tokens)
        for r in rejected:
            self.rejections.add(r.stage, r.task, r.reason, r.details)
        splits = make_splits(kept, cfg.seed, name, cfg.data.n_extraction, cfg.data.n_calibration,
                             cfg.data.n_evaluation, allow_reduced=cfg.data.allow_reduced_splits)
        if splits.reduced:
            self.rejections.add("data", name, "pools_reduced", {"n_items": len(kept), "sizes": {
                "extraction": len(splits.extraction), "calibration": len(splits.calibration), "evaluation": len(splits.evaluation)}})
        st = TaskState(name=name, splits=splits)
        write_json(out / "splits.json", {"n_items_total": len(task.items), "n_items_kept": len(kept),
                                         "n_items_rejected": len(rejected), "params": task.params, **splits.as_dict()})
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
                return self._finish_task(st, q, out)
        if self.stop_after == "fewshot":
            return self._finish_task(st, q, out)

        # 3. extraction --------------------------------------------------
        seeds = extract_differences(self.backend, cfg.prompt, splits.extraction, cfg.seed, name, cfg.extraction.n_seeds)
        st.directions = directions_from_differences(seeds, self.layers, cfg.extraction)
        ext = {
            "seeds": [{"seed_index": s.seed_index, "seed": s.seed, "metrics": s.metrics} for s in seeds],
            "layers": {str(l): d.summary() for l, d in st.directions.items()},
            "all_layer_stability": {},
        }
        # exploratory diagnostic: stability at every read point, not only candidates
        all_dirs = directions_from_differences(seeds, list(range(self.backend.n_layers + 1)), cfg.extraction)
        ext["all_layer_stability"] = {str(l): d.stability for l, d in all_dirs.items()}
        ext["all_layer_pooled_evr"] = {str(l): d.pooled_explained_variance_ratio for l, d in all_dirs.items()}
        write_json(out / "extraction.json", ext)
        np.savez_compressed(out / "directions.npz",
                            layers=np.array(self.layers),
                            pooled=np.stack([st.directions[l].direction for l in self.layers]),
                            per_seed=np.stack([st.directions[l].seed_directions for l in self.layers]))
        st.stable_layers = [l for l in self.layers if st.directions[l].stability >= cfg.qualification.min_stability]
        q["stability"] = {str(l): st.directions[l].stability for l in self.layers}
        q["stable_layers"] = st.stable_layers
        q["gates"]["stability"] = bool(st.stable_layers)
        log.info("[%s] stability per candidate layer: %s", name, {l: round(st.directions[l].stability, 3) for l in self.layers})
        if not st.stable_layers:
            self.rejections.add("stability", name, "no_stable_candidate_layer",
                                {"stability": q["stability"], "min_stability": cfg.qualification.min_stability})
            if cfg.qualification.enforce:
                return self._finish_task(st, q, out)
            st.stable_layers = list(self.layers)
        if self.stop_after == "extraction":
            return self._finish_task(st, q, out)

        # 4. calibration -------------------------------------------------
        cal = calibrate(self.backend, cfg.prompt, cfg.calibration, cfg.evaluation,
                        {l: st.directions[l] for l in st.stable_layers}, splits.calibration, cfg.seed, name)
        st.calibration = cal
        write_json(out / "calibration.json", cal.as_dict())
        q["gates"]["calibration"] = cal.selected is not None
        if cal.selected is None:
            self.rejections.add("calibration", name, "no_reliable_intervention", {"reason": cal.reason})
            if cfg.qualification.enforce:
                return self._finish_task(st, q, out)
            best = max(cal.grid, key=lambda g: g.test.mean_diff)
            st.selection = Selection(best.layer, best.rho, best.alpha)
            st.fallback_selection = True
            log.warning("[%s] gates not enforced: falling back to best grid point layer %d rho %.3g", name, best.layer, best.rho)
        else:
            st.selection = cal.selected
        q["selection"] = {**st.selection.__dict__, "fallback": st.fallback_selection}
        log.info("[%s] selected layer %d, rho %.3g (alpha %.3g)%s", name, st.selection.layer, st.selection.rho,
                 st.selection.alpha, " [FALLBACK]" if st.fallback_selection else "")

        # 5. held-out steering vs matched random controls ---------------
        sel = st.selection
        v = st.directions[sel.layer].direction
        st.steered = self.backend.run(zs_prompts, interventions=[Intervention(sel.layer, v, sel.alpha)], capture=True)
        brng = rng_for(cfg.seed, "bootstrap", name)
        test = paired_bootstrap_test(st.steered.logprob_per_token, st.base.logprob_per_token, brng, n_boot=cfg.calibration.n_boot)
        controls = matched_random_controls(rng_for(cfg.seed, "random_controls", name), v,
                                           cfg.evaluation.n_random_controls, tuple(cfg.evaluation.random_control_kinds))
        rand_rows = []
        for kind, u in controls:
            r = self.backend.run(zs_prompts, interventions=[Intervention(sel.layer, u, sel.alpha)], capture=True)
            st.randoms.append((kind, u, r))
            rt = paired_bootstrap_test(r.logprob_per_token, st.base.logprob_per_token, brng, n_boot=cfg.calibration.n_boot)
            rand_rows.append({"kind": kind, "mean_diff": rt.mean_diff, "p_value": rt.p_value, "metrics": r.metrics_dict()})
        null = compare_to_null(test.mean_diff, np.array([r["mean_diff"] for r in rand_rows]))
        ev = {
            "selection": q["selection"],
            "baseline": st.base.metrics_dict(),
            "steered": st.steered.metrics_dict(),
            "steering_test": test.__dict__,
            "random_controls": rand_rows,
            "random_comparison": null.__dict__,
            "per_example": {
                "baseline_logprob_per_token": st.base.logprob_per_token.tolist(),
                "steered_logprob_per_token": st.steered.logprob_per_token.tolist(),
                "baseline_exact_match": st.base.exact_match.tolist(),
                "steered_exact_match": st.steered.exact_match.tolist(),
            },
        }
        write_json(out / "evaluation.json", ev)
        q["steering"] = {"test": test.__dict__, "random_comparison": null.__dict__}
        q["gates"]["steering"] = bool(test.p_value <= cfg.qualification.steering_alpha and test.mean_diff > 0)
        q["gates"]["random_controls"] = bool(null.p_upper <= cfg.qualification.random_control_max_p)
        log.info("[%s] held-out steering: mean d(lp/tok)=%.4f p=%.4f; vs %d random: z=%.2f p=%.3f", name,
                 test.mean_diff, test.p_value, len(rand_rows), null.z, null.p_upper)
        if not q["gates"]["steering"]:
            self.rejections.add("steering", name, "no_reliable_heldout_steering", {"test": test.__dict__})
        if not q["gates"]["random_controls"]:
            self.rejections.add("random_controls", name, "steering_not_above_random_controls", {"comparison": null.__dict__})
        st.qualified = all(q["gates"].values())
        if not st.qualified and cfg.qualification.enforce:
            return self._finish_task(st, q, out)
        if self.stop_after == "steering":
            return self._finish_task(st, q, out)

        # 6. layerwise ---------------------------------------------------
        prng = rng_for(cfg.seed, "profile_bootstrap", name)
        assert st.base.residuals is not None and st.steered.residuals is not None
        st.profile = compute_profile(st.base.residuals, st.steered.residuals, v, sel.layer, cfg.evaluation, prng)
        rand_profiles = []
        for kind, u, r in st.randoms:
            assert r.residuals is not None
            rand_profiles.append(compute_profile(st.base.residuals, r.residuals, u, sel.layer, cfg.evaluation, prng))
        st.comparison = compare_to_random(st.profile, rand_profiles)
        st.signature = profile_signature(st.profile, st.comparison, cfg.analysis)
        write_json(out / "layerwise.json", {
            "real": st.profile.as_dict(),
            "random_controls": [{"kind": k, **p.as_dict()} for (k, _, _), p in zip(st.randoms, rand_profiles)],
            "comparison": st.comparison,
            "signature": st.signature,
        })
        np.savez_compressed(out / "layerwise_arrays.npz", **profile_arrays(st.profile))
        log.info("[%s] profile: cum log G %.3f, final alignment %.3f, d_eff %.1f->%.1f, labels %s", name,
                 st.signature["cumulative_log_gain"], st.signature["alignment_final"] or float("nan"),
                 st.signature["d_eff_first"] or float("nan"), st.signature["d_eff_final"] or float("nan"),
                 st.signature["labels"])
        if self.stop_after == "layerwise":
            return self._finish_task(st, q, out)

        # 7. exploratory -------------------------------------------------
        try:
            self._exploratory(st, v, prng)
        except Exception as e:
            log.error("[%s] exploratory stage failed: %s", name, e)
            self.rejections.add("error", name, f"exploratory: {type(e).__name__}: {e}", {"traceback": traceback.format_exc()})
        return self._finish_task(st, q, out)

    def _finish_task(self, st: TaskState, q: dict[str, Any], out: Path) -> TaskState:
        q["qualified"] = bool(st.qualified)
        q["enforce"] = self.cfg.qualification.enforce
        st.qualification = q
        write_json(out / "qualification.json", q)
        return st

    # ------------------------------------------------------------------ #

    def _exploratory(self, st: TaskState, v: np.ndarray, prng: np.random.Generator) -> None:
        cfg = self.cfg
        assert st.calibration is not None and st.selection is not None and st.profile is not None
        assert st.base is not None and st.steered is not None and st.base.residuals is not None
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
                p = compute_profile(st.base.residuals, r.residuals, v, alt.layer, cfg.evaluation, prng)
                rob["profiles"][label] = {"selection": alt.__dict__, "metrics": r.metrics_dict(), "profile": p.as_dict()}
                rob["rank_correlations"][label] = {
                    m: spearman(st.profile.metric_curve(m), p.metric_curve(m))
                    for m in ("log_gain", "d_eff", "new_subspace", "new_subspace_uncentered", "conversion", "magnitude")
                }
            write_json(out / "strength_robustness.json", rob)

        if cfg.exploratory.block_ablation.enabled:
            blocks = select_blocks(st.profile, cfg.exploratory.block_ablation)
            delta = st.steered.residuals.astype(np.float64) - st.base.residuals.astype(np.float64)
            abl = block_ablation(self.backend, st.eval_prompts, st.base, st.steered, delta, sel.layer, v, sel.alpha,
                                 blocks, prng, n_boot=cfg.evaluation.n_boot)
            abl["criterion"] = cfg.exploratory.block_ablation.criterion
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
            "model": self.metadata.get("model", {}).get("name"),
            "tasks": {
                n: {
                    "qualified": s.qualified,
                    "gates": s.qualification.get("gates", {}),
                    "selection": s.qualification.get("selection"),
                    "labels": None if s.signature is None else s.signature["labels"],
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

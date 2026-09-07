"""End-to-end orchestration of the measurement-validation pilot.

Two entry points share all of their machinery:

``run_validation``
    task qualification only -- few-shot ICL performance, control-direction
    extraction and cross-seed stability, intervention calibration, held-out
    causal steering, and the matched-random-control comparison.

``run_pilot``
    validation followed by the full downstream layerwise measurement, the
    random-control null distributions, the exploratory analyses and figures.

Every automatic rejection is written to ``rejections.jsonl`` with the numbers
that produced it.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import torch

from . import controls as controls_mod
from . import calibration, extraction, figures, layerwise, mathx, profiles
from .config import Config
from .extraction import DirectionSpec
from .metadata import run_metadata
from .model import BlockPatch, LanguageModel
from .prompts import build_eval_prompts
from .serialization import RunDirectory
from .tasks import load_task, split_task

STAGES = (
    "fewshot_qualification",
    "direction_stability",
    "intervention_calibration",
    "heldout_steering",
    "random_control_comparison",
)


@dataclass
class TaskReport:
    task: str
    qualified: bool = False
    failed_stage: str | None = None
    stages: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)
    direction: DirectionSpec | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "qualified": self.qualified,
            "failed_stage": self.failed_stage,
            "stages": self.stages,
            "error": self.error,
            **self.payload,
        }


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # cuBLAS GEMMs remain nondeterministic-tolerant


def task_rng(base_seed: int, task: str, purpose: str) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(entropy=base_seed, spawn_key=(mathx.stable_key(purpose, task),))
    )


# --------------------------------------------------------------------------
# per-task validation
# --------------------------------------------------------------------------


def validate_task(cfg: Config, rundir: RunDirectory, lm: LanguageModel, task_name: str) -> TaskReport:
    rep = TaskReport(task=task_name)
    task = load_task(task_name)
    splits = split_task(
        task,
        cfg.data.n_extraction,
        cfg.data.n_calibration,
        cfg.data.n_eval,
        seed=cfg.run.seed,
    )
    rep.payload["splits"] = {k: [i.id for i in v] for k, v in splits.as_dict().items()}
    tmpl = cfg.data.template
    compute = dict(batch_size=cfg.compute.batch_size, max_seq_len=cfg.compute.max_seq_len)

    # -- stage 1: does the model do the task at all, with ICL? ------------
    qual_prompts = build_eval_prompts(
        task_rng(cfg.run.seed, task_name, "qualification"),
        tmpl,
        list(splits.evaluation),
        list(splits.evaluation),
        cfg.data.n_shot_qualification,
    )
    qual = lm.score(qual_prompts, **compute).summary()
    ok = (
        qual["accuracy"] >= cfg.qualification.min_fewshot_accuracy
        and qual["target_logprob_per_token"] >= cfg.qualification.min_target_logprob
    )
    rep.stages["fewshot_qualification"] = {
        "passed": ok,
        "n_shot": cfg.data.n_shot_qualification,
        "behavior": qual,
        "thresholds": {
            "min_fewshot_accuracy": cfg.qualification.min_fewshot_accuracy,
            "min_target_logprob": cfg.qualification.min_target_logprob,
        },
    }
    rundir.log(
        f"  [{task_name}] {cfg.data.n_shot_qualification}-shot accuracy="
        f"{qual['accuracy']:.3f} logprob/token={qual['target_logprob_per_token']:.3f}"
    )
    if not ok:
        rep.failed_stage = "fewshot_qualification"
        rundir.reject("fewshot_qualification", task_name, "below ICL accuracy/logprob threshold", qual)
        return rep

    # -- stage 2: control-direction extraction and cross-seed stability ---
    candidate_layers = sorted(
        {lm.layer_index_from_fraction(f) for f in cfg.extraction.candidate_layer_fracs}
    )
    ext = extraction.extract_directions(
        lm, cfg, task_name, list(splits.extraction), list(splits.extraction), candidate_layers
    )
    stable = [
        l
        for l in candidate_layers
        if ext.layers[l].stability_min >= cfg.qualification.min_direction_stability
    ]
    rep.stages["direction_stability"] = {
        "passed": bool(stable),
        "candidate_layers": candidate_layers,
        "stable_layers": stable,
        "threshold": cfg.qualification.min_direction_stability,
        "per_layer": {str(l): ext.layers[l].to_dict() for l in candidate_layers},
    }
    rundir.log(
        f"  [{task_name}] stability(min) per layer: "
        + ", ".join(f"{l}:{ext.layers[l].stability_min:.2f}" for l in candidate_layers)
    )
    if not stable:
        rep.failed_stage = "direction_stability"
        rundir.reject(
            "direction_stability", task_name, "no candidate layer met cross-seed stability",
            {"per_layer_min": {str(l): ext.layers[l].stability_min for l in candidate_layers}},
        )
        return rep

    # -- stage 3: intervention calibration on the calibration split -------
    calib_prompts = build_eval_prompts(
        task_rng(cfg.run.seed, task_name, "calibration"),
        tmpl,
        list(splits.calibration),
        list(splits.calibration),
        cfg.data.n_shot_eval,
    )
    calib_resid = lm.capture(calib_prompts, **compute)
    median_norms = extraction.median_residual_norms(calib_resid)
    ext.baseline_norms["calibration"] = [float(x) for x in median_norms]
    baseline_calib = lm.score(calib_prompts, **compute)
    calib = calibration.sweep(
        lm, cfg, task_name, ext, calib_prompts, baseline_calib, median_norms, stable
    )
    rep.stages["intervention_calibration"] = {"passed": calib.selected is not None, **calib.to_dict()}
    if calib.selected is None:
        rep.failed_stage = "intervention_calibration"
        rundir.reject("intervention_calibration", task_name, calib.rejected_reason or "no selection", {})
        return rep
    sel = calib.selected
    rundir.log(
        f"  [{task_name}] selected layer={sel.layer} rho={sel.rho} alpha={sel.alpha:.3f} "
        f"({calib.metric_name} {calib.baseline_metric:.3f} -> {sel.metric:.3f})"
    )

    direction = DirectionSpec(
        layer=sel.layer,
        vector=ext.layers[sel.layer].consensus.astype(np.float32),
        alpha=sel.alpha,
        label="control",
    )
    rep.direction = direction

    # -- stage 4: held-out causal steering --------------------------------
    eval_prompts = build_eval_prompts(
        task_rng(cfg.run.seed, task_name, "evaluation"),
        tmpl,
        list(splits.evaluation),
        list(splits.evaluation),
        cfg.data.n_shot_eval,
    )
    metric = cfg.intervention.selection_metric
    base_eval = lm.score(eval_prompts, **compute)
    steer_eval = lm.score(eval_prompts, intervention=direction, **compute)
    improvement = steer_eval.metric(metric) - base_eval.metric(metric)
    passed_steer = improvement >= cfg.qualification.min_steering_improvement
    rep.stages["heldout_steering"] = {
        "passed": bool(passed_steer),
        "metric": metric,
        "baseline": base_eval.summary(),
        "steered": steer_eval.summary(),
        "improvement": float(improvement),
        "threshold": cfg.qualification.min_steering_improvement,
    }
    rundir.log(
        f"  [{task_name}] held-out {metric}: {base_eval.metric(metric):.3f} -> "
        f"{steer_eval.metric(metric):.3f} (delta {improvement:+.3f})"
    )
    if not passed_steer:
        rep.failed_stage = "heldout_steering"
        rundir.reject(
            "heldout_steering", task_name, "held-out steering improvement below threshold",
            rep.stages["heldout_steering"],
        )
        return rep

    # -- stage 5: matched random controls ---------------------------------
    rng = controls_mod.control_rng(cfg.run.seed, task_name, direction.layer)
    ctrl_specs = controls_mod.build_random_controls(
        rng, direction, cfg.controls.n_random, cfg.controls.kinds
    )
    ctrl_rows = []
    for spec in ctrl_specs:
        out = lm.score(eval_prompts, intervention=spec, **compute)
        ctrl_rows.append(
            {"label": spec.label, "behavior": out.summary(),
             "improvement": float(out.metric(metric) - base_eval.metric(metric))}
        )
    ctrl_impr = np.array([r["improvement"] for r in ctrl_rows])
    p_value = mathx.empirical_p_value(improvement, ctrl_impr)
    z = mathx.z_against_null(improvement, ctrl_impr)
    passed_ctrl = p_value <= cfg.qualification.max_control_p_value
    rep.stages["random_control_comparison"] = {
        "passed": bool(passed_ctrl),
        "n_controls": len(ctrl_specs),
        "kinds": cfg.controls.kinds,
        "control_improvement_mean": float(np.mean(ctrl_impr)),
        "control_improvement_max": float(np.max(ctrl_impr)),
        "p_value": p_value,
        "z": z,
        "threshold": cfg.qualification.max_control_p_value,
        "controls": ctrl_rows,
    }
    rundir.log(
        f"  [{task_name}] vs {len(ctrl_specs)} matched random controls: "
        f"p={p_value:.3f} z={z:.2f} (control mean delta {np.mean(ctrl_impr):+.3f})"
    )
    if not passed_ctrl:
        rep.failed_stage = "random_control_comparison"
        rundir.reject(
            "random_control_comparison", task_name,
            "steering effect not separated from matched random controls",
            {"p_value": p_value, "z": z, "improvement": improvement},
        )
        return rep

    rep.qualified = True
    rep.payload["extraction"] = ext.to_dict()
    rep.payload["control_specs"] = [
        {"label": s.label, "layer": s.layer, "alpha": s.alpha} for s in ctrl_specs
    ]
    rep.payload["_runtime"] = {
        "eval_prompt_ids": [p.item_id for p in eval_prompts],
        "n_shot_eval": cfg.data.n_shot_eval,
    }
    # objects needed downstream by the layerwise stage (not serialized directly)
    rep.payload["_objects"] = {
        "calibration": calib,
        "eval_prompts": eval_prompts,
        "control_specs": ctrl_specs,
        "baseline_behavior": base_eval,
        "steered_behavior": steer_eval,
        "extraction": ext,
    }
    return rep


# --------------------------------------------------------------------------
# layerwise stage
# --------------------------------------------------------------------------


def measure_task_layerwise(
    cfg: Config, rundir: RunDirectory, lm: LanguageModel, rep: TaskReport
) -> dict:
    direction = rep.direction
    objs = rep.payload["_objects"]
    eval_prompts = objs["eval_prompts"]
    compute = dict(batch_size=cfg.compute.layerwise_batch_size, max_seq_len=cfg.compute.max_seq_len)

    base_resid = lm.capture(eval_prompts, **compute)
    steer_resid = lm.capture(eval_prompts, intervention=direction, **compute)
    m = layerwise.measure(
        cfg, "control", direction.layer, direction.alpha, direction.vector, base_resid, steer_resid
    )

    control_ms = []
    for spec in objs["control_specs"]:
        resid = lm.capture(eval_prompts, intervention=spec, **compute)
        control_ms.append(
            layerwise.measure(cfg, spec.label, spec.layer, spec.alpha, spec.vector, base_resid, resid)
        )
    null = layerwise.aggregate_controls(control_ms)

    core = {
        "task": rep.task,
        "direction": {"layer": direction.layer, "alpha": direction.alpha},
        "measurement": m.to_dict(),
        "random_control_null": null,
        "real_vs_null": _real_vs_null(m, null),
    }
    rundir.write_json(f"core/layerwise__{rep.task}.json", core)
    rundir.write_npz(
        f"core/layerwise__{rep.task}.npz",
        S=m.S, G=m.G, C=m.C, d_eff=m.d_eff, d90=m.d90, N=m.N,
        N_uncentered=m.N_uncentered, control_alignment=m.control_alignment,
        direction=direction.vector,
    )
    objs["measurement"] = m
    objs["baseline_residuals"] = base_resid
    objs["deltas"] = steer_resid - base_resid
    return core


def _real_vs_null(m: layerwise.LayerwiseMeasurement, null: dict) -> dict:
    """Per-layer standardized position of the real direction in the control null."""
    with layerwise.quiet_nan_reductions():
        return _real_vs_null_inner(m, null)


def _real_vs_null_inner(m: layerwise.LayerwiseMeasurement, null: dict) -> dict:
    obs = {
        "S": np.nanmedian(m.S, axis=1),
        "log_G": np.nanmedian(np.log(m.G), axis=1),
        "C": np.nanmedian(m.C, axis=1),
        "d_eff": m.d_eff,
        "d90": m.d90,
        "N": m.N,
        "N_uncentered": m.N_uncentered,
    }
    out = {}
    for key, values in obs.items():
        if key not in null:
            continue
        samples = np.array(
            [[np.nan if v is None else v for v in row] for row in null[key]["samples"]], float
        )
        zs, ps = [], []
        for l in range(samples.shape[1]):
            zs.append(mathx.z_against_null(values[l], samples[:, l]))
            ps.append(mathx.empirical_p_value(values[l], samples[:, l]))
        out[key] = {
            "observed": [None if not np.isfinite(v) else float(v) for v in values[: samples.shape[1]]],
            "z": [None if not np.isfinite(v) else float(v) for v in zs],
            "p_one_sided": [None if not np.isfinite(v) else float(v) for v in ps],
        }
    return out


# --------------------------------------------------------------------------
# exploratory analyses
# --------------------------------------------------------------------------


def exploratory_task(cfg: Config, rundir: RunDirectory, lm: LanguageModel, rep: TaskReport) -> dict:
    objs = rep.payload["_objects"]
    m: layerwise.LayerwiseMeasurement = objs["measurement"]
    out: dict = {"task": rep.task}

    if cfg.layerwise.also_measure_strongest:
        out["strength_robustness"] = _strength_robustness(cfg, rundir, lm, rep, objs, m)

    if cfg.exploratory.profile_labels:
        out["profile"] = profiles.classify(m)
        rundir.log(f"  [{rep.task}] exploratory profile label: {out['profile']['label']}")

    ab = cfg.exploratory.block_ablation
    if ab.enabled:
        out["block_ablation"] = _block_ablation(cfg, lm, rep, m, objs, ab.top_k_layers)

    rundir.write_json(f"exploratory/{rep.task}.json", out)
    return out


def _strength_robustness(cfg, rundir, lm, rep, objs, m) -> dict:
    """Repeat the layerwise measurement at the strongest reliable strength.

    Same layer and same direction; only ``alpha`` changes. If the profile is a
    property of the control direction rather than of the injection magnitude,
    the two measurements should agree in shape.
    """
    calib = objs["calibration"]
    same_layer = [p for p in calib.points if p.layer == rep.direction.layer and p.reliable]
    if not same_layer:
        return {"available": False, "reason": "no other reliable strength at the selected layer"}
    strongest = max(same_layer, key=lambda p: p.improvement)
    if abs(strongest.alpha - rep.direction.alpha) < 1e-9:
        return {"available": False, "reason": "the selected point is already the strongest"}
    spec = rep.direction.scaled(strongest.alpha)
    compute = dict(batch_size=cfg.compute.layerwise_batch_size, max_seq_len=cfg.compute.max_seq_len)
    resid = lm.capture(objs["eval_prompts"], intervention=spec, **compute)
    m2 = layerwise.measure(
        cfg, "control_strongest", spec.layer, spec.alpha, spec.vector,
        objs["baseline_residuals"], resid,
    )
    behavior = lm.score(
        objs["eval_prompts"], intervention=spec,
        batch_size=cfg.compute.batch_size, max_seq_len=cfg.compute.max_seq_len,
    )
    metric = cfg.intervention.selection_metric
    with layerwise.quiet_nan_reductions():
        agreement = {
            "log_G_spearman": _rank_corr(
                np.nanmedian(np.log(m.G), axis=1), np.nanmedian(np.log(m2.G), axis=1)
            ),
            "d_eff_spearman": _rank_corr(m.d_eff, m2.d_eff),
            "N_spearman": _rank_corr(m.N, m2.N),
        }
    rundir.write_json(
        f"exploratory/layerwise_strongest__{rep.task}.json",
        {"task": rep.task, "measurement": m2.to_dict()},
    )
    return {
        "available": True,
        "selected_rho": calib.selected.rho,
        "strongest_rho": strongest.rho,
        "strongest_alpha": strongest.alpha,
        "heldout_metric": behavior.metric(metric),
        "heldout_improvement": behavior.metric(metric)
        - objs["baseline_behavior"].metric(metric),
        "profile_agreement_with_selected": agreement,
        "profile": profiles.classify(m2),
    }


def _rank_corr(a, b) -> float | None:
    """Spearman correlation over the layers where both series are defined."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / denom) if denom > 0 else None


def _block_ablation(cfg, lm, rep, m, objs, top_k: int) -> dict:
    """Causal validation of high-conversion / high-new-subspace blocks.

    For each selected block ``l`` we (a) remove ``b_l(x)`` from the block's
    output during the steered run (necessity) and (b) inject ``b_l(x)`` during
    an unsteered run (sufficiency), and report the fraction of the steering
    behavioural improvement that is lost or reproduced.
    """
    deltas = objs["deltas"]  # (L+1, N, d)
    blocks_resp = mathx.blockwise_response(deltas)  # (L, N, d)
    metric = cfg.intervention.selection_metric
    base_metric = objs["baseline_behavior"].metric(metric)
    steer_metric = objs["steered_behavior"].metric(metric)
    gain = steer_metric - base_metric

    scores = {}
    for l in range(m.layer, m.n_layers):
        c = np.nanmedian(m.C[l])
        n = m.N_uncentered[l]
        scores[l] = float(np.nan_to_num(c) + np.nan_to_num(n))
    chosen = sorted(sorted(scores, key=lambda l: -scores[l])[:top_k])

    compute = dict(batch_size=cfg.compute.batch_size, max_seq_len=cfg.compute.max_seq_len)
    rows = []
    for l in chosen:
        vecs = np.asarray(blocks_resp[l], dtype=np.float32)
        removed = lm.score(
            objs["eval_prompts"], intervention=rep.direction,
            block_patch=BlockPatch(layer=l, vectors=vecs, sign=-1.0), **compute,
        )
        restored = lm.score(
            objs["eval_prompts"], intervention=None,
            block_patch=BlockPatch(layer=l, vectors=vecs, sign=+1.0), **compute,
        )
        rows.append({
            "block": l,
            "selection_score": scores[l],
            "median_conversion": float(np.nanmedian(m.C[l])),
            "new_subspace_uncentered": float(m.N_uncentered[l]),
            "removed_metric": removed.metric(metric),
            "restored_metric": restored.metric(metric),
            "fraction_of_gain_lost": float((steer_metric - removed.metric(metric)) / gain)
            if gain != 0 else None,
            "fraction_of_gain_restored": float((restored.metric(metric) - base_metric) / gain)
            if gain != 0 else None,
            "removed_behavior": removed.summary(),
            "restored_behavior": restored.summary(),
        })
    return {
        "metric": metric,
        "baseline_metric": base_metric,
        "steered_metric": steer_metric,
        "steering_gain": gain,
        "selection": "top-k blocks by median C_l + N_l (uncentered)",
        "blocks": rows,
    }


# --------------------------------------------------------------------------
# top-level drivers
# --------------------------------------------------------------------------


def _open_run(cfg: Config, command: str) -> RunDirectory:
    rundir = RunDirectory.create(cfg, command)
    rundir.write_yaml("config.resolved.yaml", cfg.to_dict())
    rundir.log(f"run directory: {rundir.root}")
    return rundir


def execute(cfg: Config, command: str, do_layerwise: bool) -> dict:
    t0 = time.time()
    rundir = _open_run(cfg, command)
    seed_everything(cfg.run.seed)
    rundir.log(f"loading model backend={cfg.model.backend} {cfg.model.name_or_path}")
    lm = LanguageModel.from_config(cfg)
    meta = run_metadata(cfg, {"command": command, "model": lm.metadata()})
    rundir.write_json("metadata.json", meta)
    rundir.log(
        f"model: {lm.metadata()['name_or_path']} layers={lm.n_layers} d_model={lm.d_model} "
        f"params={lm.metadata()['n_parameters']:,} device={lm.device} dtype={lm.dtype}"
    )

    reports: dict[str, TaskReport] = {}
    for task_name in cfg.data.tasks:
        rundir.log(f"task {task_name}: validating")
        try:
            reports[task_name] = validate_task(cfg, rundir, lm, task_name)
        except Exception as exc:  # a broken task must not abort the whole run
            rep = TaskReport(task=task_name, error=f"{type(exc).__name__}: {exc}")
            rep.failed_stage = "exception"
            reports[task_name] = rep
            rundir.reject("exception", task_name, repr(exc), {"traceback": traceback.format_exc()})

    qualified = [t for t, r in reports.items() if r.qualified]
    rundir.log(f"qualified tasks ({len(qualified)}/{len(reports)}): {qualified}")

    layerwise_results: dict[str, dict] = {}
    exploratory_results: dict[str, dict] = {}
    measurements: dict[str, layerwise.LayerwiseMeasurement] = {}
    figure_paths: list[str] = []

    if do_layerwise:
        for task_name in qualified:
            rundir.log(f"task {task_name}: layerwise measurement")
            try:
                layerwise_results[task_name] = measure_task_layerwise(
                    cfg, rundir, lm, reports[task_name]
                )
                measurements[task_name] = reports[task_name].payload["_objects"]["measurement"]
                exploratory_results[task_name] = exploratory_task(cfg, rundir, lm, reports[task_name])
            except Exception as exc:
                reports[task_name].error = f"layerwise: {type(exc).__name__}: {exc}"
                rundir.reject("layerwise", task_name, repr(exc), {"traceback": traceback.format_exc()})

    # -- serialization ---------------------------------------------------
    for name, rep in reports.items():
        rep.payload.pop("_objects", None)
        rundir.write_json(f"core/validation__{name}.json", rep.to_dict())

    if cfg.figures.enabled:
        figure_paths = _make_figures(cfg, rundir, reports, measurements, layerwise_results)

    summary = {
        "command": command,
        "run_name": cfg.run.name,
        "run_directory": str(rundir.root),
        "model": lm.metadata()["name_or_path"],
        "n_layers": lm.n_layers,
        "d_model": lm.d_model,
        "seed": cfg.run.seed,
        "git_commit": meta["git"]["commit"],
        "config_fingerprint": cfg.fingerprint(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "tasks_requested": list(cfg.data.tasks),
        "tasks_qualified": qualified,
        "task_status": {
            t: {
                "qualified": r.qualified,
                "failed_stage": r.failed_stage,
                "error": r.error,
                "direction_layer": r.direction.layer if r.direction else None,
                "alpha": r.direction.alpha if r.direction else None,
                "heldout_improvement": r.stages.get("heldout_steering", {}).get("improvement"),
                "control_p_value": r.stages.get("random_control_comparison", {}).get("p_value"),
            }
            for t, r in reports.items()
        },
        "profile_labels": {
            t: v.get("profile", {}).get("label") for t, v in exploratory_results.items()
        },
        "figures": figure_paths,
        "layerwise_completed": sorted(layerwise_results),
    }
    rundir.write_json("summary.json", summary)
    rundir.log(f"done in {summary['elapsed_seconds']}s; summary written")
    return summary


def _make_figures(cfg, rundir, reports, measurements, layerwise_results) -> list[str]:
    paths: list[str] = []
    dpi, ext = cfg.figures.dpi, cfg.figures.format
    try:
        extractions = {
            t: r.payload["extraction"] for t, r in reports.items() if "extraction" in r.payload
        }
        p = figures.stability_figure(rundir, extractions, dpi=dpi, ext=ext)
        if p:
            paths.append(p)
        for t, r in reports.items():
            calib = r.stages.get("intervention_calibration")
            if calib and calib.get("grid"):
                paths.append(figures.calibration_figure(rundir, t, calib, dpi=dpi, ext=ext))
        for t, m in measurements.items():
            null = layerwise_results.get(t, {}).get("random_control_null", {})
            paths += figures.per_task_figures(rundir, t, m, null, dpi=dpi, ext=ext)
            p = figures.real_vs_random(rundir, t, m, null, dpi=dpi, ext=ext)
            if p:
                paths.append(p)
        paths += figures.heatmaps(rundir, measurements, dpi=dpi, ext=ext)
    except Exception as exc:
        rundir.reject("figures", "all", repr(exc), {"traceback": traceback.format_exc()})
    return paths


def run_validation(cfg: Config) -> dict:
    return execute(cfg, "validate", do_layerwise=False)


def run_pilot(cfg: Config) -> dict:
    return execute(cfg, "pilot", do_layerwise=True)

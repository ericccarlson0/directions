"""The source test (docs/DECISIONS.md D38): which sublayer writes the increments that turn the injected
direction into the model's natural task direction, and is that sublayer necessary?

For every task, the learned vector at the primary layer and its canonical strength, on the held-out prompts:

* three captured passes (the unsteered zero-shot run, the steered run, the demonstration run) record every
  block's attention write and MLP write at the query token (``ModelBackend.run(capture_sublayers=True)``);
* each block's steered increment (steered minus unsteered) and natural increment (demonstrations minus none)
  is split into its attention and MLP parts, and each part's component along the prompt's natural difference
  at the read point the write lands on is the *aligning* write of that sublayer;
* over the window from the injection layer to the hand-over read point (the landmark run, D37) the aligning
  writes are summed per sublayer and shared;
* the selected heads' share of the attention write is read from the per-head outputs (exact on pre-norm blocks,
  the pre-norm attention output on OLMo 3 / Gemma 4);
* random directions injected at the same norm give the split of a perturbation that carries no task;
* causally, each sublayer's aligning component (and its whole increment) is removed from the steered run over
  the window, by subtracting it at the read point it lands on (D29 mechanics), against random per-example
  directions of the same norms; and, one block at a time, each block's aligning component alone.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .config import SourceConfig
from .model import ForwardResult, Intervention, ModelBackend, environment_metadata, set_torch_determinism
from .profiling import Profiler
from .prompts import few_shot_prompt, zero_shot_prompt
from .runinfo import git_info, read_json, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .stats import bootstrap_median_ci_rows, paired_excess_test
from .trajectories import TaskInputs, _load_run, _load_task, alpha_from_calibration, unit_vectors

# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #


def unit_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), eps)


def along(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """The signed component of the rows of ``X`` along the unit rows ``U`` (same shape), ``(..., )``."""
    return (np.asarray(X, dtype=np.float64) * U).sum(-1)


def row_cosines(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return (A * B).sum(-1) / np.maximum(np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1), eps)


def decompose(base: ForwardResult, steered: ForwardResult, icl: ForwardResult, layer: int) -> dict[str, np.ndarray]:
    """Per block ``l >= layer`` and prompt: the attention and MLP parts of the steered and the natural increment,
    their components along the natural difference at read point ``l + 1``, norms, cosines between the steered
    and the natural part, and the exactness of the split (attention + MLP against the residual increment)."""
    assert base.residuals is not None and steered.residuals is not None and icl.residuals is not None
    assert base.attn_writes is not None and steered.attn_writes is not None and icl.attn_writes is not None
    assert base.mlp_writes is not None and steered.mlp_writes is not None and icl.mlp_writes is not None
    L, N, _ = base.attn_writes.shape
    ref = icl.residuals.astype(np.float64) - base.residuals.astype(np.float64)  # (L+1, N, d)
    U = unit_rows(ref)
    delta = steered.residuals.astype(np.float64) - base.residuals.astype(np.float64)
    dA = steered.attn_writes.astype(np.float64) - base.attn_writes.astype(np.float64)
    dM = steered.mlp_writes.astype(np.float64) - base.mlp_writes.astype(np.float64)
    nA = icl.attn_writes.astype(np.float64) - base.attn_writes.astype(np.float64)
    nM = icl.mlp_writes.astype(np.float64) - base.mlp_writes.astype(np.float64)
    keys = ["att_along", "mlp_along", "nat_att_along", "nat_mlp_along", "att_norm", "mlp_norm", "nat_att_norm", "nat_mlp_norm",
            "cos_att", "cos_mlp", "exactness", "increment_along", "natural_norm"]
    out = {k: np.full((L, N), np.nan) for k in keys}
    for l in range(layer, L):
        u = U[l + 1]
        out["att_along"][l] = along(dA[l], u)
        out["mlp_along"][l] = along(dM[l], u)
        out["nat_att_along"][l] = along(nA[l], u)
        out["nat_mlp_along"][l] = along(nM[l], u)
        out["att_norm"][l] = np.linalg.norm(dA[l], axis=-1)
        out["mlp_norm"][l] = np.linalg.norm(dM[l], axis=-1)
        out["nat_att_norm"][l] = np.linalg.norm(nA[l], axis=-1)
        out["nat_mlp_norm"][l] = np.linalg.norm(nM[l], axis=-1)
        out["cos_att"][l] = row_cosines(dA[l], nA[l])
        out["cos_mlp"][l] = row_cosines(dM[l], nM[l])
        inc = delta[l + 1] - delta[l]
        out["exactness"][l] = np.linalg.norm(dA[l] + dM[l] - inc, axis=-1) / np.maximum(np.linalg.norm(inc, axis=-1), 1e-12)
        out["increment_along"][l] = along(inc, u)
        out["natural_norm"][l] = np.linalg.norm(ref[l + 1], axis=-1)
    return out


def window_shares(att_along: np.ndarray, mlp_along: np.ndarray, blocks: list[int]) -> dict[str, np.ndarray]:
    """Per prompt: the sums of the aligning writes over ``blocks`` for each sublayer and attention's share of the
    total (NaN where the total is not positive)."""
    if not blocks:
        n = att_along.shape[1]
        return {"att_sum": np.full(n, np.nan), "mlp_sum": np.full(n, np.nan), "att_share": np.full(n, np.nan)}
    a = att_along[blocks].sum(0)
    m = mlp_along[blocks].sum(0)
    tot = a + m
    share = np.where(tot > 0, a / np.where(tot > 0, tot, 1.0), np.nan)
    return {"att_sum": a, "mlp_sum": m, "att_share": share}


def share_verdict(median_share: float | None, ci: tuple[float, float] | None) -> str:
    """The D38 reading of a window share: attention-written (median ≥ 0.6, CI above 0.5), MLP-written (≤ 0.4,
    CI below 0.5), else mixed."""
    if median_share is None or ci is None or not np.isfinite(median_share):
        return "undetermined"
    lo, hi = ci
    if median_share >= 0.6 and lo > 0.5:
        return "attention"
    if median_share <= 0.4 and hi < 0.5:
        return "mlp"
    return "mixed"


def head_writes(head_outputs_delta: np.ndarray, W: np.ndarray, head_dim: int, heads: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The pre-norm attention writes of the selected ``heads`` and of all heads, from per-head output changes
    ``(N, n_heads, head_dim_padded)`` and the output projection ``W`` ``(d, n_heads * head_dim)``; ``(N, d)`` each."""
    N, n_heads, _ = head_outputs_delta.shape
    Z = np.asarray(head_outputs_delta, dtype=np.float64)[:, :, :head_dim].reshape(N, n_heads * head_dim)
    all_w = Z @ W.T
    mask = np.zeros(n_heads * head_dim, dtype=bool)
    for h in heads:
        mask[h * head_dim:(h + 1) * head_dim] = True
    sel_w = (Z * mask) @ W.T
    return sel_w, all_w


def _rows_summary(X: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float) -> dict[str, list[float | None]]:
    """Row-wise medians with bootstrap CIs over the finite entries of each row (None for an empty row)."""
    X = np.asarray(X, dtype=np.float64)
    out: dict[str, list[float | None]] = {"median": [], "ci_low": [], "ci_high": []}
    for row in X:
        v = row[np.isfinite(row)]
        if v.size == 0:
            out["median"].append(None); out["ci_low"].append(None); out["ci_high"].append(None)
            continue
        b = bootstrap_median_ci_rows(v[None, :], rng, n_boot=n_boot, alpha=alpha)
        out["median"].append(float(b["median"][0])); out["ci_low"].append(float(b["low"][0])); out["ci_high"].append(float(b["high"][0]))
    return out


def _mean_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float) -> tuple[float, float]:
    """A percentile bootstrap interval of the mean of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def _scalar_summary(x: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    b = bootstrap_median_ci_rows(x[None, :], rng, n_boot=n_boot, alpha=alpha)
    return {"median": float(b["median"][0]), "ci_low": float(b["low"][0]), "ci_high": float(b["high"][0]), "n": int(x.size)}


# --------------------------------------------------------------------------- #
# The test
# --------------------------------------------------------------------------- #


class SourceTest:
    def __init__(self, cfg: SourceConfig, fv_run: Path, learned_run: Path, landmark_run: Path | None, run_id: str | None,
                 config_path: str | None) -> None:
        self.cfg = cfg
        self.fv_root, self.learned_root = Path(fv_run), Path(learned_run)
        self.landmark_root = Path(landmark_run) if landmark_run else None
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_source_{self.learned_root.name}"
        self.root = Path(cfg.output_dir) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = setup_logging(self.root / "log.txt")
        self.run_cfg, self.learned_meta = _load_run(self.learned_root)
        if cfg.device:
            from dataclasses import replace

            self.run_cfg = replace(self.run_cfg, model=replace(self.run_cfg.model, device=cfg.device))
        self.metadata_seed = cfg.seed
        self.prof = Profiler()
        self.metadata: dict[str, Any] = {
            "version": __version__, "config": asdict(cfg),
            "config_path": config_path, "run_id": run_id, "git": git_info(), "environment": environment_metadata(),
            "runs": {"learned": {"path": str(self.learned_root), "run_id": self.learned_meta.get("run_id")},
                     "fv": {"path": str(self.fv_root)}, "landmark": {"path": str(self.landmark_root) if self.landmark_root else None}},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.landmarks = read_json(self.landmark_root / "landmarks" / "landmarks.json") if self.landmark_root and (self.landmark_root / "landmarks" / "landmarks.json").exists() else None
        if self.landmark_root and self.landmarks is None and (self.landmark_root / "landmarks.json").exists():
            self.landmarks = read_json(self.landmark_root / "landmarks.json")

    def run(self) -> Path:
        set_torch_determinism(derive_seed(self.cfg.seed, "torch"))
        with self.prof.section("total"):
            self._run()
        self.metadata["timings_seconds"] = self.prof.timings()
        self.metadata["profile"] = self.prof.report()
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("profile (sections >= 1 s):\n%s", self.prof.table())
        self.log.info("done in %.1fs -> %s", self.metadata["timings_seconds"]["total"], self.root)
        return self.root

    def _run(self) -> None:
        with self.prof.section("model_load"):
            self.backend = ModelBackend(self.run_cfg.model, run_seed=self.run_cfg.seed)
        self.prof.backend = self.backend
        self.metadata["model"] = self.backend.metadata()
        names = sorted(p.name for p in (self.learned_root / "core" / "tasks").iterdir() if p.is_dir())
        summary: dict[str, Any] = {"tasks": {}, "skipped": {}}
        for name in names:
            inputs = _load_task(name, self.learned_root, self.fv_root, self.run_cfg)
            if inputs is None:
                summary["skipped"][name] = "no learned-vector selection in the learned run"
                continue
            with self.prof.section("task", name):
                summary["tasks"][name] = self._task(inputs)
        write_json(self.root / "core" / "summary.json", summary)
        self.log.info("summary:\n%s", format_summary(summary))

    # ------------------------------------------------------------------ #

    def _handover(self, name: str) -> int | None:
        if not self.landmarks:
            return None
        t = self.landmarks.get("tasks", {}).get(name)
        if not t:
            return None
        return ((t.get("handover") or {}).get("learned") or {}).get("read_point")

    def _task(self, inputs: TaskInputs) -> dict[str, Any]:
        cfg, name, pc = self.cfg, inputs.name, inputs.prompt_cfg
        backend = self.backend
        L, d = backend.n_layers, backend.hidden_size
        sel = inputs.learned_qual["selection"]
        layer = int(sel["layer"])
        la = inputs.learned_arrays
        layers = [int(l) for l in la["layers"]]
        i = layers.index(layer)
        direction = np.asarray(la["learned"][i], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        strength = alpha_from_calibration(inputs.learned_cal, layer) or {"alpha": float(la["learned_radii"][i]), "rho": 1.0, "source": "radius"}
        alpha = float(strength["alpha"])
        ev = inputs.evaluation
        zs = [zero_shot_prompt(pc, x) for x in ev]
        rng = rng_for(cfg.seed, "source_fewshot", name)
        fs = [few_shot_prompt(pc, ev, x, rng) for x in ev]
        brng = rng_for(cfg.seed, "source_bootstrap", name)
        inject = Intervention(layer, direction, alpha)

        with self.prof.section("passes", name):
            base = backend.run(zs, capture=True, capture_sublayers=True, capture_heads=True)
            steered = backend.run(zs, [inject], capture=True, capture_sublayers=True, capture_heads=True)
            icl = backend.run(fs, capture=True, capture_sublayers=True)
        if cfg.determinism_check and "determinism_check" not in self.metadata:
            again = backend.run(zs, [inject], capture=True, capture_sublayers=True)
            identical = bool(np.array_equal(again.residuals, steered.residuals) and np.array_equal(again.attn_writes, steered.attn_writes)
                             and np.array_equal(again.mlp_writes, steered.mlp_writes) and np.array_equal(again.logprob_sum, steered.logprob_sum))
            self.metadata["determinism_check"] = {"task": name, "layer": layer, "identical": identical}
            self.log.info("[%s] determinism check: a repeated steered pass with sublayer capture is %s", name, "bit-identical" if identical else "NOT identical")
            if not identical:
                raise RuntimeError("determinism check failed: a repeated steered pass differs")
        base_lp, steered_lp = base.logprob_per_token, steered.logprob_per_token
        full = float(np.mean(steered_lp - base_lp))
        dec = decompose(base, steered, icl, layer)
        handover = self._handover(name)
        self.log.info("[%s] layer %d alpha %.2f, full effect %+.3f nats/tok, hand-over read point %s; split exactness median %.2e (max %.2e)",
                      name, layer, alpha, full, handover, float(np.nanmedian(dec["exactness"])), float(np.nanmax(dec["exactness"])))

        # random directions at the same norm: the split of a perturbation carrying no task
        rand_keys = ("att_along", "mlp_along", "att_norm", "mlp_norm")
        rand: dict[str, list[np.ndarray]] = {k: [] for k in rand_keys}
        with self.prof.section("random", name):
            for j in range(cfg.n_controls):
                R = unit_vectors(rng_for(cfg.seed, "source_random", name, j), 1, d)[0]
                r = backend.run(zs, [Intervention(layer, R, alpha)], capture=True, capture_sublayers=True)
                dr = decompose(base, r, icl, layer)
                for k in rand_keys:
                    rand[k].append(dr[k])
        rand_mean = {k: np.nanmean(np.stack(v), axis=0) for k, v in rand.items()}  # (L, N), mean over controls

        result: dict[str, Any] = {
            "task": name, "layer": layer, "alpha": alpha, "strength": strength, "n_examples": len(ev), "n_layers": L,
            "full_effect": full, "handover_read_point": handover, "attention_output_normed": backend.attention_output_normed,
            "per_block": {}, "windows": {}, "heads": None, "removal": {}, "per_block_removal": {},
        }
        blocks = list(range(layer, L))
        result["per_block"]["blocks"] = blocks
        for k in ("att_along", "mlp_along", "nat_att_along", "nat_mlp_along", "att_norm", "mlp_norm", "nat_att_norm", "nat_mlp_norm",
                  "cos_att", "cos_mlp", "exactness", "increment_along", "natural_norm"):
            result["per_block"][k] = _rows_summary(dec[k][blocks], brng, cfg.n_boot, cfg.ci_alpha)
        for k in rand_keys:
            result["per_block"]["random_" + k] = _rows_summary(rand_mean[k][blocks], brng, cfg.n_boot, cfg.ci_alpha)

        # window shares
        def window(name_: str, bl: list[int]) -> dict[str, Any]:
            st = window_shares(dec["att_along"], dec["mlp_along"], bl)
            nat = window_shares(dec["nat_att_along"], dec["nat_mlp_along"], bl)
            rnd = window_shares(rand_mean["att_along"], rand_mean["mlp_along"], bl)
            out = {"blocks": bl}
            for lab, s in (("steered", st), ("natural", nat), ("random", rnd)):
                out[lab] = {k: _scalar_summary(v, brng, cfg.n_boot, cfg.ci_alpha) for k, v in s.items()}
                # the share of the summed medians (robust where per-prompt totals are small)
                a, m = float(np.nanmedian(s["att_sum"])), float(np.nanmedian(s["mlp_sum"]))
                out[lab]["share_of_medians"] = a / (a + m) if (a + m) > 0 else None
                sh = out[lab]["att_share"]
                out[lab]["verdict"] = share_verdict(sh["median"], (sh["ci_low"], sh["ci_high"]) if sh["ci_low"] is not None else None)
            return out

        if handover is not None:
            result["windows"]["to_handover"] = window("to_handover", [l for l in blocks if l + 1 <= handover])
            result["windows"]["after_handover"] = window("after_handover", [l for l in blocks if l + 1 > handover])
        result["windows"]["all"] = window("all", blocks)

        # the selected heads' share of the attention write (pre-norm; exact on pre-norm blocks)
        if inputs.fv_arrays is not None and "fv_heads" in inputs.fv_arrays and base.head_outputs is not None and steered.head_outputs is not None:
            heads_by_layer: dict[int, list[int]] = {}
            for lh in np.asarray(inputs.fv_arrays["fv_heads"]):
                heads_by_layer.setdefault(int(lh[0]), []).append(int(lh[1]))
            ref = icl.residuals.astype(np.float64) - base.residuals.astype(np.float64)
            U = unit_rows(ref)
            per_layer = {}
            sel_tot = np.zeros(len(ev)); all_tot = np.zeros(len(ev)); nsel = 0
            for l, hs in sorted(heads_by_layer.items()):
                if l < layer:
                    continue
                W = backend.attn_out[l].weight.detach().to("cpu").double().numpy()
                dz = steered.head_outputs[l] - base.head_outputs[l]
                sel_w, all_w = head_writes(dz, W, backend.head_dims[l], hs)
                s_al, a_al = along(sel_w, U[l + 1]), along(all_w, U[l + 1])
                per_layer[str(l)] = {"heads": hs, "selected_along": _scalar_summary(s_al, brng, cfg.n_boot, cfg.ci_alpha),
                                     "all_heads_along": _scalar_summary(a_al, brng, cfg.n_boot, cfg.ci_alpha)}
                sel_tot += s_al; all_tot += a_al; nsel += 1
            share = np.where(all_tot > 0, sel_tot / np.where(all_tot > 0, all_tot, 1.0), np.nan) if nsel else np.full(len(ev), np.nan)
            result["heads"] = {"n_selected": int(sum(len(v) for v in heads_by_layer.values())), "layers": sorted(heads_by_layer),
                               "exact": not backend.attention_output_normed, "per_layer": per_layer,
                               "selected_share_over_head_blocks": _scalar_summary(share, brng, cfg.n_boot, cfg.ci_alpha),
                               "selected_sum": _scalar_summary(sel_tot, brng, cfg.n_boot, cfg.ci_alpha),
                               "all_heads_sum": _scalar_summary(all_tot, brng, cfg.n_boot, cfg.ci_alpha)}

        # removals over the window to the hand-over (and, per block, the aligning component alone)
        ref = icl.residuals.astype(np.float64) - base.residuals.astype(np.float64)
        U = unit_rows(ref)
        dA = steered.attn_writes.astype(np.float64) - base.attn_writes.astype(np.float64)
        dM = steered.mlp_writes.astype(np.float64) - base.mlp_writes.astype(np.float64)

        def vectors(sub: str, variant: str, l: int) -> np.ndarray:
            X = dA[l] if sub == "att" else dM[l]
            if variant == "full":
                return X
            return along(X, U[l + 1])[:, None] * U[l + 1]

        def run_removal(edits: list[tuple[int, np.ndarray]]) -> np.ndarray:
            ivs = [inject] + [Intervention(l + 1, (-v).astype(np.float32), 1.0) for l, v in edits]
            r = backend.run(zs, ivs)
            return r.logprob_per_token - base_lp

        if handover is not None:
            wblocks = [l for l in blocks if l + 1 <= handover]
            with self.prof.section("removal", name):
                for sub in ("att", "mlp"):
                    for variant in cfg.variants:
                        edits = [(l, vectors(sub, variant, l)) for l in wblocks]
                        real = run_removal(edits)
                        ctl = []
                        for j in range(cfg.n_edit_controls):
                            crng = rng_for(cfg.seed, "source_removal_control", name, sub, variant, j)
                            cedits = []
                            for l, v in edits:
                                Rm = crng.standard_normal(v.shape)
                                Rm *= np.linalg.norm(v, axis=-1, keepdims=True) / np.maximum(np.linalg.norm(Rm, axis=-1, keepdims=True), 1e-12)
                                cedits.append((l, Rm))
                            ctl.append(run_removal(cedits))
                        C = np.stack(ctl)
                        test = paired_excess_test(-real, -C, brng, n_boot=cfg.n_boot)
                        key = f"{sub}_{variant}"
                        result["removal"][key] = {
                            "blocks": wblocks, "n_controls": cfg.n_edit_controls,
                            "retained": None if full == 0 else float(np.mean(real) / full),
                            "retained_ci": [c / full for c in _mean_ci(real, brng, cfg.n_boot, cfg.ci_alpha)] if full else None,
                            "random_retained": None if full == 0 else float(np.mean(C) / full),
                            "excess_vs_random": test.__dict__,
                            "edit_norm_median": float(np.median([np.linalg.norm(v, axis=-1) for _, v in edits])) if edits else None,
                        }
                        self.log.info("[%s] remove %s (%s) over blocks %s: retained %.2f (random %.2f, p %.3f)", name, sub, variant,
                                      f"{wblocks[0]}..{wblocks[-1]}" if wblocks else "-", result["removal"][key]["retained"] or float("nan"),
                                      result["removal"][key]["random_retained"] or float("nan"), test.p_value)
        if cfg.per_block:
            with self.prof.section("per_block_removal", name):
                for sub in ("att", "mlp"):
                    rows = []
                    for l in blocks:
                        real = run_removal([(l, vectors(sub, "along", l))])
                        rows.append({"block": l, "read_point": l + 1, "retained": None if full == 0 else float(np.mean(real) / full)})
                    result["per_block_removal"][sub] = rows
        return result


def format_summary(summary: dict[str, Any]) -> str:
    lines = []
    for name, t in summary.get("tasks", {}).items():
        w = t["windows"].get("to_handover")
        line = f"{name:20s} L{t['layer']:2d} hand-over {t['handover_read_point']!s:>4} effect {t['full_effect']:+.2f}"
        if w:
            st, nat, rnd = w["steered"], w["natural"], w["random"]
            line += (f" | to hand-over: attention share steered {st['att_share']['median']!s:>6} [{st['verdict']}] natural {nat['att_share']['median']!s:>6} [{nat['verdict']}]"
                     f" random {rnd['att_share']['median']!s:>6}")
        rm = t.get("removal", {})
        if rm:
            line += " | removal retained: " + ", ".join(f"{k} {v['retained']:.2f} (rnd {v['random_retained']:.2f}, p {v['excess_vs_random']['p_value']:.3f})"
                                                        for k, v in rm.items() if v["retained"] is not None)
        if t.get("heads"):
            line += f" | selected heads' share {t['heads']['selected_share_over_head_blocks']['median']!s}"
        lines.append(line)
    return "\n".join(lines)


def run_source(cfg: SourceConfig, fv_run: str | Path, learned_run: str | Path, landmark_run: str | Path | None = None,
               run_id: str | None = None, config_path: str | None = None) -> Path:
    return SourceTest(cfg, Path(fv_run), Path(learned_run), Path(landmark_run) if landmark_run else None, run_id, config_path).run()

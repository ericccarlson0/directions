"""The operator's natural direction at the operator token, added to max runs and read with the variable map
(docs/DECISIONS.md D58; D57 step 3).

The direction v[m] is the mean, over max prompts and their min twins, of the twin's state minus the max run's at the
operator token at read point m. The switch curve adds v[m] at the operator token of max runs at every read point,
beside the twin's whole state swapped in at the same site and random directions of v[m]'s norm. The steered map adds
v[m_inj] to min expressions shown as max and maps them with D45's swaps (sources steered the same way); D57's operator
swap becomes the unsteer swap, the same prompt without the direction.

    uv run directions opsteer --config configs/opsteer_qwen3_8b.yaml --run-id opsteer_qwen3_8b_seed20260916
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._version import __version__
from .config import OpsteerConfig
from .determinism import compare_arrays
from .model import Intervention, ModelBackend, SitePatch, environment_metadata, gpu_peak_memory, set_torch_determinism
from .runinfo import RejectionLog, git_info, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .prompts import Prompt
from .varmap import (DIGITS, SOURCE_TYPES, Expr, all_exprs, balanced_demos, bootstrap_rate, draw_source, operator_readouts,
                     plot_task, prediction, predictions, query_tokens, reading, render)

MAP_TYPES = SOURCE_TYPES["maxmin"]  # "op" is the unsteer swap here
DRAWN_TYPES = tuple(k for k in MAP_TYPES if k != "op")


def shown(e: Expr) -> Expr:
    """The expression as the prompt writes it: with ``max``."""
    return dataclasses.replace(e, op="max")


def twin(e: Expr) -> Expr:
    """The same digits under the other operator."""
    return dataclasses.replace(e, op="min" if e.op == "max" else "max")


def sample_opsteer(seed: int, n_dir: int, n_bases: int, n_shots: int, n_demo_pool: int,
                   rejections: RejectionLog | None = None) -> dict[str, Any]:
    """D58's draws: D57's demonstration pool; the direction set (max expressions, each with its own demonstrations);
    then the bases (min expressions shown as max, their digits outside the direction set), each with one source per
    drawn swap type (min expressions, D45's rules) and its demonstrations."""
    exprs = all_exprs("maxmin")
    order = rng_for(seed, "opsteer", "split").permutation(len(exprs))
    pool = [exprs[i] for i in order[:n_demo_pool]]
    rest = [exprs[i] for i in order[n_demo_pool:]]
    dir_set = [e for e in rest if e.op == "max"][:n_dir]
    dir_demos = [balanced_demos(pool, n_shots, rng_for(seed, "opsteer", "dir_demos", i)) for i in range(len(dir_set))]
    used = {(e.a, e.b, e.c, e.d) for e in dir_set}
    mins = [e for e in rest if e.op == "min"]
    bases, sources, demos, skipped = [], [], [], 0
    for base in mins:
        if len(bases) == n_bases:
            break
        if (base.a, base.b, base.c, base.d) in used:
            continue
        i = len(bases)
        srcs = {k: draw_source(k, base, mins, rng_for(seed, "opsteer", "source", i, skipped, k)) for k in DRAWN_TYPES}
        if any(v is None for v in srcs.values()):
            skipped += 1
            if rejections is not None:
                rejections.add("source", "maxmin", "no valid source of some type",
                               {"base": base.text(), "missing": [k for k, v in srcs.items() if v is None]})
            continue
        bases.append(base)
        sources.append(srcs)
        demos.append(balanced_demos(pool, n_shots, rng_for(seed, "opsteer", "demos", i)))
    info = {"n_expressions": len(exprs), "n_dir": len(dir_set), "n_bases": len(bases), "skipped_without_source": skipped}
    return {"dir_set": dir_set, "dir_demos": dir_demos, "bases": bases, "sources": sources, "demos": demos, "info": info}


def window(lo: np.ndarray, threshold: float = 0.5) -> int | None:
    """The last read point m with ``lo >= threshold`` at every read point 0..m (None if not even at 0)."""
    below = np.flatnonzero(np.asarray(lo) < threshold)
    if len(below) == 0:
        return len(lo) - 1
    return int(below[0]) - 1 if below[0] > 0 else None


def mean_readouts(rates: dict[str, Any], labels: list[str], read_points: list[int], from_rp: int) -> dict[str, Any]:
    """D58's readouts by the means (beside D45's lower bounds): the first read point with the unsteer swap's mean
    flip at least 0.5 at the answer token, the last at an operator token (from ``from_rp`` on), and the first with
    both sums' mean flips at least 0.5 at the answer token."""
    ans = len(labels) - 1
    op_cols = [j for j, lab in enumerate(labels) if lab == "op"]
    rps = np.asarray(read_points)

    def first(mask: np.ndarray) -> int | None:
        idx = np.flatnonzero(mask)
        return int(rps[idx[0]]) if len(idx) else None

    out: dict[str, Any] = {"R_O": None, "O_out": None, "G": None}
    if "op" in rates:
        m = np.asarray(rates["op"]["flip"]["mean"])
        out["R_O"] = first(m[:, ans] >= 0.5)
        at_op = (m[:, op_cols] >= 0.5).any(axis=1) & (rps >= from_rp)
        out["O_out"] = int(rps[np.flatnonzero(at_op)[-1]]) if at_op.any() else None
    if "S1" in rates and "S2" in rates:
        both = (np.asarray(rates["S1"]["flip"]["mean"])[:, ans] >= 0.5) & (np.asarray(rates["S2"]["flip"]["mean"])[:, ans] >= 0.5)
        out["G"] = first(both)
    return out


class OpSteer:
    def __init__(self, cfg: OpsteerConfig, run_id: str | None = None, config_path: str | None = None) -> None:
        self.cfg = cfg
        slug = cfg.model.name.split("/")[-1].lower() if cfg.model.backend == "hf" else "toy"
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_opsteer_{slug}_seed{cfg.seed}"
        self.root = Path(cfg.output_dir) / run_id
        if self.root.exists():
            raise FileExistsError(f"run directory already exists: {self.root}")
        (self.root / "core").mkdir(parents=True)
        (self.root / "figures").mkdir()
        self.log = setup_logging(self.root / "log.txt")
        self.rejections = RejectionLog(self.root / "rejections.jsonl")
        self.metadata: dict[str, Any] = {
            "version": __version__, "config": asdict(cfg), "config_path": config_path, "run_id": run_id, "seed": cfg.seed,
            "git": git_info(), "environment": environment_metadata(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run(self) -> Path:
        cfg = self.cfg
        set_torch_determinism(derive_seed(cfg.seed, "torch"))
        self.backend = ModelBackend(cfg.model, run_seed=cfg.seed)
        self.metadata["model"] = self.backend.metadata()
        self.tid = self._digit_ids()
        L = self.backend.n_layers
        if cfg.inject_read_point > L:
            raise ValueError(f"inject_read_point {cfg.inject_read_point} beyond the model's {L} read points")
        draw = sample_opsteer(cfg.seed, cfg.n_dir, cfg.n_bases, cfg.n_shots, cfg.n_demo_pool, self.rejections)
        self.metadata["sampling"] = draw["info"]
        t0 = time.time()
        direction = self._direction(draw["dir_set"], draw["dir_demos"])
        self.metadata.setdefault("timings_seconds", {})["direction"] = round(time.time() - t0, 1)
        t0 = time.time()
        setup = self._setup(draw)
        curve = self._curve(setup, direction["v"])
        self.metadata["timings_seconds"]["curve"] = round(time.time() - t0, 1)
        t0 = time.time()
        steered = self._map(setup, direction["v"])
        self.metadata["timings_seconds"]["map"] = round(time.time() - t0, 1)
        arrays = {"direction": direction.pop("v").cpu().numpy().astype(np.float32), **curve.pop("_arrays"), **steered.pop("_arrays")}
        np.savez_compressed(self.root / "core" / "opsteer_arrays.npz", **arrays)
        result = {"direction": direction, "curve": curve, "map": steered, "labels": setup["labels"],
                  "inject_read_point": cfg.inject_read_point}
        write_json(self.root / "core" / "opsteer.json", result)
        summary = {"W_steer": curve["W_steer"], "W_swap": curve["W_swap"], "null_switch_max_hi": curve["null_switch_max_hi"],
                   "switch_at_inject": {a: curve["arms"][a]["switch"]["mean"][cfg.inject_read_point] for a in ("steer", "swap")},
                   "map": {k: steered[k] for k in ("accuracy", "n_kept", "operator", "operator_means") if k in steered}}
        write_json(self.root / "core" / "summary.json", summary)
        if cfg.figures:
            try:
                self._figures(result)
            except Exception as exc:  # a figure never fails a run
                self.log.warning("figure failed: %s", exc)
        self.metadata["gpu_peak_memory_gib"] = gpu_peak_memory()
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("done -> %s", self.root)
        return self.root

    def _digit_ids(self) -> np.ndarray:
        ids = []
        for k in range(10):
            enc = self.backend.tokenizer.encode(str(k), add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"the digit {k} is not a single token on this tokenizer: {enc}")
            ids.append(enc[0])
        return np.array(ids, dtype=np.int64)

    def _op_position(self, prompt: Prompt, query: Expr) -> tuple[int, list[int], list[int], list[str]]:
        """The operator token (the last token of the operator word) and the query's positions, ids and labels."""
        pos, ids, labels = query_tokens(self.backend, prompt, query, "op")
        j = max(i for i, lab in enumerate(labels) if lab == "op")
        return pos[j], pos, ids, labels

    def _steer(self, v: torch.Tensor, m: int, op_pos: np.ndarray) -> Intervention:
        return Intervention(m, v[m].cpu().numpy().astype(np.float32), 1.0, np.asarray(op_pos, dtype=np.int64))

    # ------------------------------------------------------------------ #

    def _direction(self, dir_set: list[Expr], dir_demos: list[list[Expr]]) -> dict[str, Any]:
        """v[m]: the mean over the direction set of the min twin's state minus the max run's at the operator token."""
        backend, log = self.backend, self.log
        maxp = [render(d, e) for d, e in zip(dir_demos, dir_set)]
        minp = [render(d, twin(e)) for d, e in zip(dir_demos, dir_set)]
        pos, keep = [], []
        for i, e in enumerate(dir_set):
            p_max, *_ = self._op_position(maxp[i], e)
            p_min, *_ = self._op_position(minp[i], twin(e))
            keep.append(p_max == p_min)
            pos.append(p_max)
            if p_max != p_min:
                self.rejections.add("align", "direction", "the twin's operator token is elsewhere", {"expr": e.text()})
        idx = np.flatnonzero(keep)
        pos = np.asarray(pos, dtype=np.int64)[idx][:, None]
        h_max = backend.capture_sites([maxp[i] for i in idx], pos)[:, :, 0].float()  # (n, L+1, d)
        h_min = backend.capture_sites([minp[i] for i in idx], pos)[:, :, 0].float()
        diff = h_min - h_max
        v = diff.mean(dim=0)  # (L+1, d) float32
        norm = v.norm(dim=-1)
        state = h_max.norm(dim=-1).mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(diff, v[None].expand_as(diff), dim=-1).mean(dim=0)
        step = max(1, len(norm) // 8)
        log.info("direction from %d pairs, every %d read points: norm %s; mean cosine of the pairs' differences with it %s",
                 len(idx), step, [round(float(x), 2) for x in norm[::step]], [round(float(x), 3) for x in cos[::step]])
        del h_max, h_min, diff
        return {"v": v, "n_pairs": len(idx), "norm": norm.cpu().numpy().tolist(),
                "state_norm": state.cpu().numpy().tolist(), "ratio": (norm / state).cpu().numpy().tolist(),
                "pair_cosine": cos.cpu().numpy().tolist()}

    def _setup(self, draw: dict[str, Any]) -> dict[str, Any]:
        """Prompts, positions and the unsteered runs: each base shown as max, its natural min twin, each drawn source
        shown as max; aligned token by token (only the digits may differ, and the operator for the twin)."""
        backend = self.backend
        bases, sources, demos = draw["bases"], draw["sources"], draw["demos"]
        base_prompts = [render(demos[i], shown(b)) for i, b in enumerate(bases)]
        twin_prompts = [render(demos[i], b) for i, b in enumerate(bases)]
        _, pos0, _, labels = self._op_position(base_prompts[0], shown(bases[0]))
        P = len(pos0)
        positions = np.zeros((len(bases), P), dtype=np.int64)
        op_pos = np.zeros(len(bases), dtype=np.int64)
        keep = np.ones(len(bases), dtype=bool)
        src_prompts: dict[str, list[Prompt | None]] = {k: [] for k in DRAWN_TYPES}
        for i, b in enumerate(bases):
            op_i, pos, ids, _ = self._op_position(base_prompts[i], shown(b))
            _, tpos, tids, _ = self._op_position(twin_prompts[i], b)
            ok = len(pos) == P and tpos == pos and all(x == y for x, y, lab in zip(ids, tids, labels) if lab != "op")
            if ok:
                positions[i], op_pos[i] = pos, op_i
            for k in DRAWN_TYPES:
                sp = render(demos[i], shown(sources[i][k]))
                _, spos, sids, _ = self._op_position(sp, shown(sources[i][k]))
                if not (ok and spos == pos and all(x == y for x, y, lab in zip(ids, sids, labels) if lab not in DIGITS)):
                    ok = False
                src_prompts[k].append(sp)
            if not ok:
                keep[i] = False
                self.rejections.add("align", "maxmin", "base, twin or source misaligned", {"base": b.text()})
        tid = self.tid
        plain = backend.run(base_prompts, token_ids=tid)  # the max runs, unsteered
        natural = backend.run(twin_prompts, token_ids=tid)  # the min twins
        max_ok = plain.query_argmax == tid[[shown(b).Y for b in bases]]
        min_ok = natural.query_argmax == tid[[b.Y for b in bases]]
        self.log.info("%d bases (%d aligned); unsteered max accuracy %.3f, natural min accuracy %.3f, both %d", len(bases),
                      int(keep.sum()), float(max_ok[keep].mean()), float(min_ok[keep].mean()), int((keep & max_ok & min_ok).sum()))
        return {"bases": bases, "sources": sources, "base_prompts": base_prompts, "twin_prompts": twin_prompts,
                "src_prompts": src_prompts, "positions": positions, "op_pos": op_pos, "keep": keep, "labels": labels,
                "plain_argmax": plain.query_argmax, "max_ok": max_ok, "min_ok": min_ok,
                "accuracy": {"unsteered_max": float(max_ok[keep].mean()), "natural_min": float(min_ok[keep].mean())}}

    def _curve(self, s: dict[str, Any], v: torch.Tensor) -> dict[str, Any]:
        """At every read point m, on the bases whose max and min versions are both answered: v[m] added at the
        operator token, the twin's whole state swapped in there, and ``n_null`` random directions of v[m]'s norm."""
        cfg, backend, log, tid = self.cfg, self.backend, self.log, self.tid
        rows = np.flatnonzero(s["keep"] & s["max_ok"] & s["min_ok"] if cfg.gate else s["keep"])
        n, L = len(rows), backend.n_layers
        M = L + 1
        prompts = [s["base_prompts"][i] for i in rows]
        op_pos = s["op_pos"][rows]
        y_min = tid[[s["bases"][i].Y for i in rows]]
        y_max = tid[[shown(s["bases"][i]).Y for i in rows]]
        out_am: dict[str, np.ndarray] = {}
        out_am["steer"] = np.stack([backend.run(prompts, interventions=[self._steer(v, m, op_pos)], token_ids=tid).query_argmax
                                    for m in range(M)], axis=1)  # (n, M)
        twin_states = backend.capture_sites([s["twin_prompts"][i] for i in rows], op_pos[:, None])  # (n, M, 1, d)
        ex, mm = np.meshgrid(np.arange(n), np.arange(M), indexing="ij")
        ex, mm = ex.ravel(), mm.ravel()
        dev = twin_states.device
        vals = twin_states[torch.as_tensor(ex, device=dev), torch.as_tensor(mm, device=dev), 0]
        res = backend.run([prompts[e] for e in ex], site_patch=SitePatch(mm, op_pos[ex], vals), token_ids=tid)
        out_am["swap"] = res.query_argmax.reshape(n, M)
        del twin_states, vals
        norms = v.norm(dim=-1).cpu().numpy()
        for k in range(cfg.n_null):
            r = rng_for(cfg.seed, "opsteer", "null", k).standard_normal(v.shape).astype(np.float32)
            r *= (norms / np.maximum(np.linalg.norm(r, axis=1), 1e-12))[:, None]
            rv = torch.as_tensor(r)
            out_am[f"null{k}"] = np.stack([backend.run(prompts, interventions=[self._steer(rv, m, op_pos)], token_ids=tid).query_argmax
                                           for m in range(M)], axis=1)
        seed = derive_seed(cfg.seed, "opsteer", "curve", "bootstrap")
        arms: dict[str, Any] = {}
        for i, (a, am) in enumerate(out_am.items()):
            sw, st = am == y_min[:, None], am == y_max[:, None]
            arms[a] = {"switch": bootstrap_rate(sw, np.random.default_rng(seed + 3 * i), cfg.n_boot, cfg.alpha),
                       "stay": bootstrap_rate(st, np.random.default_rng(seed + 3 * i + 1), cfg.n_boot, cfg.alpha),
                       "other": bootstrap_rate(~sw & ~st, np.random.default_rng(seed + 3 * i + 2), cfg.n_boot, cfg.alpha)}
        diff = (out_am["swap"] == y_min[:, None]).astype(np.float64) - (out_am["steer"] == y_min[:, None])
        gap = bootstrap_rate(diff, np.random.default_rng(seed + 99), cfg.n_boot, cfg.alpha)
        W_steer, W_swap = window(arms["steer"]["switch"]["lo"]), window(arms["swap"]["switch"]["lo"])
        null_hi = max((float(np.max(arms[f"null{k}"]["switch"]["hi"])) for k in range(cfg.n_null)), default=None)
        log.info("switch curve on %d bases: W_steer %s, W_swap %s; null switch upper bound at most %s", n, W_steer, W_swap, null_hi)
        log.info("switch rate, direction / swap: %s", [(m, round(float(arms["steer"]["switch"]["mean"][m]), 2),
                                                        round(float(arms["swap"]["switch"]["mean"][m]), 2)) for m in range(0, M, 2)])
        return {"n": int(n), "read_points": list(range(M)), "arms": arms, "swap_minus_steer": gap, "W_steer": W_steer,
                "W_swap": W_swap, "null_switch_max_hi": null_hi,
                "_arrays": {"curve__rows": rows, **{f"curve__{a}": am.astype(np.int32) for a, am in out_am.items()}}}

    def _map(self, s: dict[str, Any], v: torch.Tensor) -> dict[str, Any]:
        """D45's map of the steered runs at ``inject_read_point``: drawn sources steered the same way, the unsteer swap
        for the operator."""
        cfg, backend, log, tid = self.cfg, self.backend, self.log, self.tid
        m0 = cfg.inject_read_point
        bases, sources, positions, op_pos, labels = s["bases"], s["sources"], s["positions"], s["op_pos"], s["labels"]
        P = positions.shape[1]
        keep = s["keep"]
        steer_all = self._steer(v, m0, op_pos)
        base_run = backend.run(s["base_prompts"], interventions=[steer_all], token_ids=tid)
        base_ok = keep & (base_run.query_argmax == tid[[b.Y for b in bases]])
        accuracy = float(base_ok[keep].mean())
        if not cfg.gate:
            base_ok = keep.copy()
        log.info("steered at read point %d: the min answer on %.3f of the aligned bases", m0, accuracy)
        src_ok = {"op": s["max_ok"]}
        for k in DRAWN_TYPES:
            r = backend.run(s["src_prompts"][k], interventions=[steer_all], token_ids=tid)
            src_ok[k] = r.query_argmax == tid[[sources[i][k].Y for i in range(len(bases))]]
        type_rows = {k: np.flatnonzero(base_ok & (src_ok[k] if cfg.gate else True))[: cfg.max_per_type] for k in MAP_TYPES}
        log.info("bases kept per type %s", {k: len(x) for k, x in type_rows.items()})
        L = backend.n_layers
        read_points = list(range(L + 1)) if cfg.read_points is None else [m for m in cfg.read_points if 0 <= m <= L]
        M = len(read_points)

        def steer_rows(rows: np.ndarray) -> Intervention:
            return Intervention(m0, steer_all.vectors, 1.0, op_pos[rows])

        if cfg.identity_check:
            rows = np.flatnonzero(base_ok)
            bp = [s["base_prompts"][i] for i in rows]
            own = backend.capture_sites(bp, positions[rows], interventions=[steer_rows(rows)])
            mid = read_points[M // 2]
            ref = backend.run(bp, interventions=[steer_rows(rows)], token_ids=tid)
            for j in (0, P // 2, P - 1):
                got = backend.run(bp, interventions=[steer_rows(rows)], token_ids=tid,
                                  site_patch=SitePatch(np.full(len(rows), mid), positions[rows, j], own[:, mid, j]))
                same = bool(np.array_equal(got.token_logprobs, ref.token_logprobs) and np.array_equal(got.query_argmax, ref.query_argmax))
                self.metadata.setdefault("identity_check", {})[f"read_point_{mid}/token_{j}"] = same
                if not same:
                    raise RuntimeError(f"identity check failed: a steered run's own state at read point {mid}, token {j} changed it")
            del own
        out_rates: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}
        n_kept: dict[str, int] = {}
        bseed = derive_seed(cfg.seed, "opsteer", "map", "bootstrap")
        for k in MAP_TYPES:
            rows = type_rows[k]
            n = len(rows)
            n_kept[k] = n
            if n < 2:
                log.info("%s: fewer than two bases kept; no map", k)
                continue
            base_rows = [s["base_prompts"][i] for i in rows]
            if k == "op":  # the unsteer swap: the same prompt without the direction
                src_states = backend.capture_sites(base_rows, positions[rows])
            else:
                src_states = backend.capture_sites([s["src_prompts"][k][i] for i in rows], positions[rows],
                                                   interventions=[steer_rows(rows)])
            ex, mm, jj = np.meshgrid(np.arange(n), np.arange(M), np.arange(P), indexing="ij")
            ex, mm, jj = ex.ravel(), mm.ravel(), jj.ravel()
            rp = np.asarray(read_points)[mm]
            dev = src_states.device
            values = src_states[torch.as_tensor(ex, device=dev), torch.as_tensor(rp, device=dev), torch.as_tensor(jj, device=dev)]
            prompts = [base_rows[e] for e in ex]
            patch = SitePatch(rp, positions[rows][ex, jj], values)
            steer = Intervention(m0, steer_all.vectors, 1.0, op_pos[rows][ex])
            res = backend.run(prompts, site_patch=patch, interventions=[steer], token_ids=tid)
            if cfg.determinism_check and "swap" not in self.metadata.get("determinism_check", {}):
                bs = backend.cfg.batch_size
                sub = SitePatch(rp[:bs], positions[rows][ex[:bs], jj[:bs]], values[:bs])
                sub_steer = Intervention(m0, steer_all.vectors, 1.0, op_pos[rows][ex[:bs]])
                a = backend.run(prompts[:bs], site_patch=sub, interventions=[sub_steer], token_ids=tid)
                b = backend.run(prompts[:bs], site_patch=sub, interventions=[sub_steer], token_ids=tid)
                cmp = compare_arrays({"lp": a.token_logprobs, "am": a.query_argmax}, {"lp": b.token_logprobs, "am": b.query_argmax})
                self.metadata.setdefault("determinism_check", {})["swap"] = {"identical": cmp["identical"]}
                if not cmp["identical"]:
                    raise RuntimeError("determinism check failed: a repeated swap batch differs")
            del values, src_states
            am = res.query_argmax.reshape(n, M, P)
            pred = np.array([shown(bases[i]).Y if k == "op" else prediction(k, bases[i], sources[i][k]) for i in rows])
            flip = am == tid[pred][:, None, None]
            change = am != base_run.query_argmax[rows][:, None, None]
            rates: dict[str, Any] = {"n": n, "flip": bootstrap_rate(flip, np.random.default_rng(bseed + 1), cfg.n_boot, cfg.alpha),
                                     "change": bootstrap_rate(change, np.random.default_rng(bseed + 2), cfg.n_boot, cfg.alpha)}
            if k == "both":
                hyp = {h: np.array([predictions(bases[i], sources[i][k])[h] for i in rows])
                       for h in predictions(bases[rows[0]], sources[rows[0]][k])}
                rates["hypotheses"] = {h: bootstrap_rate(am == tid[x][:, None, None], np.random.default_rng(bseed + 4), cfg.n_boot, cfg.alpha)
                                       for h, x in hyp.items()}
            out_rates[k] = rates
            arrays[f"map__{k}__flip"], arrays[f"map__{k}__argmax"], arrays[f"map__{k}__bases"] = flip, am.astype(np.int32), rows
            log.info("%s: %d bases, %d swaps; max flip rate %.2f, at the answer token's last read point %.2f", k, n, len(prompts),
                     float(rates["flip"]["mean"].max()), float(rates["flip"]["mean"][-1, -1]))
        if not out_rates:
            log.info("no swap type kept two bases: no steered map")
            return {"read_points": read_points, "labels": labels, "accuracy": accuracy, **s["accuracy"], "n_kept": n_kept,
                    "_arrays": arrays}
        read = reading(out_rates, "maxmin", cfg.present_flip, cfg.value_change)
        operator = operator_readouts(read, labels, read_points) if "present_O" in read else None
        means = mean_readouts(out_rates, labels, read_points, m0)
        log.info("the added operator: %s; by the means: %s", operator, means)
        return {"read_points": read_points, "labels": labels, "accuracy": accuracy, **s["accuracy"], "n_kept": n_kept,
                "rates": out_rates, "reading": read, "operator": operator, "operator_means": means,
                "examples": [{"base": shown(b).text(), "Y_min": b.Y, "Y_max": shown(b).Y,
                              "sources": {k: shown(sources[i][k]).text() for k in DRAWN_TYPES}} for i, b in enumerate(bases)],
                "_arrays": arrays}

    def _figures(self, result: dict[str, Any]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mc = self.cfg.model
        name = mc.name.split("/")[-1] if mc.backend == "hf" else f"toy {mc.toy.family}"
        c = result["curve"]
        rps = np.asarray(c["read_points"])
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        colours = {"swap": "#8a8985", "steer": "#2a78d6"}
        for a, lab in (("swap", "twin's state swapped in"), ("steer", "direction v[m] added")):
            r = c["arms"][a]["switch"]
            ax.fill_between(rps, r["lo"], r["hi"], color=colours[a], alpha=0.18, lw=0)
            ax.plot(rps, r["mean"], color=colours[a], lw=1.6, label=lab)
        for k in range(self.cfg.n_null):
            ax.plot(rps, c["arms"][f"null{k}"]["switch"]["mean"], color="#d9a521", lw=1, ls="--",
                    label="random direction, same norm" if k == 0 else None)
        ax.axvline(self.cfg.inject_read_point, color="#0b0b0b", lw=0.6, ls=":")
        ax.set_xlabel("read point at the operator token", fontsize=8)
        ax.set_ylabel("answer switched to min", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"max runs, min direction at the operator token ({name}; {c['n']} bases)", fontsize=8.5, loc="left")
        fig.savefig(self.root / "figures" / "curve.png", dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(fig)
        m = result["map"]
        if "rates" not in m:
            return
        plot_task({"task": "maxmin", "template": f"max(a+b, c+d) = , min direction added at the operator token, read point "
                   f"{result['inject_read_point']}", "labels": m["labels"], "read_points": m["read_points"], "rates": m["rates"],
                   "reading": m["reading"], "n_kept": m["n_kept"], "accuracy": m["accuracy"]},
                  self.root / "figures" / "map.png", name, op_title="unsteer (the same prompt without the direction)\nanswer = the max answer")


def run_opsteer(cfg: OpsteerConfig, run_id: str | None = None, config_path: str | None = None) -> Path:
    return OpSteer(cfg, run_id=run_id, config_path=config_path).run()

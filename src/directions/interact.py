"""Where control becomes computation: the operator and the operands changed independently and together
(docs/DECISIONS.md D61).

For a base expression and a pair of factors (X, Z), four prompts share the base's demonstrations: the base (00), X
changed (10), Z changed (01) and both changed (11). While the two factors are carried separately, the state of the
joint change is the sum of the single changes, h11 = h10 + h01 - h00; once they are combined it is not. At every read
point the additive state is swapped into the base run at every query token, and the read point at which it stops
giving 11's answer is where the two have been combined. Three pairs: the operator x both sums (the test), the first x
the second sum (a pair the comparison must combine), and the operator x digits that keep both sums (a pair that
should not combine: the null for the method).

    uv run directions interact --config configs/interact_qwen3_8b.yaml --run-id interact_qwen3_8b_seed20260916
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._version import __version__
from .config import InteractConfig
from .determinism import compare_arrays
from .model import (
    ModelBackend,
    SitePatch,
    environment_metadata,
    gpu_peak_memory,
    set_torch_determinism,
)
from .opset import GROUP_LABELS, GROUPS, token_groups
from .opsteer import op_position, twin
from .runinfo import RejectionLog, git_info, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .varmap import DIGITS, Expr, all_exprs, balanced_demos, bootstrap_rate, render

PAIRS = ("OS", "S1S2", "OD")
PAIR_LABELS = {"OS": "operator x both sums", "S1S2": "first sum x second sum", "OD": "operator x digits (sums kept)"}
PATCHES = ("joint", "additive") + tuple(f"add_{g}" for g in GROUPS)
CATS = ("11", "10", "01", "00", "other")  # in this order: an answer that is 11's counts as 11 (O x D: 11's is 10's)


def factor_sources(pair: str, base: Expr, rest: list[Expr], pool: set[Expr], rng: np.random.Generator
                   ) -> tuple[Expr, Expr, Expr] | None:
    """The prompts 10, 01 and 11 of ``pair`` for ``base`` (D61), from the expressions outside the demonstration pool;
    every prompt with a changed operator must be outside it too. None when the base has no such sources."""
    t = twin(base)
    if pair in ("OS", "OD") and t in pool:
        return None
    if pair == "OS":
        cands = [c for c in rest if c.op == base.op and c.S1 != base.S1 and c.S2 != base.S2 and twin(c) not in pool
                 and len({base.Y, t.Y, c.Y, twin(c).Y}) == 4]
        if not cands:
            return None
        c = cands[int(rng.integers(len(cands)))]
        return t, c, twin(c)
    if pair == "OD":
        cands = [c for c in rest if c.op == base.op and c.S1 == base.S1 and c.S2 == base.S2
                 and (c.a, c.b) != (base.a, base.b) and (c.c, c.d) != (base.c, base.d) and twin(c) not in pool]
        if not cands:
            return None
        c = cands[int(rng.integers(len(cands)))]
        return t, c, twin(c)
    if pair == "S1S2":
        c1s = [c for c in rest if c.op == base.op and (c.c, c.d) == (base.c, base.d) and c.S1 != base.S1]
        c2s = [c for c in rest if c.op == base.op and (c.a, c.b) == (base.a, base.b) and c.S2 != base.S2]
        combos = []
        for c1 in c1s:
            for c2 in c2s:
                j = Expr(c1.a, c1.b, c2.c, c2.d, op=base.op)
                # 11's answer distinct from the other three (pairwise distinct answers are impossible under max and
                # min: one bracket's change always leaves the answer where another change puts it; D61)
                if j.S1 != j.S2 and j not in pool and j.Y not in {base.Y, c1.Y, c2.Y}:
                    combos.append((c1, c2, j))
        if not combos:
            return None
        return combos[int(rng.integers(len(combos)))]
    raise ValueError(f"unknown factor pair {pair!r}")


def draw_interact(seed: int, n_bases: int, n_shots: int, n_demo_pool: int, pairs: list[str],
                  rejections: RejectionLog | None = None) -> dict[str, Any]:
    """D61's draws: D57's split and demonstration pool; bases in D57's order, each with the three prompts of every
    pair and its own balanced demonstrations; a base without the sources of some pair is skipped and counted."""
    exprs = all_exprs("maxmin")
    order = rng_for(seed, "varmap", "maxmin", "split").permutation(len(exprs))  # D57's split
    pool_list = [exprs[i] for i in order[:n_demo_pool]]
    rest = [exprs[i] for i in order[n_demo_pool:]]
    pool = set(pool_list)
    bases, sources, demos, skipped = [], [], [], 0
    for base in rest:
        if len(bases) == n_bases:
            break
        i = len(bases)
        srcs = {p: factor_sources(p, base, rest, pool, rng_for(seed, "interact", "source", i, skipped, p)) for p in pairs}
        if any(v is None for v in srcs.values()):
            skipped += 1
            if rejections is not None:
                rejections.add("source", "maxmin", "no sources for some pair",
                               {"base": base.text(), "missing": [p for p, v in srcs.items() if v is None]})
            continue
        bases.append(base)
        sources.append(srcs)
        demos.append(balanced_demos(pool_list, n_shots, rng_for(seed, "interact", "demos", i)))
    return {"bases": bases, "sources": sources, "demos": demos,
            "info": {"n_expressions": len(exprs), "n_bases": len(bases), "skipped_without_sources": skipped}}


def conversion_depth(lo: np.ndarray, hi: np.ndarray, read_points: list[int], threshold: float) -> dict[str, Any]:
    """K: the first read point at which the additive patch fails (its share's upper bound below ``threshold``), and
    whether it held (lower bound at least ``threshold``) at the first read point."""
    fails = np.flatnonzero(np.asarray(hi) < threshold)
    return {"K": int(read_points[fails[0]]) if len(fails) else None, "holds_at_first": bool(np.asarray(lo)[0] >= threshold)}


def reading(K: dict[str, int | None], localization: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """D61's reading from the conversion depths of the three pairs."""
    k_os, k_s, k_od = K.get("OS"), K.get("S1S2"), K.get("OD")
    if k_os is None:
        main = "never (the operator stays additively separable from the operands)"
    elif k_s is None:
        main = "undecided (the operands' pair never fails)"
    elif abs(k_os - k_s) <= 2:
        main = "where the operands are combined"
    elif k_os <= k_s - 3:
        main = "earlier than the operands"
    else:
        main = "later than the operands"
    null_ok = not (k_od is not None and k_os is not None and k_od <= k_os + 2)
    return {"K": K, "control_combined": main, "null_clear": null_ok, "localization": localization or {}}


class Interaction:
    def __init__(self, cfg: InteractConfig, run_id: str | None = None, config_path: str | None = None) -> None:
        self.cfg = cfg
        slug = cfg.model.name.split("/")[-1].lower() if cfg.model.backend == "hf" else "toy"
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_interact_{slug}_seed{cfg.seed}"
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
        self.read_points = list(range(L + 1)) if cfg.read_points is None else [m for m in cfg.read_points if 0 <= m <= L]
        draw = draw_interact(cfg.seed, cfg.n_bases, cfg.n_shots, cfg.n_demo_pool, cfg.pairs, self.rejections)
        self.metadata["sampling"] = draw["info"]
        result: dict[str, Any] = {"read_points": self.read_points, "patches": list(PATCHES), "categories": list(CATS),
                                  "pairs": {}}
        arrays: dict[str, np.ndarray] = {}
        for pair in cfg.pairs:
            t0 = time.time()
            res, arr = self._pair(pair, draw)
            result["pairs"][pair] = res
            arrays.update({f"{pair}__{k}": v for k, v in arr.items()})
            self.metadata.setdefault("timings_seconds", {})[pair] = round(time.time() - t0, 1)
        K = {p: r.get("K") for p, r in result["pairs"].items()}
        loc = {}
        if K.get("OS") is not None and "OS" in result["pairs"]:
            r = result["pairs"]["OS"]
            for m in range(K["OS"] - 2, K["OS"] + 3):
                if m in self.read_points:
                    i = self.read_points.index(m)
                    shares = {g: float(r["rates"][f"add_{g}"]["11"]["mean"][i]) for g in GROUPS}
                    loc[str(m)] = {"shares": shares, "lowest": min(shares, key=shares.get)}
        result["reading"] = reading(K, loc)
        self.log.info("conversion depths %s; the control is combined %s; null clear %s; localization %s", K,
                      result["reading"]["control_combined"], result["reading"]["null_clear"],
                      {m: v["lowest"] for m, v in loc.items()})
        write_json(self.root / "core" / "interact.json", result)
        np.savez_compressed(self.root / "core" / "interact_arrays.npz", **arrays)
        write_json(self.root / "core" / "summary.json",
                   {"reading": result["reading"], "n": {p: r["n"] for p, r in result["pairs"].items()}})
        if cfg.figures:
            try:
                self._figure(result)
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

    # ------------------------------------------------------------------ #

    def _pair(self, pair: str, draw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        cfg, backend, log, tid = self.cfg, self.backend, self.log, self.tid
        bases, demos = draw["bases"], draw["demos"]
        quads = [(b, *draw["sources"][i][pair]) for i, b in enumerate(bases)]  # 00, 10, 01, 11
        prompts = [[render(demos[i], e) for i, e in enumerate(col)] for col in zip(*quads)]
        _, pos0, _, labels = op_position(backend, prompts[0][0], bases[0])
        P = len(pos0)
        positions = np.zeros((len(bases), P), dtype=np.int64)
        keep = np.zeros(len(bases), dtype=bool)
        for i in range(len(bases)):
            toks = [op_position(backend, prompts[k][i], quads[i][k]) for k in range(4)]
            _, pos, ids, lab = toks[0]
            same = len(pos) == P and lab == labels and all(
                t[1] == pos and all(x == y for x, y, la in zip(ids, t[2], labels) if la != "op" and la not in DIGITS)
                for t in toks[1:])
            if same:
                positions[i], keep[i] = pos, True
            else:
                self.rejections.add("align", pair, "the four prompts misaligned", {"base": bases[i].text()})
        Y = np.array([[e.Y for e in q] for q in quads])  # (n, 4): 00, 10, 01, 11
        answered = np.stack([backend.run(prompts[k], token_ids=tid).query_argmax == tid[Y[:, k]] for k in range(4)], axis=1)
        ok = keep & answered.all(axis=1)
        rows = np.flatnonzero(ok if cfg.gate else keep)
        n = len(rows)
        log.info("%s: %d bases, %d aligned; answered 00 %.3f, 10 %.3f, 01 %.3f, 11 %.3f; %d with all four", pair,
                 len(bases), int(keep.sum()), *[float(answered[keep, k].mean()) for k in range(4)], int(ok.sum()))
        groups = token_groups(labels)
        out: dict[str, Any] = {"n": int(n), "labels": labels, "groups": groups,
                               "accuracy": {k: float(answered[keep, j].mean()) for j, k in enumerate(("00", "10", "01", "11"))}}
        if n < 2:
            log.info("%s: fewer than two bases kept", pair)
            return {**out, "K": None}, {}
        rps = self.read_points
        M = len(rps)
        if cfg.identity_check and "identity_check" not in self.metadata:  # the base's own states at every token
            r = rows[: min(n, backend.cfg.batch_size)]
            bp = [prompts[0][i] for i in r]
            own = backend.capture_sites(bp, positions[r])
            mid = rps[M // 2]
            ref = backend.run(bp, token_ids=tid)
            got = backend.run(bp, token_ids=tid, site_patch=SitePatch(np.full(len(r), mid), positions[r], own[:, mid]))
            same = bool(np.array_equal(got.token_logprobs, ref.token_logprobs) and np.array_equal(got.query_argmax, ref.query_argmax))
            self.metadata["identity_check"] = {f"{pair}/read_point_{mid}/query": same}
            if not same:
                raise RuntimeError("identity check failed: a run's own query states swapped back changed it")
            del own
        T = len(PATCHES)
        am = np.full((T, n, M), -1, dtype=np.int64)
        ratio = np.zeros((n, M, P), dtype=np.float32)
        group_mask = {g: torch.zeros(P, dtype=torch.bool) for g in GROUPS}
        for g in GROUPS:
            group_mask[g][groups[g]] = True
        for c0 in range(0, n, cfg.chunk):
            rc = rows[c0 : c0 + cfg.chunk]
            nc = len(rc)
            h = [backend.capture_sites([prompts[k][i] for i in rc], positions[rc]) for k in range(4)]  # (nc, L+1, P, d)
            dev, dtype = h[0].device, h[0].dtype
            base_rows = [prompts[0][i] for i in rc]
            for mi, m in enumerate(rps):
                h00, h10, h01, h11 = (x[:, m].float() for x in h)
                inter = (h11 - h10 - h01 + h00).norm(dim=-1)
                ratio[c0 : c0 + nc, mi] = (inter / ((h10 - h00).norm(dim=-1) + (h01 - h00).norm(dim=-1) + 1e-6)).cpu().numpy()
                add = (h10 + h01 - h00).to(dtype)
                joint = h[3][:, m]
                vals = [joint, add]
                for g in GROUPS:
                    v = joint.clone()
                    v[:, group_mask[g].to(dev)] = add[:, group_mask[g].to(dev)]
                    vals.append(v)
                values = torch.cat(vals, dim=0)  # (T * nc, P, d)
                prompts_t = base_rows * T
                patch = SitePatch(np.full(T * nc, m), np.tile(positions[rc], (T, 1)), values)
                res = backend.run(prompts_t, site_patch=patch, token_ids=tid)
                if cfg.determinism_check and "patch" not in self.metadata.get("determinism_check", {}):
                    bs = min(backend.cfg.batch_size, T * nc)
                    sub = SitePatch(np.full(bs, m), np.tile(positions[rc], (T, 1))[:bs], values[:bs])
                    a = backend.run(prompts_t[:bs], site_patch=sub, token_ids=tid)
                    b = backend.run(prompts_t[:bs], site_patch=sub, token_ids=tid)
                    cmp = compare_arrays({"lp": a.token_logprobs, "am": a.query_argmax}, {"lp": b.token_logprobs, "am": b.query_argmax})
                    self.metadata.setdefault("determinism_check", {})["patch"] = {"identical": cmp["identical"]}
                    if not cmp["identical"]:
                        raise RuntimeError("determinism check failed: a repeated patch batch differs")
                am[:, c0 : c0 + nc, mi] = res.query_argmax.reshape(T, nc)
                del values, vals, add, joint
            del h
            log.info("%s: %d of %d bases patched", pair, min(c0 + nc, n), n)
        # categories: 11's answer first, then 10's, 01's, 00's, other
        Yt = tid[Y[rows]]  # (n, 4) token ids of 00, 10, 01, 11
        order = {"11": 3, "10": 1, "01": 2, "00": 0}
        cat = np.full(am.shape, len(CATS) - 1, dtype=np.int8)
        for ci in reversed(range(len(CATS) - 1)):
            cat[am == Yt[None, :, order[CATS[ci]]][:, :, None]] = ci
        seed = derive_seed(cfg.seed, "interact", pair, "bootstrap")
        rates: dict[str, Any] = {}
        for ti, name in enumerate(PATCHES):
            rates[name] = {c: bootstrap_rate(cat[ti] == ci, np.random.default_rng(seed + 10 * ti + ci), cfg.n_boot, cfg.alpha)
                           for ci, c in enumerate(CATS)}
        K = conversion_depth(rates["additive"]["11"]["lo"], rates["additive"]["11"]["hi"], rps, cfg.sufficient)
        check0 = float((cat[1, :, 0] == cat[0, :, 0]).mean()) if rps[0] == 0 else None
        log.info("%s: the additive patch gives 11's answer at read points %s: %s; K %s (held at the first read point: %s); "
                 "additive = joint at read point 0 on %s of the bases", pair, rps[::4],
                 [round(float(x), 2) for x in rates["additive"]["11"]["mean"][::4]], K["K"], K["holds_at_first"], check0)
        out.update({"rates": rates, **K, "additive_equals_joint_at_0": check0,
                    "ratio_mean": ratio.mean(axis=0), "examples": [[e.text() for e in quads[i]] for i in rows]})
        return out, {"argmax": am.astype(np.int32), "rows": rows, "ratio": ratio}

    def _figure(self, result: dict[str, Any]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mc = self.cfg.model
        name = mc.name.split("/")[-1] if mc.backend == "hf" else f"toy {mc.toy.family}"
        pairs = [p for p, r in result["pairs"].items() if r.get("rates")]
        if not pairs:
            return
        rps = np.asarray(result["read_points"])
        fig, axes = plt.subplots(2, len(pairs), figsize=(5.2 * len(pairs), 7.4), squeeze=False,
                                 gridspec_kw={"height_ratios": [1.3, 1]})
        cols = {"joint": "#8a8985", "additive": "#2a78d6"}
        gcol = {"O": "#d9a521", "F": "#1baf7a", "S": "#0f7a52", "P": "#c2417f", "E": "#eb6834", "A": "#0b0b0b"}
        for c, p in enumerate(pairs):
            r = result["pairs"][p]
            ax = axes[0, c]
            for k, lab in (("joint", "joint state h11"), ("additive", "additive h10 + h01 - h00")):
                f = r["rates"][k]["11"]
                ax.fill_between(rps, f["lo"], f["hi"], color=cols[k], alpha=0.15, lw=0)
                ax.plot(rps, f["mean"], color=cols[k], lw=1.8, label=lab)
            for g in GROUPS:
                ax.plot(rps, r["rates"][f"add_{g}"]["11"]["mean"], color=gcol[g], lw=0.9, ls="--",
                        label=f"joint, additive at {g} ({GROUP_LABELS[g]})")
            if r.get("K") is not None:
                ax.axvline(r["K"], color="#52514e", lw=0.7, ls=":")
            ax.axhline(0.5, color="#52514e", lw=0.5, ls=":")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{PAIR_LABELS[p]} ({r['n']} bases): share giving 11's answer; K = {r.get('K')}", fontsize=8, loc="left")
            ax.set_xlabel("read point", fontsize=8)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.legend(fontsize=6, frameon=False, loc="lower left")
            ax = axes[1, c]
            im = ax.imshow(np.asarray(r["ratio_mean"]), aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks(range(len(r["labels"])))
            ax.set_xticklabels(r["labels"], fontsize=6.5)
            step = max(1, len(rps) // 8)
            ax.set_yticks(range(0, len(rps), step))
            ax.set_yticklabels(rps[::step], fontsize=7)
            ax.set_title("interaction ratio |h11 - h10 - h01 + h00| / (|h10 - h00| + |h01 - h00|)", fontsize=7.5, loc="left")
            fig.colorbar(im, ax=ax, fraction=0.03)
        fig.suptitle(f"The operator and the operands changed independently and together (D61), {name}; "
                     f"the control is combined {result['reading']['control_combined']}", fontsize=8.5, x=0.01, ha="left")
        fig.savefig(self.root / "figures" / "interact.png", dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(fig)


def run_interact(cfg: InteractConfig, run_id: str | None = None, config_path: str | None = None) -> Path:
    return Interaction(cfg, run_id=run_id, config_path=config_path).run()

"""Splitting the block where the sums appear (docs/DECISIONS.md D47).

D46 found the sums of ``(a+b) - (c+d) = `` linearly readable at the answer token from the block that brings in the
operands (or the next), and the answer one to three blocks later. A block does two things to the answer token in
order: attention reads the earlier tokens and adds a summary, then the MLP works on the token's own vector and adds
its result. Part 1 runs the D46 probe on the state before a block, after its attention and after its MLP (and on
the two writes alone), so the block's gain in a readable sum is split between attention and MLP. Part 2 swaps
single attention heads of the answer token between the D45 bases and their sources: which heads carry the first
bracket, which the second, and whether the two arrive through separate heads (lanes) or one channel.

    uv run directions varsplit --config configs/varsplit_qwen3_4b.yaml --run-id varsplit_qwen3_4b_seed20260916
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .config import VarsplitConfig
from .determinism import compare_arrays
from .model import HeadPatch, ModelBackend, environment_metadata, set_torch_determinism
from .probes import conditional_probe
from .runinfo import RejectionLog, git_info, setup_logging, write_json
from .seeds import derive_seed
from .varmap import (
    SOURCE_TYPES,
    bootstrap_rate,
    bootstrap_ratio,
    prediction,
    render,
    sample_bases,
)
from .varprobe import TARGETS, _jsonable, sample_probe_set, target_spec, variables

VALUE_TYPES = ("S1", "S2", "both")  # the source types whose prediction differs from the base answer
SAME_OF = {"S1": "S1_same", "S2": "S2_same"}


def state_sequence(blocks: list[int]) -> tuple[list[str], list[tuple[str, int]]]:
    """Names and recipes of the probed states over consecutive ``blocks``: h(b0-1), then mid(b), h(b) per block.

    A recipe is ("h", read point) or ("mid", block); mid(b) = h(b-1) + the attention write of block b."""
    names, recipe = [f"h{blocks[0] - 1}"], [("h", blocks[0] - 1)]
    for b in blocks:
        names += [f"mid{b}", f"h{b}"]
        recipe += [("mid", b), ("h", b)]
    return names, recipe


def attention_share(correct: np.ndarray, names: list[str], block: int, rng: np.random.Generator, n_boot: int,
                    alpha: float) -> dict[str, Any]:
    """The share of a block's accuracy gain delivered by its attention, with a paired bootstrap interval:
    sum(c_mid - c_prev) / sum(c_h - c_prev) over examples, ``correct`` (S, N) bool per state."""
    i_prev, i_mid, i_h = names.index(f"h{block - 1}"), names.index(f"mid{block}"), names.index(f"h{block}")
    c = correct.astype(np.float64)
    num, den = c[i_mid] - c[i_prev], c[i_h] - c[i_prev]
    r = bootstrap_ratio(num, den, rng, n_boot, alpha)
    return {"gain": float(den.mean()), "attention_gain": float(num.mean()), "share": float(r["mean"]),
            "lo": float(r["lo"]), "hi": float(r["hi"])}


def read_share(s: dict[str, Any], gain_min: float) -> str:
    if s["gain"] < gain_min:
        return "no gain"
    if s["lo"] > 0.5:
        return "attention"
    if s["hi"] < 0.5:
        return "mlp"
    return "mixed"


def read_heads(types: dict[str, Any], carry_lower: float, not_carry_upper: float, digit_change: float) -> dict[str, Any]:
    """Carriers, lanes or one channel from the per-head transfers of one block (D47)."""
    lo = {k: np.asarray(types[k]["transfer"]["lo"])[:-1] for k in ("S1", "S2") if k in types}
    hi = {k: np.asarray(types[k]["transfer"]["hi"])[:-1] for k in ("S1", "S2") if k in types}
    if len(lo) < 2:
        return {"reading": "undecided"}
    carry = {k: lo[k] >= carry_lower for k in lo}
    only1 = carry["S1"] & (hi["S2"] < not_carry_upper)
    only2 = carry["S2"] & (hi["S1"] < not_carry_upper)
    any_carrier = carry["S1"] | carry["S2"]
    if not any_carrier.any():
        reading = "no carrier"
    elif only1.any() and only2.any():
        reading = "lanes"
    elif (carry["S1"] == carry["S2"]).all():
        reading = "one channel"
    else:
        reading = "mixed"
    out = {"reading": reading, "carries_S1": np.flatnonzero(carry["S1"]).tolist(),
           "carries_S2": np.flatnonzero(carry["S2"]).tolist(), "S1_only": np.flatnonzero(only1).tolist(),
           "S2_only": np.flatnonzero(only2).tolist()}
    for k, same in SAME_OF.items():
        if same in types and k in carry:
            ch_lo = np.asarray(types[same]["change"]["lo"])[:-1]
            out[f"{k}_carriers_holding_a_digit"] = [int(h) for h in np.flatnonzero(carry[k]) if ch_lo[h] >= digit_change]
    return out


class VarSplit:
    def __init__(self, cfg: VarsplitConfig, run_id: str | None = None, config_path: str | None = None) -> None:
        self.cfg = cfg
        slug = cfg.model.name.split("/")[-1].lower() if cfg.model.backend == "hf" else "toy"
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_varsplit_{slug}_seed{cfg.seed}"
        self.root = Path(cfg.output_dir) / run_id
        if self.root.exists():
            raise FileExistsError(f"run directory already exists: {self.root}")
        (self.root / "core" / "tasks").mkdir(parents=True)
        (self.root / "figures").mkdir()
        self.log = setup_logging(self.root / "log.txt")
        self.rejections = RejectionLog(self.root / "rejections.jsonl")
        self.metadata: dict[str, Any] = {
            "version": __version__, "config": asdict(cfg), "config_path": config_path, "run_id": run_id, "seed": cfg.seed,
            "git": git_info(), "environment": environment_metadata(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run(self) -> Path:
        set_torch_determinism(derive_seed(self.cfg.seed, "torch"))
        self.backend = ModelBackend(self.cfg.model, run_seed=self.cfg.seed)
        L = self.backend.n_layers
        if max(self.cfg.split_blocks + self.cfg.head_blocks) > L:
            raise ValueError(f"a block beyond the model's {L}")
        self.metadata["model"] = self.backend.metadata()
        self.digit_ids = np.array([self._single_token(str(k)) for k in range(10)], dtype=np.int64)
        summary: dict[str, Any] = {"tasks": {}}
        for task in self.cfg.expressions:
            t0 = time.time()
            result = {"task": task, "gather": self.cfg.gather, "probe": self._probe_part(task), "heads": self._head_part(task)}
            td = self.root / "core" / "tasks" / task
            write_json(td / "varsplit.json", _jsonable(result))
            if self.cfg.figures and "skipped" not in result["probe"]:
                try:
                    mc = self.cfg.model
                    plot_task(result, self.root / "figures" / f"{task}.png",
                              mc.name.split("/")[-1] if mc.backend == "hf" else f"toy {mc.toy.family}")
                except Exception as exc:  # a figure never fails a run
                    self.log.warning("%s: figure failed: %s", task, exc)
            summary["tasks"][task] = {
                "shares": {t: {b: v["reading"] for b, v in r["shares"].items()} for t, r in result["probe"].get("targets", {}).items()},
                "heads": {b: v["reading"]["reading"] for b, v in result["heads"].get("blocks", {}).items()},
            }
            self.metadata.setdefault("timings_seconds", {})[task] = round(time.time() - t0, 1)
        write_json(self.root / "core" / "summary.json", summary)
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        self.log.info("done -> %s", self.root)
        return self.root

    def _single_token(self, text: str) -> int:
        enc = self.backend.tokenizer.encode(text, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"{text!r} is not a single token on this tokenizer: {enc}")
        return enc[0]

    # ------------------------------------------------------------------ #
    # Part 1: the sub-layer probe

    def _probe_part(self, task: str) -> dict[str, Any]:
        cfg, backend, log = self.cfg, self.backend, self.log
        exprs, demos, info = sample_probe_set(cfg.seed, task, cfg.n_examples, cfg.n_shots, cfg.n_demo_pool)
        self.metadata.setdefault("sampling", {}).setdefault(task, {})["probe"] = info
        prompts = [render(demos[i], e) for i, e in enumerate(exprs)]
        res = backend.run(prompts, capture=True, capture_sublayers=True, token_ids=self.digit_ids)
        v = variables(exprs)
        correct = res.query_argmax == self.digit_ids[v["Y"]]
        accuracy = float(correct.mean())
        log.info("%s: few-shot accuracy %.3f on %d expressions", task, accuracy, len(exprs))
        if accuracy < cfg.min_accuracy:
            self.rejections.add("gate", task, "few-shot accuracy below the minimum", {"accuracy": accuracy})
            return {"skipped": "few-shot accuracy below the minimum", "accuracy": accuracy}
        if cfg.determinism_check and "capture" not in self.metadata.get("determinism_check", {}):
            bs = backend.cfg.batch_size
            again = backend.run(prompts[:bs], capture=True, capture_sublayers=True, token_ids=self.digit_ids)
            cmp = compare_arrays({"r": res.residuals[:, :bs], "a": res.attn_writes[:, :bs], "m": res.mlp_writes[:, :bs]},
                                 {"r": again.residuals, "a": again.attn_writes, "m": again.mlp_writes})
            self.metadata.setdefault("determinism_check", {})["capture"] = {"identical": cmp["identical"]}
            if not cmp["identical"]:
                raise RuntimeError("determinism check failed: a repeated capture batch differs")
        keep = correct if cfg.gate else np.ones(len(exprs), dtype=bool)
        vk = {k: x[keep] for k, x in v.items()}
        blocks = list(cfg.split_blocks)
        lo = blocks[0] - 1  # only the window's layers are kept: read points lo..blocks[-1], blocks' writes
        resid = res.residuals[lo : blocks[-1] + 1][:, keep]
        attn = res.attn_writes[lo : blocks[-1]][:, keep]
        mlp = res.mlp_writes[lo : blocks[-1]][:, keep]
        del res

        def h(m: int) -> np.ndarray:  # the state at read point m
            return resid[m - lo]

        def w(kind: str, b: int) -> np.ndarray:  # block b's attention or MLP write
            return (attn if kind == "attn" else mlp)[b - 1 - lo]

        # the residual step of each block is its two writes (float32 sums of bf16 states)
        step = np.stack([h(b) - h(b - 1) for b in blocks])
        parts = np.stack([w("attn", b) + w("mlp", b) for b in blocks])
        rel = np.linalg.norm(step - parts, axis=-1) / np.maximum(np.linalg.norm(step, axis=-1), 1e-12)
        additivity = {"median": float(np.median(rel)), "max": float(rel.max())}
        names, recipe = state_sequence(blocks)
        states = np.stack([h(m) if kind == "h" else h(m - 1) + w("attn", m) for kind, m in recipe]).astype(np.float32)
        wnames = [f"{kind}{b}" for b in blocks for kind in ("attn", "mlp")]
        writes = np.stack([w(kind, b) for b in blocks for kind in ("attn", "mlp")])
        del resid, attn, mlp
        targets: dict[str, Any] = {}
        for name in TARGETS[task]:
            t0 = time.time()
            y, cond, allowed = target_spec(name, task, vk)
            kw = dict(n_folds=cfg.n_folds, lambdas=cfg.lambdas, n_perm=cfg.n_perm, alpha=cfg.alpha, device=backend.device)
            st = conditional_probe(states, y, cond, allowed, seed=derive_seed(cfg.seed, "varsplit", task, name, "states"),
                                   per_example=True, **kw)
            wr = conditional_probe(writes, y, cond, allowed, seed=derive_seed(cfg.seed, "varsplit", task, name, "writes"), **kw)
            shares = {}
            for b in blocks:
                sh = attention_share(st["correct"], names, b, np.random.default_rng(derive_seed(cfg.seed, "varsplit", task, name, b)),
                                     cfg.n_boot, cfg.alpha)
                shares[b] = sh | {"reading": read_share(sh, cfg.share_gain_min)}
            targets[name] = {"states": {k: x for k, x in st.items() if k not in ("null", "correct")},
                             "writes": {k: x for k, x in wr.items() if k != "null"}, "shares": shares}
            log.info("%s/%s: %.0fs; accuracy along %s: %s; attention shares %s", task, name, time.time() - t0, names,
                     np.round(st["accuracy"], 2).tolist(), {b: (round(s["gain"], 2), round(s["share"], 2), s["reading"])
                                                            for b, s in shares.items()})
        return {"accuracy": accuracy, "n_kept": int(keep.sum()), "states": names, "writes": wnames, "blocks": blocks,
                "additivity": additivity, "targets": targets}

    # ------------------------------------------------------------------ #
    # Part 2: single heads swapped

    def _head_part(self, task: str) -> dict[str, Any]:
        cfg, backend, log = self.cfg, self.backend, self.log
        bases, sources, demos, info = sample_bases(cfg.seed, task, cfg.n_bases, cfg.n_shots, cfg.n_demo_pool, self.rejections)
        self.metadata.setdefault("sampling", {}).setdefault(task, {})["heads"] = info
        types = SOURCE_TYPES[task]
        tid = self.digit_ids
        H = backend.n_heads
        Y = np.array([b.Y for b in bases])
        base_prompts = [render(demos[i], b) for i, b in enumerate(bases)]
        base = backend.run(base_prompts, capture_heads=True, token_ids=tid)
        base_ok = base.query_argmax == tid[Y]
        accuracy = float(base_ok.mean())
        if accuracy < cfg.min_accuracy:
            return {"skipped": "few-shot accuracy below the minimum", "accuracy": accuracy}
        src: dict[str, Any] = {}
        for k in types:
            r = backend.run([render(demos[i], sources[i][k]) for i in range(len(bases))], capture_heads=True, token_ids=tid)
            src[k] = {"heads": r.head_outputs, "lp": r.token_logprobs,
                      "ok": r.query_argmax == tid[[sources[i][k].Y for i in range(len(bases))]]}
        if cfg.determinism_check and "noop_patch" not in self.metadata.get("determinism_check", {}):
            bs = min(backend.cfg.batch_size, len(bases))
            b0 = cfg.head_blocks[0]
            plain = backend.run(base_prompts[:bs], token_ids=tid)
            noop = backend.run(base_prompts[:bs], token_ids=tid,
                               head_patches=[HeadPatch(b0 - 1, 0, base.head_outputs[b0 - 1, :bs, 0])])
            cmp = compare_arrays({"lp": plain.token_logprobs}, {"lp": noop.token_logprobs})
            self.metadata.setdefault("determinism_check", {})["noop_patch"] = {"identical": cmp["identical"]}
            if not cmp["identical"]:
                raise RuntimeError("identity check failed: a no-op head patch changed the batch")
        blocks_out: dict[int, Any] = {}
        for b in cfg.head_blocks:
            per_type: dict[str, Any] = {}
            for k in types:
                rows = np.flatnonzero(base_ok & src[k]["ok"]) if cfg.gate else np.arange(len(bases))
                if len(rows) == 0:
                    per_type[k] = {"n": 0}
                    continue
                out = self._swap_heads(b, k, rows, base_prompts, base.head_outputs, src[k]["heads"])
                lp, am = out  # (n, H+2, 10), (n, H+2): heads 0..H-1, all heads, no-op
                ref_ok = am[:, H + 1] == base.query_argmax[rows]  # the gated answer (Y) when the gate is on
                if (~ref_ok).any():
                    self.rejections.add("noop", task, "no-op head patch misses the gated answer",
                                        {"block": b, "type": k, "bases": rows[~ref_ok].tolist()})
                rows, lp, am = rows[ref_ok], lp[ref_ok], am[ref_ok]
                n = len(rows)
                rng = np.random.default_rng(derive_seed(cfg.seed, "varsplit", task, "heads", b, k))
                entry: dict[str, Any] = {"n": n}
                if n >= 2 and k in VALUE_TYPES:
                    pred = np.array([prediction(k, bases[i], sources[i][k]) for i in rows])
                    yb = Y[rows]
                    ar = np.arange(n)
                    d = lp[ar, :, pred] - lp[ar, :, yb]  # (n, H+2)
                    d_src = src[k]["lp"][rows][ar, pred] - src[k]["lp"][rows][ar, yb]
                    num = d[:, : H + 1] - d[:, [H + 1]]
                    den = np.broadcast_to((d_src - d[:, H + 1])[:, None], num.shape)
                    entry["transfer"] = bootstrap_ratio(num, den, rng, cfg.n_boot, cfg.alpha)
                    entry["flip"] = bootstrap_rate(am[:, : H + 1] == tid[pred][:, None], rng, cfg.n_boot, cfg.alpha)
                elif n >= 2:
                    entry["change"] = bootstrap_rate(am[:, : H + 1] != tid[Y[rows]][:, None], rng, cfg.n_boot, cfg.alpha)
                    entry["logp_drop"] = (lp[np.arange(n), H + 1, Y[rows]][:, None]
                                          - lp[np.arange(n)[:, None], np.arange(H + 1)[None, :], Y[rows][:, None]]).mean(axis=0)
                per_type[k] = entry
            reading = read_heads(per_type, cfg.carry_lower, cfg.not_carry_upper, cfg.digit_change)
            sums = {k: float(np.asarray(per_type[k]["transfer"]["mean"])[:H].sum()) for k in VALUE_TYPES
                    if "transfer" in per_type.get(k, {})}
            alls = {k: float(np.asarray(per_type[k]["transfer"]["mean"])[H]) for k in VALUE_TYPES if "transfer" in per_type.get(k, {})}
            blocks_out[b] = {"types": per_type, "reading": reading, "sum_of_heads": sums, "all_heads": alls}
            log.info("%s: heads of block %d: %s (S1 carriers %s, S2 carriers %s); all heads %s, sum of heads %s", task, b,
                     reading.get("reading"), reading.get("carries_S1"), reading.get("carries_S2"),
                     {k: round(x, 2) for k, x in alls.items()}, {k: round(x, 2) for k, x in sums.items()})
        return {"accuracy": accuracy, "n_heads": H, "blocks": blocks_out}

    def _swap_heads(self, block: int, kind: str, rows: np.ndarray, base_prompts: list, base_heads: np.ndarray,
                    src_heads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per base row: each head swapped alone, all heads at once, and a no-op (head 0 given its own output), in one
        set of batches. Returns the digit log-probabilities (n, H+2, 10) and the argmax (n, H+2)."""
        backend, tid = self.backend, self.digit_ids
        H, layer = backend.n_heads, block - 1
        n = len(rows)
        per = H + 2
        prompts = [base_prompts[i] for i in rows for _ in range(per)]
        heads = np.zeros((H, n * per), dtype=np.int64)
        values = np.zeros((H, n * per, base_heads.shape[-1]), dtype=np.float32)
        for r, i in enumerate(rows):
            o = r * per
            for h in range(H):  # head h alone: every patch object writes head h's source output
                heads[:, o + h] = h
                values[:, o + h] = src_heads[layer, i, h]
            heads[:, o + H] = np.arange(H)  # all heads at once
            values[:, o + H] = src_heads[layer, i]
            heads[:, o + H + 1] = 0  # no-op: head 0 given its own output
            values[:, o + H + 1] = base_heads[layer, i, 0]
        patches = [HeadPatch(layer, heads[j], values[j]) for j in range(H)]
        out = backend.run(prompts, head_patches=patches, token_ids=tid)
        if self.cfg.determinism_check and "head_patch" not in self.metadata.get("determinism_check", {}):
            bs = min(backend.cfg.batch_size, len(prompts))
            again = backend.run(prompts[:bs], head_patches=[HeadPatch(layer, heads[j][:bs], values[j][:bs]) for j in range(H)],
                                token_ids=tid)
            cmp = compare_arrays({"lp": out.token_logprobs[:bs]}, {"lp": again.token_logprobs})
            self.metadata.setdefault("determinism_check", {})["head_patch"] = {"identical": cmp["identical"]}
            if not cmp["identical"]:
                raise RuntimeError("determinism check failed: a repeated head-patch batch differs")
        return out.token_logprobs.reshape(n, per, 10), out.query_argmax.reshape(n, per)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

SERIES = "#2a78d6"
NULL_FILL, NULL_LINE = "#e4e3df", "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"
ATTN_BAND = "#f7ece0"  # a warm tint for the attention steps, apart from the grey null band
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def plot_task(result: dict[str, Any], path: Path, model_name: str) -> None:
    """Top: the probe along h(b-1) -> after attention -> after MLP for each block (attention steps shaded), for S1 given
    Y and Y. Bottom: each head's transfer at the answer token for the source types, one heatmap per block."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    probe, heads = result["probe"], result["heads"]
    names = probe["states"]
    x = np.arange(len(names))
    hb = list(heads.get("blocks", {}).keys())
    fig = plt.figure(figsize=(10.5, 3.4 + 1.25 * len(hb)))
    gs = fig.add_gridspec(1 + len(hb), 2, height_ratios=[2.2] + [1.0] * len(hb), hspace=0.75, wspace=0.18, top=0.9, bottom=0.12)
    for j, t in enumerate(("S1|Y", "Y")):
        if t not in probe["targets"]:
            continue
        ax = fig.add_subplot(gs[0, j])
        st = probe["targets"][t]["states"]
        acc, mean, sd = (np.asarray(st[k], dtype=float) for k in ("accuracy", "null_mean", "null_sd"))
        for i, nm in enumerate(names):
            if nm.startswith("mid"):
                ax.axvspan(i - 1, i, color=ATTN_BAND, lw=0)
        ax.fill_between(x, mean, np.minimum(1, mean + st["threshold"] * sd), color=NULL_FILL, lw=0)
        ax.plot(x, mean, color=NULL_LINE, lw=0.9, ls=(0, (3, 2)))
        ax.plot(x, acc, color=SERIES, lw=1.4, marker="o", ms=3.2)
        ax.set_xticks(x)
        ax.set_xticklabels([nm.replace("mid", "+attn ").replace("h", "") for nm in names], fontsize=6.5, rotation=0)
        ax.set_ylim(0, 1)
        label = "S1 given Y (the first sum, beyond the answer)" if t == "S1|Y" else "Y (the answer)"
        ax.set_title(f"probe: {label}", fontsize=8, loc="left", color=INK)
        for b, s in probe["targets"][t]["shares"].items():
            if s["reading"] != "no gain":
                i = names.index(f"h{b}")
                ax.text(i - 0.5, 1.02, f"{s['share']:.2f}", fontsize=6.5, color=INK2, ha="center", va="bottom",
                        transform=ax.get_xaxis_transform())
        ax.tick_params(labelsize=6.5, length=2, colors=INK2)
        ax.grid(axis="y", color="#ecebe7", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("read point (answer token)", fontsize=7, color=INK2)
    handles = [plt.Line2D([], [], color=SERIES, lw=1.4, marker="o", ms=3.2), plt.Line2D([], [], color=NULL_LINE, lw=0.9, ls=(0, (3, 2))),
               plt.Rectangle((0, 0), 1, 1, color=NULL_FILL), plt.Rectangle((0, 0), 1, 1, color=ATTN_BAND)]
    fig.legend(handles, ["probe accuracy", "null mean", "family-wise null band", "a block's attention step (the next is its MLP)"],
               loc="upper left", bbox_to_anchor=(0.06, 0.035), ncol=4, fontsize=6.5, frameon=False)
    fig.text(0.06, 0.0, "'+attn b' = the state after block b's attention, before its MLP. Numbers above a block: "
             "attention's share of that block's gain in probe accuracy.", fontsize=6.5, color=INK2, ha="left", va="top")
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    for r, b in enumerate(hb):
        ax = fig.add_subplot(gs[1 + r, :])
        types = [k for k in VALUE_TYPES if "transfer" in heads["blocks"][b]["types"].get(k, {})]
        mat = np.stack([np.asarray(heads["blocks"][b]["types"][k]["transfer"]["mean"], dtype=float) for k in types])
        im = ax.imshow(np.clip(mat, 0, 1), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        H = heads["n_heads"]
        ax.set_xticks(list(range(0, H, 4)) + [H])
        ax.set_xticklabels([str(h) for h in range(0, H, 4)] + ["all"], fontsize=6.5)
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels([f"swap from source changing {k}" if k != "both" else "swap from source changing both" for k in types],
                           fontsize=6.5)
        ax.axvline(H - 0.5, color="#fcfcfb", lw=3)
        rd = heads["blocks"][b]["reading"]
        ax.set_title(f"block {b}: each head's transfer at the answer token (0 = nothing moves, 1 = as the source)  -  "
                     f"reading: {rd.get('reading')}", fontsize=7.5, loc="left", color=INK)
        ax.tick_params(length=2, colors=INK2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if r == len(hb) - 1:
            ax.set_xlabel("head", fontsize=7, color=INK2)
    cax = fig.add_axes([0.92, 0.08, 0.012, 0.18])
    cb = fig.colorbar(im, cax=cax) if hb else None
    if cb is not None:
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_visible(False)
    task = "(a+b) - (c+d) = " if result["task"] == "two_lane" else "(a+b) - c = "
    fig.suptitle(f"Splitting the block where the sums appear: {task.strip()}  ({model_name}; D45 gather at read point "
                 f"{result.get('gather')})", fontsize=8.5, x=0.01, ha="left")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def run_varsplit(cfg: VarsplitConfig, run_id: str | None = None, config_path: str | None = None) -> Path:
    return VarSplit(cfg, run_id, config_path).run()

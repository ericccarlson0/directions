"""The variable map (docs/DECISIONS.md D45): where the sub-results of an arithmetic expression live in the network.

For ``(a+b) - (c+d) = `` the sub-results are S1 = a+b, S2 = c+d and the answer Y = S1 - S2 (a single digit). For each
base expression and each kind of source (one that changes only S1, only S2, both, or the digits of one bracket with
its sum kept), the residual stream of the base run is swapped with the source run's at one (read point, query token)
site, the rest of the forward pass being the model's own, and the answer is read. Where a swap carries the source's
value of a variable into the answer, that variable is causally present at the site; where swapping in other digits
with the same sum changes nothing, it is held there as a value. The one-lane twin ``(a+b) - c = `` is the built-in
control. Whole-vector swaps: two variables present at one site may be two variables or already their combination
(D45; separating them is the next step, subspace swaps).

    uv run directions varmap --config configs/varmap_qwen3_4b.yaml --run-id varmap_qwen3_4b_seed20260916
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._version import __version__
from .config import VarmapConfig
from .determinism import compare_arrays
from .model import ModelBackend, SitePatch, environment_metadata, set_torch_determinism
from .prompts import Prompt
from .runinfo import RejectionLog, git_info, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .tasks import Item

# --------------------------------------------------------------------------- #
# The tasks and their high-level causal model
# --------------------------------------------------------------------------- #

TASKS = ("two_lane", "one_lane")
SOURCE_TYPES = {"two_lane": ("S1", "S2", "S1_same", "S2_same", "both"), "one_lane": ("S1", "S2", "S1_same", "both")}
# the variable each value-changing type moves, and the same-value type that tests it as a value
VALUE_TEST = {"S1": "S1_same", "S2": "S2_same"}


@dataclass(frozen=True)
class Expr:
    """``(a+b) - (c+d)`` (two lanes) or ``(a+b) - c`` (one lane, ``d`` None)."""

    a: int
    b: int
    c: int
    d: int | None = None

    @property
    def S1(self) -> int:
        return self.a + self.b

    @property
    def S2(self) -> int:
        return self.c + (self.d or 0)

    @property
    def Y(self) -> int:
        return self.S1 - self.S2

    def text(self) -> str:
        second = f"({self.c}+{self.d})" if self.d is not None else f"{self.c}"
        return f"({self.a}+{self.b}) - {second}"

    def digit_slots(self) -> dict[int, str]:
        """Character index in :meth:`text` -> the digit's name."""
        slots = {1: "a", 3: "b"}
        slots.update({9: "c", 11: "d"} if self.d is not None else {8: "c"})
        return slots


def all_exprs(task: str) -> list[Expr]:
    """Every expression of the task whose answer is a single digit 0..9."""
    if task == "two_lane":
        out = [Expr(a, b, c, d) for a, b, c, d in itertools.product(range(10), repeat=4)]
    elif task == "one_lane":
        out = [Expr(a, b, c) for a, b, c in itertools.product(range(10), repeat=3)]
    else:
        raise ValueError(f"unknown varmap task {task!r}")
    return [e for e in out if 0 <= e.Y <= 9]


def predictions(base: Expr, src: Expr) -> dict[str, int]:
    """The answer the swapped run gives if the site holds: the source's S1 only, its S2 only, both (its answer), or
    nothing of the source (the base answer)."""
    return {"S1": src.S1 - base.S2, "S2": base.S1 - src.S2, "both": src.Y, "none": base.Y}


def prediction(kind: str, base: Expr, src: Expr) -> int:
    """The answer a swap of ``kind`` predicts under its own hypothesis (D45)."""
    p = predictions(base, src)
    return {"S1": p["S1"], "S2": p["S2"], "both": p["both"], "S1_same": base.Y, "S2_same": base.Y}[kind]


def valid_source(kind: str, base: Expr, src: Expr) -> bool:
    if src == base:
        return False
    same_first = (src.a, src.b) == (base.a, base.b)
    same_second = (src.c, src.d) == (base.c, base.d)
    if kind == "S1":
        return same_second and src.S1 != base.S1
    if kind == "S2":
        return same_first and src.S2 != base.S2
    if kind == "S1_same":
        return same_second and not same_first and src.S1 == base.S1
    if kind == "S2_same":
        return same_first and not same_second and src.S2 == base.S2
    if kind == "both":
        p = predictions(base, src)
        vals = [p["S1"], p["S2"], p["both"], p["none"]]
        return src.S1 != base.S1 and src.S2 != base.S2 and all(0 <= v <= 9 for v in vals) and len(set(vals)) == 4
    raise ValueError(f"unknown source type {kind!r}")


def draw_source(kind: str, base: Expr, pool: Sequence[Expr], rng: np.random.Generator) -> Expr | None:
    cands = [e for e in pool if valid_source(kind, base, e)]
    return cands[int(rng.integers(len(cands)))] if cands else None


def render(demos: Sequence[Expr], query: Expr) -> Prompt:
    """``{expr} = {Y}`` per demonstration, one per line, then ``{expr} = `` with the digit as the target."""
    lines = [f"{e.text()} = {e.Y}" for e in demos]
    prompt = "\n".join(lines + [f"{query.text()} = "])
    return Prompt(prompt=prompt, target=f"{query.Y}", query=Item(query.text(), str(query.Y)),
                  demos=tuple(Item(e.text(), str(e.Y)) for e in demos))


def query_chars(query: Expr) -> str:
    return f"{query.text()} = "


# --------------------------------------------------------------------------- #
# Readouts
# --------------------------------------------------------------------------- #


def _resample_counts(n: int, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """How often each of ``n`` examples is drawn in each of ``n_boot`` resamples with replacement, ``(n_boot, n)``."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)


def bootstrap_rate(x: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float) -> dict[str, np.ndarray]:
    """Mean over the first axis of ``x`` (examples) with a percentile bootstrap interval; ``x`` is ``(n, ...)``."""
    x = np.asarray(x, dtype=np.float64)
    n, shape = x.shape[0], x.shape[1:]
    boots = (_resample_counts(n, rng, n_boot) @ x.reshape(n, -1) / n).reshape((n_boot,) + shape)
    return {"mean": x.mean(axis=0), "lo": np.quantile(boots, alpha / 2, axis=0), "hi": np.quantile(boots, 1 - alpha / 2, axis=0)}


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float) -> dict[str, np.ndarray]:
    """The ratio of means sum(num)/sum(den) over the first axis, with a percentile bootstrap interval."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    n, shape = num.shape[0], num.shape[1:]
    w = _resample_counts(n, rng, n_boot)
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = ((w @ num.reshape(n, -1)) / (w @ den.reshape(n, -1))).reshape((n_boot,) + shape)
        est = num.sum(axis=0) / den.sum(axis=0)
    return {"mean": est, "lo": np.nanquantile(boots, alpha / 2, axis=0), "hi": np.nanquantile(boots, 1 - alpha / 2, axis=0)}


def reading(rates: dict[str, dict[str, Any]], task: str, present_flip: float, value_change: float) -> dict[str, np.ndarray]:
    """The D45 reading per site from the rate maps: presence (flip-rate lower bound >= ``present_flip``), held as a
    value (present, and the same-value swap's change-rate upper bound <= ``value_change``), and the count of S1 and
    S2 present. A type without a map (too few bases kept) reads as absent, and a value test without one as undecided
    (neither value nor digits)."""
    shape = np.asarray(next(iter(rates.values()))["flip"]["lo"]).shape
    out: dict[str, np.ndarray] = {}
    for var in ("S1", "S2"):
        present = (np.asarray(rates[var]["flip"]["lo"]) >= present_flip) if var in rates else np.zeros(shape, dtype=bool)
        out[f"present_{var}"] = present
        test = VALUE_TEST[var]
        if test in rates:
            value = present & (np.asarray(rates[test]["change"]["hi"]) <= value_change)
            out[f"value_{var}"] = value
            out[f"digits_{var}"] = present & ~value
    out["count"] = out["present_S1"].astype(int) + out["present_S2"].astype(int)
    return out


def query_tokens(backend: ModelBackend, prompt: Prompt, query: Expr) -> tuple[list[int], list[int], list[str]]:
    """The query's token positions (absolute indices), its token ids, and a label per token in which each operand
    digit is named by its slot (a, b, c, d) and the answer position is ``␣`` (the trailing space)."""
    qtext = query_chars(query)
    context_chars = len(prompt.prompt) - len(qtext)
    ids, _ = backend._encode_prompt(prompt)
    _, end = backend.context_span(prompt, context_chars)
    q_ids = ids[end:]
    labels, cursor = [], 0
    slots = query.digit_slots()
    for tid in q_ids:
        piece = backend.tokenizer.decode([tid])
        start = qtext.find(piece, cursor) if piece else cursor
        if start != cursor:
            raise ValueError(f"query tokens do not tile the query text {qtext!r}")
        lab = "".join(slots.get(cursor + k, ch) for k, ch in enumerate(piece))
        labels.append("␣" if lab == " " and cursor + len(piece) == len(qtext) else lab)
        cursor += len(piece)
    if cursor != len(qtext):
        raise ValueError(f"query tokens cover {cursor} of {len(qtext)} characters of {qtext!r}")
    return list(range(end, len(ids))), list(q_ids), labels


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


class VariableMap:
    def __init__(self, cfg: VarmapConfig, run_id: str | None = None, config_path: str | None = None) -> None:
        self.cfg = cfg
        slug = cfg.model.name.split("/")[-1].lower() if cfg.model.backend == "hf" else "toy"
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_varmap_{slug}_seed{cfg.seed}"
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
        self.metadata["model"] = self.backend.metadata()
        self.digit_ids = self._digit_ids()
        summary: dict[str, Any] = {"tasks": {}, "skipped": {}}
        for task in self.cfg.expressions:
            t0 = time.time()
            res = self._task(task)
            if "skipped" in res:
                summary["skipped"][task] = res["skipped"]
            else:
                summary["tasks"][task] = res["summary"]
            self.metadata.setdefault("timings_seconds", {})[task] = round(time.time() - t0, 1)
        write_json(self.root / "core" / "summary.json", summary)
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

    def _sample(self, task: str) -> tuple[list[Expr], list[dict[str, Expr | None]], list[list[Expr]]]:
        """The demonstration pool, then the bases in a seeded order, each with one source per type; an expression
        that has no valid source of some type (a sum with a single digit pair, e.g. 0 or 18) is skipped and counted."""
        cfg = self.cfg
        exprs = all_exprs(task)
        rng = rng_for(cfg.seed, "varmap", task, "split")
        order = rng.permutation(len(exprs))
        demo_pool = [exprs[i] for i in order[: cfg.n_demo_pool]]
        rest = [exprs[i] for i in order[cfg.n_demo_pool :]]
        bases, sources, demos, skipped = [], [], [], 0
        for base in rest:
            if len(bases) == cfg.n_bases:
                break
            i = len(bases)
            srcs = {k: draw_source(k, base, rest, rng_for(cfg.seed, "varmap", task, "source", i, skipped, k)) for k in SOURCE_TYPES[task]}
            if any(v is None for v in srcs.values()):
                skipped += 1
                self.rejections.add("source", task, "no valid source of some type", {"base": base.text(),
                                    "missing": [k for k, v in srcs.items() if v is None]})
                continue
            bases.append(base)
            sources.append(srcs)
            r = rng_for(cfg.seed, "varmap", task, "demos", i)
            demos.append([demo_pool[j] for j in r.choice(len(demo_pool), size=cfg.n_shots, replace=False)])
        self.metadata.setdefault("sampling", {})[task] = {"n_expressions": len(exprs), "skipped_without_source": skipped,
                                                          "n_bases": len(bases)}
        return bases, sources, demos

    def _task(self, task: str) -> dict[str, Any]:
        cfg, backend, log = self.cfg, self.backend, self.log
        bases, sources, demos = self._sample(task)
        types = SOURCE_TYPES[task]
        base_prompts = [render(demos[i], b) for i, b in enumerate(bases)]
        pos0, _ids0, labels = query_tokens(backend, base_prompts[0], bases[0])
        P = len(pos0)
        positions = np.zeros((len(bases), P), dtype=np.int64)
        keep = np.ones(len(bases), dtype=bool)
        src_prompts: dict[str, list[Prompt | None]] = {k: [] for k in types}
        for i, base in enumerate(bases):
            pos, ids, _ = query_tokens(backend, base_prompts[i], base)
            if len(pos) != P:
                keep[i] = False
                self.rejections.add("align", task, "query token count differs", {"base": base.text(), "n": len(pos), "expected": P})
                for k in types:
                    src_prompts[k].append(None)
                continue
            positions[i] = pos
            for k in types:
                src = sources[i][k]
                if src is None:
                    keep[i] = False
                    self.rejections.add("source", task, f"no valid {k} source", {"base": base.text()})
                    src_prompts[k].append(None)
                    continue
                sp = render(demos[i], src)
                spos, sids, _ = query_tokens(backend, sp, src)
                aligned = spos == pos and all(x == y for x, y, lab in zip(ids, sids, labels) if lab not in ("a", "b", "c", "d"))
                if not aligned:
                    keep[i] = False
                    self.rejections.add("align", task, f"{k} source misaligned", {"base": base.text(), "source": src.text()})
                src_prompts[k].append(sp)
        log.info("%s: %d bases, %d query tokens %s, %d aligned with every source", task, len(bases), P, labels, int(keep.sum()))

        # base and source runs: the answer and the gate
        tid = self.digit_ids
        base_run = backend.run(base_prompts, token_ids=tid)
        base_ok = base_run.query_argmax == tid[[b.Y for b in bases]]
        accuracy = float(base_ok.mean())
        log.info("%s: few-shot accuracy %.3f on %d bases", task, accuracy, len(bases))
        if accuracy < cfg.min_accuracy:
            self.rejections.add("gate", task, "few-shot accuracy below the minimum", {"accuracy": accuracy, "min": cfg.min_accuracy})
            return {"skipped": {"reason": "few-shot accuracy below the minimum", "accuracy": accuracy}}
        src_runs = {}
        for k in types:
            idx = [i for i in range(len(bases)) if src_prompts[k][i] is not None]
            r = backend.run([src_prompts[k][i] for i in idx], token_ids=tid)
            lp = np.full((len(bases), 10), np.nan, dtype=np.float32)
            am = np.full(len(bases), -1, dtype=np.int64)
            lp[idx], am[idx] = r.token_logprobs, r.query_argmax
            src_runs[k] = {"lp": lp, "argmax": am}
            ok = np.zeros(len(bases), dtype=bool)
            ok[idx] = r.query_argmax == tid[[sources[i][k].Y for i in idx]]
            src_runs[k]["ok"] = ok
        # the gate, per swap type: a base enters a type's map when the model answers it and that type's source
        # correctly (D45); the identity check uses the bases it answers correctly
        base_mask = keep & (base_ok if cfg.gate else True)
        type_rows = {k: np.where(base_mask & (src_runs[k]["ok"] if cfg.gate else True))[0] for k in types}
        for i in np.where(keep & ~base_mask)[0]:
            self.rejections.add("gate", task, "base answered wrongly", {"base": bases[i].text()})
        for k in types:
            for i in np.where(base_mask & ~np.isin(np.arange(len(bases)), type_rows[k]))[0]:
                self.rejections.add("gate", task, f"{k} source answered wrongly", {"base": bases[i].text(), "source": sources[i][k].text()})
        log.info("%s: bases kept per type %s (of %d)", task, {k: len(v) for k, v in type_rows.items()}, len(bases))
        base_rows_idx = np.where(base_mask)[0]
        if len(base_rows_idx) < 2:
            return {"skipped": {"reason": "fewer than two bases answered correctly", "accuracy": accuracy}}

        L = backend.n_layers
        read_points = list(range(L + 1)) if cfg.read_points is None else [m for m in cfg.read_points if 0 <= m <= L]
        M = len(read_points)

        # checks (D23): a swap of a run's own state reproduces it bit-exactly; a repeated swap batch is identical
        if cfg.identity_check:
            rows = base_rows_idx
            base_rows = [base_prompts[i] for i in rows]
            own = backend.capture_sites(base_rows, positions[rows])
            mid = read_points[M // 2]
            ref = backend.run(base_rows, token_ids=tid)
            for j in (0, P // 2, P - 1):
                got = backend.run(base_rows, token_ids=tid, site_patch=SitePatch(np.full(len(rows), mid), positions[rows, j], own[:, mid, j]))
                same = bool(np.array_equal(got.token_logprobs, ref.token_logprobs) and np.array_equal(got.query_argmax, ref.query_argmax))
                self.metadata.setdefault("identity_check", {})[f"{task}/read_point_{mid}/token_{j}"] = same
                if not same:
                    raise RuntimeError(f"identity check failed: swapping the run's own state at read point {mid}, token {j} changed it")
            del own

        out_rates: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}
        n_kept: dict[str, int] = {}
        brng_seed = derive_seed(cfg.seed, "varmap", task, "bootstrap")
        for k in types:
            rows = type_rows[k]
            n = len(rows)
            n_kept[k] = n
            if n < 2:
                log.info("%s/%s: fewer than two bases kept; no map", task, k)
                continue
            base_rows = [base_prompts[i] for i in rows]
            base_lp = base_run.token_logprobs[rows]
            y_base = np.array([bases[i].Y for i in rows])
            src_list = [src_prompts[k][i] for i in rows]
            src_states = backend.capture_sites(src_list, positions[rows])  # (n, L+1, P, d)
            # one row per (example, read point, token): example-major so that a batch spans few prompts
            ex, mm, jj = np.meshgrid(np.arange(n), np.arange(M), np.arange(P), indexing="ij")
            ex, mm, jj = ex.ravel(), mm.ravel(), jj.ravel()
            rp = np.asarray(read_points)[mm]
            dev = src_states.device
            values = src_states[torch.as_tensor(ex, device=dev), torch.as_tensor(rp, device=dev), torch.as_tensor(jj, device=dev)]
            prompts = [base_rows[e] for e in ex]
            patch = SitePatch(rp, positions[rows][ex, jj], values)
            res = backend.run(prompts, site_patch=patch, token_ids=tid)
            if cfg.determinism_check and "swap" not in self.metadata.get("determinism_check", {}):
                bs = backend.cfg.batch_size
                sub = SitePatch(rp[:bs], positions[rows][ex[:bs], jj[:bs]], values[:bs])
                a = backend.run(prompts[:bs], site_patch=sub, token_ids=tid)
                b = backend.run(prompts[:bs], site_patch=sub, token_ids=tid)
                cmp = compare_arrays({"lp": a.token_logprobs, "am": a.query_argmax}, {"lp": b.token_logprobs, "am": b.query_argmax})
                self.metadata.setdefault("determinism_check", {})["swap"] = {"identical": cmp["identical"]}
                if not cmp["identical"]:
                    raise RuntimeError("determinism check failed: a repeated swap batch differs")
            del values, src_states
            lp = res.token_logprobs.reshape(n, M, P, 10)
            am = res.query_argmax.reshape(n, M, P)
            pred = np.array([prediction(k, bases[i], sources[i][k]) for i in rows])
            flip = am == tid[pred][:, None, None]
            change = am != base_run.query_argmax[rows][:, None, None]  # against the base run's own answer (= Y when gated)
            rates: dict[str, Any] = {"n": n, "flip": bootstrap_rate(flip, np.random.default_rng(brng_seed + 1), cfg.n_boot, cfg.alpha),
                                     "change": bootstrap_rate(change, np.random.default_rng(brng_seed + 2), cfg.n_boot, cfg.alpha)}
            arrays[f"{k}/flip"], arrays[f"{k}/argmax"], arrays[f"{k}/bases"] = flip, am.astype(np.int32), rows
            if not k.endswith("_same"):
                # transfer: the shift of logit(prediction) - logit(base answer), against the source run's own shift
                ar = np.arange(n)
                d_base = base_lp[ar, pred] - base_lp[ar, y_base]
                d_src = src_runs[k]["lp"][rows][ar, pred] - src_runs[k]["lp"][rows][ar, y_base]
                ia, im, ij = ar[:, None, None], np.arange(M)[None, :, None], np.arange(P)[None, None, :]
                d_sw = lp[ia, im, ij, pred[:, None, None]] - lp[ia, im, ij, y_base[:, None, None]]
                num = d_sw - d_base[:, None, None]
                den = np.broadcast_to((d_src - d_base)[:, None, None], num.shape)
                rates["transfer"] = bootstrap_ratio(num, den, np.random.default_rng(brng_seed + 3), cfg.n_boot, cfg.alpha)
                arrays[f"{k}/transfer_num"], arrays[f"{k}/transfer_den"] = num.astype(np.float32), (d_src - d_base).astype(np.float32)
            if k == "both":
                hyp = {h: np.array([predictions(bases[i], sources[i]["both"])[h] for i in rows]) for h in ("S1", "S2", "both", "none")}
                rates["hypotheses"] = {h: bootstrap_rate(am == tid[v][:, None, None], np.random.default_rng(brng_seed + 4), cfg.n_boot, cfg.alpha)
                                       for h, v in hyp.items()}
            out_rates[k] = rates
            log.info("%s/%s: %d bases, %d swaps; max flip rate %.2f, at the answer token's last read point %.2f", task, k, n, len(prompts),
                     float(rates["flip"]["mean"].max()), float(rates["flip"]["mean"][-1, -1]))

        read = reading(out_rates, task, cfg.present_flip, cfg.value_change)
        td = self.root / "core" / "tasks" / task
        result = {
            "task": task, "template": "(a+b) - (c+d) = " if task == "two_lane" else "(a+b) - c = ", "labels": labels,
            "read_points": read_points, "n_bases": len(bases), "n_kept": n_kept, "accuracy": accuracy,
            "source_accuracy": {k: float(src_runs[k]["ok"][keep].mean()) for k in types},
            "examples": [{"base": bases[i].text(), "Y": bases[i].Y, "sources": {k: sources[i][k].text() for k in types},
                          "kept": [k for k in types if i in set(type_rows[k].tolist())]} for i in range(len(bases))],
            "rates": out_rates, "reading": read,
            "thresholds": {"present_flip": cfg.present_flip, "value_change": cfg.value_change},
        }
        write_json(td / "varmap.json", result)
        np.savez_compressed(td / "varmap_arrays.npz", **{k.replace("/", "__"): v for k, v in arrays.items()})
        if cfg.figures:
            try:
                mc = self.cfg.model
                plot_task(result, self.root / "figures" / f"{task}.png",
                          mc.name.split("/")[-1] if mc.backend == "hf" else f"toy {mc.toy.family}")
            except Exception as exc:  # a figure never fails a run
                log.warning("%s: figure failed: %s", task, exc)
        summ = {"accuracy": accuracy, "n_kept": n_kept, "labels": labels,
                "max_count": int(read["count"].max()),
                "sites_with_both": int((read["count"] == 2).sum()),
                "first_value_S1": _first_site(read.get("value_S1"), read_points, labels),
                "first_value_S2": _first_site(read.get("value_S2"), read_points, labels)}
        return {"summary": summ}


def _first_site(mask: np.ndarray | None, read_points: list[int], labels: list[str]) -> dict[str, Any] | None:
    """The shallowest site where ``mask`` holds (read point, token), or None."""
    if mask is None or not np.asarray(mask).any():
        return None
    m, j = np.argwhere(np.asarray(mask))[0]
    return {"read_point": int(read_points[m]), "token": int(j), "label": labels[j]}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential blue, light -> dark
CAT = {"none": "#ecebe7", "S1": "#2a78d6", "S2": "#eb6834", "both": "#1baf7a"}


def plot_task(result: dict[str, Any], path: Path, model_name: str) -> None:
    """Heatmaps over (read point x query token). Left block: the flip rates of the value-changing swaps (top row)
    and the change rates of the same-value swaps (bottom row), on one colour scale. Right column: the count of
    variables present, and which prediction a swap from a source changing both brackets produces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    seq = LinearSegmentedColormap.from_list("seq", SEQ)
    labels = result["labels"]
    rps = result["read_points"]
    rates, read = result["rates"], result["reading"]
    two = result["task"] == "two_lane"
    grid = [[("S1", "flip", "swap S1\nanswer follows source S1"),
             ("S2", "flip", "swap S2\nanswer follows source S2" if two else "swap c\nanswer follows source c"),
             ("both", "flip", "swap both\nanswer = source answer")],
            [("S1_same", "change", "same S1, other digits\nanswer changes"),
             ("S2_same", "change", "same S2, other digits\nanswer changes"), None]]
    w = 0.17 * len(labels) + 0.5
    h = 0.085 * len(rps) + 0.9
    H = 2 * h + 1.9
    fig = plt.figure(figsize=(4 * w + 1.4, H))
    gs = fig.add_gridspec(2, 6, width_ratios=[w, w, w, 0.07 * w, 0.25 * w, w], wspace=0.12, hspace=1.15 / h, top=1 - 1.05 / H)

    def style(ax, ylabels: bool) -> None:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.tick_params(length=2, labelsize=6.5, colors="#52514e")
        step = max(1, len(rps) // 8)
        ax.set_yticks(range(0, len(rps), step))
        ax.set_yticklabels([rps[i] for i in range(0, len(rps), step)] if ylabels else [])
        for sp in ax.spines.values():
            sp.set_color("#d9d8d4")
            sp.set_linewidth(0.6)

    im = None
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            if cell is None or cell[0] not in rates:
                ax.axis("off")
                continue
            k, key, title = cell
            im = ax.imshow(np.asarray(rates[k][key]["mean"]), aspect="auto", cmap=seq, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=7.5, loc="left", color="#0b0b0b")
            style(ax, c == 0)
            if c == 0:
                ax.set_ylabel("read point (layer)", fontsize=7.5)
    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.set_label("rate (mean over bases)", fontsize=7)
    cbar.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 5])
    ax.imshow(np.asarray(read["count"]), aspect="auto", cmap=ListedColormap([CAT["none"], "#9ec5f4", "#184f95"]), vmin=-0.5,
              vmax=2.5, interpolation="nearest")
    ax.set_title(f"variables present\n(of S1, {'S2' if two else 'c'})", fontsize=7.5, loc="left")
    style(ax, True)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=col) for col in (CAT["none"], "#9ec5f4", "#184f95")], ["0", "1", "2"],
              fontsize=6, loc="upper left", bbox_to_anchor=(0, -0.36 / h), ncol=3, frameon=False, handlelength=1)
    ax = fig.add_subplot(gs[1, 5])
    if "both" in rates and "hypotheses" in rates["both"]:
        hyp = rates["both"]["hypotheses"]
        stack = np.stack([np.asarray(hyp[x]["mean"]) for x in ("none", "S1", "S2", "both")])  # (4, M, P)
        win = np.where(stack.max(axis=0) >= 0.5, stack.argmax(axis=0), 0)
        ax.imshow(win, aspect="auto", cmap=ListedColormap([CAT["none"], CAT["S1"], CAT["S2"], CAT["both"]]), vmin=-0.5,
                  vmax=3.5, interpolation="nearest")
        ax.set_title("swap both: which\nprediction wins (>= 0.5)", fontsize=7.5, loc="left")
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=CAT[x]) for x in ("S1", "S2", "both", "none")],
                  [x.replace("S2", "S2" if two else "c") for x in ("S1' - S2", "S1 - S2'", "S1' - S2'", "base / none")], fontsize=6, loc="upper left",
                  bbox_to_anchor=(0, -0.36 / h), ncol=2, frameon=False, handlelength=1)
        style(ax, True)
    else:
        ax.axis("off")
    kept = ", ".join(f"{k} {v}" for k, v in result["n_kept"].items())
    fig.suptitle(f"Variable map: {result['template'].strip()}   ({model_name})\n"
                 f"few-shot accuracy {result['accuracy']:.2f}; bases kept per swap: {kept}", fontsize=8.5, x=0.01, y=1 - 0.1 / H, ha="left", va="top")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def run_varmap(cfg: VarmapConfig, run_id: str | None = None, config_path: str | None = None) -> Path:
    return VariableMap(cfg, run_id, config_path).run()

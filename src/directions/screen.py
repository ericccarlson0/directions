"""The task screen for a conditional task (docs/DECISIONS.md D62): which surface form the models do few-shot, without a
shortcut and with an aligned tokenization, before a full test is preregistered.

``IF(g>h, a+b, c+d)`` keeps D55's two sums and makes the choice between them a control computed from separate digits.
Seven forms: max (D55's baseline), the conditional in three surfaces (``IF``, the C ternary, Python's conditional
expression, where the condition follows the first branch), with the comparison's sign mixed, with the condition given
as a digit, and with a condition that is itself a sum. Per form: accuracy, per cell (which branch is chosen x whether
the chosen sum is the larger), the shortcut, branch and sign gaps, the kinds of error, and the tokenization.

    uv run directions screen --config configs/screen_qwen3_8b.yaml --run-id screen_qwen3_8b_seed20260916
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .config import SCREEN_FORMATS, ScreenConfig
from .determinism import compare_arrays
from .model import (
    ModelBackend,
    environment_metadata,
    gpu_peak_memory,
    set_torch_determinism,
)
from .prompts import Prompt
from .runinfo import git_info, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .tasks import Item
from .varmap import bootstrap_rate

FORMATS = SCREEN_FORMATS
SELECTION_ORDER = ("if_mixed", "if_gt", "ternary", "python")  # D62's rule: the first that passes on the 8B and the 14B
SUM_PAIRS = [(a, b) for a in range(10) for b in range(10) if a + b <= 9]  # the 55 single-digit sums' digit pairs


@dataclass(frozen=True)
class CondExpr:
    """One query of a form: the condition's digits ``g, h`` (or ``k``, the given condition), its sign ``cmp``, and the
    branches ``a+b`` and ``c+d``. ``max`` has no condition: it chooses the larger sum."""

    form: str
    a: int
    b: int
    c: int
    d: int
    g: int = 0
    h: int = 0
    cmp: str = ">"
    k: int = 0

    @property
    def S1(self) -> int:
        return self.a + self.b

    @property
    def S2(self) -> int:
        return self.c + self.d

    @property
    def C(self) -> int:
        """1 when the first branch is chosen."""
        if self.form == "max":
            return int(self.S1 > self.S2)
        if self.form == "if_given":
            return int(self.k == 1)
        if self.form == "if_sum":
            return int(self.g + self.h > 9)
        return int(self.g < self.h) if self.cmp == "<" else int(self.g > self.h)

    @property
    def Y(self) -> int:
        return self.S1 if self.C else self.S2

    @property
    def other(self) -> int:
        """The sum not chosen."""
        return self.S2 if self.C else self.S1

    @property
    def M(self) -> int:
        """1 when the chosen sum is the larger (always so on max)."""
        return int(self.Y > self.other)

    @property
    def cell(self) -> tuple[Any, ...]:
        if self.form == "max":
            return (self.C,)
        if self.form == "if_mixed":
            return (self.cmp, self.C, self.M)
        return (self.C, self.M)

    def parts(self) -> list[tuple[str, str | None]]:
        """The text as (piece, slot) pairs: the slot names the digit or sign a piece is (None for the template)."""
        b1 = [(str(self.a), "a"), ("+", None), (str(self.b), "b")]
        b2 = [(str(self.c), "c"), ("+", None), (str(self.d), "d")]
        cond = [(str(self.g), "g"), (self.cmp, "cmp"), (str(self.h), "h")]
        if self.form == "max":
            return [("max(", None), *b1, (", ", None), *b2, (")", None)]
        if self.form == "if_given":
            return [("IF(", None), (str(self.k), "k"), (", ", None), *b1, (", ", None), *b2, (")", None)]
        if self.form == "if_sum":
            cond = [(str(self.g), "g"), ("+", None), (str(self.h), "h"), (">9", None)]
        if self.form == "ternary":
            return [*cond, (" ? ", None), *b1, (" : ", None), *b2]
        if self.form == "python":
            return [("(", None), *b1, (" if ", None), *cond, (" else ", None), *b2, (")", None)]
        return [("IF(", None), *cond, (", ", None), *b1, (", ", None), *b2, (")", None)]

    def text(self) -> str:
        return "".join(p for p, _ in self.parts())

    def slots(self) -> dict[int, str]:
        """Character index in :meth:`text` -> the slot of the digit or sign there."""
        out, i = {}, 0
        for piece, slot in self.parts():
            if slot is not None:
                out.update({i + j: slot for j in range(len(piece))})
            i += len(piece)
        return out


def cells(form: str) -> list[tuple[Any, ...]]:
    if form == "max":
        return [(1,), (0,)]
    if form == "if_mixed":
        return [(s, c, m) for s in (">", "<") for c in (1, 0) for m in (1, 0)]
    return [(c, m) for c in (1, 0) for m in (1, 0)]


def random_expr(form: str, rng: np.random.Generator) -> CondExpr | None:
    """One expression drawn uniformly from the form's valid ones (None for a rejected draw)."""
    (a, b), (c, d) = SUM_PAIRS[int(rng.integers(55))], SUM_PAIRS[int(rng.integers(55))]
    g, h = int(rng.integers(10)), int(rng.integers(10))
    cmp = ("<", ">")[int(rng.integers(2))] if form == "if_mixed" else ">"
    e = CondExpr(form, a, b, c, d, g=g, h=h, cmp=cmp, k=int(rng.integers(2)))
    if e.S1 == e.S2 or (g == h and form not in ("max", "if_given", "if_sum")):
        return None
    if form in ("max", "if_given"):  # no condition digits: fixed, so that equal draws are one expression
        e = CondExpr(form, a, b, c, d, k=e.k)
    return e


def draw_balanced(form: str, n: int, rng: np.random.Generator, exclude: set[str]) -> list[CondExpr]:
    """``n`` distinct expressions, ``n / len(cells)`` per cell, none of whose texts is in ``exclude``, in a random
    order."""
    cs = cells(form)
    quota = {c: n // len(cs) for c in cs}
    out: dict[str, CondExpr] = {}
    while any(quota.values()):
        e = random_expr(form, rng)
        if e is None or e.text() in exclude or e.text() in out or not quota.get(e.cell):
            continue
        out[e.text()] = e
        quota[e.cell] -= 1
    items = list(out.values())
    return [items[i] for i in rng.permutation(len(items))]


def draw_form(seed: int, form: str, n_queries: int, n_demo_pool: int) -> tuple[list[CondExpr], list[CondExpr]]:
    """The demonstration pool, then the queries (disjoint from it), both balanced over the cells."""
    pool = draw_balanced(form, n_demo_pool, rng_for(seed, "screen", form, "pool"), set())
    queries = draw_balanced(form, n_queries, rng_for(seed, "screen", form, "queries"), {e.text() for e in pool})
    return pool, queries


def demos_for(pool: list[CondExpr], n_shots: int, rng: np.random.Generator) -> list[CondExpr]:
    """``n_shots`` demonstrations from the pool, the same number per cell, in a random order."""
    cs = cells(pool[0].form)
    by_cell = [[e for e in pool if e.cell == c] for c in cs]
    per = n_shots // len(cs)
    chosen = [x[j] for x in by_cell for j in rng.choice(len(x), size=per, replace=False)]
    return [chosen[j] for j in rng.permutation(len(chosen))]


def render(demos: list[CondExpr], query: CondExpr) -> Prompt:
    """D45's rendering: ``{expr} = {Y}`` per demonstration, one per line, then ``{expr} = ``."""
    lines = [f"{e.text()} = {e.Y}" for e in demos]
    return Prompt(prompt="\n".join(lines + [f"{query.text()} = "]), target=f"{query.Y}", query=Item(query.text(), str(query.Y)),
                  demos=tuple(Item(e.text(), str(e.Y)) for e in demos))


def query_labels(backend: ModelBackend, prompt: Prompt, query: CondExpr) -> list[str]:
    """The query's tokens, each labelled by its text with every digit or sign named by its slot (g, cmp, h, k, a, b, c,
    d); the trailing space, the answer position, is ``␣``. Equal lists across queries mean token-aligned prompts."""
    return query_tokens(backend, prompt, query)[2]


def query_tokens(backend: ModelBackend, prompt: Prompt, query: CondExpr) -> tuple[list[int], list[int], list[str]]:
    """The query's token positions (absolute indices), its token ids and its labels (:func:`query_labels`)."""
    qtext = f"{query.text()} = "
    ids, _ = backend._encode_prompt(prompt)
    _, end = backend.context_span(prompt, len(prompt.prompt) - len(qtext))
    slots, labels, cursor = query.slots(), [], 0
    for tid in ids[end:]:
        piece = backend.tokenizer.decode([tid])
        if not piece or qtext.find(piece, cursor) != cursor:
            raise ValueError(f"query tokens do not tile the query text {qtext!r}")
        lab = "".join(f"<{slots[cursor + j]}>" if cursor + j in slots else ch for j, ch in enumerate(piece))
        labels.append("␣" if piece == " " and cursor + 1 == len(qtext) else lab)
        cursor += len(piece)
    if cursor != len(qtext):
        raise ValueError(f"query tokens cover {cursor} of {len(qtext)} characters of {qtext!r}")
    return list(range(end, len(ids))), list(ids[end:]), labels


def _boot_means(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return rng.multinomial(len(x), np.full(len(x), 1.0 / len(x)), size=n_boot) @ x / len(x)


def gap(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float) -> dict[str, float]:
    """mean(x) - mean(y) with a percentile interval, the two groups resampled independently."""
    if len(x) == 0 or len(y) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    d = _boot_means(x, rng, n_boot) - _boot_means(y, rng, n_boot)
    return {"mean": float(np.mean(x) - np.mean(y)), "lo": float(np.quantile(d, alpha / 2)), "hi": float(np.quantile(d, 1 - alpha / 2))}


def _rate(x: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float) -> dict[str, float]:
    return {k: float(v) for k, v in bootstrap_rate(np.asarray(x, dtype=np.float64), rng, n_boot, alpha).items()}


def measures(queries: list[CondExpr], argmax: np.ndarray, tid: np.ndarray, rng: np.random.Generator, n_boot: int,
             alpha: float) -> dict[str, Any]:
    """Accuracy overall and per cell, the gaps, and the kinds of error."""
    Y = np.array([e.Y for e in queries])
    other = np.array([e.other for e in queries])
    correct = argmax == tid[Y]
    C, M = np.array([e.C for e in queries]), np.array([e.M for e in queries])
    out: dict[str, Any] = {"n": len(queries), "accuracy": _rate(correct, rng, n_boot, alpha), "cells": {}}
    for c in cells(queries[0].form):
        sel = np.array([e.cell == c for e in queries])
        out["cells"]["/".join(map(str, c))] = {"n": int(sel.sum()), **_rate(correct[sel], rng, n_boot, alpha)}
    gaps = {"branch": gap(correct[C == 1], correct[C == 0], rng, n_boot, alpha)}
    if queries[0].form != "max":
        gaps["shortcut"] = gap(correct[M == 1], correct[M == 0], rng, n_boot, alpha)
    if queries[0].form == "if_mixed":
        S = np.array([e.cmp for e in queries])
        gaps["sign"] = gap(correct[S == ">"], correct[S == "<"], rng, n_boot, alpha)
    out["gaps"] = gaps
    wrong = ~correct
    digit = np.isin(argmax, tid)
    out["errors"] = {"n": int(wrong.sum()), "other_branch": int((wrong & (argmax == tid[other])).sum()),
                     "other_digit": int((wrong & digit & (argmax != tid[other])).sum()), "not_a_digit": int((wrong & ~digit).sum())}
    return out


def passes(m: dict[str, Any], aligned: bool, accuracy_lo: float, gap_max: float) -> dict[str, Any]:
    """D62's criteria: accuracy's lower bound, every gap's interval within +-gap_max, one tokenization."""
    gaps_ok = all(-gap_max <= g["lo"] and g["hi"] <= gap_max for g in m["gaps"].values())
    acc_ok = m["accuracy"]["lo"] >= accuracy_lo
    return {"accuracy": bool(acc_ok), "gaps": bool(gaps_ok), "aligned": bool(aligned), "pass": bool(acc_ok and gaps_ok and aligned)}


class Screen:
    def __init__(self, cfg: ScreenConfig, run_id: str | None = None, config_path: str | None = None) -> None:
        self.cfg = cfg
        slug = cfg.model.name.split("/")[-1].lower() if cfg.model.backend == "hf" else "toy"
        run_id = run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_screen_{slug}_seed{cfg.seed}"
        self.root = Path(cfg.output_dir) / run_id
        if self.root.exists():
            raise FileExistsError(f"run directory already exists: {self.root}")
        (self.root / "core").mkdir(parents=True)
        self.log = setup_logging(self.root / "log.txt")
        self.metadata: dict[str, Any] = {
            "version": __version__, "config": asdict(cfg), "config_path": config_path, "run_id": run_id, "seed": cfg.seed,
            "git": git_info(), "environment": environment_metadata(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run(self) -> Path:
        cfg, log = self.cfg, self.log
        set_torch_determinism(derive_seed(cfg.seed, "torch"))
        self.backend = backend = ModelBackend(cfg.model, run_seed=cfg.seed)
        self.metadata["model"] = backend.metadata()
        tid = self._digit_ids()
        result: dict[str, Any] = {"criteria": {"accuracy_lo": cfg.accuracy_lo, "gap_max": cfg.gap_max}, "forms": {}}
        arrays: dict[str, np.ndarray] = {}
        summary: dict[str, Any] = {}
        for form in cfg.formats:
            pool, queries = draw_form(cfg.seed, form, cfg.n_queries, cfg.n_demo_pool)
            result["forms"][form] = {"example_query": f"{queries[0].text()} = {queries[0].Y}"}
            for k in cfg.n_shots:
                t0 = time.time()
                prompts = [render(demos_for(pool, k, rng_for(cfg.seed, "screen", form, k, "demos", i)), q)
                           for i, q in enumerate(queries)]
                labels = [query_labels(backend, p, q) for p, q in zip(prompts, queries)]
                aligned = all(lab == labels[0] for lab in labels)
                res = backend.run(prompts, token_ids=tid)
                self._determinism(prompts, tid, res)
                m = measures(queries, res.query_argmax, tid, rng_for(cfg.seed, "screen", form, k, "bootstrap"), cfg.n_boot, cfg.alpha)
                m.update({"aligned": aligned, "labels": labels[0], "n_misaligned": int(sum(lab != labels[0] for lab in labels)),
                          "passes": passes(m, aligned, cfg.accuracy_lo, cfg.gap_max)})
                if k == cfg.n_shots[0]:
                    m["example_prompt"] = prompts[0].prompt
                result["forms"][form][str(k)] = m
                summary.setdefault(form, {})[str(k)] = {"accuracy": m["accuracy"], **m["passes"]}
                arrays[f"{form}__{k}__argmax"] = res.query_argmax.astype(np.int64)
                arrays[f"{form}__{k}__digit_logprobs"] = res.token_logprobs.astype(np.float32)
                self.metadata.setdefault("timings_seconds", {})[f"{form}/{k}"] = round(time.time() - t0, 1)
                a, g = m["accuracy"], m["gaps"]
                log.info("%s, %d shots: accuracy %.3f [%.3f, %.3f]; cells %s; gaps %s; errors %s; aligned %s -> %s", form, k,
                         a["mean"], a["lo"], a["hi"], {c: round(v["mean"], 2) for c, v in m["cells"].items()},
                         {n: [round(v["mean"], 3), round(v["lo"], 3), round(v["hi"], 3)] for n, v in g.items()}, m["errors"],
                         aligned, "pass" if m["passes"]["pass"] else "fail")
            arrays[f"{form}__Y"] = np.array([e.Y for e in queries])
            arrays[f"{form}__other"] = np.array([e.other for e in queries])
            arrays[f"{form}__C"] = np.array([e.C for e in queries])
            arrays[f"{form}__M"] = np.array([e.M for e in queries])
        write_json(self.root / "core" / "screen.json", result)
        np.savez_compressed(self.root / "core" / "screen_arrays.npz", **arrays)
        write_json(self.root / "core" / "summary.json", summary)
        self.metadata["gpu_peak_memory_gib"] = gpu_peak_memory()
        self.metadata["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.root / "metadata.json", self.metadata)
        log.info("done -> %s", self.root)
        return self.root

    def _digit_ids(self) -> np.ndarray:
        ids = []
        for k in range(10):
            enc = self.backend.tokenizer.encode(str(k), add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"the digit {k} is not a single token on this tokenizer: {enc}")
            ids.append(enc[0])
        return np.array(ids, dtype=np.int64)

    def _determinism(self, prompts: list[Prompt], tid: np.ndarray, res: Any) -> None:
        """Once per run: the first batch again, bit-identical to the first pass (D23)."""
        if not self.cfg.determinism_check or "determinism_check" in self.metadata:
            return
        bs = min(self.backend.cfg.batch_size, len(prompts))
        again = self.backend.run(prompts[:bs], token_ids=tid)
        cmp = compare_arrays({"lp": res.token_logprobs[:bs], "am": res.query_argmax[:bs]},
                             {"lp": again.token_logprobs, "am": again.query_argmax})
        self.metadata["determinism_check"] = {"identical": cmp["identical"]}
        if not cmp["identical"]:
            raise RuntimeError("determinism check failed: a repeated batch differs")


def run_screen(cfg: ScreenConfig, run_id: str | None = None, config_path: str | None = None) -> Path:
    return Screen(cfg, run_id=run_id, config_path=config_path).run()

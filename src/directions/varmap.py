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
from .model import ModelBackend, SitePatch, environment_metadata, gpu_peak_memory, set_torch_determinism
from .prompts import Prompt
from .runinfo import RejectionLog, git_info, setup_logging, write_json
from .seeds import derive_seed, rng_for
from .tasks import Item

# --------------------------------------------------------------------------- #
# The tasks and their high-level causal model
# --------------------------------------------------------------------------- #

TASKS = ("two_lane", "one_lane", "three_lane", "max", "maxmin")
SOURCE_TYPES = {"two_lane": ("S1", "S2", "S1_same", "S2_same", "both"), "one_lane": ("S1", "S2", "S1_same", "both"),
                "three_lane": ("S1", "S2", "S3", "S1_same", "S2_same", "S3_same", "all"),
                "max": ("S1", "S2", "S1_same", "S2_same", "both"),
                # D57: the operator swap (the other of max and min, the same digits) and D55's swaps on the base's operator
                "maxmin": ("op", "S1", "S2", "S1_same", "S2_same", "both")}
# the variable each value-changing type moves, and the same-value type that tests it as a value
VALUE_TEST = {"S1": "S1_same", "S2": "S2_same", "S3": "S3_same"}
# the sub-results counted at a site (D45; three brackets D51)
BRACKETS = {"two_lane": ("S1", "S2"), "one_lane": ("S1", "S2"), "three_lane": ("S1", "S2", "S3"), "max": ("S1", "S2"),
            "maxmin": ("S1", "S2")}
TEMPLATES = {"two_lane": "(a+b) - (c+d) = ", "one_lane": "(a+b) - c = ", "three_lane": "(a+b) - (c+d) - (e+f) = ",
             "max": "max(a+b, c+d) = ", "maxmin": "op(a+b, c+d) = "}
OPERATORS = ("max", "min")  # the selection tasks (D55, D57): one operator token chooses the larger or the smaller sum
DIGITS = ("a", "b", "c", "d", "e", "f")


@dataclass(frozen=True)
class Expr:
    """``(a+b) - (c+d)`` (two lanes), ``(a+b) - c`` (one lane, ``d`` None), ``(a+b) - (c+d) - (e+f)`` (three lanes,
    D51) or, with ``op`` "max" or "min", ``max(a+b, c+d)`` (D55) and ``min(a+b, c+d)`` (D57)."""

    a: int
    b: int
    c: int
    d: int | None = None
    e: int | None = None
    f: int | None = None
    op: str = "-"

    @property
    def S1(self) -> int:
        return self.a + self.b

    @property
    def S2(self) -> int:
        return self.c + (self.d or 0)

    @property
    def S3(self) -> int:
        """The third bracket's sum (0 without one)."""
        return 0 if self.e is None else self.e + (self.f or 0)

    @property
    def T(self) -> int:
        """S1 - S2, the left-to-right intermediate of three brackets (D51)."""
        return self.S1 - self.S2

    @property
    def U(self) -> int:
        """S2 + S3, the subtrahends merged first (D51)."""
        return self.S2 + self.S3

    @property
    def C(self) -> int:
        """The comparison bit of max (D55): 1 when the first sum is the larger."""
        return int(self.S1 > self.S2)

    @property
    def L(self) -> int:
        """The smaller sum of max (D55)."""
        return min(self.S1, self.S2)

    @property
    def Y(self) -> int:
        if self.op == "max":
            return max(self.S1, self.S2)
        if self.op == "min":
            return min(self.S1, self.S2)
        return self.S1 - self.S2 - self.S3

    def text(self) -> str:
        if self.op in OPERATORS:
            return f"{self.op}({self.a}+{self.b}, {self.c}+{self.d})"
        second = f"({self.c}+{self.d})" if self.d is not None else f"{self.c}"
        third = f" - ({self.e}+{self.f})" if self.e is not None else ""
        return f"({self.a}+{self.b}) - {second}{third}"

    def digit_slots(self) -> dict[int, str]:
        """Character index in :meth:`text` -> the digit's name."""
        if self.op in OPERATORS:
            return {4: "a", 6: "b", 9: "c", 11: "d"}
        slots = {1: "a", 3: "b"}
        slots.update({9: "c", 11: "d"} if self.d is not None else {8: "c"})
        if self.e is not None:
            slots.update({17: "e", 19: "f"})
        return slots


def all_exprs(task: str) -> list[Expr]:
    """Every expression of the task whose answer is a single digit 0..9 (in the order of the digits' product)."""
    if task == "two_lane":
        out = [Expr(a, b, c, d) for a, b, c, d in itertools.product(range(10), repeat=4)]
    elif task == "one_lane":
        out = [Expr(a, b, c) for a, b, c in itertools.product(range(10), repeat=3)]
    elif task == "three_lane":
        D = np.array(list(itertools.product(range(10), repeat=6)), dtype=np.int64)
        y = D[:, 0] + D[:, 1] - D[:, 2] - D[:, 3] - D[:, 4] - D[:, 5]
        return [Expr(*map(int, row)) for row in D[(y >= 0) & (y <= 9)]]
    elif task == "max":  # D55: both sums single digits and unequal (the comparison has a winner)
        return [Expr(a, b, c, d, op="max") for a, b, c, d in itertools.product(range(10), repeat=4)
                if a + b <= 9 and c + d <= 9 and a + b != c + d]
    elif task == "maxmin":  # D57: max's expressions, each with either operator
        return [Expr(a, b, c, d, op=op) for op in OPERATORS for a, b, c, d in itertools.product(range(10), repeat=4)
                if a + b <= 9 and c + d <= 9 and a + b != c + d]
    else:
        raise ValueError(f"unknown varmap task {task!r}")
    return [e for e in out if 0 <= e.Y <= 9]


def predictions(base: Expr, src: Expr) -> dict[str, int]:
    """The answer the swapped run gives if the site holds: the source's S1 only, its S2 only, both (its answer), or
    nothing of the source (the base answer). Three brackets (D51): each sum alone, the intermediates T (S1 and S2
    together) and U (S2 and S3 together), all (the source's answer) or none. Max (D55): each sum alone, the
    comparison alone (C: the base's sum that the source's comparison picks), both or none. Min (D57): the same,
    with the base's operation choosing the smaller sum."""
    if base.op == "max":
        return {"S1": max(src.S1, base.S2), "S2": max(base.S1, src.S2), "C": base.S1 if src.C else base.S2,
                "both": src.Y, "none": base.Y}
    if base.op == "min":
        return {"S1": min(src.S1, base.S2), "S2": min(base.S1, src.S2), "C": base.S2 if src.C else base.S1,
                "both": min(src.S1, src.S2), "none": base.Y}
    if base.e is None:
        return {"S1": src.S1 - base.S2, "S2": base.S1 - src.S2, "both": src.Y, "none": base.Y}
    return {"S1": src.S1 - base.S2 - base.S3, "S2": base.S1 - src.S2 - base.S3, "S3": base.S1 - base.S2 - src.S3,
            "T": src.T - base.S3, "U": base.S1 - src.U, "all": src.Y, "none": base.Y}


def prediction(kind: str, base: Expr, src: Expr) -> int:
    """The answer a swap of ``kind`` predicts under its own hypothesis (D45); the operator swap, the other operation's
    answer on the same digits (D57)."""
    if kind == "op":
        return src.Y
    if kind.endswith("_same"):
        return base.Y
    return predictions(base, src)[kind]


def valid_source(kind: str, base: Expr, src: Expr) -> bool:
    if src == base:
        return False
    if kind == "op":  # D57: the same digits, the other operator
        return (src.a, src.b, src.c, src.d) == (base.a, base.b, base.c, base.d) and src.op != base.op
    if src.op != base.op:  # every other swap keeps the operator (only maxmin mixes operators)
        return False
    same_first = (src.a, src.b) == (base.a, base.b)
    same_second = (src.c, src.d) == (base.c, base.d)
    if base.e is not None:
        return bool(source_masks(kind, base, np.array([[src.a, src.b, src.c, src.d, src.e, src.f]]))[0])
    if base.op in OPERATORS:  # D55, D57: a value-changing swap must move the answer under its own prediction
        p = predictions(base, src)
        if kind == "S1":
            return same_second and src.S1 != base.S1 and p["S1"] != base.Y
        if kind == "S2":
            return same_first and src.S2 != base.S2 and p["S2"] != base.Y
        if kind == "both":
            return src.S1 != base.S1 and src.S2 != base.S2 and src.Y != base.Y
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


def source_masks(kind: str, base: Expr, D: np.ndarray) -> np.ndarray:
    """Three brackets (D51): which candidates (digits ``D``, (N, 6), every one a valid expression) are sources of
    ``kind`` for ``base``. One bracket's sum changes with the other brackets' digits kept (S1, S2, S3); a bracket's
    digits change with its sum and the other brackets kept (Sk_same); or every sum changes with the seven
    predictions single digits and pairwise distinct (all)."""
    S = [D[:, 0] + D[:, 1], D[:, 2] + D[:, 3], D[:, 4] + D[:, 5]]
    bS = [base.S1, base.S2, base.S3]
    bd = [(base.a, base.b), (base.c, base.d), (base.e, base.f)]
    same = [(D[:, 2 * j] == bd[j][0]) & (D[:, 2 * j + 1] == bd[j][1]) for j in range(3)]
    if kind in ("S1", "S2", "S3"):
        j = int(kind[1]) - 1
        others = [same[i] for i in range(3) if i != j]
        return others[0] & others[1] & (S[j] != bS[j])
    if kind in ("S1_same", "S2_same", "S3_same"):
        j = int(kind[1]) - 1
        others = [same[i] for i in range(3) if i != j]
        return others[0] & others[1] & ~same[j] & (S[j] == bS[j])
    if kind == "all":
        Y = S[0] - S[1] - S[2]
        vals = [S[0] - bS[1] - bS[2], bS[0] - S[1] - bS[2], bS[0] - bS[1] - S[2], S[0] - S[1] - bS[2],
                bS[0] - S[1] - S[2], Y, np.full(len(D), base.Y)]
        ok = (S[0] != bS[0]) & (S[1] != bS[1]) & (S[2] != bS[2])
        for v in vals:
            ok &= (v >= 0) & (v <= 9)
        for i in range(len(vals)):
            for k in range(i + 1, len(vals)):
                ok &= vals[i] != vals[k]
        return ok
    raise ValueError(f"unknown source type {kind!r}")


def draw_source(kind: str, base: Expr, pool: Sequence[Expr], rng: np.random.Generator,
                digits: np.ndarray | None = None) -> Expr | None:
    """One valid source of ``kind`` from ``pool``, uniformly (three brackets: from the pool's digit array)."""
    if digits is not None:
        cands = np.flatnonzero(source_masks(kind, base, digits))
        return pool[int(cands[int(rng.integers(len(cands)))])] if len(cands) else None
    cands = [e for e in pool if valid_source(kind, base, e)]
    return cands[int(rng.integers(len(cands)))] if cands else None


def sample_bases(seed: int, task: str, n_bases: int, n_shots: int, n_demo_pool: int, rejections: RejectionLog | None = None
                 ) -> tuple[list[Expr], list[dict[str, Expr]], list[list[Expr]], dict[str, int]]:
    """The D45 draw: the demonstration pool, then the bases in a seeded order, each with one source per type and its
    demonstrations (shared by the base and its sources). An expression without a valid source of some type (a sum
    with a single digit pair, e.g. 0 or 18) is skipped and counted (and logged to ``rejections``)."""
    exprs = all_exprs(task)
    rng = rng_for(seed, "varmap", task, "split")
    order = rng.permutation(len(exprs))
    demo_pool = [exprs[i] for i in order[:n_demo_pool]]
    rest = [exprs[i] for i in order[n_demo_pool:]]
    bases, sources, demos, skipped = [], [], [], 0
    digits = (np.array([[e.a, e.b, e.c, e.d, e.e, e.f] for e in rest], dtype=np.int64) if task == "three_lane" else None)
    for base in rest:
        if len(bases) == n_bases:
            break
        i = len(bases)
        srcs = {k: draw_source(k, base, rest, rng_for(seed, "varmap", task, "source", i, skipped, k), digits)
                for k in SOURCE_TYPES[task]}
        if any(v is None for v in srcs.values()):
            skipped += 1
            if rejections is not None:
                rejections.add("source", task, "no valid source of some type", {"base": base.text(),
                               "missing": [k for k, v in srcs.items() if v is None]})
            continue
        bases.append(base)
        sources.append(srcs)
        r = rng_for(seed, "varmap", task, "demos", i)
        demos.append(balanced_demos(demo_pool, n_shots, r) if task == "maxmin"
                     else [demo_pool[j] for j in r.choice(len(demo_pool), size=n_shots, replace=False)])
    return bases, sources, demos, {"n_expressions": len(exprs), "skipped_without_source": skipped, "n_bases": len(bases)}


def balanced_demos(pool: Sequence[Expr], n_shots: int, rng: np.random.Generator) -> list[Expr]:
    """D57: half of the demonstrations of each operator, in a random order, so that the query's operator token is
    what sets the operation."""
    if n_shots % 2:
        raise ValueError("maxmin needs an even number of demonstrations (half of each operator)")
    by_op = [[e for e in pool if e.op == op] for op in OPERATORS]
    if min(len(x) for x in by_op) < n_shots // 2:
        raise ValueError("the demonstration pool holds too few expressions of an operator")
    chosen = [x[j] for x in by_op for j in rng.choice(len(x), size=n_shots // 2, replace=False)]
    return [chosen[j] for j in rng.permutation(n_shots)]


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
    for var in BRACKETS[task]:
        present = (np.asarray(rates[var]["flip"]["lo"]) >= present_flip) if var in rates else np.zeros(shape, dtype=bool)
        out[f"present_{var}"] = present
        test = VALUE_TEST[var]
        if test in rates:
            value = present & (np.asarray(rates[test]["change"]["hi"]) <= value_change)
            out[f"value_{var}"] = value
            out[f"digits_{var}"] = present & ~value
    out["count"] = sum(out[f"present_{var}"].astype(int) for var in BRACKETS[task])
    if "op" in rates:  # D57: the operator, present where swapping the site's state flips the operation
        out["present_O"] = np.asarray(rates["op"]["flip"]["lo"]) >= present_flip
    return out


def operator_readouts(read: dict[str, np.ndarray], labels: list[str], read_points: list[int]) -> dict[str, Any]:
    """D57's readouts of the operator map: R_O (the first read point with O present at the answer token), G (the first
    with both sums present there), O_out (the last with O present at an operator token) and the route (the other
    tokens where O is present, with the read points)."""
    ans = len(labels) - 1
    op_cols = [j for j, lab in enumerate(labels) if lab == "op"]
    present = np.asarray(read["present_O"])
    count = np.asarray(read["count"])

    def first(mask: np.ndarray) -> int | None:
        idx = np.flatnonzero(mask)
        return int(read_points[idx[0]]) if len(idx) else None

    at_op = present[:, op_cols].any(axis=1) if op_cols else np.zeros(len(read_points), dtype=bool)
    route = {f"{j}:{labels[j]}": [int(read_points[m]) for m in np.flatnonzero(present[:, j])]
             for j in range(len(labels)) if j != ans and j not in op_cols and present[:, j].any()}
    out = {"R_O": first(present[:, ans]), "G": first(count[:, ans] == 2),
           "O_out": int(read_points[np.flatnonzero(at_op)[-1]]) if at_op.any() else None, "route": route}
    R, G, O_out = out["R_O"], out["G"], out["O_out"]
    if R is None or G is None:
        out["reading"] = "undecided (no R_O or no G)"
    elif route and min(min(v) for v in route.values()) < R:
        out["reading"] = "relayed"
    elif R <= G - 2:
        out["reading"] = "carried ahead and held"
    elif R >= G + 2:
        out["reading"] = "late"
    elif O_out is not None and O_out >= R - 1:
        out["reading"] = "taken up where the operands meet"
    else:
        out["reading"] = "relayed (spread over context)"
    return out


def query_tokens(backend: ModelBackend, prompt: Prompt, query: Expr, op_label: str | None = None
                 ) -> tuple[list[int], list[int], list[str]]:
    """The query's token positions (absolute indices), its token ids, and a label per token in which each operand
    digit is named by its slot (a, b, c, d) and the answer position is ``␣`` (the trailing space). With
    ``op_label`` (maxmin, D57), the tokens of the operator word are labelled so, whichever operator it is."""
    qtext = query_chars(query)
    context_chars = len(prompt.prompt) - len(qtext)
    ids, _ = backend._encode_prompt(prompt)
    _, end = backend.context_span(prompt, context_chars)
    q_ids = ids[end:]
    labels, cursor = [], 0
    slots = query.digit_slots()
    op_end = len(query.op) if op_label is not None and query.op in OPERATORS else 0
    for tid in q_ids:
        piece = backend.tokenizer.decode([tid])
        start = qtext.find(piece, cursor) if piece else cursor
        if start != cursor:
            raise ValueError(f"query tokens do not tile the query text {qtext!r}")
        if cursor < op_end:
            labels.append(op_label)
            cursor += len(piece)
            continue
        lab = "".join(slots.get(cursor + k, ch) for k, ch in enumerate(piece))
        if lab in DIGITS and not any(cursor + k in slots for k in range(len(piece))):
            lab = f"'{lab}'"  # a letter of the template (max's "a", D55), not the digit slot of that name
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
        self.metadata["gpu_peak_memory_gib"] = gpu_peak_memory()  # the run's peak: the headroom a change can use
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
        cfg = self.cfg
        bases, sources, demos, info = sample_bases(cfg.seed, task, cfg.n_bases, cfg.n_shots, cfg.n_demo_pool, self.rejections)
        self.metadata.setdefault("sampling", {})[task] = info
        return bases, sources, demos

    def _task(self, task: str) -> dict[str, Any]:
        cfg, backend, log = self.cfg, self.backend, self.log
        bases, sources, demos = self._sample(task)
        types = SOURCE_TYPES[task]
        base_prompts = [render(demos[i], b) for i, b in enumerate(bases)]
        op_label = "op" if task == "maxmin" else None  # D57: the operator token, whichever operator
        pos0, _ids0, labels = query_tokens(backend, base_prompts[0], bases[0], op_label)
        P = len(pos0)
        positions = np.zeros((len(bases), P), dtype=np.int64)
        keep = np.ones(len(bases), dtype=bool)
        src_prompts: dict[str, list[Prompt | None]] = {k: [] for k in types}
        for i, base in enumerate(bases):
            pos, ids, _ = query_tokens(backend, base_prompts[i], base, op_label)
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
                spos, sids, _ = query_tokens(backend, sp, src, op_label)
                # only the digits may differ, and the operator token in an operator swap (D57)
                aligned = spos == pos and all(x == y for x, y, lab in zip(ids, sids, labels)
                                              if lab not in DIGITS and not (k == "op" and lab == "op"))
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
        by_op = ({op: float(base_ok[[i for i, b in enumerate(bases) if b.op == op]].mean()) for op in OPERATORS}
                 if task == "maxmin" else None)
        if by_op:
            log.info("%s: few-shot accuracy per operator %s", task, {k: round(v, 3) for k, v in by_op.items()})
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
            if k in ("both", "all"):
                hyp = {h: np.array([predictions(bases[i], sources[i][k])[h] for i in rows])
                       for h in predictions(bases[rows[0]], sources[rows[0]][k])}
                rates["hypotheses"] = {h: bootstrap_rate(am == tid[v][:, None, None], np.random.default_rng(brng_seed + 4), cfg.n_boot, cfg.alpha)
                                       for h, v in hyp.items()}
            out_rates[k] = rates
            log.info("%s/%s: %d bases, %d swaps; max flip rate %.2f, at the answer token's last read point %.2f", task, k, n, len(prompts),
                     float(rates["flip"]["mean"].max()), float(rates["flip"]["mean"][-1, -1]))

        read = reading(out_rates, task, cfg.present_flip, cfg.value_change)
        operator = operator_readouts(read, labels, read_points) if "present_O" in read else None
        if operator is not None:
            log.info("%s: the operator: %s", task, operator)
        td = self.root / "core" / "tasks" / task
        result = {
            "task": task, "template": TEMPLATES[task], "labels": labels,
            "read_points": read_points, "n_bases": len(bases), "n_kept": n_kept, "accuracy": accuracy,
            "source_accuracy": {k: float(src_runs[k]["ok"][keep].mean()) for k in types},
            "examples": [{"base": bases[i].text(), "Y": bases[i].Y, "sources": {k: sources[i][k].text() for k in types},
                          "kept": [k for k in types if i in set(type_rows[k].tolist())]} for i in range(len(bases))],
            "rates": out_rates, "reading": read,
            "thresholds": {"present_flip": cfg.present_flip, "value_change": cfg.value_change},
            **({"accuracy_by_op": by_op, "operator": operator} if task == "maxmin" else {}),
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
                "sites_with_all": int((read["count"] == len(BRACKETS[task])).sum()),
                "first_value_S1": _first_site(read.get("value_S1"), read_points, labels),
                "first_value_S2": _first_site(read.get("value_S2"), read_points, labels),
                **({"first_value_S3": _first_site(read.get("value_S3"), read_points, labels)} if task == "three_lane" else {}),
                **({"accuracy_by_op": by_op, "operator": operator} if task == "maxmin" else {})}
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
CAT = {"none": "#ecebe7", "S1": "#2a78d6", "S2": "#eb6834", "S3": "#8a5cd0", "T": "#d9a521", "U": "#c2417f",
       "both": "#1baf7a", "all": "#1baf7a", "C": "#d9a521"}
HYP_LABEL = {"S1": "S1' - S2", "S2": "S1 - S2'", "both": "S1' - S2'", "none": "base / none",
             "S1_3": "S1' only", "S2_3": "S2' only", "S3_3": "S3' only", "T_3": "T' (S1', S2')", "U_3": "U' (S2', S3')",
             "all_3": "all (source answer)"}
HYP_LABEL_MAX = {"S1": "max(S1', S2)", "S2": "max(S1, S2')", "C": "C' on the base's sums", "both": "max(S1', S2')",
                 "none": "base / none"}
HYP_LABEL_MAXMIN = {"S1": "f(S1', S2)", "S2": "f(S1, S2')", "C": "C' on the base's sums", "both": "f(S1', S2')",
                    "none": "base / none"}  # D57: f, the base's operation


def plot_task(result: dict[str, Any], path: Path, model_name: str) -> None:
    """Heatmaps over (read point x query token). Left block: the flip rates of the value-changing swaps (top row)
    and the change rates of the same-value swaps (bottom row), on one colour scale. Right column: the count of
    variables present, and which prediction a swap from a source changing every bracket produces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    seq = LinearSegmentedColormap.from_list("seq", SEQ)
    labels = result["labels"]
    rps = result["read_points"]
    rates, read = result["rates"], result["reading"]
    task = result["task"]
    three = task == "three_lane"
    brackets = BRACKETS[task]
    combined = "all" if three else "both"
    second = "S2" if task != "one_lane" else "c"
    names = {"S1": "S1", "S2": second, "S3": "S3"}
    top = [(b, "flip", f"swap {names[b]}\nanswer follows source {names[b]}") for b in brackets]
    top.append((combined, "flip", f"swap {combined}\nanswer = source answer"))
    bottom = [(f"{b}_same", "change", f"same {names[b]}, other digits\nanswer changes") for b in brackets]
    bottom.append(("op", "flip", "swap the operator (max <-> min)\nanswer = the other operation's") if "op" in rates
                  else None)
    grid = [top, bottom]
    ncol = len(top)
    w = 0.17 * len(labels) + 0.5
    h = 0.085 * len(rps) + 0.9
    H = 2 * h + 1.9
    fig = plt.figure(figsize=((ncol + 1) * w + 1.4, H))
    gs = fig.add_gridspec(2, ncol + 3, width_ratios=[w] * ncol + [0.07 * w, 0.25 * w, w], wspace=0.12, hspace=1.15 / h,
                          top=1 - 1.05 / H)

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
    cax = fig.add_subplot(gs[:, ncol])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.set_label("rate (mean over bases)", fontsize=7)
    cbar.outline.set_visible(False)

    nb = len(brackets)
    counts = [CAT["none"]] + SEQ[2 : 2 + 2 * nb : 2][: nb - 1] + ["#184f95"]
    ax = fig.add_subplot(gs[0, ncol + 2])
    ax.imshow(np.asarray(read["count"]), aspect="auto", cmap=ListedColormap(counts), vmin=-0.5, vmax=nb + 0.5,
              interpolation="nearest")
    ax.set_title(f"variables present\n(of {', '.join(names[b] for b in brackets)})", fontsize=7.5, loc="left")
    style(ax, True)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=col) for col in counts], [str(i) for i in range(nb + 1)],
              fontsize=6, loc="upper left", bbox_to_anchor=(0, -0.36 / h), ncol=nb + 1, frameon=False, handlelength=1)
    ax = fig.add_subplot(gs[1, ncol + 2])
    if combined in rates and "hypotheses" in rates[combined]:
        hyp = rates[combined]["hypotheses"]
        order = ["none"] + [x for x in hyp if x != "none"]
        stack = np.stack([np.asarray(hyp[x]["mean"]) for x in order])  # (H, M, P)
        win = np.where(stack.max(axis=0) >= 0.5, stack.argmax(axis=0), 0)
        ax.imshow(win, aspect="auto", cmap=ListedColormap([CAT[x] for x in order]), vmin=-0.5, vmax=len(order) - 0.5,
                  interpolation="nearest")
        ax.set_title(f"swap {combined}: which\nprediction wins (>= 0.5)", fontsize=7.5, loc="left")
        legend = [x for x in order[1:]] + ["none"]
        text = [HYP_LABEL_MAX[x] if task == "max" else HYP_LABEL_MAXMIN[x] if task == "maxmin"
                else HYP_LABEL[f"{x}_3"] if three and x != "none" else HYP_LABEL[x] for x in legend]
        if task == "one_lane":
            text = [t.replace("S2", "c") for t in text]
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=CAT[x]) for x in legend], text, fontsize=6, loc="upper left",
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

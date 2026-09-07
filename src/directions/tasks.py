"""Deterministic mapping tasks, item filtering and the three disjoint query pools."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import data
from .data_extra import PARTICIPLE_DOUBLING, PARTICIPLE_EXCEPTIONS
from .seeds import rng_for


@dataclass(frozen=True)
class Item:
    input: str
    output: str


@dataclass
class Task:
    name: str
    items: list[Item]
    params: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Task builders
# --------------------------------------------------------------------------- #


def _check_params(name: str, params: dict[str, Any], allowed: dict[str, Any]) -> dict[str, Any]:
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(f"unknown params for task {name!r}: {sorted(unknown)}; allowed {sorted(allowed)}")
    out = dict(allowed)
    out.update(params)
    return out


def _from_pairs(name: str, pairs: list[tuple[str, str]]) -> Callable[[dict[str, Any]], Task]:
    def build(params: dict[str, Any]) -> Task:
        _check_params(name, params, {})
        return Task(name, [Item(i, o) for i, o in data.dedupe(pairs)])

    return build


def _arithmetic(params: dict[str, Any]) -> Task:
    p = _check_params("arithmetic", params, {"operation": "add", "operand": 3, "min": 0, "max": 300})
    op, k = p["operation"], int(p["operand"])
    ops = {"add": lambda n: n + k, "subtract": lambda n: n - k, "multiply": lambda n: n * k}
    if op not in ops:
        raise ValueError(f"unknown arithmetic operation {op!r}; choose from {sorted(ops)}")
    items = [Item(str(n), str(ops[op](n))) for n in range(int(p["min"]), int(p["max"]) + 1)]
    return Task("arithmetic", items, params=p)


def _add_two(params: dict[str, Any]) -> Task:
    """Two-operand addition ``"a + b" -> a+b`` over every ordered pair in [min, max]."""
    p = _check_params("add_two", params, {"min": 2, "max": 59})
    lo, hi = int(p["min"]), int(p["max"])
    items = [Item(f"{a} + {b}", str(a + b)) for a in range(lo, hi + 1) for b in range(lo, hi + 1)]
    return Task("add_two", items, params=p)


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """English words for 0 <= n <= 999 (American style, no 'and', hyphenated tens)."""
    if not 0 <= n <= 999:
        raise ValueError("number_to_words supports 0..999")
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (f"-{_ONES[o]}" if o else "")
    h, r = divmod(n, 100)
    return f"{_ONES[h]} hundred" + (f" {number_to_words(r)}" if r else "")


def _number_to_words(params: dict[str, Any]) -> Task:
    p = _check_params("number_to_words", params, {"min": 0, "max": 999})
    items = [Item(str(n), number_to_words(n)) for n in range(int(p["min"]), int(p["max"]) + 1)]
    return Task("number_to_words", items, params=p)


_VOWELS = "aeiou"


def present_participle(verb: str) -> str:
    """Rule-based ``-ing`` form with an explicit exception table (American spelling)."""
    if verb in PARTICIPLE_EXCEPTIONS:
        return PARTICIPLE_EXCEPTIONS[verb]
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith(("ee", "ye", "oe")):
        return verb + "ing"
    if verb.endswith("e"):
        return verb[:-1] + "ing"
    syllables = len(re.findall(r"[aeiouy]+", verb))
    cvc = (
        len(verb) >= 3
        and verb[-1] not in "wxy"
        and verb[-1] not in _VOWELS
        and verb[-2] in _VOWELS
        and verb[-3] not in _VOWELS
    )
    if cvc and (syllables == 1 or verb in PARTICIPLE_DOUBLING):
        return verb + verb[-1] + "ing"
    return verb + "ing"


def _present_participle(params: dict[str, Any]) -> Task:
    _check_params("present_participle", params, {})
    verbs = [v for v, _ in data.dedupe(data.PAST_TENSE)]
    return Task("present_participle", [Item(v, present_participle(v)) for v in verbs])


def _singular(params: dict[str, Any]) -> Task:
    """Plural noun -> singular, excluding nouns whose two forms coincide."""
    _check_params("singular", params, {})
    pairs = [(p, s) for s, p in data.dedupe(data.PLURAL) if s != p]
    return Task("singular", [Item(i, o) for i, o in data.dedupe(pairs)])


def _uppercase(params: dict[str, Any]) -> Task:
    """Lower-case word -> UPPER-CASE word, over the union of the single-word lists."""
    _check_params("uppercase", params, {})
    words: set[str] = set()
    for lst in (data.PLURAL, data.PAST_TENSE, data.ANTONYM):
        for i, _ in lst:
            if i.isalpha() and i.islower():
                words.add(i)
    return Task("uppercase", [Item(w, w.upper()) for w in sorted(words)])


TASK_BUILDERS: dict[str, Callable[[dict[str, Any]], Task]] = {
    "antonym": _from_pairs("antonym", data.ANTONYM),
    "plural": _from_pairs("plural", data.PLURAL),
    "past_tense": _from_pairs("past_tense", data.PAST_TENSE),
    "en_fr": _from_pairs("en_fr", data.EN_FR),
    "arithmetic": _arithmetic,
    "add_two": _add_two,
    "number_to_words": _number_to_words,
    "present_participle": _present_participle,
    "singular": _singular,
    "uppercase": _uppercase,
}


def build_task(name: str, params: dict[str, Any] | None = None) -> Task:
    if name not in TASK_BUILDERS:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASK_BUILDERS)}")
    return TASK_BUILDERS[name](dict(params or {}))


# --------------------------------------------------------------------------- #
# Filtering and splits
# --------------------------------------------------------------------------- #


@dataclass
class Rejection:
    stage: str
    task: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def filter_items(
    task: Task,
    target_token_count: Callable[[str], int],
    target_template: str,
    max_target_tokens: int,
) -> tuple[list[Item], list[Rejection]]:
    """Drop items whose formatted target exceeds ``max_target_tokens`` tokens."""
    kept: list[Item] = []
    rejected: list[Rejection] = []
    for it in task.items:
        n = target_token_count(target_template.format(output=it.output))
        if n > max_target_tokens:
            rejected.append(
                Rejection(
                    stage="item_filter",
                    task=task.name,
                    reason="target_too_many_tokens",
                    details={"input": it.input, "output": it.output, "n_tokens": n, "max": max_target_tokens},
                )
            )
        else:
            kept.append(it)
    return kept, rejected


@dataclass
class Splits:
    extraction: list[Item]
    calibration: list[Item]
    evaluation: list[Item]
    reduced: bool = False  # True when pools were shrunk because the task had too few items

    def as_dict(self) -> dict[str, Any]:
        return {
            "extraction": [(i.input, i.output) for i in self.extraction],
            "calibration": [(i.input, i.output) for i in self.calibration],
            "evaluation": [(i.input, i.output) for i in self.evaluation],
            "reduced": self.reduced,
        }


def make_splits(
    items: list[Item],
    run_seed: int,
    task_name: str,
    n_extraction: int,
    n_calibration: int,
    n_evaluation: int,
    allow_reduced: bool = False,
) -> Splits:
    """Deterministically shuffle ``items`` and cut three disjoint pools."""
    rng = rng_for(run_seed, "split", task_name)
    order = rng.permutation(len(items))
    shuffled = [items[i] for i in order]
    need = n_extraction + n_calibration + n_evaluation
    reduced = False
    if len(shuffled) < need:
        if not allow_reduced:
            raise ValueError(
                f"task {task_name!r} has {len(shuffled)} usable items but the splits need {need}; "
                "reduce pool sizes or set data.allow_reduced_splits"
            )
        scale = len(shuffled) / need
        n_extraction = max(1, int(n_extraction * scale))
        n_calibration = max(1, int(n_calibration * scale))
        n_evaluation = max(1, int(n_evaluation * scale))
        reduced = True
    a, b = n_extraction, n_extraction + n_calibration
    return Splits(
        extraction=shuffled[:a],
        calibration=shuffled[a:b],
        evaluation=shuffled[b : b + n_evaluation],
        reduced=reduced,
    )


def sample_demonstrations(
    pool: list[Item], query: Item, n_shots: int, rng: np.random.Generator
) -> list[Item]:
    """Sample ``n_shots`` demonstrations from ``pool`` excluding ``query``.

    Demonstrations are chosen with distinct outputs where the pool allows it, so
    that a derangement of outputs changes every demonstration.
    """
    candidates = [it for it in pool if it != query]
    order = rng.permutation(len(candidates))
    chosen: list[Item] = []
    seen_outputs: set[str] = set()
    for i in order:
        it = candidates[i]
        if it.output in seen_outputs:
            continue
        chosen.append(it)
        seen_outputs.add(it.output)
        if len(chosen) == n_shots:
            return chosen
    # not enough distinct outputs: fill with the remaining candidates
    for i in order:
        it = candidates[i]
        if it not in chosen:
            chosen.append(it)
            if len(chosen) == n_shots:
                break
    if len(chosen) < n_shots:
        raise ValueError(f"pool too small for {n_shots} demonstrations (have {len(candidates)})")
    return chosen


def derangement(n: int, rng: np.random.Generator, max_tries: int = 10_000) -> np.ndarray:
    """A uniformly random permutation of ``range(n)`` with no fixed point."""
    if n < 2:
        raise ValueError("a derangement needs n >= 2")
    for _ in range(max_tries):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    raise RuntimeError("failed to sample a derangement")

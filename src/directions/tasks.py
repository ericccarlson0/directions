"""Deterministic mapping tasks, item filtering and the three disjoint query pools."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import data
from .data_extra import PARTICIPLE_DOUBLING, PARTICIPLE_EXCEPTIONS
from .seeds import derive_seed, rng_for


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
    return Task("uppercase", [Item(w, w.upper()) for w in _single_words()])


def _single_words() -> list[str]:
    """Union of the lower-case alphabetic inputs of the single-word lists (sorted, unique)."""
    words: set[str] = set()
    for lst in (data.PLURAL, data.PAST_TENSE, data.ANTONYM):
        for i, _ in lst:
            if i.isalpha() and i.islower():
                words.add(i)
    return sorted(words)


LIST_SEPARATOR = ", "


def _last_antonym(params: dict[str, Any]) -> Task:
    """Composite task after Todd et al. (2024, "Last-Antonym"): a list of ``n_words``
    words -> the antonym of the *last* one. The last word runs over the antonym
    list (so inputs are unique); the preceding words are distractors drawn without
    replacement from the other antonym inputs with a fixed generator seed, so the
    item list is deterministic and independent of the run seed."""
    p = _check_params("last_antonym", params, {"n_words": 3, "items_seed": 20260908})
    n_words = int(p["n_words"])
    if n_words < 2:
        raise ValueError("last_antonym needs n_words >= 2")
    pairs = data.dedupe(data.ANTONYM)
    inputs = [w for w, _ in pairs]
    rng = np.random.default_rng(derive_seed(int(p["items_seed"]), "task_items", "last_antonym"))
    items: list[Item] = []
    for w, a in pairs:
        pool = [x for x in inputs if x != w and x != a]
        distractors = [pool[i] for i in rng.choice(len(pool), size=n_words - 1, replace=False)]
        items.append(Item(LIST_SEPARATOR.join(distractors + [w]), a))
    return Task("last_antonym", items, params=p)


def _alphabetically_first(params: dict[str, Any]) -> Task:
    """Extractive task after Todd et al. (2024, "alphabetically_first"): a list of
    ``n_words`` distinct words -> the alphabetically first one. Word lists are
    sampled from the union of the single-word lists with a fixed generator seed;
    inputs are unique by construction."""
    p = _check_params("alphabetically_first", params, {"n_words": 3, "n_items": 500, "items_seed": 20260908})
    n_words, n_items = int(p["n_words"]), int(p["n_items"])
    if n_words < 2 or n_items < 1:
        raise ValueError("alphabetically_first needs n_words >= 2 and n_items >= 1")
    words = _single_words()
    rng = np.random.default_rng(derive_seed(int(p["items_seed"]), "task_items", "alphabetically_first"))
    seen: set[str] = set()
    items: list[Item] = []
    while len(items) < n_items:
        chosen = [words[i] for i in rng.choice(len(words), size=n_words, replace=False)]
        key = LIST_SEPARATOR.join(chosen)
        if key in seen:
            continue
        seen.add(key)
        items.append(Item(key, min(chosen)))
    return Task("alphabetically_first", items, params=p)


def _kth_word(params: dict[str, Any]) -> Task:
    """Positional selection family (docs/DECISIONS.md D39): a list of ``n_words``
    distinct words -> the ``k``-th one (1-based). The lists are sampled from the
    union of the single-word lists with a fixed generator seed and are the same
    for every ``k`` (the family shares its inputs and differs only in the target),
    so the inputs are unique and independent of the run seed."""
    p = _check_params("kth_word", params, {"n_words": 5, "k": 1, "n_items": 500, "items_seed": 20260908})
    n_words, k, n_items = int(p["n_words"]), int(p["k"]), int(p["n_items"])
    if n_words < 2 or n_items < 1:
        raise ValueError("kth_word needs n_words >= 2 and n_items >= 1")
    if not 1 <= k <= n_words:
        raise ValueError(f"kth_word needs 1 <= k <= n_words, got k={k} with n_words={n_words}")
    words = _single_words()
    rng = np.random.default_rng(derive_seed(int(p["items_seed"]), "task_items", "kth_word", n_words))
    seen: set[str] = set()
    items: list[Item] = []
    while len(items) < n_items:
        chosen = [words[i] for i in rng.choice(len(words), size=n_words, replace=False)]
        key = LIST_SEPARATOR.join(chosen)
        if key in seen:
            continue
        seen.add(key)
        items.append(Item(key, chosen[k - 1]))
    return Task("kth_word", items, params=p)


def _arithmetic_words(params: dict[str, Any]) -> Task:
    """Composite numeric->lexical task: ``n -> number_to_words(op(n))``, i.e. the
    ``arithmetic`` mapping followed by ``number_to_words`` (two of the single-step
    tasks composed, in the spirit of Todd et al. 2024, sec. 4.3)."""
    p = _check_params("arithmetic_words", params, {"operation": "add", "operand": 3, "min": 0, "max": 996})
    inner = _arithmetic({"operation": p["operation"], "operand": p["operand"], "min": p["min"], "max": p["max"]})
    items = [Item(it.input, number_to_words(int(it.output))) for it in inner.items if 0 <= int(it.output) <= 999]
    return Task("arithmetic_words", items, params=p)


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
    "last_antonym": _last_antonym,
    "alphabetically_first": _alphabetically_first,
    "arithmetic_words": _arithmetic_words,
    "kth_word": _kth_word,
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

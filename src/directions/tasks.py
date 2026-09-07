"""Deterministic mapping tasks, item filtering and the three disjoint query pools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import data
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
# Task registry
# --------------------------------------------------------------------------- #


def _from_pairs(name: str, pairs: list[tuple[str, str]]) -> Callable[[dict[str, Any]], Task]:
    def build(params: dict[str, Any]) -> Task:
        if params:
            raise ValueError(f"task {name!r} takes no params, got {params}")
        return Task(name, [Item(i, o) for i, o in data.dedupe(pairs)])

    return build


def _arithmetic(params: dict[str, Any]) -> Task:
    p = {"operation": "add", "operand": 3, "min": 0, "max": 300}
    unknown = set(params) - set(p)
    if unknown:
        raise ValueError(f"unknown arithmetic params: {sorted(unknown)}")
    p.update(params)
    op, k = p["operation"], int(p["operand"])
    ops = {
        "add": lambda n: n + k,
        "subtract": lambda n: n - k,
        "multiply": lambda n: n * k,
    }
    if op not in ops:
        raise ValueError(f"unknown arithmetic operation {op!r}; choose from {sorted(ops)}")
    items = [Item(str(n), str(ops[op](n))) for n in range(int(p["min"]), int(p["max"]) + 1)]
    return Task("arithmetic", items, params=p)


TASK_BUILDERS: dict[str, Callable[[dict[str, Any]], Task]] = {
    "antonym": _from_pairs("antonym", data.ANTONYM),
    "plural": _from_pairs("plural", data.PLURAL),
    "past_tense": _from_pairs("past_tense", data.PAST_TENSE),
    "en_fr": _from_pairs("en_fr", data.EN_FR),
    "arithmetic": _arithmetic,
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

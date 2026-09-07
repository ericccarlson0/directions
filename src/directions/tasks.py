"""Deterministic task datasets.

A task is a finite set of ``(input, output)`` pairs defining a deterministic
mapping that admits exact automatic scoring. Datasets are either shipped as
JSON under ``directions/data/`` (word-level mappings) or generated from a
seeded rule (arithmetic).

Splits are derived by a seeded permutation of the *sorted* item ids, so the
extraction / calibration / evaluation splits are disjoint and reproducible from
the config alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Callable

import numpy as np

DATA_PACKAGE = "directions.data"


@dataclass(frozen=True)
class TaskItem:
    id: str
    input: str
    output: str


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    items: tuple[TaskItem, ...]

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class TaskSplits:
    """Disjoint query pools. Demonstrations are drawn from the same split as
    the query (and never include the query itself), so no example ever appears
    on both sides of the extraction/evaluation boundary."""

    extraction: tuple[TaskItem, ...]
    calibration: tuple[TaskItem, ...]
    evaluation: tuple[TaskItem, ...]

    def as_dict(self) -> dict[str, tuple[TaskItem, ...]]:
        return {
            "extraction": self.extraction,
            "calibration": self.calibration,
            "evaluation": self.evaluation,
        }


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


def _load_json_task(filename: str, name: str, description: str) -> Task:
    text = resources.files(DATA_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
    raw = json.loads(text)
    seen: set[str] = set()
    items = []
    for entry in raw:
        if entry["id"] in seen:
            raise ValueError(f"duplicate item id {entry['id']!r} in {filename}")
        seen.add(entry["id"])
        items.append(TaskItem(id=entry["id"], input=entry["input"], output=entry["output"]))
    return Task(name=name, description=description, items=tuple(items))


def _arithmetic_add(n_items: int = 240, seed: int = 20260101) -> Task:
    """``"a + b" -> "a+b"`` for two-operand addition with distinct pairs."""
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    items = []
    while len(items) < n_items:
        a, b = int(rng.integers(2, 60)), int(rng.integers(2, 60))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        items.append(TaskItem(id=f"add_{a}_{b}", input=f"{a} + {b}", output=str(a + b)))
    return Task("arithmetic_add", "two-operand integer addition", tuple(items))


def _arithmetic_plus_k(k: int = 7, n_items: int = 240) -> Task:
    """``n -> n + k``: a mapping recoverable only from the demonstrations,
    which keeps the zero-shot baseline near the floor."""
    items = [
        TaskItem(id=f"plus{k}_{n}", input=str(n), output=str(n + k))
        for n in range(10, 10 + n_items)
    ]
    return Task(f"arithmetic_plus_{k}", f"add the constant {k}", tuple(items))


TASK_REGISTRY: dict[str, Callable[[], Task]] = {
    "antonym": lambda: _load_json_task("antonym.json", "antonym", "English word -> antonym"),
    "plural": lambda: _load_json_task("plural.json", "plural", "singular noun -> plural noun"),
    "past_tense": lambda: _load_json_task(
        "past_tense.json", "past_tense", "present-tense verb -> simple past"
    ),
    "en_fr": lambda: _load_json_task("en_fr.json", "en_fr", "English word -> French translation"),
    "arithmetic_add": _arithmetic_add,
    "arithmetic_plus_7": _arithmetic_plus_k,
}


def available_tasks() -> list[str]:
    return sorted(TASK_REGISTRY)


def load_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise KeyError(f"unknown task {name!r}; available: {available_tasks()}")
    return TASK_REGISTRY[name]()


def filter_by_target_length(
    task: Task, tokenizer, prefix: str, max_target_tokens: int
) -> tuple[Task, list[dict]]:
    """Drop items whose target does not tokenize within ``max_target_tokens``.

    Long targets make teacher-forced exact match near-impossible and dilute the
    per-token log-probability, so they are filtered automatically and every
    dropped item is reported for the run's rejection log.
    """
    kept, dropped = [], []
    for item in task.items:
        n = len(tokenizer.encode(prefix + item.output, add_special_tokens=False))
        if n <= max_target_tokens:
            kept.append(item)
        else:
            dropped.append({"id": item.id, "output": item.output, "n_target_tokens": n})
    return Task(task.name, task.description, tuple(kept)), dropped


def split_task(
    task: Task,
    n_extraction: int,
    n_calibration: int,
    n_eval: int,
    seed: int,
) -> TaskSplits:
    """Deterministic disjoint split of ``task``'s items."""
    needed = n_extraction + n_calibration + n_eval
    if needed > len(task):
        raise ValueError(
            f"task {task.name!r} has {len(task)} items but the config requests {needed} "
            f"({n_extraction} extraction + {n_calibration} calibration + {n_eval} eval)"
        )
    ordered = sorted(task.items, key=lambda it: it.id)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    picked = [ordered[i] for i in perm[:needed]]
    a, b = n_extraction, n_extraction + n_calibration
    return TaskSplits(
        extraction=tuple(picked[:a]),
        calibration=tuple(picked[a:b]),
        evaluation=tuple(picked[b:needed]),
    )

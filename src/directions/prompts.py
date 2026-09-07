"""Few-shot prompt construction and positive-vs-permuted demonstration pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PromptTemplate
from .tasks import TaskItem


@dataclass(frozen=True)
class Prompt:
    """A prompt plus the target continuation it should be scored against."""

    text: str
    target: str
    item_id: str
    kind: str  # "positive" | "permuted" | "query"
    n_shot: int


def render(template: PromptTemplate, demos: list[TaskItem], query: TaskItem) -> str:
    parts = [template.demo.format(input=d.input, output=d.output) for d in demos]
    parts.append(template.query.format(input=query.input))
    return template.separator.join(parts)


def make_prompt(
    template: PromptTemplate,
    demos: list[TaskItem],
    query: TaskItem,
    kind: str = "query",
) -> Prompt:
    return Prompt(
        text=render(template, demos, query),
        target=template.target_prefix + query.output,
        item_id=query.id,
        kind=kind,
        n_shot=len(demos),
    )


def sample_demos(
    rng: np.random.Generator,
    pool: list[TaskItem],
    n_shot: int,
    exclude_id: str | None = None,
) -> list[TaskItem]:
    """Sample ``n_shot`` demonstrations, never reusing the query item itself."""
    candidates = [it for it in pool if it.id != exclude_id]
    if n_shot > len(candidates):
        raise ValueError(f"cannot draw {n_shot} demos from a pool of {len(candidates)}")
    idx = rng.choice(len(candidates), size=n_shot, replace=False)
    return [candidates[int(i)] for i in idx]


def derangement(rng: np.random.Generator, n: int) -> np.ndarray:
    """A permutation of ``range(n)`` with no fixed point.

    Used for the "permuted" negative prompts: every demonstration keeps its
    input but receives another demonstration's output, so the *only* difference
    from the positive prompt is that the input->output mapping is broken.
    """
    if n < 2:
        raise ValueError("a derangement requires n >= 2")
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def make_pair(
    rng: np.random.Generator,
    template: PromptTemplate,
    demos: list[TaskItem],
    query: TaskItem,
) -> tuple[Prompt, Prompt]:
    """Positive (correct demos) and permuted (deranged demo outputs) prompts."""
    perm = derangement(rng, len(demos))
    permuted_demos = [
        TaskItem(id=d.id, input=d.input, output=demos[int(j)].output)
        for d, j in zip(demos, perm)
    ]
    return (
        make_prompt(template, demos, query, kind="positive"),
        make_prompt(template, permuted_demos, query, kind="permuted"),
    )


def build_pairs(
    rng: np.random.Generator,
    template: PromptTemplate,
    pool: list[TaskItem],
    queries: list[TaskItem],
    n_shot: int,
) -> list[tuple[Prompt, Prompt]]:
    """One positive/permuted pair per query, with independently sampled demos."""
    if n_shot < 2:
        raise ValueError("paired extraction needs n_shot >= 2 so demo outputs can be deranged")
    pairs = []
    for q in queries:
        demos = sample_demos(rng, pool, n_shot, exclude_id=q.id)
        pairs.append(make_pair(rng, template, demos, q))
    return pairs


def build_eval_prompts(
    rng: np.random.Generator,
    template: PromptTemplate,
    pool: list[TaskItem],
    queries: list[TaskItem],
    n_shot: int,
) -> list[Prompt]:
    """Prompts used for behavioural evaluation (``n_shot`` may be 0)."""
    out = []
    for q in queries:
        demos = sample_demos(rng, pool, n_shot, exclude_id=q.id) if n_shot else []
        out.append(make_prompt(template, demos, q, kind="query"))
    return out

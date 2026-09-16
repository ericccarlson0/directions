"""Prompt construction: few-shot, zero-shot and positive/permuted demonstration pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PromptConfig
from .tasks import Item, derangement, sample_demonstrations


@dataclass(frozen=True)
class Prompt:
    """A prompt string and its (separately tokenised) target continuation."""

    prompt: str
    target: str
    query: Item
    demos: tuple[Item, ...]
    # When set, only the target tokens that differ from this text's tokens (under the left and the right
    # alignment; ``ModelBackend._scored_tokens``) are scored (docs/DECISIONS.md D30): the input rendered as a
    # target, for ``target_scoring: changed_tokens``.
    score_reference: str | None = None


def score_reference(cfg: PromptConfig, query: Item) -> str | None:
    return cfg.target_template.format(output=query.input) if cfg.target_scoring == "changed_tokens" else None


def _render(cfg: PromptConfig, demos: list[Item], query: Item) -> str:
    parts: list[str] = []
    if cfg.instruction:
        parts.append(cfg.instruction)
    parts.extend(cfg.demo_template.format(input=d.input, output=d.output) for d in demos)
    parts.append(cfg.query_template.format(input=query.input))
    return cfg.separator.join(parts)


def zero_shot_prompt(cfg: PromptConfig, query: Item) -> Prompt:
    return Prompt(
        prompt=_render(cfg, [], query),
        target=cfg.target_template.format(output=query.output),
        query=query,
        demos=(),
        score_reference=score_reference(cfg, query),
    )


def few_shot_prompt(cfg: PromptConfig, pool: list[Item], query: Item, rng: np.random.Generator) -> Prompt:
    demos = sample_demonstrations(pool, query, cfg.n_shots, rng)
    return Prompt(
        prompt=_render(cfg, demos, query),
        target=cfg.target_template.format(output=query.output),
        query=query,
        demos=tuple(demos),
        score_reference=score_reference(cfg, query),
    )


def paired_prompts(
    cfg: PromptConfig, pool: list[Item], query: Item, rng: np.random.Generator
) -> tuple[Prompt, Prompt]:
    """Positive (correct demos) and permuted (deranged demo outputs) prompts.

    Both share the same demonstration inputs and the same query; the permuted
    prompt's demonstration outputs are a derangement of the positive ones.
    """
    demos = sample_demonstrations(pool, query, cfg.n_shots, rng)
    perm = derangement(len(demos), rng)
    deranged = [Item(d.input, demos[j].output) for d, j in zip(demos, perm)]
    target = cfg.target_template.format(output=query.output)
    ref = score_reference(cfg, query)
    pos = Prompt(_render(cfg, demos, query), target, query, tuple(demos), ref)
    neg = Prompt(_render(cfg, deranged, query), target, query, tuple(deranged), ref)
    return pos, neg

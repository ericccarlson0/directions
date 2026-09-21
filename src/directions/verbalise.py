"""The verbalisation test (docs/DECISIONS.md D36): read residual-stream directions through a Jacobian lens.

A Jacobian lens (Anthropic's ``jlens``; the Neuronpedia repository ``neuronpedia/jacobian-lens`` publishes
fitted lenses) holds one matrix ``J_l`` per block, the average Jacobian of the final residual with respect
to the output of block ``l``; the readout of a vector ``h`` at that block is ``unembed(J_l h)`` with the
model's own final norm and unembedding. The lens is indexed by *block output*, so our read point ``m`` (the
input of block ``m``; ``m = L`` the output of the last block) uses ``J_{m-1}``, and the last read point is the
plain logit lens (identity).

This module holds the pure parts (the lens file, the index mapping, the readout bookkeeping, the task-word
vocabulary and the probability masses); ``scripts/verbalise.py`` runs them on a finished trajectories run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

# Words that name each task, read at the first token of the space-prefixed form. Exploratory: the list is
# fixed before the readouts are seen and is not tuned to them.
TASK_WORDS: dict[str, list[str]] = {
    "antonym": ["opposite", "opposites", "antonym", "antonyms", "contrary", "reverse", "inverse", "negation"],
    "last_antonym": ["opposite", "opposites", "antonym", "antonyms", "contrary", "reverse", "last", "final"],
    "plural": ["plural", "plurals", "pluralize", "many", "multiple", "several"],
    "singular": ["singular", "single", "one", "individual"],
    "past_tense": ["past", "tense", "yesterday", "ago", "previously", "preterite"],
    "present_participle": ["participle", "gerund", "continuous", "progressive", "ing", "currently", "ongoing"],
    "uppercase": ["uppercase", "capital", "capitals", "capitalized", "capitalize", "CAPS", "Uppercase"],
    "number_to_words": ["words", "spelled", "spell", "written", "word", "letters", "numeral"],
    "arithmetic_words": ["words", "spelled", "plus", "add", "three", "sum"],
    "add_3": ["three", "add", "plus", "sum", "addition", "increment"],
    "arithmetic": ["three", "add", "plus", "sum", "addition", "increment"],
}


@dataclass
class Lens:
    """The fitted matrices, keyed by block index (the block whose output they read)."""

    jacobians: dict[int, torch.Tensor]
    d_model: int
    n_prompts: int

    @classmethod
    def load(cls, path: str) -> "Lens":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in ck:
            raise ValueError(f"{path} is not a Jacobian-lens file (keys {sorted(ck)})")
        return cls(jacobians={int(k): v.float() for k, v in ck["J"].items()}, d_model=int(ck["d_model"]),
                   n_prompts=int(ck["n_prompts"]))

    def matrix_for_read_point(self, m: int, n_layers: int) -> torch.Tensor | None:
        """``J`` for read point ``m`` of an ``n_layers`` model: the lens of block ``m - 1`` (its output is the
        input of block ``m``); ``None`` (identity) at the last read point, which is the final residual itself.
        Raises for read point 0 (the embeddings; no lens) and for blocks the lens does not cover."""
        if m == n_layers:
            return None
        if m <= 0:
            raise ValueError("read point 0 (the embeddings) has no lens")
        if (m - 1) not in self.jacobians:
            raise ValueError(f"the lens has no matrix for block {m - 1} (fitted blocks {sorted(self.jacobians)})")
        return self.jacobians[m - 1]


def transport(lens: Lens | None, h: np.ndarray, m: int, n_layers: int) -> torch.Tensor:
    """``J_{m-1} h`` as a float32 tensor (or ``h`` itself for the logit lens / the last read point)."""
    x = torch.as_tensor(np.asarray(h, dtype=np.float32))
    if lens is None:
        return x
    J = lens.matrix_for_read_point(m, n_layers)
    return x if J is None else x @ J.T


def word_token_ids(tokenizer: Any, words: list[str]) -> dict[str, int]:
    """The first token of each space-prefixed word (the form a completion would take)."""
    out: dict[str, int] = {}
    for w in words:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if ids:
            out[w] = int(ids[0])
    return out


def readout_summary(logprobs: np.ndarray, decode: Callable[[list[int]], str], top: int, task_ids: dict[str, int],
                    answer_ids: np.ndarray) -> dict[str, Any]:
    """Top tokens and the probability masses on the task words and on the answer tokens."""
    order = np.argsort(-logprobs)[:top]
    p = np.exp(logprobs)
    ids = sorted(set(task_ids.values()))
    ans = np.unique(np.asarray(answer_ids, dtype=np.int64))
    return {
        "top": [(decode([int(i)]), float(logprobs[i])) for i in order],
        "task_mass": float(p[ids].sum()) if ids else 0.0,
        "task_best": (max(task_ids, key=lambda w: logprobs[task_ids[w]]) if task_ids else None),
        "task_best_rank": (int((logprobs > max(logprobs[i] for i in ids)).sum()) + 1) if ids else None,
        "answer_mass": float(p[ans].sum()) if ans.size else 0.0,
        "answer_best_rank": (int((logprobs > logprobs[ans].max()).sum()) + 1) if ans.size else None,
        "entropy": float(-(p * logprobs).sum()),
    }

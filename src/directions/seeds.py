"""Deterministic sub-seed derivation.

Sub-seeds are derived from the run seed through a stable digest (SHA-256), never
through Python's per-process-salted ``hash()``.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(run_seed: int, *names: object) -> int:
    """Return a stable 63-bit integer seed for ``(run_seed, *names)``.

    ``names`` may be any objects with a stable ``str``/``repr`` (strings, ints).
    """
    key = "|".join([str(int(run_seed))] + [repr(n) for n in names])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def rng_for(run_seed: int, *names: object) -> np.random.Generator:
    """A NumPy generator seeded from :func:`derive_seed`."""
    return np.random.default_rng(derive_seed(run_seed, *names))

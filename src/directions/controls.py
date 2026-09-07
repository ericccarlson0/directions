"""Matched random control directions.

Every control is matched to the validated control direction on intervention
layer and intervention norm (identical ``alpha``, and both vectors are unit
norm). Two families are used:

``isotropic``
    a uniformly random unit vector; the generic "some direction of the same
    size" null.

``orthogonal``
    a uniformly random unit vector in the orthogonal complement of the control
    direction; rules out the possibility that an isotropic control accidentally
    carries a component along the control direction.
"""

from __future__ import annotations

import numpy as np

from . import mathx
from .extraction import DirectionSpec


def control_rng(base_seed: int, task: str, layer: int) -> np.random.Generator:
    ss = np.random.SeedSequence(
        entropy=base_seed, spawn_key=(mathx.stable_key("controls", task), layer)
    )
    return np.random.default_rng(ss)


def build_random_controls(
    rng: np.random.Generator,
    direction: DirectionSpec,
    n_random: int,
    kinds: list[str],
) -> list[DirectionSpec]:
    """``n_random`` controls matched to ``direction``, split evenly across kinds."""
    if not kinds:
        raise ValueError("controls.kinds must be non-empty")
    d = direction.vector.shape[0]
    v = mathx.normalize(direction.vector)
    out: list[DirectionSpec] = []
    for i in range(n_random):
        kind = kinds[i % len(kinds)]
        if kind == "isotropic":
            w = mathx.random_unit(rng, d)
        elif kind == "orthogonal":
            w = mathx.random_unit_orthogonal(rng, v)
        else:
            raise ValueError(f"unknown control kind {kind!r}")
        out.append(
            DirectionSpec(
                layer=direction.layer,
                vector=w.astype(np.float32),
                alpha=direction.alpha,
                label=f"{kind}_{i:02d}",
            )
        )
    return out

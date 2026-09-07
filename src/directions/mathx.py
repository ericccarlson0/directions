"""Numerical / statistical primitives used by the pilot.

Everything here operates on ``numpy`` arrays in float64 (statistics are always
computed in high precision even though model execution is bf16, per
``docs/EXPERIMENT.md``).

Conventions
-----------
A "stack" matrix ``D`` has shape ``(n_examples, d_model)``: one row per
held-out example, matching the ``D_l`` / ``B_l`` definitions in the experiment
specification.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

EPS = 1e-12


def as_f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------
# direction handling
# --------------------------------------------------------------------------


def normalize(v: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Scale ``v`` to unit L2 norm.

    Raises for a (numerically) zero vector rather than silently returning
    garbage: a zero control direction always indicates an upstream bug.
    """
    v = as_f64(v)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        raise ValueError(f"cannot normalize vector with norm {n!r}")
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = as_f64(a), as_f64(b)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def abs_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Sign-invariant alignment; PCA directions are only defined up to sign."""
    return abs(cosine(a, b))


def orient_like(v: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Fix the arbitrary sign of ``v`` so it points along ``reference``."""
    return -v if cosine(v, reference) < 0 else v


def project_out(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove from ``v`` its component in the row space of ``basis``.

    ``basis`` may be 1-D (single direction) or 2-D ``(k, d)``. The basis is
    orthonormalized internally, so callers need not pre-condition it.
    """
    v = as_f64(v)
    basis = as_f64(basis)
    if basis.ndim == 1:
        basis = basis[None, :]
    if basis.size == 0:
        return v
    q, _ = np.linalg.qr(basis.T)  # (d, k) orthonormal columns
    return v - q @ (q.T @ v)


def random_unit(rng: np.random.Generator, d: int) -> np.ndarray:
    """Isotropic random unit vector (normalized Gaussian)."""
    return normalize(rng.standard_normal(d))


def random_unit_orthogonal(
    rng: np.random.Generator, v: np.ndarray, max_tries: int = 32
) -> np.ndarray:
    """Random unit vector orthogonal to ``v``.

    Sampled isotropically then projected onto ``v``'s orthogonal complement and
    renormalized; in high dimensions the projection barely changes the vector,
    but the result is *exactly* orthogonal, which is what the control condition
    requires.
    """
    v = normalize(v)
    for _ in range(max_tries):
        w = rng.standard_normal(v.shape[0])
        w = w - np.dot(w, v) * v
        if np.linalg.norm(w) > 1e-8:
            return normalize(w)
    raise RuntimeError("failed to sample a vector orthogonal to v")


# --------------------------------------------------------------------------
# spectra
# --------------------------------------------------------------------------


def singular_values(D: np.ndarray, center: bool = True) -> np.ndarray:
    """Singular values of ``D`` (rows = examples), optionally mean-centered.

    ``docs/EXPERIMENT.md`` requires centering across examples before the SVD
    for the effective-dimensionality metrics.
    """
    D = as_f64(D)
    if D.ndim != 2:
        raise ValueError(f"expected a 2-D stack, got shape {D.shape}")
    if center:
        D = D - D.mean(axis=0, keepdims=True)
    return np.linalg.svd(D, compute_uv=False)


def effective_rank(sv: np.ndarray) -> float:
    r"""Participation-ratio effective rank.

    .. math:: d_{eff} = \frac{(\sum_i \sigma_i^2)^2}{\sum_i \sigma_i^4}

    Equals ``k`` for ``k`` equal singular values and ``1`` for a rank-1 matrix.
    """
    sv = as_f64(sv)
    s2 = sv**2
    denom = float(np.sum(s2**2))
    if denom <= EPS:
        return 0.0
    return float(np.sum(s2) ** 2 / denom)


def d_frac(sv: np.ndarray, frac: float = 0.9) -> int:
    """Smallest number of singular directions explaining ``frac`` of variance."""
    sv = as_f64(sv)
    s2 = sv**2
    total = float(np.sum(s2))
    if total <= EPS:
        return 0
    csum = np.cumsum(s2) / total
    # ``>= frac - tiny`` guards against float error when one PC holds it all.
    return int(np.searchsorted(csum, frac - 1e-12) + 1)


def top_subspace_basis(D: np.ndarray, frac: float = 0.9, center: bool = True) -> np.ndarray:
    """Orthonormal basis (rows) for the top-``d_frac`` right-singular subspace of ``D``."""
    D = as_f64(D)
    if center:
        D = D - D.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(D, full_matrices=False)
    k = d_frac(s, frac)
    return vt[:k]


@dataclass(frozen=True)
class SpectrumMetrics:
    effective_rank: float
    d90: int
    total_variance: float
    top_explained_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def spectrum_metrics(D: np.ndarray, frac: float = 0.9, center: bool = True) -> SpectrumMetrics:
    sv = singular_values(D, center=center)
    s2 = sv**2
    total = float(np.sum(s2))
    return SpectrumMetrics(
        effective_rank=effective_rank(sv),
        d90=d_frac(sv, frac),
        total_variance=total,
        top_explained_ratio=float(s2[0] / total) if total > EPS else 0.0,
    )


def pc1(D: np.ndarray, center: bool = False) -> tuple[np.ndarray, float]:
    """First principal component of the rows of ``D``.

    Returns ``(unit_direction, explained_variance_ratio)``.

    ``center=False`` (the pilot default, see ``docs/DECISIONS.md``) makes this
    the top right-singular vector of the raw paired-difference matrix, which
    retains the shared mean component that carries the task signal.
    """
    D = as_f64(D)
    if D.ndim != 2:
        raise ValueError(f"expected a 2-D stack, got shape {D.shape}")
    if center:
        D = D - D.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(D, full_matrices=False)
    s2 = s**2
    total = float(np.sum(s2))
    ratio = float(s2[0] / total) if total > EPS else 0.0
    return normalize(vt[0]), ratio


# --------------------------------------------------------------------------
# layerwise response decomposition
# --------------------------------------------------------------------------


def row_norms(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(as_f64(X), axis=-1)


def blockwise_response(deltas: np.ndarray) -> np.ndarray:
    r"""``b_l = \delta_{l+1} - \delta_l`` along the layer axis.

    ``deltas`` has shape ``(n_layers_plus_1, n_examples, d_model)``; the result
    has shape ``(n_layers, n_examples, d_model)``.
    """
    deltas = as_f64(deltas)
    if deltas.ndim != 3:
        raise ValueError(f"expected (layers, examples, d_model), got {deltas.shape}")
    return deltas[1:] - deltas[:-1]


def perturbation_magnitude(delta: np.ndarray, base: np.ndarray) -> np.ndarray:
    r"""``S_l(x) = \|\delta_l(x)\| / \|h_l^{base}(x)\|`` per example."""
    return row_norms(delta) / np.maximum(row_norms(base), EPS)


def layer_gain(delta: np.ndarray, delta_next: np.ndarray) -> np.ndarray:
    r"""``G_l(x) = \|\delta_{l+1}(x)\| / \|\delta_l(x)\|`` per example."""
    return row_norms(delta_next) / np.maximum(row_norms(delta), EPS)


def block_conversion(delta: np.ndarray, block: np.ndarray) -> np.ndarray:
    r"""``C_l(x) = \|b_l(x)\| / \|\delta_l(x)\|`` per example."""
    return row_norms(block) / np.maximum(row_norms(delta), EPS)


def new_subspace_fraction(D: np.ndarray, B: np.ndarray, frac: float = 0.9) -> float:
    r"""Fraction of block response lying outside the incoming perturbation subspace.

    .. math::
        N_l = \frac{\|B_l (I - P_l)\|_F^2}{\|B_l\|_F^2}

    where ``P_l`` projects onto the top right-singular subspace of the centered
    ``D_l`` explaining ``frac`` of its variance.
    """
    D, B = as_f64(D), as_f64(B)
    denom = float(np.sum(B**2))
    if denom <= EPS:
        return 0.0
    basis = top_subspace_basis(D, frac=frac)  # (k, d) orthonormal rows
    if basis.size == 0:
        return 1.0
    residual = B - (B @ basis.T) @ basis
    return float(np.sum(residual**2) / denom)


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.median,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap ``(point_estimate, lo, hi)`` over examples."""
    values = as_f64(values).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"),) * 3
    point = float(statistic(values))
    if values.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    stats = statistic(values[idx], axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def summarize(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Median + bootstrap CI + basic descriptive statistics."""
    v = as_f64(values).ravel()
    finite = v[np.isfinite(v)]
    med, lo, hi = bootstrap_ci(v, np.median, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "n": int(finite.size),
        "median": med,
        "ci_lo": lo,
        "ci_hi": hi,
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
    }


def paired_bootstrap_p(
    differences, n_boot: int = 2000, seed: int = 0
) -> float:
    """One-sided bootstrap p-value for ``mean(differences) > 0``, paired by example.

    Used to decide whether a calibration grid point *reliably* improves the
    behavioural metric rather than merely clearing a fixed threshold: with a few
    dozen evaluation examples a threshold alone is comfortably inside sampling
    noise. Add-one smoothed, so the value is never exactly zero.
    """
    d = as_f64(differences).ravel()
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float((np.sum(means <= 0.0) + 1) / (n_boot + 1))


def empirical_p_value(observed: float, null_samples) -> float:
    """One-sided ``P(null >= observed)`` with add-one smoothing.

    Used to place a real control direction against its matched random-control
    distribution. Add-one smoothing keeps the p-value from ever being exactly
    zero with a finite number of controls.
    """
    null = as_f64(null_samples).ravel()
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((np.sum(null >= observed) + 1) / (null.size + 1))


def z_against_null(observed: float, null_samples) -> float:
    """Standardized distance of ``observed`` from the null-control distribution."""
    null = as_f64(null_samples).ravel()
    null = null[np.isfinite(null)]
    if null.size < 2 or not np.isfinite(observed):
        return float("nan")
    sd = float(np.std(null, ddof=1))
    if sd <= EPS:
        return float("nan")
    return float((observed - float(np.mean(null))) / sd)


def stable_key(*parts) -> int:
    """Process-independent integer key for seeding.

    ``hash()`` on strings is salted per interpreter process (PYTHONHASHSEED), so
    using it inside a seed would make runs irreproducible across invocations.
    This derives the key from a BLAKE2b digest instead.
    """
    import hashlib

    blob = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(blob, digest_size=4).digest(), "big") % (2**31)

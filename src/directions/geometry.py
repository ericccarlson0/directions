"""Geometric utilities for control directions and residual-stream perturbations.

All functions operate on NumPy arrays and compute in float64. Conventions follow
``docs/EXPERIMENT.md``:

* residual read points are indexed ``0..L`` (``L`` transformer blocks);
* ``delta[l]`` is the steering-induced perturbation at read point ``l``;
* ``b[l] = delta[l+1] - delta[l]`` is block ``l``'s response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-30


# --------------------------------------------------------------------------- #
# Direction handling
# --------------------------------------------------------------------------- #


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return ``v`` scaled to unit L2 norm along ``axis`` (float64)."""
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("cannot normalize a zero vector")
    return v / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def rowwise_abs_cosine(rows: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``|cos(rows[i], v)|`` for every row; zero rows give 0."""
    rows = np.asarray(rows, dtype=np.float64)
    v = normalize(v)
    norms = np.linalg.norm(rows, axis=1)
    out = np.zeros(rows.shape[0])
    nz = norms > 0
    out[nz] = np.abs(rows[nz] @ v) / norms[nz]
    return out


@dataclass(frozen=True)
class PCADirection:
    direction: np.ndarray  # unit vector, sign-aligned with the mean difference
    explained_variance_ratio: float  # sigma_1^2 / sum sigma_i^2 (of the matrix given)
    cos_with_mean: float  # cos(PC1, mean row)
    singular_values: np.ndarray


def pca_direction(diffs: np.ndarray, center: bool = False) -> PCADirection:
    """First principal direction of a difference matrix ``diffs`` (n, d).

    With ``center=False`` (the preregistered default) this is the top right
    singular vector of the *raw* matrix, so the shared (mean) component is
    retained. The sign is chosen so that ``cos(PC1, mean(diffs)) >= 0``.
    """
    X = np.asarray(diffs, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError("diffs must be a 2-D array with at least one row")
    mean = X.mean(axis=0)
    if center:
        X = X - mean
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    if mean @ v < 0:
        v = -v
    total = float(np.sum(s**2))
    evr = float(s[0] ** 2 / total) if total > 0 else 0.0
    return PCADirection(
        direction=normalize(v),
        explained_variance_ratio=evr,
        cos_with_mean=cosine(v, mean),
        singular_values=s,
    )


def pairwise_abs_cosines(directions: np.ndarray) -> np.ndarray:
    """Upper-triangular pairwise ``|cos|`` between rows of ``directions`` (k, d)."""
    D = normalize(np.asarray(directions, dtype=np.float64))
    k = D.shape[0]
    return np.array([abs(float(D[i] @ D[j])) for i in range(k) for j in range(i + 1, k)])


def stability(directions: np.ndarray) -> float:
    """Minimum pairwise ``|cos|`` between seed directions (1.0 for a single row)."""
    if np.asarray(directions).shape[0] < 2:
        return 1.0
    return float(np.min(pairwise_abs_cosines(directions)))


# --------------------------------------------------------------------------- #
# Random controls
# --------------------------------------------------------------------------- #


def random_unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Uniform random unit vector in R^dim (isotropic)."""
    while True:
        g = rng.standard_normal(dim)
        n = np.linalg.norm(g)
        if n > 0:
            return g / n


def random_orthogonal_unit_vector(rng: np.random.Generator, v: np.ndarray) -> np.ndarray:
    """Uniform random unit vector in the orthogonal complement of ``v``."""
    v = normalize(v)
    while True:
        g = rng.standard_normal(v.shape[0])
        g = g - (g @ v) * v
        n = np.linalg.norm(g)
        if n > 0:
            return g / n


def covariance_matched_unit_vector(rng: np.random.Generator, H: np.ndarray) -> np.ndarray:
    """Random unit vector drawn from the (centered) covariance of the rows of ``H`` (n, d).

    ``u ∝ H_cᵀ g`` with ``g ~ N(0, I_n)`` has covariance proportional to
    ``H_cᵀ H_c``, so it lies in the subspace the residual stream actually
    occupies, unlike an isotropic direction.
    """
    Hc = np.asarray(H, dtype=np.float64)
    Hc = Hc - Hc.mean(axis=0, keepdims=True)
    while True:
        g = rng.standard_normal(Hc.shape[0])
        u = Hc.T @ g
        n = np.linalg.norm(u)
        if n > 0:
            return u / n


def matched_random_controls(
    rng: np.random.Generator, v: np.ndarray, n: int, kinds: tuple[str, ...] = ("isotropic", "orthogonal")
) -> list[tuple[str, np.ndarray]]:
    """``n`` unit-norm random controls, alternating through ``kinds``.

    The caller scales them by the same absolute strength as the real control.
    """
    out: list[tuple[str, np.ndarray]] = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        if kind == "isotropic":
            u = random_unit_vector(rng, np.asarray(v).shape[0])
        elif kind == "orthogonal":
            u = random_orthogonal_unit_vector(rng, v)
        else:
            raise ValueError(f"unknown random-control kind: {kind}")
        out.append((kind, u))
    return out


# --------------------------------------------------------------------------- #
# Spectra and dimensionality
# --------------------------------------------------------------------------- #


def _singular_values_sq(X: np.ndarray) -> np.ndarray:
    """Squared singular values of ``X`` (n, d), descending; via the n x n Gram matrix when n < d."""
    n, d = X.shape
    if n < d:
        w = np.linalg.eigvalsh(X @ X.T)[::-1]
        return np.clip(w, 0.0, None)
    return np.linalg.svd(X, compute_uv=False) ** 2


def _top_right_singular_vectors(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(s^2 descending, top-k right singular vectors as rows) via the Gram trick when n < d."""
    n, d = X.shape
    if n < d:
        w, U = np.linalg.eigh(X @ X.T)
        order = np.argsort(w)[::-1]
        w, U = np.clip(w[order], 0.0, None), U[:, order]
        s = np.sqrt(w[:k])
        keep = s > 0
        V = (X.T @ U[:, :k])[:, keep] / s[keep]
        return w, V.T
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    return s**2, vt[:k]


def effective_rank(X: np.ndarray, center: bool = True) -> float:
    """Participation-ratio effective rank ``(sum s^2)^2 / sum s^4`` of ``X`` (n, d).

    Returns ``nan`` when the (centered) matrix is exactly zero.
    """
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s2 = _singular_values_sq(X)
    denom = float(np.sum(s2**2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(s2) ** 2 / denom)


def d90(X: np.ndarray, frac: float = 0.9, center: bool = True) -> int | None:
    """Minimum number of singular directions explaining ``frac`` of variance.

    Returns ``None`` when the (centered) matrix is exactly zero.
    """
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s2 = _singular_values_sq(X)
    total = float(np.sum(s2))
    if total == 0.0:
        return None
    cum = np.cumsum(s2) / total
    return int(np.searchsorted(cum, frac - 1e-12) + 1)


@dataclass(frozen=True)
class Spectrum:
    effective_rank: float  # nan if zero matrix
    d90: int | None
    total_variance: float  # sum of squared singular values (of the matrix analysed)
    top_subspace: np.ndarray  # (k, d) orthonormal rows explaining >= frac variance


def spectrum(X: np.ndarray, frac: float = 0.9, center: bool = True) -> Spectrum:
    """Effective rank, d90, total variance and the top-``frac`` subspace of ``X``."""
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s2 = _singular_values_sq(X)
    total = float(np.sum(s2))
    if total == 0.0:
        return Spectrum(float("nan"), None, 0.0, np.zeros((0, X.shape[1])))
    cum = np.cumsum(s2) / total
    k = int(np.searchsorted(cum, frac - 1e-12) + 1)
    eff = float(total**2 / np.sum(s2**2))
    _, top = _top_right_singular_vectors(X, k)
    return Spectrum(eff, k, total, top)


def new_subspace_fraction(B: np.ndarray, top_subspace: np.ndarray) -> float:
    """``||B (I - P)||_F^2 / ||B||_F^2`` where ``P`` projects onto ``top_subspace`` rows.

    Returns ``nan`` when ``B`` is exactly zero.
    """
    B = np.asarray(B, dtype=np.float64)
    total = float(np.sum(B**2))
    if total == 0.0:
        return float("nan")
    if top_subspace.shape[0] == 0:
        return 1.0
    proj = (B @ top_subspace.T) @ top_subspace
    resid = B - proj
    return float(np.sum(resid**2) / total)


# --------------------------------------------------------------------------- #
# Layerwise per-example metrics
# --------------------------------------------------------------------------- #


def block_responses(delta: np.ndarray) -> np.ndarray:
    """``b[l] = delta[l+1] - delta[l]`` for ``delta`` shaped (L+1, n, d) -> (L, n, d)."""
    delta = np.asarray(delta, dtype=np.float64)
    return delta[1:] - delta[:-1]


@dataclass(frozen=True)
class LayerwiseExampleMetrics:
    """Per-example, per-read-point quantities; ``nan`` where undefined."""

    magnitude: np.ndarray  # S_l(x) = ||delta_l|| / ||h_l^base||, shape (L+1, n)
    gain: np.ndarray  # G_l(x) = ||delta_{l+1}|| / ||delta_l||, shape (L, n)
    log_gain: np.ndarray  # log G_l(x), shape (L, n)
    conversion: np.ndarray  # C_l(x) = ||b_l|| / ||delta_l||, shape (L, n)
    alignment: np.ndarray  # A_l(x) = |cos(delta_l, v)|, shape (L+1, n)
    delta_norm: np.ndarray  # ||delta_l(x)||, shape (L+1, n)
    block_norm: np.ndarray  # ||b_l(x)||, shape (L, n)


def layerwise_example_metrics(
    delta: np.ndarray, base: np.ndarray, v: np.ndarray
) -> LayerwiseExampleMetrics:
    """Compute S, G, log G, C, A per example from ``delta``/``base`` of shape (L+1, n, d)."""
    delta = np.asarray(delta, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    v = normalize(v)
    dn = np.linalg.norm(delta, axis=2)  # (L+1, n)
    bn = np.linalg.norm(base, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = np.where(bn > 0, dn / bn, np.nan)
        b = block_responses(delta)
        b_norm = np.linalg.norm(b, axis=2)  # (L, n)
        gain = np.where(dn[:-1] > 0, dn[1:] / dn[:-1], np.nan)
        log_gain = np.log(gain)
        conversion = np.where(dn[:-1] > 0, b_norm / dn[:-1], np.nan)
        proj = np.abs(delta @ v)  # (L+1, n)
        alignment = np.where(dn > 0, proj / dn, np.nan)
    return LayerwiseExampleMetrics(
        magnitude=magnitude,
        gain=gain,
        log_gain=log_gain,
        conversion=conversion,
        alignment=alignment,
        delta_norm=dn,
        block_norm=b_norm,
    )

"""Conditional linear probes with permutation nulls (docs/DECISIONS.md D46).

Can a linear readout of a state recover a target T among the values consistent with a conditioning variable C?
The readout is ridge regression on one-hot targets in its kernel (dual) form, with the features z-scored on the
training fold and the ridge chosen in each training fold by its closed-form leave-one-out accuracy; the prediction
is the best-scoring class among those ``allowed`` for the example (the values consistent with its C). The null
permutes T within the classes of C and reruns everything, λ choice included; the same permutations serve every
read point, so a maximum over read points gives a family-wise threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def permute_within(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``y`` with its values shuffled among the examples of each group (each group's multiset kept)."""
    out = np.array(y, copy=True)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = y[idx[rng.permutation(len(idx))]]
    return out


def fold_ids(n: int, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """A fold per example, balanced, in a seeded order."""
    return (np.arange(n) % n_folds)[rng.permutation(n)]


def _standardise(xtr: torch.Tensor, xte: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """z-score with the training fold's statistics; dimensions that are constant on it are dropped."""
    mu = xtr.mean(0)
    sd = xtr.std(0, unbiased=False)
    keep = sd > 1e-6 * max(float(sd.max()), 1e-30)
    if not bool(keep.any()):
        return xtr[:, :0], xte[:, :0]
    return (xtr[:, keep] - mu[keep]) / sd[keep], (xte[:, keep] - mu[keep]) / sd[keep]


def _restricted_correct(scores: torch.Tensor, allowed: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per label set, the number of examples whose best allowed class is its label.

    scores (n, C, S), allowed (n, C) bool, labels (S, n) -> (S,)"""
    masked = scores.masked_fill(~allowed[:, :, None], float("-inf"))
    pred = masked.argmax(dim=1)  # (n, S)
    return (pred == labels.T).sum(dim=0)


def ridge_cv_accuracy(x: torch.Tensor, labels: torch.Tensor, allowed: torch.Tensor, folds: np.ndarray,
                      lambdas: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-validated restricted accuracy of the kernel ridge probe for several label sets at once.

    x (n, d) float64; labels (S, n) int64 class indices (row 0 the true labels, the rest permuted); allowed (n, C)
    bool. Returns the accuracy per label set (S,) and the index of the λ chosen per (fold, label set) (F, S)."""
    n, C = allowed.shape
    S = labels.shape[0]
    onehot = torch.zeros(n, C, S, dtype=x.dtype, device=x.device)
    onehot.scatter_(1, labels.T[:, None, :], 1.0)
    onehot = onehot.reshape(n, C * S)
    correct = torch.zeros(S, dtype=torch.int64, device=x.device)
    chosen = []
    for f in np.unique(folds):
        tr = torch.as_tensor(np.flatnonzero(folds != f), device=x.device)
        te = torch.as_tensor(np.flatnonzero(folds == f), device=x.device)
        xtr, xte = _standardise(x[tr], x[te])
        d = max(xtr.shape[1], 1)
        ktr = xtr @ xtr.T / d
        kte = xte @ xtr.T / d
        s, U = torch.linalg.eigh(ktr)
        s = s.clamp_min(0)
        ttr = onehot[tr]
        mean = ttr.mean(0, keepdim=True)
        tc = ttr - mean
        A = U.T @ tc  # (ntr, C*S)
        U2 = U * U
        loo_correct, test_correct = [], []
        for lam in lambdas:
            w = s / (s + lam)
            fit = U @ (w[:, None] * A)  # H tc
            h = (U2 @ w)[:, None]  # diag H
            loo = tc - (tc - fit) / (1 - h) + mean  # leave-one-out predictions
            loo_correct.append(_restricted_correct(loo.reshape(-1, C, S), allowed[tr], labels[:, tr]))
            alpha = U @ (A / (s + lam)[:, None])
            pred = kte @ alpha + mean
            test_correct.append(_restricted_correct(pred.reshape(-1, C, S), allowed[te], labels[:, te]))
        loo_c = torch.stack(loo_correct)  # (n_lambda, S)
        # the best leave-one-out accuracy; ties go to the strongest ridge (the last of an increasing grid)
        best = len(lambdas) - 1 - torch.flip(loo_c, dims=[0]).argmax(dim=0)
        test_c = torch.stack(test_correct)
        correct += test_c.gather(0, best[None, :])[0]
        chosen.append(best)
    return correct.to(torch.float64) / n, torch.stack(chosen)


def conditional_probe(features: Any, y: np.ndarray, cond: np.ndarray, allowed: np.ndarray, *, n_folds: int = 5,
                      lambdas: Sequence[float] = (0.1, 1.0, 10.0, 100.0), n_perm: int = 200, alpha: float = 0.05,
                      seed: int = 0, device: str | torch.device = "cpu", read_points: Sequence[int] | None = None) -> dict:
    """The conditional probe of T = ``y`` given C = ``cond`` at every read point of ``features`` (M, N, d).

    ``allowed`` (N, C) marks the classes consistent with each example's C. Returns, per read point, the accuracy,
    the null's mean and s.d., z, the family-wise threshold (95th percentile of the permutations' maximum z over
    read points), whether T is carried, the family-wise p-value and the λ choices; ``null`` holds the permutation
    accuracies (M, n_perm)."""
    y = np.asarray(y, dtype=np.int64)
    cond = np.asarray(cond)
    allowed = np.asarray(allowed, dtype=bool)
    if not allowed[np.arange(len(y)), y].all():
        raise ValueError("an example's own target is not among its allowed classes")
    rng = np.random.default_rng(seed)
    folds = fold_ids(len(y), n_folds, rng)
    sets = np.stack([y] + [permute_within(y, cond, rng) for _ in range(n_perm)])
    dev = torch.device(device)
    labels = torch.as_tensor(sets, device=dev)
    allowed_t = torch.as_tensor(allowed, device=dev)
    rps = range(len(features)) if read_points is None else read_points
    acc, lam_idx = [], []
    for m in rps:
        x = torch.as_tensor(np.asarray(features[m]), device=dev).to(torch.float64)
        a, ch = ridge_cv_accuracy(x, labels, allowed_t, folds, lambdas)
        acc.append(a.cpu().numpy())
        lam_idx.append(ch.cpu().numpy())
    acc_all = np.stack(acc)  # (M, 1 + n_perm)
    return probe_statistics(acc_all[:, 0], acc_all[:, 1:], alpha) | {
        "lambda_chosen": np.asarray(lambdas)[np.stack(lam_idx)[:, :, 0]],  # (M, n_folds), the true labels' choice
    }


def probe_statistics(observed: np.ndarray, null: np.ndarray, alpha: float = 0.05) -> dict:
    """z against the null per read point and the family-wise threshold from the maximum z over read points.

    observed (M,), null (M, P): the same P permutations at every read point."""
    mean = null.mean(axis=1)
    sd = null.std(axis=1)
    safe = np.where(sd > 1e-12, sd, np.inf)
    z = np.where(sd > 1e-12, (observed - mean) / safe, 0.0)
    zp = np.where(sd[:, None] > 1e-12, (null - mean[:, None]) / safe[:, None], 0.0)
    maxz = zp.max(axis=0)
    threshold = float(np.quantile(maxz, 1 - alpha))
    p = (1 + (maxz[None, :] >= z[:, None]).sum(axis=1)) / (1 + null.shape[1])
    return {"accuracy": observed, "null_mean": mean, "null_sd": sd, "excess": observed - mean, "z": z,
            "threshold": threshold, "carried": z > threshold, "p_fwer": p, "null": null}


def match_rates(pred: np.ndarray, candidate: np.ndarray, eligible: np.ndarray, *, n_perm: int = 200,
                alpha: float = 0.05, seed: int = 0) -> dict:
    """How often a per-example prediction ``pred`` (M, N) equals a candidate value (N,), over the eligible examples,
    against the candidates shuffled across those examples (the same shuffles at every read point)."""
    idx = np.flatnonzero(eligible)
    if len(idx) == 0:
        M = pred.shape[0]
        return {"rate": np.full(M, np.nan), "n": 0}
    rng = np.random.default_rng(seed)
    p = pred[:, idx]
    cand = candidate[idx]
    observed = (p == cand[None, :]).mean(axis=1)
    null = np.stack([(p == cand[rng.permutation(len(idx))][None, :]).mean(axis=1) for _ in range(n_perm)], axis=1)
    st = probe_statistics(observed, null, alpha)
    return {"rate": observed, "n": len(idx), **{k: st[k] for k in ("null_mean", "null_sd", "excess", "z", "threshold",
                                                                        "carried", "p_fwer")}}

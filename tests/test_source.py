"""directions.source: the pure functions of the source test (D38)."""

import numpy as np

from directions.model import ForwardResult
from directions.source import along, decompose, head_writes, share_verdict, unit_rows, window_shares


def _result(residuals, attn, mlp):
    n = residuals.shape[1]
    z = np.zeros(n)
    return ForwardResult(logprob_sum=z, logprob_per_token=z, exact_match=z.astype(bool), first_token_margin=z, n_target_tokens=np.ones(n, dtype=int),
                         residuals=residuals.astype(np.float32), attn_writes=attn.astype(np.float32), mlp_writes=mlp.astype(np.float32))


def _consistent(rng, L, N, d):
    """Residuals built from random writes so that attention + MLP writes equal the residual increments exactly."""
    attn = rng.standard_normal((L, N, d))
    mlp = rng.standard_normal((L, N, d))
    res = np.zeros((L + 1, N, d))
    res[0] = rng.standard_normal((N, d))
    for l in range(L):
        res[l + 1] = res[l] + attn[l] + mlp[l]
    return res, attn, mlp


def test_decompose_splits_the_increment_exactly_and_aligns_along_the_natural_difference():
    rng = np.random.default_rng(0)
    L, N, d = 4, 5, 8
    rb, ab, mb = _consistent(rng, L, N, d)
    ri, ai, mi = _consistent(rng, L, N, d)
    # the steered run: the base plus an attention-only change at block 2 that points along the natural difference
    ref = ri - rb
    u = unit_rows(ref[3])
    rs, as_, ms = rb.copy(), ab.copy(), mb.copy()
    as_[2] = ab[2] + 2.0 * u
    rs[3:] += 2.0 * u  # the write lands at read point 3 and persists (no later change)
    dec = decompose(_result(rb, ab, mb), _result(rs, as_, ms), _result(ri, ai, mi), layer=1)
    assert np.isnan(dec["att_along"][0]).all()  # below the injection layer
    assert np.allclose(dec["att_along"][2], 2.0) and np.allclose(dec["mlp_along"][2], 0.0)
    assert np.allclose(dec["att_along"][3], 0.0, atol=1e-5)
    assert dec["exactness"][2].max() < 1e-5  # blocks 1 and 3 have a zero increment: their ratio is rounding over rounding
    assert np.allclose(dec["increment_along"][2], 2.0)
    assert np.allclose(dec["nat_att_along"][2], along(ai[2] - ab[2], u))
    assert dec["att_norm"][2].shape == (N,) and np.allclose(dec["att_norm"][2], 2.0)


def test_window_shares_and_verdict():
    att = np.array([[np.nan] * 3, [1.0, 2.0, 0.5], [1.0, 1.0, 0.5], [0.0, 0.0, 0.0]])
    mlp = np.array([[np.nan] * 3, [1.0, 0.0, 0.5], [0.0, 1.0, -1.5], [0.0, 0.0, 0.0]])
    s = window_shares(att, mlp, [1, 2])
    assert np.allclose(s["att_sum"], [2.0, 3.0, 1.0]) and np.allclose(s["mlp_sum"], [1.0, 1.0, -1.0])
    assert np.allclose(s["att_share"][:2], [2 / 3, 3 / 4]) and np.isnan(s["att_share"][2])  # a non-positive total gives no share
    assert np.isnan(window_shares(att, mlp, [])["att_share"]).all()
    assert share_verdict(0.7, (0.55, 0.9)) == "attention"
    assert share_verdict(0.3, (0.1, 0.45)) == "mlp"
    assert share_verdict(0.7, (0.45, 0.9)) == "mixed"
    assert share_verdict(None, None) == "undetermined"


def test_head_writes_sum_over_selected_heads_through_the_output_projection():
    rng = np.random.default_rng(1)
    N, n_heads, hd, d = 3, 4, 2, 5
    dz = rng.standard_normal((N, n_heads, hd + 1))  # padded to a wider head: the padding must be ignored
    W = rng.standard_normal((d, n_heads * hd))
    sel, allw = head_writes(dz, W, hd, [1, 3])
    expect_all = dz[:, :, :hd].reshape(N, -1) @ W.T
    expect_sel = sum(dz[:, h, :hd] @ W[:, h * hd:(h + 1) * hd].T for h in (1, 3))
    assert np.allclose(allw, expect_all) and np.allclose(sel, expect_sel)

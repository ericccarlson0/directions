"""DAS at the answer token (docs/DECISIONS.md D48): the single-token path against the full-sequence swaps, the pairs,
the training step and a toy run."""

import json

import numpy as np
import pytest
import torch

from directions.config import ModelConfig, ToyModelConfig, load_das_config
from directions.das import (
    ISOLATE,
    TASK_ISOLATE,
    TASK_SOURCE_KINDS,
    Fit,
    LastToken,
    digits_of,
    run_das,
    sample_splits,
    source_mask,
    targets,
)
from directions.model import ModelBackend, SitePatch
from directions.varmap import all_exprs, render


@pytest.fixture(scope="module", params=["qwen3", "olmo3", "gemma4"])
def backend(request):
    cfg = ModelConfig(backend="toy", dtype="float32", device="cpu", batch_size=8, toy=ToyModelConfig(family=request.param))
    return ModelBackend(cfg, run_seed=5)


@pytest.fixture(scope="module")
def prompts_and_sources():
    ex = all_exprs("two_lane")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ex), 12, replace=False)
    demos = [ex[i] for i in rng.choice(len(ex), 2, replace=False)]
    bases = [ex[i] for i in pick[:6]]
    srcs = [ex[i] for i in pick[6:]]
    return [render(demos, b) for b in bases], [render(demos, s) for s in srcs]


def _last_pos(backend, prompts):
    return np.array([len(backend._encode_prompt(p)[0]) - 1 for p in prompts])


def test_single_token_path_matches_the_full_forward(backend, prompts_and_sources):
    prompts, _ = prompts_and_sources
    full = backend.run(prompts, capture_logprobs=True)
    lt = LastToken(backend, prompts)
    with torch.no_grad():
        got = torch.log_softmax(lt.logits(), dim=1).numpy()
    np.testing.assert_allclose(got, full.query_logprobs, rtol=0, atol=2e-4)
    with torch.no_grad():  # the cache is restored after each pass
        np.testing.assert_allclose(torch.log_softmax(lt.logits(), dim=1).numpy(), got, rtol=0, atol=1e-6)


def test_swaps_on_the_single_token_path_match_site_patches(backend, prompts_and_sources):
    prompts, src_prompts = prompts_and_sources
    L = backend.n_layers
    src = backend.run(src_prompts, capture=True).residuals  # (L+1, N, d) at the answer token
    pos = _last_pos(backend, prompts)
    lt = LastToken(backend, prompts)
    rng = np.random.default_rng(1)
    B = np.linalg.qr(rng.normal(size=(backend.hidden_size, 3)))[0].T  # (3, d) orthonormal rows
    for m in (1, L - 1, L):
        vals = src[m]
        for basis in (None, B):
            ref = backend.run(prompts, capture_logprobs=True,
                              site_patch=SitePatch(np.full(len(prompts), m), pos, vals, basis)).query_logprobs
            with torch.no_grad():
                got = lt.logits(m, torch.as_tensor(vals), None if basis is None else torch.as_tensor(B, dtype=torch.float32))
            np.testing.assert_allclose(torch.log_softmax(got, dim=1).numpy(), ref, rtol=0, atol=2e-4)


def test_a_fit_learns_on_a_fixed_batch(backend, prompts_and_sources):
    prompts, src_prompts = prompts_and_sources
    L = backend.n_layers
    src = torch.as_tensor(backend.run(src_prompts, capture=True).residuals[L])
    lt = LastToken(backend, prompts)
    with torch.no_grad():
        whole = lt.logits(L, src, None)
        y = whole.argmax(dim=1)  # what the whole swap at the last read point says: reachable by a subspace
        floor = float(torch.nn.functional.cross_entropy(whole, y))
    k = backend.hidden_size // 2
    f = Fit(L, "Y", k, backend.hidden_size, backend.device, 7, 0.05)
    losses = []
    for _ in range(60):
        with torch.enable_grad():
            loss = torch.nn.functional.cross_entropy(lt.logits(L, src, f.basis()), y)
            loss.backward()
        assert f.W.grad is not None and float(f.W.grad.abs().sum()) > 0
        f.opt.step()
        f.opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    # training closes most of the gap between the untrained subspace and the whole swap
    assert losses[-1] - floor < 0.5 * (losses[0] - floor)
    Bm = f.basis().detach()
    torch.testing.assert_close(Bm @ Bm.T, torch.eye(k), rtol=0, atol=1e-5)


def test_sources_satisfy_their_definitions():
    ex = all_exprs("two_lane")
    D = digits_of(ex)
    rng = np.random.default_rng(3)
    for i in rng.choice(len(ex), 25, replace=False):
        b = ex[i]
        m = source_mask("cause", b, D)
        for j in np.flatnonzero(m)[:20]:
            s = ex[j]
            t = [b.Y] + [targets(v, b, s) for v in ("S1", "S2", "Y", "a")]
            assert len(set(t)) == 5 and all(0 <= x <= 9 for x in t)
            assert s.a != b.a and s.b != b.b and s.S1 != b.S1 and s.S2 != b.S2
        for j in np.flatnonzero(source_mask("same_S1", b, D))[:10]:
            assert ex[j].S1 == b.S1 and (ex[j].c, ex[j].d) == (b.c, b.d) and (ex[j].a, ex[j].b) != (b.a, b.b)
        for j in np.flatnonzero(source_mask("same_Y", b, D))[:10]:
            assert ex[j].Y == b.Y and ex[j].S1 != b.S1
    assert set(ISOLATE) == {"S1", "S2", "Y", "a"}


def test_splits_are_disjoint_and_seeded():
    kinds = {"train": ("cause",), "val": ("cause",), "test": ("cause", "only_S2", "same_S1")}
    a, _ = sample_splits(1, {"train": 30, "val": 10, "test": 10}, 2, 8, kinds)
    b, _ = sample_splits(1, {"train": 30, "val": 10, "test": 10}, 2, 8, kinds)
    sets = {k: {e.text() for e in v.bases} for k, v in a.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert [e.text() for e in a["test"].sources["same_S1"]] == [e.text() for e in b["test"].sources["same_S1"]]


def test_toy_run_writes_the_grid(tmp_path):
    cfg = load_das_config("configs/smoke_toy_das.yaml")
    cfg.output_dir = str(tmp_path)
    root = run_das(cfg, run_id="toy")
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["determinism_check"]["training"]["identical"] and meta["determinism_check"]["evaluation"]["identical"]
    r = json.loads((root / "core" / "das.json").read_text())
    assert set(r["cells"]) == {f"{s}/{v}" for s in cfg.sites for v in cfg.variables}
    for c in r["cells"].values():
        assert c["k_star"] in cfg.ks and set(c["grid"]) == {str(k) for k in cfg.ks}
    assert (root / "figures" / "das.png").exists()
    bases = np.load(root / "core" / "das_bases.npz")
    for s in cfg.sites:
        for v in cfg.variables:
            for k in cfg.ks:
                B = bases[f"{s}/{v}/{k}"]
                assert B.shape[0] == k and np.allclose(B @ B.T, np.eye(k), atol=1e-5)
    assert set(np.load(root / "core" / "das_predictions.npz").files)


def test_three_lane_sources_satisfy_their_definitions():
    """Three brackets (D51): one Cause source tells the eight hypotheses apart; the Isolate sources change what they
    say and keep the rest."""
    ex = all_exprs("three_lane")
    D = digits_of(ex)
    assert D.shape == (113575, 6)
    rng = np.random.default_rng(4)
    for i in rng.choice(len(ex), 12, replace=False):
        b = ex[i]
        for j in np.flatnonzero(source_mask("cause", b, D))[:15]:
            s = ex[j]
            t = [b.Y] + [targets(v, b, s) for v in ("S1", "S2", "S3", "T", "U", "Y", "a")]
            assert len(set(t)) == 8 and all(0 <= x <= 9 for x in t)
            assert s.a != b.a and s.b != b.b and s.S1 != b.S1 and s.S2 != b.S2 and s.S3 != b.S3
            assert targets("T", b, s) == s.S1 - s.S2 - b.S3 and targets("U", b, s) == b.S1 - s.S2 - s.S3
        checks = {
            "only_S3": lambda s: (s.a, s.b, s.c, s.d) == (b.a, b.b, b.c, b.d) and s.S3 != b.S3,
            "same_S3": lambda s: (s.a, s.b, s.c, s.d) == (b.a, b.b, b.c, b.d) and s.S3 == b.S3 and (s.e, s.f) != (b.e, b.f),
            "same_T": lambda s: (s.e, s.f) == (b.e, b.f) and s.T == b.T and s.S1 != b.S1 and s.S2 != b.S2,
            "same_U": lambda s: (s.a, s.b) == (b.a, b.b) and s.U == b.U and s.S2 != b.S2 and s.S3 != b.S3,
            "same_Y": lambda s: s.Y == b.Y and s.S1 != b.S1 and s.S2 != b.S2 and s.S3 != b.S3,
        }
        for kind, ok in checks.items():
            for j in np.flatnonzero(source_mask(kind, b, D))[:10]:
                assert ok(ex[j]), kind
    iso = TASK_ISOLATE["three_lane"]
    assert set(iso) == {"S1", "S2", "S3", "T", "U", "Y", "a"} and TASK_ISOLATE["two_lane"] is ISOLATE
    assert set(k for v in iso.values() for k in v) | {"cause"} == set(TASK_SOURCE_KINDS["three_lane"])


def test_three_lane_toy_run(tmp_path):
    cfg = load_das_config("configs/smoke_toy_das.yaml")
    cfg.output_dir, cfg.task, cfg.variables = str(tmp_path), "three_lane", ["S1", "S3", "T", "U", "Y"]
    cfg.sites, cfg.control_site, cfg.ks, cfg.null_k, cfg.epochs = [1, 4], 1, [2], 2, 1
    root = run_das(cfg, run_id="toy3")
    r = json.loads((root / "core" / "das.json").read_text())
    assert r["task"] == "three_lane" and set(r["cells"]) == {f"{s}/{v}" for s in (1, 4) for v in cfg.variables}
    iso = r["cells"]["4/T"]["grid"]["2"]["isolate"]
    assert set(iso) == {"only_S3", "same_T"} and "sums_held" in r["per_site"]["4"]
    assert (root / "figures" / "das.png").exists()


# --------------------------------------------------------------------------- #
# The joint swap at several tokens (D53)
# --------------------------------------------------------------------------- #


def _tail_geometry(backend, prompts, P, tail):
    """Absolute positions of the last ``tail`` tokens' indices ``P`` (relative to the tail) per prompt."""
    T = np.array([len(backend._encode_prompt(p)[0]) for p in prompts])
    return T[:, None] - tail + np.asarray(P)[None, :]


def test_the_tail_path_without_a_swap_matches_the_full_forward(backend, prompts_and_sources):
    prompts, _ = prompts_and_sources
    full = backend.run(prompts, capture_logprobs=True)
    lt = LastToken(backend, prompts, n_tail=7, swap_idx=[0, 3, 6])
    with torch.no_grad():
        got = torch.log_softmax(lt.logits(), dim=1).numpy()
        again = torch.log_softmax(lt.logits(), dim=1).numpy()
    np.testing.assert_allclose(got, full.query_logprobs, rtol=0, atol=2e-4)
    np.testing.assert_allclose(again, got, rtol=0, atol=1e-6)  # the cache is restored after a multi-token pass


def test_joint_swaps_on_the_tail_path_match_set_site_patches(backend, prompts_and_sources):
    """The tail path's joint swap (concatenated states, one basis) against the native full pass with a set of
    positions swapped (D52's site patch): the whole states, and a basis that is a different subspace per token
    (block-diagonal), which the set patch applies as one basis per token."""
    prompts, src_prompts = prompts_and_sources
    L = backend.n_layers
    tail, P = 7, [0, 3, 6]
    pos = _tail_geometry(backend, prompts, P, tail)
    src = backend.capture_sites(src_prompts, pos)  # (N, L+1, P, d), the sources share the base's token geometry
    d = backend.hidden_size
    lt = LastToken(backend, prompts, n_tail=tail, swap_idx=P)
    shared = LastToken(backend, prompts, n_tail=tail, swap_idx=P, shared=True)
    rng = np.random.default_rng(2)
    B1 = np.linalg.qr(rng.normal(size=(d, 3)))[0].T  # (3, d)
    joint = np.zeros((3 * len(P), d * len(P)), dtype=np.float32)
    for i in range(len(P)):
        joint[3 * i : 3 * i + 3, d * i : d * (i + 1)] = B1  # the same 3-dim subspace at each token
    for m in (1, L - 1, L):
        vals = src[:, m]  # (N, P, d)
        for basis in (None, B1):
            ref = backend.run(prompts, capture_logprobs=True,
                              site_patch=SitePatch(np.full(len(prompts), m), pos, vals, basis)).query_logprobs
            jb = None if basis is None else torch.as_tensor(joint)
            with torch.no_grad():
                got = lt.logits(m, vals.reshape(len(prompts), -1).float(), jb)
            np.testing.assert_allclose(torch.log_softmax(got, dim=1).numpy(), ref, rtol=0, atol=2e-4)
            if basis is not None:  # D54: one (k, d) subspace shared by the tokens is the set patch's per-token basis
                with torch.no_grad():
                    sh = shared.logits(m, vals.reshape(len(prompts), -1).float(), torch.as_tensor(B1, dtype=torch.float32))
                np.testing.assert_allclose(torch.log_softmax(sh, dim=1).numpy(), ref, rtol=0, atol=2e-4)


def test_a_joint_basis_inside_one_token_equals_that_tokens_swap(backend, prompts_and_sources):
    """A joint basis supported on one token's block swaps that token alone: the same as the one-token tail swap."""
    prompts, src_prompts = prompts_and_sources
    L = backend.n_layers
    tail, P = 5, [0, 4]
    pos = _tail_geometry(backend, prompts, P, tail)
    src = backend.capture_sites(src_prompts, pos)[:, L - 1]  # (N, P, d)
    d = backend.hidden_size
    B1 = np.linalg.qr(np.random.default_rng(3).normal(size=(d, 4)))[0].T.astype(np.float32)
    joint = np.zeros((4, 2 * d), dtype=np.float32)
    joint[:, d:] = B1  # the answer token's block only
    with torch.no_grad():
        a = LastToken(backend, prompts, n_tail=tail, swap_idx=P).logits(L - 1, src.reshape(len(prompts), -1).float(),
                                                                        torch.as_tensor(joint))
        b = LastToken(backend, prompts).logits(L - 1, src[:, 1].float(), torch.as_tensor(B1))
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=0, atol=2e-4)


def test_three_lane_joint_toy_run(tmp_path):
    cfg = load_das_config("configs/smoke_toy_das.yaml")
    cfg.output_dir, cfg.task, cfg.variables = str(tmp_path), "three_lane", ["S1", "T", "Y"]
    cfg.sites, cfg.control_site, cfg.ks, cfg.null_k, cfg.epochs = [2, 4], None, [2], 2, 1
    cfg.positions = "non_digit"
    root = run_das(cfg, run_id="toy3j")
    r = json.loads((root / "core" / "das.json").read_text())
    sw = r["swap"]
    labels = sw["labels"]
    assert r["positions"] == "non_digit" and r["control_clean"] is None
    assert all(labels[j] not in "abcdef" for j in sw["tokens"]) and sw["tokens"][-1] == len(labels) - 1
    assert min(sw["tokens"]) > labels.index("a") and sw["n_tail"] == len(labels) - min(sw["tokens"])
    d = ModelBackend(cfg.model, run_seed=cfg.seed).hidden_size
    assert np.load(root / "core" / "das_bases.npz")["4/T/2"].shape == (2, len(sw["tokens"]) * d)  # one basis, concatenated
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["determinism_check"]["training"]["identical"] and meta["determinism_check"]["evaluation"]["identical"]
    assert (root / "figures" / "das.png").exists()


def test_three_lane_two_token_shared_toy_run(tmp_path):
    """D54: the answer token and = with one subspace shared by both; the kept bases are (k, d)."""
    cfg = load_das_config("configs/smoke_toy_das.yaml")
    cfg.output_dir, cfg.task, cfg.variables = str(tmp_path), "three_lane", ["S1", "U", "Y"]
    cfg.sites, cfg.control_site, cfg.ks, cfg.null_k, cfg.epochs = [1, 4], 1, [2], 2, 1
    cfg.positions, cfg.shared_basis = "answer_eq", True
    root = run_das(cfg, run_id="toy3s")
    r = json.loads((root / "core" / "das.json").read_text())
    sw = r["swap"]
    assert r["shared_basis"] and [sw["labels"][j].strip() for j in sw["tokens"]] == ["=", "␣"] and sw["n_tail"] == 2
    d = ModelBackend(cfg.model, run_seed=cfg.seed).hidden_size
    assert np.load(root / "core" / "das_bases.npz")["4/U/2"].shape == (2, d)
    assert r["control_clean"] is not None and (root / "figures" / "das.png").exists()


def test_two_passes_before_one_backward_give_the_averaged_gradient(backend, prompts_and_sources):
    """D56 runs a Cause pass and an Isolate pass of the same fit before one backward pass: the gradient must equal
    the average of the two passes' gradients taken separately (the cache restore keeps the two graphs apart)."""
    prompts, src_prompts = prompts_and_sources
    m = backend.n_layers // 2
    lt = LastToken(backend, prompts)
    src_a = torch.as_tensor(backend.run(src_prompts, capture=True).residuals[m])
    src_b = torch.as_tensor(backend.run(src_prompts[::-1], capture=True).residuals[m])
    ya = torch.arange(len(prompts)) + 3
    yb = torch.arange(len(prompts)) + 11
    f = Fit(m, "Y", 4, backend.hidden_size, backend.device, 3, 0.05)
    ce = torch.nn.functional.cross_entropy

    def grad(loss_fn):
        f.W.grad = None
        with torch.enable_grad():
            loss_fn().backward()
        return f.W.grad.clone()

    ga = grad(lambda: ce(lt.logits(m, src_a, f.basis()), ya))
    gb = grad(lambda: ce(lt.logits(m, src_b, f.basis()), yb))
    gj = grad(lambda: 0.5 * (ce(lt.logits(m, src_a, f.basis()), ya) + ce(lt.logits(m, src_b, f.basis()), yb)))
    assert float(ga.abs().sum()) > 0 and float(gb.abs().sum()) > 0 and not torch.allclose(ga, gb)
    torch.testing.assert_close(gj, 0.5 * (ga + gb), rtol=1e-5, atol=1e-7)

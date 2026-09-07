import numpy as np
import pytest

from directions.ablation import select_blocks
from directions.analysis import assign_labels, profile_signature
from directions.config import AnalysisConfig, BlockAblationConfig, EvaluationConfig
from directions.geometry import normalize
from directions.layerwise import compare_to_random, compute_profile


def _synthetic(rng, L=6, n=12, d=16, ls=2, gain=1.0):
    base = rng.standard_normal((L + 1, n, d)) * 5
    v = normalize(rng.standard_normal(d))
    steered = base.copy()
    delta = np.zeros((L + 1, n, d))
    delta[ls] = 3.0 * v[None, :]
    for l in range(ls + 1, L + 1):
        delta[l] = gain * delta[l - 1] + 0.5 * rng.standard_normal((n, d))
    steered += delta
    return base, steered, v


def test_compute_profile_shapes_and_nulls():
    rng = np.random.default_rng(0)
    base, steered, v = _synthetic(rng)
    cfg = EvaluationConfig(n_boot=50)
    p = compute_profile(base, steered, v, 2, cfg, np.random.default_rng(1))
    assert p.pre_intervention_max_abs_delta == 0.0
    assert np.isnan(p.d_eff[:3]).all() and not np.isnan(p.d_eff[3:]).any()
    assert np.isnan(p.new_subspace[:3]).all() and not np.isnan(p.new_subspace_uncentered[2])
    assert p.summaries["alignment"]["median"][2] == pytest.approx(1.0)
    assert p.noise_floor["centered_variance_at_intervention_layer"] == pytest.approx(0.0, abs=1e-20)
    assert 0 <= p.new_subspace_uncentered[2] <= 1


def test_select_blocks_by_value_and_by_z():
    rng = np.random.default_rng(0)
    base, steered, v = _synthetic(rng)
    cfg = EvaluationConfig(n_boot=50)
    prof = compute_profile(base, steered, v, 2, cfg, np.random.default_rng(1))
    randoms = []
    for k in range(4):
        u = normalize(rng.standard_normal(16))
        _, st_r, _ = _synthetic(np.random.default_rng(10 + k))
        randoms.append(compute_profile(base, base + (st_r - base), u, 2, cfg, np.random.default_rng(2)))
    comp = compare_to_random(prof, randoms)
    by_value = select_blocks(prof, BlockAblationConfig(n_blocks=2, criterion="conversion", rank_by="value"))
    assert len(by_value) == 2 and all(b >= 2 for b in by_value)
    conv = prof.metric_curve("conversion")
    assert set(by_value) == set(int(i) for i in np.argsort(-np.nan_to_num(conv, nan=-np.inf))[:2])
    by_z = select_blocks(prof, BlockAblationConfig(n_blocks=2, criterion="conversion", rank_by="z"), comp)
    assert len(by_z) == 2 and all(b >= 2 for b in by_z)
    with pytest.raises(ValueError):
        select_blocks(prof, BlockAblationConfig(rank_by="z"))


def test_labels_from_signature_rules():
    cfg = AnalysisConfig()
    conserved = {
        "alignment_final": 0.95, "cumulative_log_gain": 0.1, "cumulative_log_gain_ci": [-0.1, 0.3],
        "conversion_dominant_share": 0.2, "conversion_entropy_ratio": 0.5, "conversion_centre_of_mass": 0.3,
        "d_eff_ratio_final_over_first": 1.0, "new_subspace_uncentered_z_mean": 0.0,
    }
    assert assign_labels(conserved, cfg) == ["conserved_transmission"]
    delayed = {**conserved, "alignment_final": 0.1, "conversion_dominant_share": 0.6, "conversion_centre_of_mass": 0.9}
    assert assign_labels(delayed, cfg) == ["localized_transformation", "delayed_activation"]
    cascade = {**conserved, "alignment_final": 0.1, "conversion_entropy_ratio": 0.95,
               "cumulative_log_gain": 2.0, "cumulative_log_gain_ci": [1.5, 2.5],
               "d_eff_ratio_final_over_first": 2.0, "new_subspace_uncentered_z_mean": 3.0}
    assert assign_labels(cascade, cfg) == ["cascade", "amplification", "dimensional_expansion", "new_direction_creation"]


def test_profile_signature_runs_end_to_end():
    rng = np.random.default_rng(3)
    base, steered, v = _synthetic(rng, gain=1.3)
    cfg = EvaluationConfig(n_boot=50)
    prof = compute_profile(base, steered, v, 2, cfg, np.random.default_rng(1))
    sig = profile_signature(prof, None, AnalysisConfig())
    assert sig["intervention_layer"] == 2 and sig["n_downstream_blocks"] == 4
    assert len(sig["conversion_mass"]) == 4 and abs(sum(sig["conversion_mass"]) - 1) < 1e-9
    assert sig["cumulative_log_gain"] > 0
    assert isinstance(sig["labels"], list)

"""directions.family: the within-family geometry (D39)."""

import numpy as np

from directions.family import family_geometry, format_geometry, order_test, pairwise_cosines, seed_spread


def test_pairwise_cosines_and_seed_spread():
    C = pairwise_cosines([np.array([1.0, 0.0]), np.array([0.0, 2.0]), np.array([1.0, 1.0])])
    assert np.allclose(np.diag(C), 1) and abs(C[0, 1]) < 1e-12 and abs(C[0, 2] - np.sqrt(0.5)) < 1e-12
    s = seed_spread(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]))
    assert s["max"] == 1.0 and s["min"] == 0.0 and s["median"] == 0.0


def test_order_test_reads_a_string_as_ordered_and_a_cloud_as_not():
    rng = np.random.default_rng(0)
    # a string: vectors on a curve, cosine falling with the parameter distance
    base = rng.standard_normal((2, 64))
    ks = [1, 2, 3, 5, 10]
    string = [np.cos(0.15 * k) * base[0] + np.sin(0.15 * k) * base[1] for k in ks]
    r = order_test(pairwise_cosines(string), ks)
    assert r["exact"] and r["n_perm"] == 120 and r["rho"] > 0.9 and r["p"] <= 0.05 and r["ordered"]
    # a cloud: independent random vectors
    cloud = [rng.standard_normal(64) for _ in ks]
    r = order_test(pairwise_cosines(cloud), ks)
    assert r["p"] > 0.05 or r["rho"] < 0.5
    # too few labels
    assert order_test(np.eye(2), [1, 2])["rho"] is None
    # the Monte Carlo path
    r = order_test(pairwise_cosines(string), ks, max_exact=3, n_perm=500)
    assert not r["exact"] and r["n_perm"] == 500 and r["p"] <= 0.05


def test_family_geometry_reads_run_directories(tmp_path):
    rng = np.random.default_rng(1)
    layers = np.array([3, 5])
    labels, params = ["kth_1", "kth_2", "kth_3", "kth_4", "kth_5"], [1, 2, 3, 4, 5]
    base = rng.standard_normal((2, 32))
    for lab, k in zip(labels, params):
        d = tmp_path / "learned" / "core" / "tasks" / lab
        d.mkdir(parents=True)
        v = np.cos(0.2 * k) * base[0] + np.sin(0.2 * k) * base[1]
        learned = np.stack([v, v + 0.01 * rng.standard_normal(32)])
        per_seed = np.stack([np.stack([v + 0.3 * rng.standard_normal(32) for _ in range(3)]) for _ in layers])
        np.savez(d / "directions.npz", layers=layers, learned=learned, learned_per_seed=per_seed)
        f = tmp_path / "fv" / "core" / "tasks" / lab
        f.mkdir(parents=True)
        if k != 5:  # one label without a head mean
            np.savez(f / "directions.npz", fv_direction=base[0] + 0.05 * k * base[1])
            (f / "qualification.json").write_text('{"selection": {"layer": 4}}')
    (tmp_path / "learned" / "core" / "tasks" / "kth_6").mkdir()  # a label without vectors is reported missing
    g = family_geometry(tmp_path / "learned", labels + ["kth_6"], params + [6], fv_run=tmp_path / "fv")
    assert g["labels"] == labels and g["missing"] == ["kth_6"]
    r = g["per_layer"]["3"]
    assert r["order"]["ordered"] and r["cross_label"]["median"] > r["seed_spread_median_over_labels"]
    assert r["cross_below_seed_spread"] is False
    assert g["head_mean"]["labels"] == labels[:4] and g["head_mean"]["selected_layers"]["kth_1"] == 4
    assert g["head_mean"]["order"]["rho"] is not None
    text = format_geometry(g, "toy")
    assert "layer 3" in text and "head means" in text

"""The staging test's geometry and readings (docs/DECISIONS.md D42) on synthetic data."""

import numpy as np
import pytest

from directions.config import load_staging_config
from directions.staging import (apply_steps, lens_verdict, masking_verdict, reference_verdict, residualise, rowwise_cos,
                                specific_excess, step_function, windows)


def test_residualise_and_specific_excess_read_the_coefficients():
    """e_perp's two cosines have the signs of the least-squares coefficients of delta on (d_g, d_c), whatever the
    angle between the references; the plain cosines of D41 barely separate them."""
    rng = np.random.default_rng(0)
    n, d = 200, 64
    shared = rng.normal(size=(n, d))
    d_g = shared + 0.3 * rng.normal(size=(n, d))  # ~0.95 aligned with d_c, as D41's references were
    d_c = shared + 0.3 * rng.normal(size=(n, d))
    assert np.median(rowwise_cos(d_g, d_c)) > 0.9
    r = residualise(d_g, d_c)
    np.testing.assert_allclose(np.sum(r * d_c, axis=1), 0, atol=1e-9)
    toward_g = 1.0 * d_g - 0.2 * d_c
    e, cg, cc = specific_excess(toward_g, d_g, d_c)
    assert (cg > 0).all() and (cc < 0).all() and (e > 0).all()
    e2, _, _ = specific_excess(-0.2 * d_g + 1.0 * d_c, d_g, d_c)
    assert (e2 < 0).all()
    # D41's statistic on the same perturbation is several times smaller: the shared part dominates both cosines
    plain = rowwise_cos(toward_g, d_g) - rowwise_cos(toward_g, d_c)
    assert np.median(e) > 3 * np.median(plain) > 0
    # batched over read points gives the per-row answer
    stack = np.stack([toward_g, -toward_g])
    np.testing.assert_allclose(specific_excess(stack, np.stack([d_g, d_g]), np.stack([d_c, d_c]))[0][0], e)


def test_zero_reference_gives_nan_not_error():
    z = np.zeros((3, 4))
    assert np.isnan(rowwise_cos(z, np.ones((3, 4)))).all()
    np.testing.assert_array_equal(residualise(np.ones((3, 4)), z), np.ones((3, 4)))


def test_windows_and_verdicts():
    assert windows([True, True, False, True, False, True, True, True]) == [0, 5, 6]
    F, T = False, True
    assert reference_verdict([F, F, F, F], [T, T, F, F])["verdict"] == "composed_only"
    assert reference_verdict([T, T, F, F, F], [F, F, F, F, F])["verdict"] == "component_only"
    assert reference_verdict([T, T, F, F, F], [F, F, F, T, T])["verdict"] == "staircase"
    # a negative window overlapping the positive one's second point does not count as after it
    assert reference_verdict([T, T, F], [F, T, T])["verdict"] == "component_only"
    # lens: the intermediate's window must come before the final's
    assert lens_verdict([T, T, F, F], [T, T, F, F], [F, F, T, T])["staged"]
    assert not lens_verdict([F, F, T, T], [F, F, T, T], [T, T, F, F])["staged"]
    assert not lens_verdict([T, T, F, F], [F, T, F, F], [F, F, T, T])["staged"]  # never positive in a window
    assert lens_verdict([T, T], [T, T], [F, F])["staged"]
    assert masking_verdict([F, T, T, F], [F, T, T, F])["staged"]
    assert not masking_verdict([F, T, T, F], [F, T, F, F])["staged"]


def test_intermediate_steps():
    assert apply_steps("a, b, hot", [step_function("last_word")]) == "hot"
    assert apply_steps("a, b, hot", [step_function("last_word"), step_function("antonym")]) == "cold"
    assert apply_steps("hot", [step_function("antonym"), step_function("uppercase")]) == "COLD"
    assert apply_steps("zzqx", [step_function("antonym")]) is None


def test_staging_config_intermediates_match_the_composed_targets():
    """Every configured intermediate, carried one more step, gives the composed task's own target (D42): the lens
    looks for the word the composition actually passes through."""
    from directions.tasks import build_task

    cfg = load_staging_config("configs/staging.yaml")
    tasks = {"upper_antonym": build_task("compose", {"steps": ["antonym", "uppercase"]}),
             "last_antonym": build_task("last_antonym"),
             "upper_last_antonym": build_task("compose", {"steps": ["last_antonym", "uppercase"]}),
             "upper_plural": build_task("compose", {"steps": ["plural", "uppercase"]}),
             "upper_last": build_task("compose", {"steps": ["kth_word", "uppercase"],
                                                  "step_params": {"kth_word": {"n_words": 3, "k": 3, "n_items": 500, "items_seed": 20260908}}})}
    last_step = {"upper_antonym": ["uppercase"], "last_antonym": ["antonym"], "upper_last_antonym": ["antonym", "uppercase"],
                 "upper_plural": ["uppercase"], "upper_last": ["uppercase"]}
    assert {c.task for c in cfg.compositions} == set(tasks)
    for c in cfg.compositions:
        first = c.intermediates[0]
        fns = [step_function(s) for s in first.steps + last_step[c.task]]
        items = tasks[c.task].items
        assert all(apply_steps(it.input, fns) == it.output for it in items), c.task


def test_staging_config_validation(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("compositions:\n  - {task: a, references: [a], intermediates: [{label: x, steps: [antonym]}]}\n")
    with pytest.raises(ValueError, match="references"):
        load_staging_config(p)
    p.write_text("compositions:\n  - {task: a, references: [b], intermediates: []}\n")
    with pytest.raises(ValueError, match="intermediate"):
        load_staging_config(p)

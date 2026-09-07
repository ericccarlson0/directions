import numpy as np
import pytest

from directions.config import PromptConfig
from directions.prompts import few_shot_prompt, paired_prompts, zero_shot_prompt
from directions.tasks import (
    TASK_BUILDERS,
    Item,
    Task,
    build_task,
    derangement,
    filter_items,
    make_splits,
    sample_demonstrations,
)


def test_all_tasks_build_with_unique_inputs_and_enough_items():
    for name in TASK_BUILDERS:
        t = build_task(name)
        inputs = [i.input for i in t.items]
        assert len(inputs) == len(set(inputs)), name
        assert len(t.items) >= 192, name  # 3 x 64 pools


def test_arithmetic_params():
    t = build_task("arithmetic", {"operation": "add", "operand": 5, "min": 0, "max": 10})
    assert [(i.input, i.output) for i in t.items][:2] == [("0", "5"), ("1", "6")]
    with pytest.raises(ValueError):
        build_task("arithmetic", {"operation": "power"})
    with pytest.raises(ValueError):
        build_task("antonym", {"x": 1})


def test_derangement_has_no_fixed_points():
    rng = np.random.default_rng(0)
    for n in (2, 3, 8, 20):
        for _ in range(20):
            p = derangement(n, rng)
            assert sorted(p) == list(range(n))
            assert not np.any(p == np.arange(n))


def test_splits_disjoint_deterministic_and_reduced():
    items = [Item(str(i), str(i * 2)) for i in range(30)]
    a = make_splits(items, 1, "t", 10, 10, 10)
    b = make_splits(items, 1, "t", 10, 10, 10)
    assert a.as_dict() == b.as_dict()
    pools = [set(i.input for i in p) for p in (a.extraction, a.calibration, a.evaluation)]
    assert not (pools[0] & pools[1]) and not (pools[1] & pools[2]) and not (pools[0] & pools[2])
    c = make_splits(items, 2, "t", 10, 10, 10)
    assert c.as_dict() != a.as_dict()  # different seed -> different partition
    with pytest.raises(ValueError):
        make_splits(items, 1, "t", 20, 20, 20)
    r = make_splits(items, 1, "t", 20, 20, 20, allow_reduced=True)
    assert r.reduced and len(r.extraction) + len(r.calibration) + len(r.evaluation) <= 30


def test_filter_items_logs_rejections():
    t = Task("x", [Item("a", "short"), Item("b", "much longer target")])
    kept, rej = filter_items(t, lambda s: len(s.split()), " {output}", 1)
    assert [k.input for k in kept] == ["a"]
    assert rej[0].reason == "target_too_many_tokens" and rej[0].details["input"] == "b"


def test_prompts_and_pairs():
    cfg = PromptConfig(n_shots=4)
    pool = [Item(f"in{i}", f"out{i}") for i in range(12)]
    q = pool[3]
    rng = np.random.default_rng(0)
    demos = sample_demonstrations(pool, q, 4, rng)
    assert len(demos) == 4 and q not in demos and len({d.output for d in demos}) == 4
    fs = few_shot_prompt(cfg, pool, q, rng)
    assert fs.prompt.endswith("Q: in3\nA:") and fs.target == " out3" and len(fs.demos) == 4
    zs = zero_shot_prompt(cfg, q)
    assert zs.prompt == "Q: in3\nA:" and zs.demos == ()
    pos, neg = paired_prompts(cfg, pool, q, rng)
    assert [d.input for d in pos.demos] == [d.input for d in neg.demos]
    assert all(p.output != n.output for p, n in zip(pos.demos, neg.demos))
    assert sorted(d.output for d in pos.demos) == sorted(d.output for d in neg.demos)
    assert pos.target == neg.target

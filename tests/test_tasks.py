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


def test_composite_and_extractive_tasks():
    from directions import data
    from directions.tasks import LIST_SEPARATOR

    antonym = dict(data.dedupe(data.ANTONYM))
    la = build_task("last_antonym")
    assert len(la.items) == len(antonym) >= 320
    for it in la.items:
        words = it.input.split(LIST_SEPARATOR)
        assert len(words) == 3 and len(set(words)) == 3 and antonym[words[-1]] == it.output
        assert it.output not in words[:-1]
    assert build_task("last_antonym").items == la.items  # deterministic, independent of the run seed
    assert build_task("last_antonym", {"n_words": 4}).items[0].input.count(LIST_SEPARATOR) == 3
    assert build_task("last_antonym", {"items_seed": 1}).items != la.items

    af = build_task("alphabetically_first")
    assert len(af.items) == 500 and len({i.input for i in af.items}) == 500
    for it in af.items:
        words = it.input.split(LIST_SEPARATOR)
        assert len(words) == 3 and len(set(words)) == 3 and it.output == sorted(words)[0]
    assert build_task("alphabetically_first").items == af.items
    assert len(build_task("alphabetically_first", {"n_items": 40, "n_words": 5}).items) == 40
    kw = {k: build_task("kth_word", {"k": k}) for k in (1, 3, 5)}
    assert len(kw[1].items) == 500 and len({i.input for i in kw[1].items}) == 500
    assert [i.input for i in kw[3].items] == [i.input for i in kw[1].items]  # the family shares its inputs
    for k, t in kw.items():
        for it in t.items:
            words = it.input.split(LIST_SEPARATOR)
            assert len(words) == 5 and len(set(words)) == 5 and it.output == words[k - 1]
    assert build_task("kth_word", {"k": 1}).items == kw[1].items
    assert build_task("kth_word", {"k": 2, "n_words": 3}).items[0].input.count(LIST_SEPARATOR) == 2
    assert build_task("kth_word", {"items_seed": 1}).items != kw[1].items
    aw = build_task("arithmetic_words")
    assert len(aw.items) == 997 and aw.items[0].input == "0" and aw.items[0].output == "three"
    assert aw.items[-1].input == "996" and aw.items[-1].output == "nine hundred ninety-nine"
    assert len(build_task("arithmetic_words", {"max": 999}).items) == 997  # outputs above 999 are dropped
    # D41: compositions map the first step's items through the later steps; uppercase is a function of the word,
    # a pair task drops the items its list does not cover
    ua = build_task("compose", {"steps": ["antonym", "uppercase"]})
    an = build_task("antonym")
    assert ua.name == "compose" and len(ua.items) == len(an.items)
    assert all(c.input == a.input and c.output == a.output.upper() for c, a in zip(ua.items, an.items))
    ul = build_task("compose", {"steps": ["kth_word", "uppercase"], "step_params": {"kth_word": {"n_words": 3, "k": 3}}})
    assert len(ul.items) == 500 and all(it.output == it.input.split(LIST_SEPARATOR)[-1].upper() for it in ul.items)
    ula = build_task("compose", {"steps": ["last_antonym", "uppercase"]})
    la = build_task("last_antonym")
    assert [(c.input, c.output) for c in ula.items] == [(a.input, a.output.upper()) for a in la.items]
    pa = build_task("compose", {"steps": ["antonym", "plural"]})
    plural = {it.input: it.output for it in build_task("plural").items}
    assert 0 < len(pa.items) < len(an.items) and all(plural[a.output] == c.output for c in pa.items for a in an.items if a.input == c.input)
    for name, bad in (("last_antonym", {"n_words": 1}), ("alphabetically_first", {"n_words": 1}),
                      ("compose", {"steps": ["antonym"]}), ("compose", {"steps": ["antonym", "compose"]}),
                      ("compose", {"steps": ["antonym", "uppercase"], "step_params": {"plural": {}}}),
                      ("compose", {"steps": ["arithmetic", "antonym"]}),
                      ("alphabetically_first", {"foo": 1}), ("kth_word", {"k": 0}), ("kth_word", {"k": 6}),
                      ("kth_word", {"n_words": 1})):
        with pytest.raises(ValueError):
            build_task(name, bad)


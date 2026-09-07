from __future__ import annotations

import numpy as np
import pytest

from directions.config import PromptTemplate
from directions.prompts import build_eval_prompts, build_pairs, derangement, sample_demos
from directions.tasks import available_tasks, load_task, split_task


@pytest.mark.parametrize("name", available_tasks())
def test_every_task_loads_with_unique_nonempty_items(name):
    task = load_task(name)
    assert len(task) >= 176, f"{name} too small for the default 64/48/64 split"
    assert len({i.id for i in task.items}) == len(task)
    assert all(i.input.strip() and i.output.strip() for i in task.items)


@pytest.mark.parametrize("name", available_tasks())
def test_task_mapping_is_deterministic(name):
    """The same input must never map to two different outputs."""
    mapping: dict[str, str] = {}
    for item in load_task(name).items:
        assert mapping.setdefault(item.input, item.output) == item.output


def test_splits_are_disjoint_and_reproducible():
    task = load_task("antonym")
    a = split_task(task, 10, 5, 10, seed=3)
    b = split_task(task, 10, 5, 10, seed=3)
    assert [i.id for i in a.extraction] == [i.id for i in b.extraction]
    ids = [{i.id for i in v} for v in a.as_dict().values()]
    assert len(ids[0] | ids[1] | ids[2]) == 25


def test_split_seed_changes_the_partition():
    task = load_task("antonym")
    a = split_task(task, 10, 5, 10, seed=0)
    b = split_task(task, 10, 5, 10, seed=1)
    assert [i.id for i in a.extraction] != [i.id for i in b.extraction]


def test_split_rejects_oversized_request():
    with pytest.raises(ValueError, match="requests"):
        split_task(load_task("antonym"), 1000, 1, 1, seed=0)


def test_derangement_has_no_fixed_point():
    rng = np.random.default_rng(0)
    for n in range(2, 12):
        perm = derangement(rng, n)
        assert sorted(perm.tolist()) == list(range(n))
        assert not np.any(perm == np.arange(n))


def test_demos_never_include_the_query():
    task = load_task("plural")
    pool = list(task.items[:20])
    rng = np.random.default_rng(0)
    demos = sample_demos(rng, pool, 5, exclude_id=pool[0].id)
    assert pool[0].id not in {d.id for d in demos}


def test_pairs_differ_only_in_demonstration_outputs():
    task = load_task("antonym")
    splits = split_task(task, 8, 2, 2, seed=0)
    pos, neg = build_pairs(
        np.random.default_rng(0), PromptTemplate(), list(splits.extraction),
        list(splits.extraction)[:1], 4,
    )[0]
    pos_lines = pos.text.split("\n\n")
    neg_lines = neg.text.split("\n\n")
    assert len(pos_lines) == len(neg_lines) == 5
    # same query, same demo inputs, different demo outputs
    assert pos_lines[-1] == neg_lines[-1]
    assert pos.target == neg.target
    for a, b in zip(pos_lines[:-1], neg_lines[:-1]):
        assert a.split("\nA:")[0] == b.split("\nA:")[0]
        assert a != b


def test_zero_shot_eval_prompt_is_the_query_alone():
    task = load_task("en_fr")
    splits = split_task(task, 4, 2, 4, seed=0)
    p = build_eval_prompts(
        np.random.default_rng(0), PromptTemplate(), list(splits.evaluation),
        list(splits.evaluation)[:1], 0,
    )[0]
    assert p.n_shot == 0
    assert "\n\n" not in p.text
    assert p.text.startswith("Q: ")


def test_pairs_require_at_least_two_shots():
    task = load_task("antonym")
    with pytest.raises(ValueError, match="n_shot >= 2"):
        build_pairs(np.random.default_rng(0), PromptTemplate(), list(task.items[:5]),
                    list(task.items[:1]), 1)

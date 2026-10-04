"""Tests for the human-validation sampling design.

The sample exists to answer "is the judge ensemble right?", so a defect in how
it is drawn does not produce an obviously wrong number -- it produces a
confident wrong one. These tests pin the properties the estimate depends on.
"""

from __future__ import annotations

from privacylens_live.base.human_validation import two_part_sample


def _cand(
    name: str, *, run: str = "c2", unanimous: bool = True, leaked: bool = False
) -> dict:
    return {
        "run_label": run,
        "task_name": name,
        "unanimous": unanimous,
        "automatic_leak_label": leaked,
        "stratum": (run, leaked),
    }


def _pool(n_unanimous: int, n_split: int) -> list[dict]:
    return [
        *[
            _cand(f"u{i}", unanimous=True, leaked=i % 2 == 0)
            for i in range(n_unanimous)
        ],
        *[_cand(f"s{i}", unanimous=False, leaked=i % 2 == 0) for i in range(n_split)],
    ]


def test_random_part_is_drawn_from_all_candidates_not_just_unanimous():
    """The property the accuracy estimate rests on.

    Keeping the two parts disjoint by drawing the random half from unanimous
    cases only is the tempting implementation and it biases the sample toward
    agreement -- inflating exactly the figure the exercise is meant to measure.
    With 50% of the pool split, a genuine random draw must contain some.
    """
    pool = _pool(n_unanimous=100, n_split=100)
    picked = two_part_sample(pool, n_random=60, n_disagreement=0, seed=1)

    assert len(picked) == 60
    n_split_in_random = sum(1 for c in picked if not c["unanimous"])
    assert n_split_in_random > 0, "random part contains no disagreements"


def test_parts_are_labelled_and_do_not_overlap():
    pool = _pool(n_unanimous=100, n_split=60)
    picked = two_part_sample(pool, n_random=40, n_disagreement=20, seed=2)

    keys = [f"{c['run_label']}::{c['task_name']}" for c in picked]
    assert len(keys) == len(set(keys)), "an item was sampled into both parts"
    assert {c["part"] for c in picked} == {"random", "disagreement"}
    assert sum(1 for c in picked if c["part"] == "random") == 40
    assert sum(1 for c in picked if c["part"] == "disagreement") == 20


def test_disagreement_part_contains_only_disagreements():
    pool = _pool(n_unanimous=100, n_split=30)
    picked = two_part_sample(pool, n_random=20, n_disagreement=10, seed=3)

    booster = [c for c in picked if c["part"] == "disagreement"]
    assert booster and all(not c["unanimous"] for c in booster)


def test_sampling_is_deterministic_for_a_seed():
    pool = _pool(n_unanimous=60, n_split=40)
    a = two_part_sample(pool, n_random=20, n_disagreement=10, seed=7)
    b = two_part_sample(pool, n_random=20, n_disagreement=10, seed=7)
    assert [c["task_name"] for c in a] == [c["task_name"] for c in b]


def test_requesting_more_than_exists_is_capped_not_an_error():
    """A condition with few disagreements must not abort the whole package."""
    pool = _pool(n_unanimous=20, n_split=3)
    picked = two_part_sample(pool, n_random=100, n_disagreement=100, seed=4)

    assert len(picked) == len(pool)
    keys = [c["task_name"] for c in picked]
    assert len(keys) == len(set(keys))

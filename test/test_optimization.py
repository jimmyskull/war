import pytest

from war.optimize import optimize_slots_config


def test_simple():
    config = optimize_slots_config(1, (1, 1), (1, 1), (1, 1))
    assert config == {
        'tasks': 1,
        'njobs_on_validation': 1,
        'njobs_on_estimator': 1,
    }


def test_cannot_reach():
    config = optimize_slots_config(32, (1, 1), (1, 1), (1, 1))
    assert config == {
        'tasks': 1,
        'njobs_on_validation': 1,
        'njobs_on_estimator': 1,
    }


def test_single():
    config = optimize_slots_config(1, (1, 10), (1, 10), (1, 10))
    assert config == {
        'tasks': 1,
        'njobs_on_validation': 1,
        'njobs_on_estimator': 1,
    }


def test_only_tasks():
    config = optimize_slots_config(9, (1, 10), (1, 1), (1, 1))
    assert config == {
        'tasks': 9,
        'njobs_on_validation': 1,
        'njobs_on_estimator': 1,
    }


def test_only_validation():
    config = optimize_slots_config(9, (1, 1), (1, 9), (1, 1))
    assert config == {
        'tasks': 1,
        'njobs_on_validation': 9,
        'njobs_on_estimator': 1,
    }


def test_only_estimator():
    config = optimize_slots_config(9, (1, 1), (1, 1), (1, 10))
    assert config == {
        'tasks': 1,
        'njobs_on_validation': 1,
        'njobs_on_estimator': 9,
    }


def test_excess():
    config = optimize_slots_config(32, (1, 32), (1, 32), (1, 32))
    assert config == {
        'tasks': 2,
        'njobs_on_validation': 4,
        'njobs_on_estimator': 4,
    }


def test_excess_over_sequential():
    config = optimize_slots_config(32, (1, 1), (1, 32), (1, 32))
    assert config == {
        'tasks': 1,
        'njobs_on_validation': 6,
        'njobs_on_estimator': 5,
    }


def test_excess_over_sequential_small_cv():
    config = optimize_slots_config(32, (1, 1), (1, 3), (1, 32))
    # We're loosing two slots that we could fill up here.
    # This is a problem in the optimization aspect, but in today's job,
    # we will still maximize the slots usage in the next iteration, by
    # running another candidate. The only exception should be when
    # working with a single sequential strategy.  For this reason, I
    # should consider to fix this issue.
    assert config == {
        'tasks': 1,
        'njobs_on_validation': 1,
        'njobs_on_estimator': 30,
    }

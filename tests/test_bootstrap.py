"""Regression tests for the duplicate-resample statistical error."""
import numpy as np
import pytest
from saerag.eval.bootstrap import _stats, paired_bootstrap


def test_repeated_rows_have_weight():
    truth = np.array(["ANSWER", "ABSTAIN"], dtype=object)
    pred = np.array(["ANSWER", "ANSWER"], dtype=object)
    closed = np.array([True, False])
    rag = np.array([False, False])
    full = _stats(truth, pred, closed, rag, np.array([0, 1]))
    repeated = _stats(truth, pred, closed, rag, np.array([0, 0, 1]))
    assert full["accuracy"] == pytest.approx(0.5)
    assert repeated["accuracy"] == pytest.approx(2 / 3)


def test_identical_routers_have_zero_differences():
    ys = ["ANSWER", "RETRIEVE", "ABSTAIN", "ANSWER"]
    acts = ["ANSWER", "RETRIEVE", "ABSTAIN", "RETRIEVE"]
    comparisons = paired_bootstrap(ys, acts, acts,
                                   [True, False, False, True],
                                   [False, True, False, False], draws=100, seed=3)
    for metric in comparisons:
        assert metric["sae_minus_dense"] == pytest.approx(0)
        assert metric["ci_low"] == pytest.approx(0)
        assert metric["ci_high"] == pytest.approx(0)


def test_bad_input_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap(["ANSWER"], [], [], [], [])

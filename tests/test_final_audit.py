"""Tests for final audit selection and gating logic."""
import pandas as pd
import pytest
import runpy
from pathlib import Path

ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/finalize_project.py"))
aggregate, select_models, wilson = (ns[k] for k in ("aggregate", "select_models", "wilson"))


def test_wilson_interval_contains_observed_rate():
    lo, hi = wilson(16, 200)
    assert lo < .08 < hi
    assert lo < .10 < hi


def test_validate_selection_only():
    data = pd.DataFrame([
        {"model": name, "budget": "full", "eval": f, "macro_f1": score, "accuracy": 0.3, "seed": 0}
        for f in ("nq_val", "triviaqa_val")
        for name, score in (("sae16k_last@L17", .6), ("sae16k_last@L22", .7), ("resid_last@L22", .75))
    ] + [{"model": "sae16k_last@L17", "budget": "full", "eval": "nq_test",
           "macro_f1": .99, "accuracy": 0.99, "seed": 0}])
    selection, _ = select_models(data)
    assert selection["sae"] == "sae16k_last@L22"  # not L17 despite test advantage
    assert selection["dense"] == "resid_last@L22"


def test_weighted_aggregate_accuracy():
    d1 = pd.DataFrame([{"router": "X", "subset": "all", "n": 10, "accuracy": 1.0,
                        "retrieval_call_rate": .2, "abstention_recall": 0.0}])
    d2 = pd.DataFrame([{"router": "X", "subset": "all", "n": 30, "accuracy": 0,
                        "retrieval_call_rate": .6, "abstention_recall": 0.0}])
    out = aggregate({"a": d1, "b": d2}, "X", files=("a", "b"))
    assert out["n"] == 40
    assert out["accuracy"] == pytest.approx(.25)
    assert out["retrieval_call_rate"] == pytest.approx(.5)

"""Metrics on hand-made routers and labels, with every expected value worked out by hand."""

import math

import pytest

from saerag.eval import metrics as M

# Six questions. closed / rag correctness and labels follow the label rule.
CLOSED = {"a": True, "b": False, "c": False, "d": False, "e": True, "f": False}
RAG = {"a": True, "b": True, "c": False, "d": True, "e": False, "f": False}
LABELS = {"a": "ANSWER", "b": "RETRIEVE", "c": "ABSTAIN", "d": "RETRIEVE", "e": "ANSWER",
          "f": "ABSTAIN"}
ACTS = {"a": "ANSWER", "b": "RETRIEVE", "c": "ANSWER", "d": "ABSTAIN", "e": "RETRIEVE",
        "f": "ABSTAIN"}
# outcomes: a T, b T, c F, d None, e F, f None


def p(ans, ret, abs_):
    return {"ANSWER": ans, "RETRIEVE": ret, "ABSTAIN": abs_}


def test_outcomes_and_accuracy():
    out = M.outcomes(ACTS, CLOSED, RAG)
    assert out == {"a": True, "b": True, "c": False, "d": None, "e": False, "f": None}
    assert M.accuracy(out) == pytest.approx(2 / 6)
    assert M.selective_accuracy(out) == pytest.approx(2 / 4)
    assert M.coverage(out) == pytest.approx(4 / 6)
    assert M.hallucination_rate(out) == pytest.approx(2 / 4)


def test_nothing_attempted_is_nan_not_crash():
    out = {"x": None, "y": None}
    assert M.accuracy(out) == 0
    assert math.isnan(M.selective_accuracy(out)) and math.isnan(M.hallucination_rate(out))


def test_retrieval_calls():
    assert M.retrieval_call_rate(ACTS) == pytest.approx(2 / 6)
    nec = M.call_necessity(ACTS, LABELS)
    # predicted RETRIEVE {b, e}, needed {b, d}
    assert nec["precision"] == pytest.approx(0.5) and nec["recall"] == pytest.approx(0.5)
    assert nec["unnecessary_call_rate"] == pytest.approx(0.5)
    assert nec["missed_call_rate"] == pytest.approx(0.5)


def test_abstention_recall():
    should = {q: lab == "ABSTAIN" for q, lab in LABELS.items()}
    r = M.abstention_recall(ACTS, should)
    assert r["abstention_recall"] == pytest.approx(1 / 2)   # f yes, c no
    assert r["false_refusal_rate"] == pytest.approx(1 / 4)  # d refused


def test_threshold_respects_false_refusal_budget():
    # 10 answerable questions with p_abstain 0.0..0.9, 2 unanswerable at 0.85 and 0.95
    p_abs = {f"n{i}": i / 10 for i in range(10)} | {"u1": 0.85, "u2": 0.95}
    should = {q: q.startswith("u") for q in p_abs}
    t = M.tune_abstain_threshold(p_abs, should, 0.10)
    refused = {q for q, v in p_abs.items() if v >= t}
    assert refused == {"n9", "u1", "u2"}  # one of ten answerable refused = 10%
    assert M.tune_abstain_threshold(p_abs, should, 0.0) > 0.9  # no answerable refused


def test_route_with_threshold():
    probs = {"x": p(0.5, 0.2, 0.3), "y": p(0.1, 0.5, 0.4), "z": p(0.3, 0.3, 0.4)}
    assert M.route(probs) == {"x": "ANSWER", "y": "RETRIEVE", "z": "ABSTAIN"}
    assert M.route(probs, 0.35) == {"x": "ANSWER", "y": "ABSTAIN", "z": "ABSTAIN"}
    assert M.route(probs, 0.9) == {"x": "ANSWER", "y": "RETRIEVE", "z": "ANSWER"}


def test_risk_coverage_and_aurc():
    out = {"a": True, "b": False, "c": True, "d": False}
    conf = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}
    rc = M.risk_coverage(out, conf)
    assert rc["coverage"] == [0.25, 0.5, 0.75, 1.0]
    assert rc["risk"] == pytest.approx([0, 1 / 2, 1 / 3, 1 / 2])
    assert rc["aurc"] == pytest.approx((0 + 1 / 2 + 1 / 3 + 1 / 2) / 4)
    # A perfect ranking (right answers first) beats a reversed one.
    good = M.risk_coverage(out, {"a": 4, "c": 3, "b": 2, "d": 1})["aurc"]
    bad = M.risk_coverage(out, {"a": 1, "c": 2, "b": 3, "d": 4})["aurc"]
    assert good < rc["aurc"] < bad


def test_risk_coverage_ties_give_one_point():
    out = {"a": True, "b": False, "c": True, "d": True}
    rc = M.risk_coverage(out, {q: 1.0 for q in out})
    assert rc["coverage"] == [1.0] and rc["risk"] == [0.25] and rc["aurc"] == pytest.approx(0.25)


def test_macro_f1_and_confusion():
    assert M.macro_f1(LABELS, LABELS) == pytest.approx(1.0)
    # per class: ANSWER P=1/2 R=1/2; RETRIEVE P=1/2 R=1/2; ABSTAIN P=1/2 R=1/2
    assert M.macro_f1(ACTS, LABELS) == pytest.approx(0.5)
    # A class absent from both labels and predictions does not drag the average down.
    two = {"x": "ANSWER", "y": "ABSTAIN"}
    assert M.macro_f1(two, two) == pytest.approx(1.0)
    cm = M.confusion(ACTS, LABELS)
    assert cm["RETRIEVE"] == {"ANSWER": 0, "RETRIEVE": 1, "ABSTAIN": 1}
    assert sum(sum(r.values()) for r in cm.values()) == 6


def test_ece():
    labels = {"x": "ANSWER", "y": "ANSWER"}
    # Perfectly calibrated hard predictions.
    assert M.expected_calibration_error({"x": p(1, 0, 0), "y": p(1, 0, 0)}, labels) == 0
    # Confidence 0.8 in one bin, accuracy 0.5 -> |0.5 - 0.8| = 0.3
    probs = {"x": p(0.8, 0.1, 0.1), "y": p(0.1, 0.8, 0.1)}
    assert M.expected_calibration_error(probs, labels) == pytest.approx(0.3)


def test_latency():
    acts = {"a": "ANSWER", "b": "RETRIEVE", "c": "ABSTAIN"}
    lat = M.route_latency(acts, {"a": 5, "b": 5, "c": 5}, {"a": 40, "b": 50, "c": 60},
                          {"a": 100, "b": 110, "c": 120}, {"a": 200, "b": 300, "c": 400})
    assert lat["a"] == {"controller": 5, "retrieval": 0.0, "generation": 100}
    assert lat["b"] == {"controller": 5, "retrieval": 50, "generation": 300}
    assert lat["c"] == {"controller": 5, "retrieval": 0.0, "generation": 0.0}
    pct = M.latency_percentiles(lat)
    assert pct["total"]["p50"] == pytest.approx(105)    # totals 105, 355, 5
    assert pct["retrieval"]["p95"] == pytest.approx(45)  # 0, 50, 0 -> linear interpolation


def test_score_oracle_is_upper_bound():
    oracle = {q: p(*(float(lab == a) for a in ("ANSWER", "RETRIEVE", "ABSTAIN")))
              for q, lab in LABELS.items()}
    should = {q: lab == "ABSTAIN" for q, lab in LABELS.items()}
    s = M.score(oracle, LABELS, CLOSED, RAG, should)
    assert s["macro_f1"] == 1 and s["hallucination_rate"] == 0
    assert s["accuracy"] == pytest.approx(4 / 6)  # every answerable question answered right
    assert s["abstention_recall"] == 1 and s["false_refusal_rate"] == 0
    assert s["retrieve_precision"] == 1 and s["ece"] == 0

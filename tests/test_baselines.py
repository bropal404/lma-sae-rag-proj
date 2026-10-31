import pandas as pd

from saerag.baselines import routers as R
from saerag.eval.handcheck import score_sheet, short_alias


def test_hard_routers():
    assert R.always_retrieve(["a"]) == {"a": {"ANSWER": 0.0, "RETRIEVE": 1.0, "ABSTAIN": 0.0}}
    assert R.never_retrieve(["a"])["a"]["ANSWER"] == 1.0
    assert R.oracle({"a": "ABSTAIN"})["a"]["ABSTAIN"] == 1.0


def test_parse_verbalised():
    assert R.parse_verbalised("ANSWER") == ("ANSWER", True)
    assert R.parse_verbalised("  retrieve.\nBecause...") == ("RETRIEVE", True)
    assert R.parse_verbalised("**Abstain**") == ("ABSTAIN", True)
    assert R.parse_verbalised("I think I should look it up") == ("RETRIEVE", False)
    probs, fail = R.verbalised({"a": "ANSWER", "b": "hmm"})
    assert probs["a"]["ANSWER"] == 1.0 and fail == 0.5


def test_text_classifier_learns_obvious_signal():
    qs = ([f"who wrote book {i}" for i in range(20)]
          + [f"what is the population of town {i}" for i in range(20)])
    ys = ["ANSWER"] * 20 + ["RETRIEVE"] * 20
    clf = R.tune_text_classifier(qs, ys, qs[::4], ys[::4])
    probs = R.predict_probs(clf, ["x", "y"], ["who wrote book 99", "population of town 99"])
    assert probs["x"]["ANSWER"] > 0.5 and probs["y"]["RETRIEVE"] > 0.5
    assert probs["x"]["ABSTAIN"] == 0.0  # class unseen in training
    assert set(R.top_terms(clf, 3)) == {"ANSWER", "RETRIEVE"}


def test_short_alias():
    assert short_alias(["AR", "Arkansas"]) and short_alias(["P.L."])
    assert not short_alias(["Paris"])


def _row(qid, label, closed, bm25, gold, v_closed="", v_bm25="", v_gold=""):
    return {"qid": qid, "unanswerable": "False", "label": label,
            "closed_output": "x", "closed_auto": closed, "verdict_closed": v_closed,
            "bm25_output": "x", "bm25_auto": bm25, "verdict_bm25": v_bm25,
            "gold_output": "x" if gold != "" else "", "gold_auto": gold, "verdict_gold": v_gold}


def test_handcheck_score():
    df = pd.DataFrame([
        _row("q1", "ANSWER", "True", "True", "True"),                # agrees
        _row("q2", "ANSWER", "True", "False", "", v_closed="n"),      # false match -> ABSTAIN
        _row("q3", "RETRIEVE", "False", "True", "True", v_bm25="y"),  # verdict agrees
        _row("q4", "ABSTAIN", "False", "False", "True", v_bm25="y"),  # missed match -> RETRIEVE
    ])
    res = score_sheet(df)
    assert res["label_errors"] == 2 and res["label_error_rate"] == 0.5
    assert {f["qid"]: f["hand"] for f in res["flagged"]} == {"q2": "ABSTAIN", "q4": "RETRIEVE"}
    assert res["match_errors"] == 2 and res["match_checks"] == 11
    assert not res["gate1_pass"]

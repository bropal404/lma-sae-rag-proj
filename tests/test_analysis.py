"""Person 2 analysis on a four-question toy split."""

import copy

import pandas as pd
import pytest

from saerag.analysis.failures import failure_type, near_miss_table, popqa_table, probe_tables
from saerag.analysis.frame import add_probe, load_frame
from saerag.analysis.tables import gate0_table, label_table, rule_change_table
from saerag.config import load_config
from saerag.labels import make_label
from saerag.schema import Generation, Passage, Question, Retrieval, write_jsonl

# closed, bm25, gold correctness per question, and whether the BM25 passage holds the answer
CASES = {
    "q:known": (True, True, True, True),
    "q:hurt": (True, False, True, False),
    "q:reader": (False, False, True, True),
    "q:miss": (False, False, True, False),
    "q:beyond": (False, False, False, False),
    "q:nogold": (False, False, None, False),
}


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    for key in ("questions", "retrieval", "generations", "labels", "results"):
        c["paths"][key] = str(tmp_path / key)
    file = "popqa_test"
    qs, ret, gold, gens, labels = [], [], [], {"closed": [], "bm25": [], "gold": []}, {}
    for i, (qid, (cl, rag, gd, has)) in enumerate(CASES.items()):
        qs.append(Question(qid, "popqa", "test", f"question {qid}?", ["Paris"], meta={"s_pop": 10 * i}))
        text = "the city of Paris" if has else "nothing useful"
        ret.append(Retrieval(qid, "bm25", False, [Passage("p1", "T", text, 1.0)]))
        if gd is not None:
            gold.append(Retrieval(qid, "gold", False, [Passage("g1", "G", "Paris", 1.0)]))
        for cond, ok in (("closed", cl), ("bm25", rag), ("gold", gd)):
            if ok is not None:
                gens[cond].append(Generation(qid, cond, "Paris" if ok else "Lyon", ok))
        for v in c["labels"]["variants"]:
            labels.setdefault(v, []).append(make_label(qid, cl, rag, gd, variant=v))
    write_jsonl(tmp_path / "questions" / f"{file}.jsonl", qs)
    write_jsonl(tmp_path / "retrieval" / "bm25" / f"{file}.jsonl", ret)
    write_jsonl(tmp_path / "retrieval" / "gold" / f"{file}.jsonl", gold)
    for cond, g in gens.items():
        write_jsonl(tmp_path / "generations" / cond / f"{file}.jsonl", g)
    for v, ls in labels.items():
        write_jsonl(tmp_path / "labels" / v / f"{file}.jsonl", ls)
    return c


def test_frame_joins_every_stage(cfg):
    df = load_frame(cfg, "popqa_test")
    assert len(df) == len(CASES)
    assert df.loc["q:nogold", "ok_gold"] is pd.NA and not df.loc["q:nogold", "gold_available"]
    assert bool(df.loc["q:reader", "bm25_has_answer"]) and not bool(df.loc["q:miss", "bm25_has_answer"])
    assert df.loc["q:miss", "label_default"] == "ABSTAIN"
    assert df.loc["q:miss", "label_miss_as_retrieve"] == "RETRIEVE"


def test_failure_types(cfg):
    df = load_frame(cfg, "popqa_test")
    got = {r.qid: failure_type(r) for r in df.itertuples()}
    assert got == {"q:known": "known, RAG keeps it", "q:hurt": "RAG hurts",
                   "q:reader": "reader failure", "q:miss": "retriever miss",
                   "q:beyond": "beyond model", "q:nogold": "no gold passage"}


def test_rule_tables(cfg):
    df = load_frame(cfg, "popqa_test")
    ch = rule_change_table(df).iloc[0]
    # reader and miss are both retriever_miss under the default rule (gold works)
    assert ch["moved to RETRIEVE"] == 2 and ch["ABSTAIN beyond_model"] == 2
    lt = label_table(df, ["miss_as_retrieve", "default"], 0.15)
    row = lt[lt.rule == "miss_as_retrieve"].iloc[0]
    assert row["ANSWER"] == pytest.approx(2 / 6) and row["RETRIEVE"] == pytest.approx(2 / 6)
    assert gate0_table(df, ["default"], 0.15).empty      # no pilot or train files here


def test_near_miss_changes_labels(cfg):
    df = load_frame(cfg, "popqa_test")
    df.loc["q:beyond", ["out_closed", "f1_closed"]] = ["Paris France", 0.67]
    t = near_miss_table(df, "miss_as_retrieve").iloc[0]
    assert t["near misses closed"] == 1
    assert t["labels that would change"] == pytest.approx(1 / 6)


def test_popqa_bins_and_probe(cfg, tmp_path):
    import numpy as np

    df = load_frame(cfg, "popqa_test")
    df["ftype"] = [failure_type(r) for r in df.itertuples()]
    t, rho = popqa_table(df, "miss_as_retrieve", np.random.default_rng(0))
    assert t.n.sum() == len(CASES) and -1 <= rho <= 1

    probe = tmp_path / "routers" / "toy"
    write_jsonl(probe / "probs_popqa_test.jsonl",
                [{"qid": q, "p": {"ANSWER": 0.6, "RETRIEVE": 0.3, "ABSTAIN": 0.1}} for q in CASES])
    dfp = add_probe(df, probe, "popqa_test")
    assert (dfp.probe_pred == "ANSWER").all()
    tabs = probe_tables(dfp, "miss_as_retrieve")
    costly = tabs["probe_costly"].iloc[0]
    assert costly.iloc[0] == pytest.approx(4 / 6)       # answered while closed-book was wrong
    conf = tabs["probe_confusion"].set_index("label")
    assert conf.loc["ANSWER", "ANSWER"] == 2

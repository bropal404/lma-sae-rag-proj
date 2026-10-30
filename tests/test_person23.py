"""Persons 2 and 3 on synthetic data: no model or GPU, but the real entry points."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from saerag import config as config_module
from saerag.features import store
from saerag.generation import make_labels, report
from saerag.generation.run import build_messages
from saerag.labels import VARIANTS
from saerag.schema import Generation, Label, Passage, Question, Retrieval, read_jsonl, write_jsonl

torch = pytest.importorskip("torch")
from saerag.features.extract import masked_mean_codes, question_mask
from saerag.features.sae import JumpReLUSAE
from saerag.probes import analyze, train

CLASSES = ("ANSWER", "RETRIEVE", "ABSTAIN")


@pytest.fixture
def cfg(tmp_path):
    base = config_module.load_config()
    for key in ("questions", "retrieval", "generations", "labels", "features"):
        base["paths"][key] = str(tmp_path / key)
    base["paths"]["results"] = str(tmp_path / "runs")
    base["model"]["tag"] = "test"
    (tmp_path / "cfg.yaml").write_text(json.dumps(base))     # JSON is valid YAML
    return base, str(tmp_path / "cfg.yaml")


def questions(n, prefix="q", **kw):
    return [Question(f"{prefix}{i}", "nq", "train", f"question {i}?", ["yes"], hole=i % 4 == 0, **kw)
            for i in range(n)]


def retrieval(qs, retriever="bm25"):
    return [Retrieval(q.qid, retriever, q.hole, [Passage("1", "T", "text", 1.0)]) for q in qs]


def write_gens(c, condition, file, flags):
    write_jsonl(config_module.path(c, "generations", condition, f"{file}.jsonl"),
                [Generation(q, condition, "x", ok) for q, ok in flags])


def test_rag_refuses_retrieval_from_another_question_build(cfg):
    c, _ = cfg
    qs = questions(4)
    write_jsonl(config_module.path(c, "questions", "f.jsonl"), qs)
    write_jsonl(config_module.path(c, "retrieval", "bm25", "f.jsonl"), retrieval(qs[:3]))
    with pytest.raises(ValueError, match="different builds"):
        build_messages(c, "bm25", "f")

    wrong = retrieval(qs)
    wrong[1].hole = not wrong[1].hole
    write_jsonl(config_module.path(c, "retrieval", "bm25", "f.jsonl"), wrong)
    with pytest.raises(ValueError, match="hole flags"):
        build_messages(c, "bm25", "f")


def test_gold_retrieval_may_be_sparse(cfg):
    c, _ = cfg
    qs = questions(4)
    write_jsonl(config_module.path(c, "questions", "f.jsonl"), qs)
    write_jsonl(config_module.path(c, "retrieval", "gold", "f.jsonl"), retrieval(qs[:2], "gold"))
    kept, msgs = build_messages(c, "gold", "f")
    assert [q.qid for q in kept] == ["q0", "q1"] and len(msgs) == 2


def test_labels_and_gate_report(cfg, monkeypatch, capsys):
    c, cpath = cfg
    qs = questions(6) + [Question("u", "selfaware", "test", "unknowable?", [], unanswerable=True)]
    write_jsonl(config_module.path(c, "questions", "f.jsonl"), qs)
    ids = [q.qid for q in qs]
    write_gens(c, "closed", "f", [(q, q in ("q0", "q1")) for q in ids])
    write_gens(c, "bm25", "f", [(q, q in ("q2", "q3")) for q in ids])
    write_gens(c, "gold", "f", [("q4", True)])         # q5 has no gold passage
    monkeypatch.setattr(sys, "argv", ["x", "--config", cpath, "--files", "f"])
    make_labels.main()

    labels = {x.qid: x for x in read_jsonl(config_module.path(c, "labels", "default", "f.jsonl"), Label)}
    assert [labels[q].label for q in ids] == ["ANSWER"] * 2 + ["RETRIEVE"] * 2 + ["ABSTAIN"] * 3
    assert labels["q4"].abstain_reason == "retriever_miss"
    assert labels["q5"].gold_correct is None and labels["q5"].abstain_reason == "beyond_model"
    assert labels["u"].abstain_reason == "unanswerable"
    other = list(read_jsonl(config_module.path(c, "labels", "miss_as_retrieve", "f.jsonl"), Label))
    assert next(x for x in other if x.qid == "q4").label == "RETRIEVE"

    text, passed = report.report(c, "f")
    assert passed and "| closed | 7 |" in text and "PASS" in text
    assert set(VARIANTS) == {"default", "miss_as_retrieve"}


def test_gate_fails_when_a_class_is_rare(cfg):
    c, _ = cfg
    qs = questions(10)
    write_jsonl(config_module.path(c, "questions", "f.jsonl"), qs)
    write_gens(c, "closed", "f", [(q.qid, True) for q in qs])
    for v in VARIANTS:
        write_jsonl(config_module.path(c, "labels", v, "f.jsonl"),
                    [Label(q.qid, v, "ANSWER", True, True, None, None, q.hole) for q in qs])
    assert not report.report(c, "f")[1]


def test_sae_encode_matches_jump_relu():
    sae = JumpReLUSAE({"w_enc": torch.eye(2), "b_enc": torch.zeros(2), "b_dec": torch.zeros(2),
                       "w_dec": torch.eye(2), "threshold": torch.tensor([1.0, 1.0])}, {})
    out = sae.encode(torch.tensor([[2.0, 0.5]]))
    assert out.tolist() == [[2.0, 0.0]]
    assert sae.decode(out).tolist() == [[2.0, 0.0]]


def test_masked_mean_codes_ignores_other_tokens():
    sae = JumpReLUSAE({"w_enc": torch.eye(1), "b_enc": torch.zeros(1), "b_dec": torch.zeros(1),
                       "threshold": torch.zeros(1)}, {})
    h = torch.tensor([[[9.0], [1.0], [3.0]], [[5.0], [9.0], [9.0]]])
    mask = torch.tensor([[False, True, True], [True, False, False]])
    assert masked_mean_codes(sae, h, mask).tolist() == [[2.0], [5.0]]


def test_question_mask_handles_padding_and_trimmed_whitespace():
    prompt = "Question: who? <end>"
    # token 0 is left padding, 1 = "Question:", 2 = "who?", 3 = "<end>"
    offsets = torch.tensor([[[0, 0], [0, 9], [10, 14], [15, 20]]])
    assert question_mask([prompt], ["who? "], offsets).tolist() == [[False, False, True, False]]
    with pytest.raises(AssertionError):
        question_mask([prompt], ["absent"], offsets)


def write_features(c, file, y, seed):
    """Sparse features: the first 3 fire by class, the rest are noise."""
    rng = np.random.default_rng(seed)
    X = sp.random(len(y), 60, density=0.05, format="lil", random_state=seed, dtype=np.float32)
    for i, k in enumerate(y):
        if rng.random() < 0.95:
            X[i, k] = 5.0
    store.save_qids(c, file, [f"{file}{i}" for i in range(len(y))])
    store.save(c, file, 9, "sae16k_last", X.tocsr())
    store.save(c, file, 9, "resid_last", np.asarray(X.todense(), dtype=np.float16))


def write_labelled(c, file, n, seed):
    y = np.arange(n) % 3
    write_features(c, file, y, seed)
    qs = [Question(f"{file}{i}", "nq", "train", f"topic{y[i]} question {i}", ["a"]) for i in range(n)]
    write_jsonl(config_module.path(c, "questions", f"{file}.jsonl"), qs)
    write_jsonl(config_module.path(c, "labels", "default", f"{file}.jsonl"),
                [Label(f"{file}{i}", "default", CLASSES[y[i]], False, False, None, None, False)
                 for i in range(n)])


def test_probe_train_and_analysis_recover_planted_features(cfg, monkeypatch, capsys):
    c, cpath = cfg
    write_labelled(c, "tr", 150, 1)
    write_labelled(c, "te", 60, 2)
    monkeypatch.setattr(sys, "argv", [
        "x", "--config", cpath, "--train", "tr", "--eval", "te", "--labels", "default",
        "--layers", "9", "--kinds", "resid_last", "sae16k_last", "--budgets", "30", "--seeds", "0", "1"])
    train.main()
    run = config_module.path(c, "results", "probes", "test", "default", "tr__te__l2")
    df = pd.read_csv(run / "summary.csv")
    assert len(df[df.budget == "full"]) == 4                 # one row per model, seeds not repeated
    assert len(df[df.budget == "30"]) == 8
    full = df[df.budget == "full"].set_index("model")
    assert full.loc["sae16k_last@L9", "macro_f1"] > 0.85 > full.loc["majority", "macro_f1"]
    probs = config_module.path(c, "results", "routers", "test__default__sae16k_last@L9__l2", "probs_te.jsonl")
    rows = list(read_jsonl(probs))
    assert len(rows) == 60 and abs(sum(rows[0]["p"].values()) - 1) < 1e-6

    monkeypatch.setattr(sys, "argv", [
        "x", "--config", cpath, "--train", "tr", "--eval", "te", "--labels", "default",
        "--layer", "9", "--kind", "sae16k_last", "--examples", "5"])
    analyze.main()
    out = config_module.path(c, "results", "probes", "test", "default", "analysis__sae16k_last@L9")
    top = pd.read_csv(out / "top_features.csv")
    assert set(top[top["rank"] == 1].feature) == {0, 1, 2}
    drop = pd.read_csv(out / "drop_top_k.csv").set_index("k")
    assert drop.loc[0, "f1_zeroed"] > 0.85
    assert drop.loc[0, "f1_zeroed"] == pytest.approx(drop.loc[0, "f1_retrained"])
    assert drop.loc[1, "f1_retrained"] < 0.6                 # the planted signal is gone
    sheet = pd.read_csv(out / "hand_check.csv")
    assert len(sheet) == 15 and sheet.verdict.isna().all()


def test_store_rejects_rows_that_do_not_match_qids(cfg):
    c, _ = cfg
    store.save_qids(c, "f", ["a", "b"])
    store.save(c, "f", 9, "resid_last", np.zeros((3, 2), dtype=np.float16))
    with pytest.raises(AssertionError, match="3 rows, 2 qids"):
        store.load(c, "f", 9, "resid_last")

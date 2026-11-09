"""One row per question, with every stage joined on qid.

Columns: qid, file, dataset, split, question, answers, hole, unanswerable, s_pop, level,
qtype; for each condition c in closed/bm25/gold: out_c, ok_c (bool, NaN when the condition
was not run for the question), f1_c; bm25_has_answer, gold_available; for each label variant v:
label_v, reason_v. Probe columns are added by add_probe().
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..answers import contains_answer, f1
from ..config import path
from ..schema import ACTIONS, Generation, Label, Question, Retrieval, read_jsonl

CONDITIONS = ("closed", "bm25", "gold")
FILES = ("pilot", "nq_train", "nq_val", "nq_test", "triviaqa_train", "triviaqa_val",
         "triviaqa_test", "popqa_test", "hotpotqa_test", "selfaware_test")
TEST_FILES = ("nq_test", "triviaqa_test", "popqa_test", "hotpotqa_test", "selfaware_test")


def available_files(cfg: dict, files=FILES) -> list[str]:
    return [f for f in files if path(cfg, "questions", f"{f}.jsonl").exists()]


def load_frame(cfg: dict, file: str) -> pd.DataFrame:
    qs = list(read_jsonl(path(cfg, "questions", f"{file}.jsonl"), Question))
    df = pd.DataFrame({
        "qid": [q.qid for q in qs], "file": file, "dataset": [q.dataset for q in qs],
        "split": [q.split for q in qs], "question": [q.question for q in qs],
        "answers": [q.answers for q in qs], "hole": [q.hole for q in qs],
        "unanswerable": [q.unanswerable for q in qs],
        "s_pop": [q.meta.get("s_pop") for q in qs],
        "level": [q.meta.get("level") for q in qs],
        "qtype": [q.meta.get("question_type") for q in qs],
    }).set_index("qid", drop=False)
    answers = df["answers"].to_dict()

    for c in CONDITIONS:
        p = path(cfg, "generations", c, f"{file}.jsonl")
        gens = {g.qid: g for g in read_jsonl(p, Generation)} if p.exists() else {}
        df[f"out_{c}"] = [gens[q].output if q in gens else None for q in df.index]
        df[f"ok_{c}"] = pd.array([gens[q].correct if q in gens else pd.NA for q in df.index],
                                 dtype="boolean")
        df[f"f1_{c}"] = [f1(gens[q].output, answers[q]) if q in gens else np.nan for q in df.index]

    p = path(cfg, "retrieval", "bm25", f"{file}.jsonl")
    if p.exists():
        has, titles = {}, {}
        k = cfg["retrieval"]["top_k"]
        for r in read_jsonl(p, Retrieval):
            has[r.qid] = bool(answers.get(r.qid)) and any(
                contains_answer(f"{x.title} {x.text}", answers[r.qid]) for x in r.passages[:k])
            titles[r.qid] = " | ".join(x.title for x in r.passages[:k])
        df["bm25_has_answer"] = pd.array([has.get(q, pd.NA) for q in df.index], dtype="boolean")
        df["bm25_titles"] = [titles.get(q) for q in df.index]
    p = path(cfg, "retrieval", "gold", f"{file}.jsonl")
    gold = {r["qid"] for r in read_jsonl(p)} if p.exists() else set()
    df["gold_available"] = df.index.isin(gold)

    for v in cfg["labels"]["variants"]:
        p = path(cfg, "labels", v, f"{file}.jsonl")
        labels = {x.qid: x for x in read_jsonl(p, Label)} if p.exists() else {}
        df[f"label_{v}"] = [labels[q].label if q in labels else None for q in df.index]
        df[f"reason_{v}"] = [labels[q].abstain_reason if q in labels else None for q in df.index]
    return df


def load_all(cfg: dict, files) -> pd.DataFrame:
    return pd.concat([load_frame(cfg, f) for f in files])


def add_probe(df: pd.DataFrame, probe_dir: Path, file: str) -> pd.DataFrame:
    """Adds probe_pred / probe_conf from runs/routers/{probe}/probs_{file}.jsonl."""
    p = Path(probe_dir) / f"probs_{file}.jsonl"
    if not p.exists():
        return df
    pred, conf = {}, {}
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            best = max(r["p"], key=r["p"].get)
            pred[r["qid"]], conf[r["qid"]] = best, r["p"][best]
    df = df.copy()
    df["probe_pred"] = df["qid"].map(pred)
    df["probe_conf"] = df["qid"].map(conf)
    return df


def shares(labels: pd.Series) -> dict[str, float]:
    labels = labels.dropna()
    n = len(labels)
    return {a: (labels == a).sum() / n if n else np.nan for a in ACTIONS}

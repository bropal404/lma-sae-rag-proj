"""Baselines B1-B4 and B8. Owner: Person 4.

Every router, baseline or probe, emits the same file so one metrics script scores all:
    runs/routers/{router_id}/probs_{file}.jsonl   {"qid", "p": {"ANSWER":..,"RETRIEVE":..,"ABSTAIN":..}}
Hard routers (B1-B4) put probability 1 on their action.

B1 always retrieve   B2 never retrieve   B3 oracle (from labels)
B4 verbalised router (prompts.verbalised_router_messages; parse the first word)
B8 question-text classifier (Adaptive-RAG style; e.g. TF-IDF + LR, then a small encoder)
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from ..schema import ACTIONS, Action

_WORD = re.compile(r"[A-Za-z]+")


def one_hot(action: Action) -> dict[str, float]:
    return {a: float(a == action) for a in ACTIONS}


def always_retrieve(qids: list[str]) -> dict[str, dict[str, float]]:
    return {q: one_hot("RETRIEVE") for q in qids}


def never_retrieve(qids: list[str]) -> dict[str, dict[str, float]]:
    return {q: one_hot("ANSWER") for q in qids}


def oracle(labels: dict[str, Action]) -> dict[str, dict[str, float]]:
    return {q: one_hot(a) for q, a in labels.items()}


# B4 ----------------------------------------------------------------------------------

def parse_verbalised(output: str, fallback: Action = "RETRIEVE") -> tuple[Action, bool]:
    """First of ANSWER/RETRIEVE/ABSTAIN in the reply, case-insensitive.

    Returns (action, parsed). Unparseable replies fall back to RETRIEVE, the safe default
    of an always-retrieve system; report the parse-failure rate next to B4's numbers.
    """
    for w in _WORD.findall(output):
        if w.upper() in ACTIONS:
            return w.upper(), True  # type: ignore[return-value]
    return fallback, False


def verbalised(outputs: dict[str, str]) -> tuple[dict[str, dict[str, float]], float]:
    """B4 probabilities from raw replies, plus the parse-failure rate."""
    parsed = {q: parse_verbalised(o) for q, o in outputs.items()}
    fails = sum(not ok for _, ok in parsed.values())
    return {q: one_hot(a) for q, (a, _) in parsed.items()}, fails / max(len(parsed), 1)


# B8 ----------------------------------------------------------------------------------

def train_text_classifier(questions: list[str], labels: list[Action], seed: int = 13,
                          C: float = 1.0) -> Pipeline:
    """TF-IDF (word 1-2 grams) + class-weighted multinomial logistic regression."""
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    return clf.fit(questions, labels)


def tune_text_classifier(train_q: list[str], train_y: list[Action], val_q: list[str],
                         val_y: list[Action], seed: int = 13,
                         grid: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)) -> Pipeline:
    """Pick C by val macro-F1, then return the model trained on train only."""
    from ..eval.metrics import macro_f1

    best, best_f1 = None, -1.0
    for C in grid:
        clf = train_text_classifier(train_q, train_y, seed, C)
        pred = clf.predict(val_q)
        f1 = macro_f1(dict(enumerate(pred)), dict(enumerate(val_y)))
        if f1 > best_f1:
            best, best_f1 = clf, f1
    return best


def predict_probs(clf: Pipeline, qids: list[str], questions: list[str]) -> dict[str, dict[str, float]]:
    """Class probabilities in the shared format; classes missing from training get 0."""
    proba = clf.predict_proba(questions)
    cols = list(clf.classes_)
    out = {}
    for q, row in zip(qids, proba, strict=True):
        out[q] = {a: float(row[cols.index(a)]) if a in cols else 0.0 for a in ACTIONS}
    return out


def top_terms(clf: Pipeline, k: int = 15) -> dict[str, list[str]]:
    """Highest-weighted n-grams per class, a sanity check that B8 is not keying on noise."""
    vec, lr = clf.steps[0][1], clf.steps[-1][1]
    vocab = np.array(vec.get_feature_names_out())
    if len(lr.classes_) == 2:  # binary LR stores one row for the second class
        w = lr.coef_[0]
        return {lr.classes_[1]: vocab[np.argsort(-w)[:k]].tolist(),
                lr.classes_[0]: vocab[np.argsort(w)[:k]].tolist()}
    return {c: vocab[np.argsort(-lr.coef_[i])[:k]].tolist() for i, c in enumerate(lr.classes_)}

"""Baselines B1-B4 and B8. Owner: Person 4.

Every router, baseline or probe, emits the same file so one metrics script scores all:
    runs/routers/{router_id}/probs_{file}.jsonl   {"qid", "p": {"ANSWER":..,"RETRIEVE":..,"ABSTAIN":..}}
Hard routers (B1-B4) put probability 1 on their action.

B1 always retrieve   B2 never retrieve   B3 oracle (from labels)
B4 verbalised router (prompts.verbalised_router_messages; parse the first word)
B8 question-text classifier (Adaptive-RAG style; e.g. TF-IDF + LR, then a small encoder)
"""

from __future__ import annotations

from ..schema import ACTIONS, Action


def one_hot(action: Action) -> dict[str, float]:
    return {a: float(a == action) for a in ACTIONS}


def always_retrieve(qids: list[str]) -> dict[str, dict[str, float]]:
    return {q: one_hot("RETRIEVE") for q in qids}


def never_retrieve(qids: list[str]) -> dict[str, dict[str, float]]:
    return {q: one_hot("ANSWER") for q in qids}


def oracle(labels: dict[str, Action]) -> dict[str, dict[str, float]]:
    return {q: one_hot(a) for q, a in labels.items()}

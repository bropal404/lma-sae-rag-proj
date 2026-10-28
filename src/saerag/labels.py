"""The cost-ordered label rule (D7).

default:           ANSWER if closed-book correct, else RETRIEVE if BM25-RAG correct, else ABSTAIN.
miss_as_retrieve:  as default, but a retriever miss (gold passage would have worked) is RETRIEVE.
Self-Aware questions are ABSTAIN by construction.
"""

from __future__ import annotations

from .schema import Action, Label

VARIANTS = ("default", "miss_as_retrieve")


def label_rule(
    closed_correct: bool,
    rag_correct: bool,
    gold_correct: bool | None,
    unanswerable: bool = False,
    variant: str = "default",
) -> tuple[Action, str | None]:
    """Returns (action, abstain_reason)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown label variant {variant!r}")
    if unanswerable:
        return "ABSTAIN", "unanswerable"
    if closed_correct:
        return "ANSWER", None
    if rag_correct:
        return "RETRIEVE", None
    if gold_correct:
        return ("RETRIEVE", None) if variant == "miss_as_retrieve" else ("ABSTAIN", "retriever_miss")
    return "ABSTAIN", "beyond_model"


def make_label(
    qid: str,
    closed_correct: bool,
    rag_correct: bool,
    gold_correct: bool | None,
    *,
    unanswerable: bool = False,
    hole: bool = False,
    variant: str = "default",
) -> Label:
    action, reason = label_rule(closed_correct, rag_correct, gold_correct, unanswerable, variant)
    return Label(qid, variant, action, closed_correct, rag_correct, gold_correct, reason, hole)

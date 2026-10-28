"""Metrics from the proposal's metrics table. Owner: Person 4.

Routing outcome per question: the router's action plus the matching generation run.
ANSWER uses the closed-book generation, RETRIEVE the BM25 (or dense) generation, ABSTAIN
counts as not answered. So end-to-end accuracy needs no new generation.
"""

from __future__ import annotations

from ..schema import Action


def outcomes(actions: dict[str, Action], closed_correct: dict[str, bool],
             rag_correct: dict[str, bool]) -> dict[str, bool | None]:
    """qid -> True/False if attempted, None if abstained."""
    out: dict[str, bool | None] = {}
    for q, a in actions.items():
        out[q] = None if a == "ABSTAIN" else (closed_correct[q] if a == "ANSWER" else rag_correct[q])
    return out


# TODO(P4), each taking the outcomes/actions/labels dicts:
#   accuracy (abstain = wrong), selective accuracy, hallucination rate (wrong / attempted,
#   reported separately on hole questions), retrieval-call rate, call-necessity P/R,
#   abstention recall at <=10% false refusal (threshold tuned on val), risk-coverage + AURC,
#   macro-F1 and ECE against labels, latency p50/p95 split by stage.

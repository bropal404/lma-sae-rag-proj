"""Index holes and the gold-passage condition (D3, D10). Owner: Person 1.

A hole question searches the index as if its evidence were missing: fetch the top
`hole_fetch_k`, drop every passage that contains an answer alias or comes from a gold
article, and keep the next `top_k`. The index itself is never modified.
"""

from __future__ import annotations

from ..answers import contains_answer
from ..schema import Passage, Question


def is_evidence(p: Passage, q: Question) -> bool:
    return contains_answer(f"{p.title} {p.text}", q.answers) or p.title in q.gold_titles


def apply_hole(candidates: list[Passage], q: Question, k: int) -> tuple[list[Passage], list[str]]:
    """Returns (top-k passages without evidence, pids removed)."""
    kept, removed = [], []
    for p in candidates:
        (removed if is_evidence(p, q) else kept).append(p)
    # TODO(P1): log questions where fewer than k passages survive.
    return kept[:k], [p.pid for p in removed]


def gold_passages(candidates: list[Passage], q: Question) -> list[Passage] | None:
    """Diagnostic gold condition.

    HotpotQA: the supporting paragraphs from the distractor set.
    NQ/TriviaQA/PopQA: proxy = highest-ranked answer-bearing passage in the top `hole_fetch_k`.
    None when no such passage exists (gold_correct is then None in the label).
    """
    if q.dataset == "hotpotqa":
        return [Passage(f"hotpot-gold-{i}", g["title"], g["text"], 0.0)
                for i, g in enumerate(q.meta["gold_paragraphs"])]
    for p in candidates:
        if contains_answer(p.text, q.answers):
            return [p]
    return None

"""Query-time index-hole filtering and diagnostic gold passages (D3, D10)."""
from __future__ import annotations

from ..answers import contains_answer, normalize
from ..schema import Passage, Question


def canonical_title(title: str) -> str:
    return normalize(title.replace('_', ' ')).strip()


def is_evidence(p: Passage, q: Question) -> bool:
    """Evidence proxy: answer alias in title/text OR a Hotpot supporting article.

    This is lexical, NOT proof that all semantic evidence is removed from Wikipedia.
    """
    aliases_present = contains_answer(f'{p.title} {p.text}', q.answers)
    supporting_titles = {canonical_title(t) for t in q.gold_titles}
    supporting_article = bool(p.title) and canonical_title(p.title) in supporting_titles
    return aliases_present or supporting_article


def apply_hole(candidates: list[Passage], q: Question, k: int) -> tuple[list[Passage], list[str]]:
    """Keep up to k non-evidence hits and log all removed candidate passage IDs."""
    if k < 1:
        raise ValueError('k must be >= 1')
    kept: list[Passage] = []
    removed: list[str] = []
    for p in candidates:
        if is_evidence(p, q):
            removed.append(p.pid)
        elif len(kept) < k:
            kept.append(p)
    return kept, removed


def gold_passages(candidates: list[Passage], q: Question) -> list[Passage] | None:
    """Independent diagnostic, computed from UNFILTERED retrieved hits.

    HotpotQA uses the actual supporting paragraphs in q.meta. Other answerable
    datasets use the first top-100 passage containing an accepted answer alias.
    This is a *proxy* for gold, not externally verified relevance annotation.
    """
    if q.unanswerable:
        return None
    if q.dataset == 'hotpotqa':
        paragraphs = q.meta.get('gold_paragraphs', [])
        if not paragraphs:
            return None
        return [
            Passage(f'hotpot-gold-{q.qid}-{i}', p['title'], p['text'], 0.0)
            for i, p in enumerate(paragraphs)
        ]
    for p in candidates:
        if contains_answer(p.text, q.answers):
            return [p]
    return None

"""Prompt templates (D4, D5). Frozen once the pilot starts: labels depend on them.

No generation prompt tells the model it may abstain. Abstention is the router's job only,
so the closed-book and RAG runs measure what the model does when it must answer.
The router's activation is read from the closed-book prompt (D5), so an ANSWER decision
can reuse that forward pass.
"""

from __future__ import annotations

from .schema import Passage

CLOSED_BOOK = (
    "Answer the question. Reply with only the answer, in a few words.\n\n"
    "Question: {question}"
)

RAG = (
    "Answer the question using the passages below. Reply with only the answer, in a few words.\n\n"
    "{passages}\n\n"
    "Question: {question}"
)

# B4 verbalised router (Person 4). Not used for labels.
VERBALISED_ROUTER = (
    "Decide how to handle the question before answering it.\n"
    "ANSWER: you are confident you know the answer.\n"
    "RETRIEVE: you would need to look it up in Wikipedia.\n"
    "ABSTAIN: it cannot be answered reliably, even with a search.\n"
    "Reply with exactly one word: ANSWER, RETRIEVE or ABSTAIN.\n\n"
    "Question: {question}"
)


def format_passages(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{i}] {p.title}\n{p.text}" for i, p in enumerate(passages, 1))


def closed_book_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": CLOSED_BOOK.format(question=question)}]


def rag_messages(question: str, passages: list[Passage]) -> list[dict]:
    content = RAG.format(question=question, passages=format_passages(passages))
    return [{"role": "user", "content": content}]


def verbalised_router_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": VERBALISED_ROUTER.format(question=question)}]

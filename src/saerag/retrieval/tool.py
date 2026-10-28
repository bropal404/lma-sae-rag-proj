"""The search tool interface (Person 1). Generation and baselines only see this."""

from __future__ import annotations

from typing import Protocol

from ..schema import Passage


class SearchTool(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Passage]:
        """Top-k passages, best first."""
        ...

    def batch_search(self, queries: dict[str, str], k: int) -> dict[str, list[Passage]]:
        """qid -> top-k passages. Used for the offline labelling runs."""
        ...

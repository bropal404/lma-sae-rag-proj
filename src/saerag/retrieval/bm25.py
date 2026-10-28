"""BM25 over the 2018 DPR Wikipedia dump via Pyserini's prebuilt index (D2). Owner: Person 1.

The prebuilt index downloads once (~10 GB) into ~/.cache/pyserini; set PYSERINI_CACHE
to put it on a shared disk. Needs Java 21+ and `uv sync --extra retrieval`.
"""

from __future__ import annotations

import json

from ..schema import Passage


class BM25Tool:
    name = "bm25"

    def __init__(self, index: str = "wikipedia-dpr-100w", threads: int = 8):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher.from_prebuilt_index(index)
        self.threads = threads

    def _passage(self, docid: str, score: float) -> Passage:
        # TODO(P1): confirm the raw format. DPR passages store '"Title"\ntext' in "contents".
        contents = json.loads(self.searcher.doc(docid).raw())["contents"]
        title, _, text = contents.partition("\n")
        return Passage(docid, title.strip('"'), text, float(score))

    def search(self, query: str, k: int) -> list[Passage]:
        return [self._passage(h.docid, h.score) for h in self.searcher.search(query, k)]

    def batch_search(self, queries: dict[str, str], k: int) -> dict[str, list[Passage]]:
        qids = list(queries)
        hits = self.searcher.batch_search([queries[q] for q in qids], qids, k, self.threads)
        return {q: [self._passage(h.docid, h.score) for h in hits[q]] for q in qids}

"""Wikipedia DPR-100w BM25 retrieval using Pyserini's existing Lucene index.

No GPU and no rebuilding of the 21M-document corpus are needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schema import Passage


def parse_passage(docid: str, score: float, raw: str | dict[str, Any]) -> Passage:
    """Convert multiple known stored-document layouts into the shared Passage schema.

    DPR indices may store title/text separately or in a single 'contents' field.
    Preserve a title if it is explicit or clearly separated by a newline.
    """
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        obj = raw
    if isinstance(obj, dict):
        title = obj.get('title') or obj.get('wikipedia_title') or ''
        body = obj.get('text') or obj.get('contents') or obj.get('body') or ''
        if isinstance(body, dict):
            body = body.get('text') or body.get('contents') or ''
    elif isinstance(obj, str):
        title, body = '', obj
    else:
        raise ValueError(f'Unexpected Lucene document type for {docid}: {type(obj).__name__}')

    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError(f'Non-text contents for Lucene document {docid}')
    if not title and '\n' in body:
        # DPR passages commonly have '"Wikipedia Article Title"\nbody'.
        title, body = body.split('\n', 1)
    title = title.strip().strip('"')
    body = body.strip()
    if not body:
        raise ValueError(f'Empty Lucene passage {docid}; inspect doc.raw()/doc.contents()')
    return Passage(str(docid), title, body, float(score))


class BM25Tool:
    name = 'bm25'

    def __init__(self, index: str = 'wikipedia-dpr-100w', threads: int = 8):
        from pyserini.search.lucene import LuceneSearcher

        if threads < 1:
            raise ValueError('threads must be >= 1')
        # Support both canonical prebuilt names and local caches manually copied to HPC.
        self.searcher = (LuceneSearcher(index) if Path(index).is_dir()
                         else LuceneSearcher.from_prebuilt_index(index))
        self.threads = threads

    def _passage(self, docid: str, score: float) -> Passage:
        doc = self.searcher.doc(docid)
        if doc is None:
            raise RuntimeError(f'BM25 hit {docid} is not fetchable from the Lucene index')
        raw = doc.raw()
        if not raw:
            raw = doc.contents()
        if not raw:
            raise RuntimeError(f'BM25 hit {docid} lacks raw/contents')
        return parse_passage(docid, score, raw)

    def search(self, query: str, k: int) -> list[Passage]:
        if k < 1:
            raise ValueError('k must be >= 1')
        if not query.strip():
            return []
        return [self._passage(h.docid, h.score) for h in self.searcher.search(query, k)]

    def batch_search(self, queries: dict[str, str], k: int) -> dict[str, list[Passage]]:
        if k < 1:
            raise ValueError('k must be >= 1')
        nonempty = {qid: q for qid, q in queries.items() if q.strip()}
        results: dict[str, list[Passage]] = {qid: [] for qid in queries}
        if not nonempty:
            return results
        qids = list(nonempty)
        hits = self.searcher.batch_search(
            [nonempty[qid] for qid in qids], qids, k, self.threads
        )
        for qid in qids:
            results[qid] = [self._passage(hit.docid, hit.score) for hit in hits.get(qid, [])]
        return results

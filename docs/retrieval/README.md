# Person 1 SAE-Gated RAG — Full-corpus retrieval handoff

**Status:** Final BM25 retrieval outputs passed structural and lexical-hole checks.

- Wikipedia: 2018 DPR 100-word passages, 21,015,324 docs, 22 Hugging Face shards.
- Source dataset: `castorini/odqa-wiki-corpora` (`wiki-text-100w-karpukhin`).
- `questions/*.jsonl`: nine benchmark splits plus a 500-question pilot (pilot questions overlap training; do not double-count).
- `retrieval/bm25/*.jsonl`: one record per question with up to three BM25 passages.
- `retrieval/gold/*.jsonl`: **sparse**: missing rows mean gold evidence unavailable in the diagnostic, NOT an empty answer. HotpotQA includes supplied supporting paragraphs; other datasets have top-100 answer-containing passage **proxies** rather than independently validated gold.
- `retrieval/stats/*.json`: dataset-level counts, index path and measured processing time.
- `manifest.json`: exact revision and per-split validation counts.

**Methodological limitations:** `hole=true` filters lexical answer-bearing passages or known supporting article titles from the **top 100 query-time retrieved hits**. It does NOT physically remove evidence from Wikipedia and does not prove all semantically sufficient evidence is absent. Do not equate query-time holes with physical index deletions. The diagnostic gold proxy is not necessarily genuine gold evidence. Manually review sampled passages before final study claims. No model generation or SAE/probe performance is asserted by this archive.

**Reproducibility:** Python sources, tests and configuration are included when available. The 11GB+ Lucene index is NOT included in this lightweight handoff; reproduce it from the pinned HF revision recorded in `manifest.json` or use the existing ADA index. For continuation on the same machine, point BM25 to the full index path in the manifest. Results have been rewritten against that full index.

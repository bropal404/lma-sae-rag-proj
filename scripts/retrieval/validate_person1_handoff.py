#!/usr/bin/env python3
"""Final verification and reproducible Person 1 retrieval handoff for SAE-Gated RAG.

Run on ADA from the project root with the same .venv used by the retrieval runner.
This validates lexical holes only; no claim of exhaustive semantic evidence deletion.
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

from pyserini.index.lucene import LuceneIndexReader
from saerag.retrieval.holes import is_evidence
from saerag.schema import Question, Retrieval, read_jsonl

EXPECTED = {
    "nq_train": 4000, "nq_val": 500, "nq_test": 1000,
    "triviaqa_train": 4000, "triviaqa_val": 500, "triviaqa_test": 1000,
    "popqa_test": 1000, "hotpotqa_test": 1000,
    "selfaware_test": 500, "pilot": 500,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "saerag-person1")
    ap.add_argument("--no-archive", action="store_true", help="Run checks without creating archive")
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    data = root / "data"
    wiki = root / "wiki-full"
    repo = root / "lma-sae-rag-proj-main"
    state = json.loads((wiki / "state.json").read_text())
    assert (wiki / "FULL_INDEX_READY").is_file(), "Missing FULL_INDEX_READY"
    assert not (wiki / "IN_PROGRESS").exists(), "An indexing shard is in progress"
    assert len(state["completed"]) == 22, "Not all Wikipedia shards committed"
    assert state["docs"] == 21015324, f"Unexpected Wikipedia count: {state['docs']}"
    stats = LuceneIndexReader(str(wiki / "index")).stats()
    assert stats["documents"] == state["docs"], "Lucene count disagrees with checkpoint"
    print(f"INDEX PASS: {stats['documents']:,} documents, 22 shards", flush=True)

    audit = {"index": str(wiki / "index"), "corpus_repo": state["repo"],
             "corpus_revision": state["revision"], "passage_count": state["docs"],
             "datasets": {}, "total_unique_dataset_questions": 13500,
             "pilot_questions": 500}
    samples = []
    for name, count in EXPECTED.items():
        qpath = data / "questions" / f"{name}.jsonl"
        bpath = data / "retrieval" / "bm25" / f"{name}.jsonl"
        gpath = data / "retrieval" / "gold" / f"{name}.jsonl"
        spath = data / "retrieval" / "stats" / f"{name}.json"
        questions = list(read_jsonl(qpath, Question))
        rows = list(read_jsonl(bpath, Retrieval))
        gold = list(read_jsonl(gpath, Retrieval))
        s = json.loads(spath.read_text())
        qmap = {q.qid: q for q in questions}
        ids = [r.qid for r in rows]
        assert len(questions) == len(qmap) == count, f"Question count/IDs: {name}"
        assert len(rows) == len(set(ids)) == count, f"BM25 count/IDs: {name}"
        assert set(ids) == set(qmap), f"Question-to-BM25 mismatch: {name}"
        assert len(gold) == len({r.qid for r in gold}), f"Duplicate gold IDs: {name}"
        assert set(r.qid for r in gold) <= set(qmap), f"Unknown gold IDs: {name}"
        assert s["questions"] == count and s["gold_available"] == len(gold), f"Stats mismatch: {name}"
        assert s["index"] == str(wiki / "index"), f"Output not from full index: {name}"
        assert not s["partial"] and s["top_k"] == 3, f"Incorrect configuration: {name}"
        holes, lexical_leaks = 0, 0
        for r in rows:
            q = qmap[r.qid]
            assert r.retriever == "bm25", f"Bad retriever: {r.qid}"
            assert r.hole == q.hole, f"Hole flag mismatch: {r.qid}"
            assert len(r.passages) <= 3, f"Too many passages: {r.qid}"
            pids = [p.pid for p in r.passages]
            assert len(set(pids)) == len(pids), f"Duplicate passage IDs: {r.qid}"
            assert not (set(pids) & set(r.removed_pids)), f"Removed ID retained: {r.qid}"
            for p in r.passages:
                assert p.pid and p.text.strip(), f"Empty passage: {r.qid}"
            if r.hole:
                holes += 1
                lexical_leaks += sum(is_evidence(p, q) for p in r.passages)
                if len(samples) < 12 and r.removed_pids and r.passages:
                    samples.append({"dataset": name, "qid": q.qid,
                                    "question": q.question, "answers": q.answers,
                                    "removed_ids": r.removed_pids[:5],
                                    "retained": [{"title": p.title, "text": p.text[:500]}
                                                 for p in r.passages[:3]]})
        assert lexical_leaks == 0, f"Lexical evidence leaks in {name}: {lexical_leaks}"
        assert holes == s["hole_questions"], f"Hole stats mismatch: {name}"
        for r in gold:
            assert r.retriever == "gold" and r.passages, f"Empty gold: {r.qid}"
        if name == "selfaware_test":
            assert not gold and all(q.unanswerable for q in questions), "SelfAware should abstain"
        audit["datasets"][name] = {"questions": count, "holes": holes,
                                   "gold_proxies_or_supporting": len(gold),
                                   "lexical_hole_leaks": lexical_leaks,
                                   "nonhole_without_gold_proxy": s["no_gold_proxy_on_nonhole"],
                                   "retrieval_seconds": s["total_elapsed_seconds"]}
        print(f"PASS {name:15s} {count:5d} questions; holes={holes}; gold={len(gold)}", flush=True)

    # Write a readable handoff note explicitly distinguishing implementation from scientific limitations.
    note = """# Person 1 SAE-Gated RAG — Full-corpus retrieval handoff

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
"""
    manifest_file = root / "person1_full_manifest.json"
    note_file = root / "person1_full_README.md"
    sample_file = root / "person1_hole_review_samples.json"
    manifest_file.write_text(json.dumps(audit, indent=2) + "\n")
    note_file.write_text(note)
    sample_file.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n")
    print(f"STRUCTURAL / LEXICAL CHECKS PASS: {sum(EXPECTED.values())} records including 500 pilot", flush=True)
    print(f"Audit samples: {sample_file}", flush=True)
    if args.no_archive:
        return
    out = root / "person1-full-wikipedia-handoff.tar.gz"
    with tarfile.open(out, "w:gz") as tf:
        tf.add(data / "questions", arcname="questions")
        tf.add(data / "retrieval", arcname="retrieval")
        for filename, arcname in ((manifest_file, "manifest.json"),
                                  (note_file, "README.md"),
                                  (sample_file, "hole_review_samples.json")):
            tf.add(filename, arcname=arcname)
        for filename in ("pyproject.toml", "uv.lock", "config.yaml", "README.md"):
            p = repo / filename
            if p.is_file():
                tf.add(p, arcname=f"repository/{filename}")
        for directory in ("src/saerag", "tests", "configs", "config"):
            p = repo / directory
            if p.is_dir():
                tf.add(p, arcname=f"repository/{directory}")
    print(f"HANDOFF PACKAGE: {out} ({out.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    print("NEXT: Run unit tests and manually audit hole_review_samples.json; do not claim semantic holes are validated.")


if __name__ == "__main__":
    main()

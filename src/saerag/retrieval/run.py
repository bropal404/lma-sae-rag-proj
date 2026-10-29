"""Offline BM25 search, index holes, and gold-condition retrieval for Person 2.

Examples:
  uv run --extra retrieval python -m saerag.retrieval.run --files pilot
  uv run --extra retrieval python -m saerag.retrieval.run --files nq_train --mode batch
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from statistics import mean
from typing import Iterable, Iterator

from ..config import load_config, path
from ..schema import Question, Retrieval, read_jsonl
from .bm25 import BM25Tool
from .holes import apply_hole, gold_passages, is_evidence


def batches(xs: Iterable[Question], size: int) -> Iterator[list[Question]]:
    it = iter(xs)
    while chunk := list(islice(it, size)):
        yield chunk


def process_question(q: Question, candidates: list, top_k: int, elapsed_ms: float | None):
    """Pure processing logic (also unit-tested without a Lucene index)."""
    if q.hole:
        selected, removed = apply_hole(candidates, q, top_k)
    else:
        selected, removed = candidates[:top_k], []
    if q.hole and any(is_evidence(p, q) for p in selected):
        raise AssertionError(f'Index-hole evidence leaked into retained passages for {q.qid}')
    bm25 = Retrieval(q.qid, 'bm25', q.hole, selected, removed, elapsed_ms)
    gold = gold_passages(candidates, q)
    golden = Retrieval(q.qid, 'gold', q.hole, gold) if gold else None
    return bm25, golden


def process_file(cfg: dict, name: str, tool: BM25Tool, args: argparse.Namespace) -> None:
    questions_file = path(cfg, 'questions', f'{name}.jsonl')
    if not questions_file.exists():
        raise FileNotFoundError(f'{questions_file} is missing. Run the data builder first.')
    out_name = f'{name}_limit{args.limit}' if args.limit else name
    bm25_file = path(cfg, 'retrieval', 'bm25', f'{out_name}.jsonl')
    gold_file = path(cfg, 'retrieval', 'gold', f'{out_name}.jsonl')
    stats_file = path(cfg, 'retrieval', 'stats', f'{out_name}.json')
    if not args.overwrite and (bm25_file.exists() or gold_file.exists() or stats_file.exists()):
        raise FileExistsError(f'Outputs already exist for {name}; use --overwrite to rerun all')
    for p in (bm25_file, gold_file, stats_file):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Open fresh temporary files; only publish completed files. No fake empty
    # 'gold' record when the diagnostic gold passage is unavailable.
    bm25_tmp = bm25_file.with_suffix('.jsonl.tmp')
    gold_tmp = gold_file.with_suffix('.jsonl.tmp')
    top_k = int(cfg['retrieval']['top_k'])
    fetch_k = int(cfg['retrieval']['hole_fetch_k'])
    if top_k < 1 or fetch_k < top_k:
        raise ValueError('require 1 <= retrieval.top_k <= retrieval.hole_fetch_k')
    counts: Counter[str] = Counter()
    latencies: list[float] = []
    batch_elapsed: list[float] = []
    start = time.perf_counter()
    seen: set[str] = set()
    try:
        with bm25_tmp.open('w', encoding='utf-8') as bm, gold_tmp.open('w', encoding='utf-8') as go:
            rows = read_jsonl(questions_file, Question)
            if args.limit:
                rows = islice(rows, args.limit)
            for chunk in batches(rows, args.batch_size):
                queries = {}
                for q in chunk:
                    if q.qid in seen:
                        raise ValueError(f'duplicate qid {q.qid} in {name}')
                    seen.add(q.qid)
                    queries[q.qid] = q.question
                # Gold proxy is also computed for non-hole questions. Thus we
                # need top-100 candidates even when only 3 are retained.
                if args.mode == 'batch':
                    st = time.perf_counter()
                    found = tool.batch_search(queries, fetch_k)
                    batch_elapsed.append((time.perf_counter() - st) * 1000)
                    measured = {q.qid: None for q in chunk}
                else:
                    found, measured = {}, {}
                    for q in chunk:
                        st = time.perf_counter()
                        found[q.qid] = tool.search(q.question, fetch_k)
                        measured[q.qid] = (time.perf_counter() - st) * 1000
                        latencies.append(measured[q.qid])
                for q in chunk:
                    candidates = found[q.qid]
                    b, g = process_question(q, candidates, top_k, measured[q.qid])
                    bm.write(json.dumps(asdict(b), ensure_ascii=False) + '\n')
                    if g is not None:
                        go.write(json.dumps(asdict(g), ensure_ascii=False) + '\n')
                        counts['gold_available'] += 1
                    counts['questions'] += 1
                    counts['hole_questions'] += int(q.hole)
                    counts['nonhole_questions'] += int(not q.hole)
                    counts['hole_removed_passages'] += len(b.removed_pids)
                    counts['hole_fewer_than_top_k'] += int(q.hole and len(b.passages) < top_k)
                    counts['zero_candidate_questions'] += int(not candidates)
                    counts['nonhole_top3_with_alias'] += int(
                        not q.hole and any(is_evidence(p, q) for p in b.passages)
                    )
                    if q.hole:
                        counts['hole_remaining_passages'] += len(b.passages)
                    if not q.unanswerable and not q.hole and g is None:
                        counts['no_gold_proxy_on_nonhole'] += 1
                if args.verbose:
                    print(f'{name}: {counts["questions"]} questions processed', flush=True)
        os.replace(bm25_tmp, bm25_file)
        os.replace(gold_tmp, gold_file)
    except BaseException:
        bm25_tmp.unlink(missing_ok=True)
        gold_tmp.unlink(missing_ok=True)
        raise
    total_sec = time.perf_counter() - start
    n = counts['questions']
    stats = {
        'file': name, 'output_stem': out_name, 'partial': bool(args.limit),
        'index': args.index, 'mode': args.mode,
        'top_k': top_k, 'hole_fetch_k': fetch_k, 'questions': n,
        'hole_questions': counts['hole_questions'],
        'nonhole_questions': counts['nonhole_questions'],
        'hole_removed_passages': counts['hole_removed_passages'],
        'hole_remaining_passages': counts['hole_remaining_passages'],
        'hole_fewer_than_top_k': counts['hole_fewer_than_top_k'],
        'gold_available': counts['gold_available'],
        'no_gold_proxy_on_nonhole': counts['no_gold_proxy_on_nonhole'],
        'zero_candidate_questions': counts['zero_candidate_questions'],
        'nonhole_top3_with_alias': counts['nonhole_top3_with_alias'],
        'total_elapsed_seconds': round(total_sec, 3),
        'mean_query_latency_ms': round(mean(latencies), 3) if latencies else None,
        'mean_batch_wall_ms': round(mean(batch_elapsed), 3) if batch_elapsed else None,
        'latency_note': (
            'Individual end-to-end retrieval latency including Lucene doc fetch; '
            'top-100 diagnostic, not deployed top-3 latency.'
            if args.mode == 'sequential' else
            'Queries executed concurrently; per-question latency deliberately unset. '
            'Batch wall time is not per-question latency.'
        ),
        'index_hole_note': 'Query-time lexical filtering within top-100, NOT a physical index deletion.'
    }
    tmp_stats = stats_file.with_suffix('.json.tmp')
    with tmp_stats.open('w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    os.replace(tmp_stats, stats_file)
    print(json.dumps(stats, indent=2), flush=True)
    print(f'Written: {bm25_file}, {gold_file}, {stats_file}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config')
    ap.add_argument('--files', nargs='+', required=True, help='Question filenames without .jsonl')
    ap.add_argument('--index', help='Pyserini prebuilt index name (config default)')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--mode', choices=['sequential', 'batch'], default='sequential')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='For smoke tests only')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config)
    args.index = args.index or cfg['retrieval']['bm25_index']
    if args.batch_size < 1 or args.limit < 0:
        ap.error('--batch-size must be >= 1 and --limit must be >= 0')
    tool = BM25Tool(args.index, threads=args.threads)
    for name in args.files:
        process_file(cfg, name, tool, args)


if __name__ == '__main__':
    main()

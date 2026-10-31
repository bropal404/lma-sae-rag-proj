"""Create deterministic QA splits and pilot, matching the repository's D8 contract.

Note: this module was missing from the uploaded archive despite TODO.md claiming
it was already completed. It is included as a compatibility repair, not a change
to the project's intended study design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from dataclasses import replace
from typing import Any

from ..config import load_config, path
from ..schema import Question, write_jsonl


def assign_hole(qid: str, seed: int = 13, fraction: float = 0.30) -> bool:
    if not 0 <= fraction <= 1:
        raise ValueError('hole fraction must be between 0 and 1')
    digest = hashlib.sha1(f'{seed}:{qid}'.encode('utf-8')).digest()
    return (int.from_bytes(digest[:8], 'big') / (1 << 64)) < fraction


def _qid(dataset: str, row: dict, source: str) -> str:
    natural_id = row.get('id') or row.get('question_id')
    if natural_id is None:
        natural_id = hashlib.sha1(row['question'].encode('utf-8')).hexdigest()[:20]
    return f'{dataset}:{natural_id}'


def _aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                value = decoded
        except (ValueError, TypeError):
            pass
    if isinstance(value, dict):
        value = value.get('aliases') or value.get('value') or value.get('normalized_value') or []
    if isinstance(value, str):
        value = [value]
    return list(dict.fromkeys(str(x).strip() for x in value if x and str(x).strip()))


def _hotpot_metadata(row: dict) -> tuple[list[str], dict]:
    support = row.get('supporting_facts') or {}
    titles = list(dict.fromkeys(support.get('title') or []))
    ctx = row.get('context') or {}
    gold_paragraphs = []
    for title, sentences in zip(ctx.get('title') or [], ctx.get('sentences') or []):
        if title in titles:
            gold_paragraphs.append({'title': title, 'text': ' '.join(sentences)})
    return titles, {
        'gold_paragraphs': gold_paragraphs,
        'level': row.get('level'),
        'question_type': row.get('type'),
    }


def to_question(dataset: str, split: str, row: dict, seed: int, frac: float) -> Question:
    title_list, meta = [], {}
    if dataset == 'nq':
        ans = _aliases(row.get('answer'))
    elif dataset == 'triviaqa':
        ans = _aliases(row.get('answer'))
    elif dataset == 'popqa':
        ans = _aliases(row.get('possible_answers'))
        meta = {'s_pop': row.get('s_pop')}
    elif dataset == 'hotpotqa':
        ans = _aliases(row.get('answer'))
        title_list, meta = _hotpot_metadata(row)
    elif dataset == 'selfaware':
        ans = []
    else:
        raise ValueError(dataset)
    qid = _qid(dataset, row, split)
    return Question(
        qid=qid, dataset=dataset, split=split, question=row['question'],
        answers=ans, unanswerable=dataset == 'selfaware',
        hole=(dataset in ('nq', 'triviaqa', 'hotpotqa') and assign_hole(qid, seed, frac)),
        gold_titles=title_list, meta=meta
    )


def select_rows(ds: Any, n: int, seed: int, excluded: set[str] | None = None,
                dataset: str = '', source: str = '') -> list[dict]:
    """Deterministic without assuming a dataset has a native stable ID."""
    excluded = set(excluded or ())
    ids = list(range(len(ds)))
    random.Random(seed).shuffle(ids)
    selected = []
    for i in ids:
        row = ds[i]
        key = _qid(dataset, row, source)
        if key in excluded:
            continue
        selected.append(row)
        excluded.add(key)
        if len(selected) == n:
            break
    if len(selected) != n:
        raise ValueError(f'needed {n} unique rows from {dataset}/{source}, found {len(selected)}')
    return selected


def build(cfg: dict, selected: list[str]) -> None:
    from datasets import load_dataset

    seed = int(cfg['seed'])
    frac = float(cfg['retrieval']['hole_fraction'])
    sizes = cfg['datasets']
    data_dir = path(cfg, 'questions')
    data_dir.mkdir(parents=True, exist_ok=True)
    pilot: list[Question] = []
    for dataset in selected:
        if dataset in ('nq', 'triviaqa'):
            repo = 'google-research-datasets/nq_open' if dataset == 'nq' else 'mandarjoshi/trivia_qa'
            subset = None if dataset == 'nq' else 'rc.nocontext'
            print(f'Loading {repo} ...', flush=True)
            src_train = load_dataset(repo, subset, split='train')
            src_test = load_dataset(repo, subset, split='validation')
            ntrain, nval, ntest = (int(sizes[dataset][k]) for k in ('train', 'val', 'test'))
            combined = select_rows(src_train, ntrain + nval, seed, dataset=dataset, source='train')
            parts = {
                'train': combined[:ntrain],
                'val': combined[ntrain:],
                'test': select_rows(src_test, ntest, seed + 2, dataset=dataset, source='validation'),
            }
            for split, rows in parts.items():
                qs = [to_question(dataset, split, r, seed, frac) for r in rows]
                write_jsonl(data_dir / f'{dataset}_{split}.jsonl', qs)
                print(f'Written {dataset}_{split}: {len(qs)}', flush=True)
                if split == 'train':
                    pilot.extend(replace(q, split='pilot') for q in qs[:int(cfg['pilot']['n']) // 2])
        elif dataset == 'popqa':
            print('Loading akariasai/PopQA ...', flush=True)
            src = load_dataset('akariasai/PopQA', split='test')
            tail = src.filter(lambda row: row['s_pop'] is not None and float(row['s_pop']) < 100)
            rows = select_rows(tail, int(sizes[dataset]['test']), seed,
                               dataset=dataset, source='test')
            write_jsonl(data_dir / 'popqa_test.jsonl',
                        [to_question(dataset, 'test', r, seed, frac) for r in rows])
            print(f'Written popqa_test: {len(rows)}')
        elif dataset == 'hotpotqa':
            print('Loading hotpotqa/hotpot_qa distractor/validation ...', flush=True)
            src = load_dataset('hotpotqa/hotpot_qa', 'distractor', split='validation')
            rows = select_rows(src, int(sizes[dataset]['test']), seed,
                               dataset=dataset, source='validation')
            write_jsonl(data_dir / 'hotpotqa_test.jsonl',
                        [to_question(dataset, 'test', r, seed, frac) for r in rows])
            print(f'Written hotpotqa_test: {len(rows)}')
        elif dataset == 'selfaware':
            print('Loading SelfAware.json from original authors ...', flush=True)
            url = 'https://raw.githubusercontent.com/yinzhangyue/SelfAware/main/data/SelfAware.json'
            with urllib.request.urlopen(url, timeout=120) as stream:
                raw = json.load(stream)
            rows = [r for r in raw['example'] if r.get('answerable') is False]
            random.Random(seed).shuffle(rows)
            n = int(sizes[dataset]['test'])
            if len(rows) < n:
                raise ValueError(f'SelfAware unanswerable only has {len(rows)} rows, expected {n}')
            write_jsonl(data_dir / 'selfaware_test.jsonl',
                        [to_question(dataset, 'test', r, seed, frac) for r in rows[:n]])
            print(f'Written selfaware_test: {n}')
        else:
            raise ValueError(f'Unknown dataset {dataset}')
    if 'nq' in selected and 'triviaqa' in selected:
        if len(pilot) != int(cfg['pilot']['n']):
            raise ValueError('Pilot size not satisfied. Use an even pilot.n.')
        write_jsonl(data_dir / 'pilot.jsonl', pilot)
        print(f'Written pilot: {len(pilot)}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config')
    ap.add_argument('--datasets', nargs='+', default=['nq', 'triviaqa', 'popqa', 'hotpotqa', 'selfaware'],
                    choices=['nq', 'triviaqa', 'popqa', 'hotpotqa', 'selfaware'])
    args = ap.parse_args()
    build(load_config(args.config), args.datasets)


if __name__ == '__main__':
    main()

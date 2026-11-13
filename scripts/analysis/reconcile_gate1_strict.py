#!/usr/bin/env python3
"""Re-score strict Gate 1 after the 2026-11-11 answer matcher correction.

Uses the unchanged human verdicts, and permits only the three known canonical
field changes for qid nq:d86072d818628efc912e. Fails without modifying files
if the rerun or the review sheet differs from the expected version.
"""
import json
from pathlib import Path

import pandas as pd
from saerag.eval.handcheck import score_sheet

root = Path.cwd()
base = root / 'results/sagemaker/handcheck'
sheet = base / 'sheet_strict.csv'
labels_dir = root / 'data/labels/miss_as_retrieve'
qid = 'nq:d86072d818628efc912e'
fields = {'label': 'label', 'closed_auto': 'closed_correct',
          'bm25_auto': 'rag_correct', 'gold_auto': 'gold_correct'}
expected = {(qid, 'label'), (qid, 'bm25_auto'), (qid, 'gold_auto')}

canonical = {}
for p in labels_dir.glob('*.jsonl'):
    with p.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                canonical[r['qid']] = r

df = pd.read_csv(sheet, encoding='utf-8-sig', keep_default_na=False)
df = df.astype({col: object for col in [*fields, "abstain_reason"]})
assert len(df) == 200 and df.qid.is_unique, 'Unexpected hand-check sample'
assert all(q in canonical for q in df.qid), 'Missing canonical labels'
normalize = lambda x: '' if x is None else str(x).strip().lower()
mismatches = {(r.qid, col)
              for _, r in df.iterrows()
              for col, field in fields.items()
              if normalize(r[col]) != normalize(canonical[r.qid][field])}
assert mismatches == expected, f'Unexpected label changes: {sorted(mismatches)}'
idx = df.index[df.qid == qid][0]
assert normalize(df.at[idx, 'verdict_bm25']) == 'y'
assert normalize(df.at[idx, 'verdict_gold']) == 'y'
for col, field in fields.items():
    value = canonical[qid][field]
    df.at[idx, col] = '' if value is None else value
df.at[idx, 'abstain_reason'] = canonical[qid]['abstain_reason'] or ''
result = score_sheet(df, variant='miss_as_retrieve')
assert (result['n'], result['label_errors'], result['match_errors'], result['match_checks']) == (200, 15, 27, 566), result
# Write only after all consistency checks have passed.
df.to_csv(sheet, index=False, encoding='utf-8-sig')
(base / 'flagged_strict.json').write_text(json.dumps(result, indent=1) + '\n')
print('STRICT HAND-CHECK: 15/200 errors (7.5%), 27/566 answer-match disagreements')
print('NOTE: The lenient hand-check is unchanged; its annotated sheet was not provided.')

# Data, label rule and Gate 0 (gemma3-4b-it)

## Data: splits, holes and retrieval coverage

| split | n | hole | unanswerable | gold passage | BM25 has answer (non-hole) | BM25 has answer (hole) |
|---|---|---|---|---|---|---|
| pilot | 500 | 31.4% | 0.0% | 85.4% | 51.9% | 0.0% |
| nq_train | 4000 | 30.2% | 0.0% | 77.1% | 37.2% | 0.0% |
| nq_val | 500 | 30.8% | 0.0% | 75.6% | 35.0% | 0.0% |
| nq_test | 1000 | 30.9% | 0.0% | 80.6% | 37.6% | 0.0% |
| triviaqa_train | 4000 | 30.3% | 0.0% | 93.2% | 71.1% | 0.0% |
| triviaqa_val | 500 | 31.4% | 0.0% | 93.4% | 72.6% | 0.0% |
| triviaqa_test | 1000 | 28.3% | 0.0% | 92.7% | 71.4% | 0.0% |
| popqa_test | 1000 | 0.0% | 0.0% | 62.8% | 40.0% | - |
| hotpotqa_test | 1000 | 30.2% | 0.0% | 100.0% | 38.0% | 0.0% |
| selfaware_test | 500 | 0.0% | 100.0% | 0.0% | 0.0% | - |

CSV: `data.csv`

## Accuracy per condition (alias containment)

| split | closed | n closed | bm25 | n bm25 | gold | n gold |
|---|---|---|---|---|---|---|
| pilot | 35.4% | 500 | 29.0% | 500 | 74.2% | 427 |
| nq_train | 18.6% | 4000 | 18.9% | 4000 | 61.5% | 3085 |
| nq_val | 17.2% | 500 | 17.6% | 500 | 60.3% | 378 |
| nq_test | 18.1% | 1000 | 19.0% | 1000 | 61.8% | 806 |
| triviaqa_train | 52.0% | 4000 | 49.6% | 4000 | 83.8% | 3728 |
| triviaqa_val | 50.6% | 500 | 50.2% | 500 | 82.0% | 467 |
| triviaqa_test | 50.4% | 1000 | 50.7% | 1000 | 83.4% | 927 |
| popqa_test | 17.2% | 1000 | 34.1% | 1000 | 70.1% | 628 |
| hotpotqa_test | 18.8% | 1000 | 21.7% | 1000 | 61.5% | 1000 |
| selfaware_test | 0.0% | 500 | 0.0% | 500 | - | 0 |

CSV: `accuracy.csv`

## Label shares under both rules (main rule: miss_as_retrieve)

| split | rule | n | ANSWER | RETRIEVE | ABSTAIN | under 15% |
|---|---|---|---|---|---|---|
| pilot | miss_as_retrieve | 500 | 35.4% | 36.4% | 28.2% | - |
| pilot | default | 500 | 35.4% | 11.2% | 53.4% | RETRIEVE |
| nq_train | miss_as_retrieve | 4000 | 18.6% | 35.3% | 46.2% | - |
| nq_train | default | 4000 | 18.6% | 11.7% | 69.8% | RETRIEVE |
| nq_val | miss_as_retrieve | 500 | 17.2% | 35.8% | 47.0% | - |
| nq_val | default | 500 | 17.2% | 10.4% | 72.4% | RETRIEVE |
| nq_test | miss_as_retrieve | 1000 | 18.1% | 36.9% | 45.0% | - |
| nq_test | default | 1000 | 18.1% | 12.2% | 69.7% | RETRIEVE |
| triviaqa_train | miss_as_retrieve | 4000 | 52.0% | 34.9% | 13.1% | ABSTAIN |
| triviaqa_train | default | 4000 | 52.0% | 18.4% | 29.6% | - |
| triviaqa_val | miss_as_retrieve | 500 | 50.6% | 32.6% | 16.8% | - |
| triviaqa_val | default | 500 | 50.6% | 16.4% | 33.0% | - |
| triviaqa_test | miss_as_retrieve | 1000 | 50.4% | 36.3% | 13.3% | ABSTAIN |
| triviaqa_test | default | 1000 | 50.4% | 21.0% | 28.6% | - |
| popqa_test | miss_as_retrieve | 1000 | 17.2% | 32.3% | 50.5% | - |
| popqa_test | default | 1000 | 17.2% | 21.8% | 61.0% | - |
| hotpotqa_test | miss_as_retrieve | 1000 | 18.8% | 47.0% | 34.2% | - |
| hotpotqa_test | default | 1000 | 18.8% | 13.1% | 68.1% | RETRIEVE |
| selfaware_test | miss_as_retrieve | 500 | 0.0% | 0.0% | 100.0% | ANSWER, RETRIEVE |
| selfaware_test | default | 500 | 0.0% | 0.0% | 100.0% | ANSWER, RETRIEVE |

CSV: `labels.csv`

## What the rule change moves

| split | n | moved to RETRIEVE | moved share | ABSTAIN beyond_model | ABSTAIN retriever_miss | ABSTAIN unanswerable |
|---|---|---|---|---|---|---|
| pilot | 500 | 126 | 25.2% | 141 | 126 | 0 |
| nq_train | 4000 | 945 | 23.6% | 1846 | 945 | 0 |
| nq_val | 500 | 127 | 25.4% | 235 | 127 | 0 |
| nq_test | 1000 | 247 | 24.7% | 450 | 247 | 0 |
| triviaqa_train | 4000 | 658 | 16.4% | 525 | 658 | 0 |
| triviaqa_val | 500 | 81 | 16.2% | 84 | 81 | 0 |
| triviaqa_test | 1000 | 153 | 15.3% | 133 | 153 | 0 |
| popqa_test | 1000 | 105 | 10.5% | 505 | 105 | 0 |
| hotpotqa_test | 1000 | 339 | 33.9% | 342 | 339 | 0 |
| selfaware_test | 500 | 0 | 0.0% | 0 | 0 | 500 |

CSV: `rule_change.csv`

## Gate 0: each class >= 15%

| set | subset | rule | n | ANSWER | RETRIEVE | ABSTAIN | gate |
|---|---|---|---|---|---|---|---|
| pilot | all | miss_as_retrieve | 500 | 35.4% | 36.4% | 28.2% | PASS |
| pilot | all | default | 500 | 35.4% | 11.2% | 53.4% | FAIL |
| pilot | nq | miss_as_retrieve | 250 | 20.8% | 38.0% | 41.2% | PASS |
| pilot | nq | default | 250 | 20.8% | 9.6% | 69.6% | FAIL |
| pilot | triviaqa | miss_as_retrieve | 250 | 50.0% | 34.8% | 15.2% | PASS |
| pilot | triviaqa | default | 250 | 50.0% | 12.8% | 37.2% | FAIL |
| train (nq+triviaqa) | all | miss_as_retrieve | 8000 | 35.3% | 35.1% | 29.6% | PASS |
| train (nq+triviaqa) | all | default | 8000 | 35.3% | 15.0% | 49.7% | PASS |
| train (nq+triviaqa) | nq | miss_as_retrieve | 4000 | 18.6% | 35.3% | 46.2% | PASS |
| train (nq+triviaqa) | nq | default | 4000 | 18.6% | 11.7% | 69.8% | FAIL |
| train (nq+triviaqa) | triviaqa | miss_as_retrieve | 4000 | 52.0% | 34.9% | 13.1% | FAIL |
| train (nq+triviaqa) | triviaqa | default | 4000 | 52.0% | 18.4% | 29.6% | PASS |

CSV: `gate0.csv`

![label shares](figs/label_shares.png)

# Generation report: pilot (gemma3-4b-it)

## Accuracy per condition

| condition | n | accuracy | non-hole | hole | sec / 1k questions |
|---|---|---|---|---|---|
| closed | 500 | 35.4% | 37.6% | 30.6% | 28.5 |
| bm25 | 500 | 29.2% | 38.2% | 9.6% | 167.3 |
| gold | 427 | 74.0% | 72.7% | 76.8% | 72.3 |

## Labels: default

| subset | n | ANSWER | RETRIEVE | ABSTAIN |
|---|---|---|---|---|
| all | 500 | 35.4% | 11.4% | 53.2% |
| non-hole | 343 | 37.6% | 16.0% | 46.4% |
| hole | 157 | 30.6% | 1.3% | 68.2% |
| nq | 250 | 20.8% | 10.0% | 69.2% |
| triviaqa | 250 | 50.0% | 12.8% | 37.2% |

Abstain reasons: {'beyond_model': 142, 'retriever_miss': 124}

## Labels: miss_as_retrieve

| subset | n | ANSWER | RETRIEVE | ABSTAIN |
|---|---|---|---|---|
| all | 500 | 35.4% | 36.2% | 28.4% |
| non-hole | 343 | 37.6% | 33.5% | 28.9% |
| hole | 157 | 30.6% | 42.0% | 27.4% |
| nq | 250 | 20.8% | 37.6% | 41.6% |
| triviaqa | 250 | 50.0% | 34.8% | 15.2% |

Abstain reasons: {'beyond_model': 142}

**Gate (each class >= 15% of the default labels): FAIL**

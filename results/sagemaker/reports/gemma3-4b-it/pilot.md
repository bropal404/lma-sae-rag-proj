# Generation report: pilot (gemma3-4b-it)

## Accuracy per condition

| condition | n | accuracy | non-hole | hole | sec / 1k questions |
|---|---|---|---|---|---|
| closed | 500 | 35.4% | 37.6% | 30.6% | 28.5 |
| bm25 | 500 | 29.8% | 39.1% | 9.6% | 167.3 |
| gold | 427 | 74.7% | 73.7% | 76.8% | 72.3 |

## Labels: default

| subset | n | ANSWER | RETRIEVE | ABSTAIN |
|---|---|---|---|---|
| all | 500 | 35.4% | 11.8% | 52.8% |
| non-hole | 343 | 37.6% | 16.6% | 45.8% |
| hole | 157 | 30.6% | 1.3% | 68.2% |
| nq | 250 | 20.8% | 10.8% | 68.4% |
| triviaqa | 250 | 50.0% | 12.8% | 37.2% |

Abstain reasons: {'beyond_model': 140, 'retriever_miss': 124}

## Labels: miss_as_retrieve

| subset | n | ANSWER | RETRIEVE | ABSTAIN |
|---|---|---|---|---|
| all | 500 | 35.4% | 36.6% | 28.0% |
| non-hole | 343 | 37.6% | 34.1% | 28.3% |
| hole | 157 | 30.6% | 42.0% | 27.4% |
| nq | 250 | 20.8% | 38.4% | 40.8% |
| triviaqa | 250 | 50.0% | 34.8% | 15.2% |

Abstain reasons: {'beyond_model': 140}

**Gate (each class >= 15% of the default labels): FAIL**

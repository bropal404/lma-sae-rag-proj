# SAE-Gated RAG: final results audit

Generated from exported results. This is an audit, not a substitute for the full paper.

## Locked experimental choices

- Primary label rule: `miss_as_retrieve`.
- SAE model: `sae16k_last@L29`; dense comparator: `resid_last@L22`.
- Both chosen by **full-budget macro-F1 on NQ/TriviaQA validation only**, not test results.

## Gate summary

| Requirement | Result | Interpretation |
|---|---|---|
| Gate 0: ≥15% each class | PASS | ANSWER 35.4%, RETRIEVE 36.6%, ABSTAIN 28.0%; data/labels/ (canonical) |
| Gate 1: <10% label error | PASS by point estimate | 15/200 = 7.5%; Wilson 95% CI [4.6%, 12.0%] |
| Gate 2 / H1: accuracy within 2 points and ≥30% fewer BM25 calls | PASS | B1 31.9% vs SAE 31.5%, BM25 calls 49.2% (reduction 50.8%) on 4000 answerable questions |
| Gate 3: latency of all chosen methods | PARTIAL | Current routed latency measures layer 17, not the validation-selected SAE layer |

## H2: SAE versus dense

Validation macro-F1: SAE 51.8%, dense 53.1% (SAE − dense -1.24 points).
Answerable-test mean per-dataset macro-F1: SAE 45.8%, dense 48.2% (SAE − dense -2.38 points).
Selection used validation; test estimates are evaluation only. This is a comparison of independently selected layers, not a paired same-layer feature ablation.

## H3: index-hole unsafe answers

Wrong answers per total hole questions (weighted across NQ, TriviaQA and HotpotQA):
- Three-way SAE: 61.0%, coverage 84.3%, wrong among attempted 72.3%.
- Two-way SAE: 82.2%, coverage 100.0%, wrong among attempted 82.2%.
The per-question unsafe-answer rate includes abstentions as safe; the separate conditional error rate above uses only attempted answers. Report both, along with coverage.

## Existing latency evidence

- `b1_always_retrieve`: p50=251.4 ms, p95=541.0 ms, BM25 calls=100.0% (n=200).
- `b4_verbalised`: p50=466.2 ms, p95=790.2 ms, BM25 calls=59.0% (n=200).
- `b8_tfidf_lr`: p50=154.5 ms, p95=411.3 ms, BM25 calls=26.0% (n=200).
- `gemma3-4b-it__miss_as_retrieve__sae16k_last@L17__l2`: p50=211.8 ms, p95=479.6 ms, BM25 calls=29.5% (n=200).
- `gemma3-4b-it__miss_as_retrieve__resid_last@L17__l2`: p50=208.5 ms, p95=457.8 ms, BM25 calls=27.5% (n=200).

## Regression and diagnostic evidence

- SAE L17, 1000 labels: macro-F1=0.533, inner-CV=0.536, seeds=3.
- SAE L17, 2000 labels: macro-F1=0.544, inner-CV=0.542, seeds=3.
- SAE L17, 4000 labels: macro-F1=0.563, inner-CV=0.545, seeds=3.
The original 1,000→2,000 degradation is not reproduced in these later seeds; the original cause remains unverified.

## Audit limitations and open items

1. Re-run `scripts/analysis/bundle_analysis.py` on the machine holding raw router probabilities: earlier paired bootstrap confidence intervals were invalid because duplicate bootstrap sample IDs were collapsed by dictionaries.
2. Time the **selected SAE layer** at batch size 1 before claiming H4 for the locked primary router.
3. The authoritative handcheck is `sheet_strict.csv` / `flagged_strict.json`. The lenient adjudication does not pass Gate 1; report that sensitivity and the confidence interval.
4. Explicitly disclose lexical top-100 index holes and retrieval-derived gold stand-ins.
5. Include reproducibility metadata: Gemma checkpoint, SAE site/width/L0, corpus revision, data splits, seeds, GPU, decoding and matcher revision.
6. Verify availability of raw outputs (`data/`, `runs/`) on SageMaker. The repository ZIP only contains the smaller exported summaries.

## Handoff readiness

Raw data directory included in snapshot: True; router probabilities included: True. Neither is needed to inspect exported CSV results, but both are needed to fully reproduce experiments and recompute bootstrap intervals.


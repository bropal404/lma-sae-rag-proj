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
| Gate 3: latency of all chosen methods | COMPLETE | Selected SAE L29 p50 234.4 ms; dense L22 p50 242.6 ms; B1 231.3 ms. H4 FAIL by predeclared median criterion. Composed stage timings, n=200. |

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

- `b1_always_retrieve`: p50=231.3 ms, p95=492.8 ms, mean=273.7 ms; BM25 calls=100.0% (n=200).
- `b2_never_retrieve`: p50=182.7 ms, p95=358.5 ms, mean=202.3 ms; BM25 calls=0.0% (n=200).
- `b3_oracle`: p50=184.9 ms, p95=411.9 ms, mean=182.1 ms; BM25 calls=43.5% (n=200).
- `b4_verbalised`: p50=427.0 ms, p95=718.2 ms, mean=438.3 ms; BM25 calls=59.0% (n=200).
- `b8_tfidf_lr`: p50=184.9 ms, p95=419.4 ms, mean=196.4 ms; BM25 calls=38.5% (n=200).
- `gemma3-4b-it__miss_as_retrieve__sae16k_last@L29__l2`: p50=234.4 ms, p95=466.6 ms, mean=229.1 ms; BM25 calls=39.0% (n=200).
- `gemma3-4b-it__miss_as_retrieve__resid_last@L22__l2`: p50=242.6 ms, p95=465.8 ms, mean=247.5 ms; BM25 calls=47.0% (n=200).

## H4: selected-controller latency

**H4 NOT SUPPORTED:** Median SAE L29 latency 234.4 ms versus always-retrieve 231.3 ms and verbalised router 427.0 ms. H4 requires the SAE median to beat both.
The selected dense L22 median is 242.6 ms. SAE improves mean latency by 16.3% over always-retrieve (229.1 versus 273.7 ms), and p95 is 466.6 versus 492.8 ms.
These are per-question composed costs from independently measured stages (GPU generation/controller plus prior BM25 retrieval timings), not integrated serving-system stopwatch measurements. The 200 timed questions are the first 100 each from NQ and TriviaQA test, not a random workload sample.

## Regression and diagnostic evidence

- SAE L17, 1000 labels: macro-F1=0.533, inner-CV=0.536, seeds=3.
- SAE L17, 2000 labels: macro-F1=0.544, inner-CV=0.542, seeds=3.
- SAE L17, 4000 labels: macro-F1=0.563, inner-CV=0.545, seeds=3.
The original 1,000→2,000 degradation is not reproduced in these later seeds; the original cause remains unverified.

## Audit limitations and remaining reporting work

1. The corrected paired bootstrap has been rerun on matcher-corrected labels and stored under `results/analysis/`; include the selected-model (SAE L29 versus dense L22) results separately from matched-layer ablations.
2. The 200-item strict handcheck reports a 7.5% error point estimate, but its Wilson 95% upper bound exceeds 10%; the lenient handcheck sensitivity is 11%.
3. Explicitly disclose lexical top-100 index holes and retrieval-derived gold stand-ins.
4. Include reproducibility metadata: Gemma checkpoint, SAE site/width/L0, corpus revision, data splits, seeds, GPU, decoding and matcher revision.
5. Large raw features and full router histories are not committed to Git; the original SageMaker persistent volume is needed for full end-to-end reproduction.
6. The final scientific report requires synthesis and references; this audit is not a finished paper.

## Handoff readiness

Raw data directory included in snapshot: True; router probabilities included: True. Neither is needed to inspect exported CSV results, but both are needed to fully reproduce experiments and recompute bootstrap intervals.


# SAE-Gated RAG: Deciding to Answer, Retrieve, or Abstain Before Generation

## Abstract (draft)
We study a three-way pre-generation controller using frozen Gemma 3 4B activations and Gemma Scope 2 sparse autoencoder codes, comparing against a dense probe on the same model. On 2018 Wikipedia retrieval over 21,015,324 passages, questions from Natural Questions, TriviaQA, PopQA, HotpotQA and SelfAware are evaluated under closed-book, BM25 and diagnostic gold conditions. The primary label rule is `miss_as_retrieve`. The quantitative conclusions below are computed from the saved score exports, not a new training run.

## 1. Research questions

H1: Accuracy within two points of always-retrieve with at least 30% fewer retrieval calls. H2: SAE versus dense macro-F1 at full data and in data-limited/transfer settings. H3: Three-way abstention reduces unsafe answers on lexical index holes. H4: Controller-based latency compared with always-retrieve and generation-time triggers.

## 2. Data and protocol

Question-only pre-generation activations are taken from frozen Gemma 3 4B. Dense probes operate on residual-stream vectors; SAE probes use pretrained Gemma Scope 2 features. Models are trained on NQ and TriviaQA training sets, with layer selection on their validation splits. The final test set consists of 1,000 questions each from NQ, TriviaQA, PopQA and HotpotQA, plus 500 unanswerable SelfAware questions.

## 3. Experimental implementation

The 2018 DPR-style Wikipedia corpus has 21,015,324 100-word passages. BM25 returns top three. Lexical index-hole cases are simulated by removing detected answer-bearing passages from the top 100 BM25 candidates at query time. This is not physical document deletion. Gold passages are often BM25-derived stand-ins, not independently annotated supporting documents. The primary three-way label rule assigns ANSWER to a correct closed-book answer, RETRIEVE to a correct BM25 answer or a gold-correct retriever miss, and ABSTAIN otherwise, with SelfAware marked unanswerable.

## 4. Results

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


## 5. Threats to validity

String matching in question answering has false positives and false negatives; the strict human handcheck estimates an error rate but its uncertainty should be retained. Index-hole filtering is lexical, gold evidence is a stand-in, selected models are validation-chosen, and the 200 timed questions may not represent all workloads. Raw end-to-end answer accuracy on SelfAware is zero by metric definition; abstention recall is the appropriate SelfAware measure.

## 6. Conclusion

Use the gate-by-gate evidence above to state supported and unsupported hypotheses. State the H4 median-latency failure despite improved mean and tail latency. Do not claim universal SAE superiority or full serving-system measurements.

## References

Reuse the vetted bibliography from `report/report_mid.typ` and add precise citations for Gemma 3, Gemma Scope 2, the datasets, BM25 and the probe methodology.

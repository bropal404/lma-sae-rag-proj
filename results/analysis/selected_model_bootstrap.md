# Selected-model paired bootstrap

Validation-selected SAE `sae16k_last@L29` versus dense `resid_last@L22`.
Decisions use the same validation-only abstention threshold as `saerag.eval.score`.

| Dataset | SAE − dense macro-F1 | 95% percentile CI |
|---|---:|---:|
| nq_test | +0.89 pp | [-2.18, +3.90] pp |
| triviaqa_test | -0.32 pp | [-4.13, +3.52] pp |
| popqa_test | -7.51 pp | [-10.06, -4.84] pp |
| hotpotqa_test | -0.04 pp | [-3.07, +2.80] pp |
| answerable_test_mean_4 | -1.74 pp | [-3.32, -0.11] pp |

The answerable-test aggregate is the unweighted mean of four 1,000-question dataset macro-F1 scores; each bootstrap draw resamples questions **within each dataset** and preserves the SAE/dense pairing.

The routed point estimate is inside the -2 pp tolerance only if it is >= -2 pp. A 95% CI that extends below -2 pp **does not establish** the non-inferiority margin.
This routed-action metric is distinct from the raw probe-grid H2 macro-F1, which may use different decision rules; report both with explicit labels.

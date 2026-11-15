# Selected-model paired bootstrap

Validation-selected SAE `sae16k_last@L29` versus dense `resid_last@L22`.
Decisions use the same validation-only abstention threshold as `saerag.eval.score`.

| Dataset | SAE − dense macro-F1 | 95% percentile CI |
|---|---:|---:|
| nq_test | +0.71 pp | [-2.29, +3.83] pp |
| triviaqa_test | -0.46 pp | [-4.25, +3.50] pp |
| popqa_test | -7.68 pp | [-10.31, -5.00] pp |
| hotpotqa_test | -0.22 pp | [-3.18, +2.65] pp |
| answerable_test_mean_4 | -1.91 pp | [-3.47, -0.28] pp |

The answerable-test aggregate is the unweighted mean of four 1,000-question dataset macro-F1 scores; each bootstrap draw resamples questions **within each dataset** and preserves the SAE/dense pairing.

The routed point estimate is inside the -2 pp tolerance only if it is >= -2 pp. A 95% CI that extends below -2 pp **does not establish** the non-inferiority margin.
This routed-action metric is distinct from the raw probe-grid H2 macro-F1, which may use different decision rules; report both with explicit labels.

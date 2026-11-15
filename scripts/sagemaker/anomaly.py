"""Why did the SAE probe score lower on 2,000 labels than on 1,000 (mid-eval report)?

Refits SAE and dense probes at one layer for several label budgets and records, per fit,
the C picked by the inner CV, the inner-CV score, iterations used (5,000 = did not converge),
non-zero weights and test scores on nq_test + triviaqa_test. Runs on the proxy label
("closed": closed-book correct or not) and on the main label rule.
Writes runs/diagnostics/anomaly.csv.
"""
import argparse

import numpy as np, pandas as pd

from saerag.config import load_config, path
from saerag.probes.train import load_block, make_model, score, subsample

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--layer", type=int, default=17)
ap.add_argument("--labels", nargs="+", default=["closed", "miss_as_retrieve"])
ap.add_argument("--budgets", nargs="+", type=int, default=[1000, 2000, 4000])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
args = ap.parse_args()
cfg = load_config(args.config)
L, kinds = args.layer, ["resid_last", "sae16k_last"]

rows = []
for labels in args.labels:
    _, y_str, _, X = load_block(cfg, ["nq_train", "triviaqa_train"], labels, [L], kinds)
    _, y2_str, _, X2 = load_block(cfg, ["nq_test", "triviaqa_test"], labels, [L], kinds)
    classes = sorted(set(y_str))
    y = np.array([classes.index(c) for c in y_str])
    y2 = np.array([classes.index(c) for c in y2_str])
    idx = np.arange(len(y))
    for k in kinds:
        name = f"{k}@L{L}"
        for budget in [*args.budgets, None]:
            for seed in (args.seeds if budget else args.seeds[:1]):
                tr = subsample(idx, y, budget, seed)
                fit = make_model(name, "l2", False, seed).fit(X[name][tr], y[tr])
                lr = fit.best_estimator_[-1]
                rows.append({"labels": labels, "model": name, "budget": len(tr), "seed": seed,
                             "C": fit.best_params_["logisticregression__C"],
                             "inner_cv_f1": fit.best_score_, "n_iter": int(np.max(lr.n_iter_)),
                             "nonzero_weights": int((lr.coef_ != 0).sum()),
                             **score(fit.predict_proba(X2[name]), y2, len(classes))})
                r = rows[-1]
                print(f"{labels} {name} n={r['budget']} seed={seed} C={r['C']} iters={r['n_iter']} "
                      f"test F1={r['macro_f1']:.3f}", flush=True)

df = pd.DataFrame(rows)
out = path(cfg, "results", "diagnostics")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "anomaly.csv", index=False)
print(df.groupby(["labels", "model", "budget"])[["macro_f1", "inner_cv_f1", "n_iter"]].mean().round(3).to_string())
print(df.groupby(["labels", "model", "budget"]).C.agg(lambda c: sorted(set(c))).to_string())

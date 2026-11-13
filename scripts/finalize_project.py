#!/usr/bin/env python3
"""Read-only audit of exported SAE-RAG results; writes a reproducible final summary.

Works from the small results/sagemaker/ files in Git, even if data/ and runs/
are on another machine. Does NOT train models, mutate labels or upload anything.

    python scripts/finalize_project.py
    python scripts/finalize_project.py --root /path/to/repo

Outputs results/final/{gate_summary.json,gate2_by_dataset.csv,
model_selection.csv,final_summary.md,report_draft.md}.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

TESTS = ("nq_test", "triviaqa_test", "popqa_test", "hotpotqa_test")
ALL_TESTS = TESTS + ("selfaware_test",)
VAL = ("nq_val", "triviaqa_val")
TAG = "gemma3-4b-it"
PRIMARY = "miss_as_retrieve"
SEED_ROUTER = "b1_always_retrieve"


def f_pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None or not math.isfinite(float(x)) else f"{x * 100:.{digits}f}%"


def f_num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(float(x)) else f"{x:.{digits}f}"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z**2 / n
    cent = (p + z**2 / (2 * n)) / den
    rad = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / den
    return cent - rad, cent + rad


def _scores(repo: Path) -> dict[str, pd.DataFrame]:
    scores = repo / "results/sagemaker/scores" / PRIMARY
    out = {}
    for name in ALL_TESTS:
        file = scores / f"{name}.csv"
        if not file.exists():
            raise FileNotFoundError(f"Missing score export: {file}. The raw runs may be on SageMaker.")
        out[name] = pd.read_csv(file)
    return out


def _probe_summary(repo: Path) -> pd.DataFrame:
    folder = repo / "results/sagemaker/probes" / TAG / PRIMARY
    matches = [f for f in folder.glob("*/summary.csv") if f.parent.name.endswith("__l2")]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one main L2 probe grid in {folder}, found {len(matches)}")
    df = pd.read_csv(matches[0])
    required = {"model", "budget", "seed", "eval", "macro_f1", "accuracy"}
    if not required.issubset(df.columns):
        raise ValueError(f"Probe summary missing {required - set(df.columns)}")
    return df


def select_models(summary: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    """Choose one SAE and one dense model from validation ONLY, using full-label results."""
    v = summary[(summary["budget"].astype(str) == "full") & summary["eval"].isin(VAL)].copy()
    table = v.groupby("model", as_index=False).agg(validation_macro_f1=("macro_f1", "mean"),
                                                   validation_files=("eval", "nunique"))
    table = table[table.validation_files == len(VAL)].copy()
    sae = table[table.model.str.fullmatch(r"sae16k_last@L\d+")]
    dense = table[table.model.str.fullmatch(r"resid_last@L\d+")]
    if sae.empty or dense.empty:
        raise ValueError("Cannot select SAE/dense probes; missing validation results")
    chosen = {"sae": sae.sort_values(["validation_macro_f1", "model"], ascending=[False, True]).iloc[0].model,
              "dense": dense.sort_values(["validation_macro_f1", "model"], ascending=[False, True]).iloc[0].model}
    table["selected"] = table.model.isin(chosen.values())
    return chosen, table.sort_values("validation_macro_f1", ascending=False)


def _router_id(model: str) -> str:
    return f"{TAG}__{PRIMARY}__{model}__l2"


def _all_row(df: pd.DataFrame, rid: str, subset: str = "all") -> pd.Series:
    x = df[(df.router == rid) & (df.subset == subset)]
    if len(x) != 1:
        raise ValueError(f"Expected 1 row for {rid} {subset}, found {len(x)}")
    return x.iloc[0]


def aggregate(scores: dict[str, pd.DataFrame], rid: str, files=TESTS) -> dict:
    """Aggregate micro-style correctness and retrieval calls across answerable test datasets.

    The stored per-dataset `accuracy` already means correct / all questions, with
    abstentions counted as not correct. Weighted means are exact up to CSV rounding.
    """
    rows = [_all_row(scores[f], rid) for f in files]
    n = sum(int(x["n"]) for x in rows)
    if not n:
        raise ValueError("empty evaluation")
    return {"router": rid, "n": n,
            "accuracy": sum(int(x["n"]) * float(x["accuracy"]) for x in rows) / n,
            "retrieval_call_rate": sum(int(x["n"]) * float(x["retrieval_call_rate"]) for x in rows) / n,
            "abstention_recall": sum(int(x["n"]) * float(x["abstention_recall"]) for x in rows
                                     if pd.notna(x["abstention_recall"])) / n}


def _pilot(repo: Path) -> tuple[dict[str, float], str]:
    # Prefer the canonical labels on the machine that actually holds data/.
    p = repo / "data/labels" / PRIMARY / "pilot.jsonl"
    if p.exists():
        counts = {k: 0 for k in ("ANSWER", "RETRIEVE", "ABSTAIN")}
        for line in p.read_text().splitlines():
            if line.strip():
                counts[json.loads(line)["label"]] += 1
        n = sum(counts.values())
        if n == 0:
            raise ValueError("Empty pilot label file")
        return {k: v / n for k, v in counts.items()}, "data/labels/ (canonical)"
    p = repo / "results/sagemaker/reports" / TAG / "pilot.md"
    if not p.exists():
        return {}, "unavailable"
    text = p.read_text()
    # The recorded generation report can predate later rescoring; explicitly label that caveat.
    section = text.split("## Labels: miss_as_retrieve", 1)
    if len(section) < 2:
        return {}, "unavailable"
    m = re.search(r"\|\s*all\s*\|\s*\d+\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%", section[1])
    if not m:
        return {}, "unavailable"
    return dict(zip(("ANSWER", "RETRIEVE", "ABSTAIN"), (float(s) / 100 for s in m.groups()))), \
        "exported pilot report (may predate latest answer rescore)"


def _handcheck(repo: Path) -> tuple[dict | None, dict | None]:
    base = repo / "results/sagemaker/handcheck"
    def load(name):
        p = base / name
        return json.loads(p.read_text()) if p.exists() else None
    return load("flagged_strict.json"), load("flagged_lenient.json")


def _h3(scores: dict[str, pd.DataFrame], threeway_id: str) -> pd.DataFrame:
    records = []
    for rid in (threeway_id, threeway_id + "__2way"):
        for f in ("nq_test", "triviaqa_test", "hotpotqa_test"):
            r = _all_row(scores[f], rid, "hole")
            coverage = float(r["coverage"])
            halluc = float(r["hallucination_rate"]) if pd.notna(r["hallucination_rate"]) else math.nan
            n = int(r["n"])
            # Unsafe/wrong attempt per all hole questions (abstention contributes zero).
            bad_per_hole = 0.0 if coverage == 0 else coverage * halluc
            records.append(dict(router=rid, file=f, n=n, coverage=coverage,
                                wrong_among_attempted=halluc,
                                wrong_per_hole_question=bad_per_hole))
    return pd.DataFrame(records)


def _h2(summary: pd.DataFrame, chosen: dict[str, str]) -> pd.DataFrame:
    # Include actual finite-budget experiments for both locked model choices.
    picks = set(chosen.values())
    df = summary[summary.model.isin(picks) & summary["eval"].isin((*VAL, *ALL_TESTS))].copy()
    # Mean across seeds for each split, retaining the label budget and model identity.
    return df.groupby(["model", "budget", "eval"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"), accuracy=("accuracy", "mean"), seeds=("seed", "nunique"))


def _anomaly(repo: Path) -> pd.DataFrame | None:
    f = repo / "results/sagemaker/diagnostics/anomaly.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d.groupby(["labels", "model", "budget"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"), inner_cv_f1=("inner_cv_f1", "mean"),
        n_iter=("n_iter", "mean"), seeds=("seed", "nunique"))


def _check_handoff(repo: Path) -> dict:
    f = repo / "artifacts/retrieval/person1-full-wikipedia-handoff.tar.gz"
    return {"retrieval_archive_present": f.exists(), "retrieval_archive_bytes": f.stat().st_size if f.exists() else 0,
            "raw_data_present": (repo / "data/labels" / PRIMARY).is_dir(),
            "router_probabilities_present": (repo / "runs/routers").is_dir(),
            "features_present": (repo / "data/features").is_dir()}


def _report(repo: Path, dest: Path, selected: dict[str, str], selection: pd.DataFrame,
            scores: dict[str, pd.DataFrame], gate2: pd.DataFrame, h2: pd.DataFrame, h3: pd.DataFrame,
            pilot: dict, pilot_source: str, strict: dict | None, lenient: dict | None,
            anomaly: pd.DataFrame | None, source: dict) -> tuple[dict, str, str]:
    primary = _router_id(selected["sae"])
    dense = _router_id(selected["dense"])
    base = gate2[gate2.router == SEED_ROUTER].iloc[0]
    selected_row = gate2[gate2.router == primary].iloc[0]
    strict_k, strict_n = (int(strict["label_errors"]), int(strict["n"])) if strict else (0, 0)
    ci = wilson(strict_k, strict_n) if strict else (math.nan, math.nan)
    gate0 = bool(pilot) and all(p >= 0.15 for p in pilot.values())
    gate1 = strict is not None and strict_n > 0 and strict_k / strict_n < .10
    gate2_ok = bool(selected_row["accuracy"] >= base["accuracy"] - .02 and
                    selected_row["retrieval_call_rate"] <= base["retrieval_call_rate"] * .70)

    h3_aggs = {}
    for rid, g in h3.groupby("router"):
        attempted = float((g.n * g.coverage).sum())
        wrong = float((g.n * g.wrong_per_hole_question).sum())
        h3_aggs[rid] = {
            "wrong_per_hole_question": float(np.average(g.wrong_per_hole_question, weights=g.n)),
            "wrong_among_attempted": wrong / attempted if attempted else None,
            "coverage": float(np.average(g.coverage, weights=g.n)),
        }
    h3_pass = h3_aggs.get(primary, {}).get("wrong_per_hole_question", math.nan) < \
        h3_aggs.get(primary + "__2way", {}).get("wrong_per_hole_question", math.nan)
    # Only claim H4 for a controller that was actually timed at the same layer.
    selected_layer = int(re.search(r"@L(\d+)", selected["sae"]).group(1))
    routed_dir = repo / "results/sagemaker/latency"
    by_layer = routed_dir / f"routed_l{selected_layer}.csv"
    routed_file = by_layer if by_layer.exists() else routed_dir / "routed.csv"
    routed = pd.read_csv(routed_file) if routed_file.exists() else pd.DataFrame()
    lat_rows = {r.router: r for _, r in routed.iterrows()} if not routed.empty else {}
    h4_timed = (primary in lat_rows and SEED_ROUTER in lat_rows and "b4_verbalised" in lat_rows
                and "routing_policy" in routed.columns
                and (routed["routing_policy"] == "validation-threshold").all())
    h4_pass = (float(lat_rows[primary].p50_ms) < float(lat_rows[SEED_ROUTER].p50_ms)
               and float(lat_rows[primary].p50_ms) < float(lat_rows["b4_verbalised"].p50_ms)) if h4_timed else None
    # H2 at full-data budget, selected families determined only on validation.
    val = selection.set_index("model").validation_macro_f1
    h2_val_gap = float(val[selected["sae"]] - val[selected["dense"]])
    comp = h2[(h2.budget.astype(str) == "full") & h2["eval"].isin(TESTS)]
    comp_mean = comp.groupby("model").macro_f1.mean().to_dict()
    h2_test_gap = float(comp_mean[selected["sae"]] - comp_mean[selected["dense"]])
    h2_near = h2_test_gap >= -.02

    gates = {
        "source_snapshot": repo.name, "label_variant": PRIMARY,
        "selection_rule": "full-label validation macro-F1 on NQ+TriviaQA val; never select by test",
        "models": {"sae": selected["sae"], "dense": selected["dense"],
                   "sae_router": primary, "dense_router": dense},
        "gate0": {"pass": gate0, "class_shares": pilot, "source": pilot_source},
        "gate1": {"pass_point_estimate": gate1, "errors": strict_k, "n": strict_n,
                  "rate": strict_k / strict_n if strict_n else None,
                  "wilson95": list(ci), "lenient_rate": lenient["label_error_rate"] if lenient else None},
        "gate2_and_H1": {"pass": gate2_ok, "answerable_test_n": int(base["n"]),
                         "b1_accuracy": float(base["accuracy"]),
                         "selected_accuracy": float(selected_row["accuracy"]),
                         "selected_retrieval_calls": float(selected_row["retrieval_call_rate"]),
                         "call_reduction": 1 - float(selected_row["retrieval_call_rate"])},
        "H2": {"sae_minus_dense_validation_macro_f1": h2_val_gap,
               "sae_minus_dense_answerable_test_macro_f1": h2_test_gap,
               "within_2_points_on_test": h2_near,
               "note": "SAE and dense selected independently on validation; a matched-layer comparison is also useful."},
        "H3": {"lower_wrong_answer_rate_on_hole_questions": bool(h3_pass), "summary": h3_aggs,
               "note": "Wrong per all hole questions includes abstentions as safe; report coverage and conditional wrong rate too."},
        "gate3_and_H4": {"selected_controller_timed": h4_timed, "h4_pass": h4_pass,
                         "existing_latency_file": str(routed_file.relative_to(repo)) if routed_file.exists() else None},
        "availability": source,
        "limitations": ["Gold passages mostly retrieval-derived stand-ins, not independently verified gold",
                        "Index holes are query-time lexical filtering of BM25 top-100, not physical corpus deletion",
                        "SelfAware all-ABSTAIN: raw end-to-end answer accuracy is zero by metric definition; use abstention recall",
                        "Handcheck n=200 point estimate does not rule out a true rate above 10%"],
    }

    lines = ["# SAE-Gated RAG: final results audit", "",
             "Generated from exported results. This is an audit, not a substitute for the full paper.", "",
             "## Locked experimental choices", "",
             f"- Primary label rule: `{PRIMARY}`.",
             f"- SAE model: `{selected['sae']}`; dense comparator: `{selected['dense']}`.",
             "- Both chosen by **full-budget macro-F1 on NQ/TriviaQA validation only**, not test results.", "",
             "## Gate summary", "",
             "| Requirement | Result | Interpretation |", "|---|---|---|",
             f"| Gate 0: ≥15% each class | {'PASS' if gate0 else 'UNVERIFIED/FAIL'} | " +
             ", ".join(f"{x} {f_pct(pilot.get(x))}" for x in ("ANSWER", "RETRIEVE", "ABSTAIN")) + f"; {pilot_source} |",
             f"| Gate 1: <10% label error | {'PASS by point estimate' if gate1 else 'UNVERIFIED/FAIL'} | " +
             (f"{strict_k}/{strict_n} = {f_pct(strict_k / strict_n)}; Wilson 95% CI "
              f"[{f_pct(ci[0])}, {f_pct(ci[1])}]" if strict_n else "Handcheck data unavailable") + " |",
             f"| Gate 2 / H1: accuracy within 2 points and ≥30% fewer BM25 calls | {'PASS' if gate2_ok else 'FAIL'} | " +
             f"B1 {f_pct(base.accuracy)} vs SAE {f_pct(selected_row.accuracy)}, " +
             f"BM25 calls {f_pct(selected_row.retrieval_call_rate)} (reduction {f_pct(1-selected_row.retrieval_call_rate)}) on {int(base.n)} answerable questions |",
             f"| Gate 3: latency of all chosen methods | {'Selected SAE timed' if h4_timed else 'PARTIAL'} | " +
             (f"SAE {lat_rows[primary].p50_ms:.1f} ms; B1 {lat_rows[SEED_ROUTER].p50_ms:.1f} ms" if h4_timed
              else "Current routed latency measures layer 17, not the validation-selected SAE layer") + " |",
             "", "## H2: SAE versus dense", "",
             f"Validation macro-F1: SAE {f_pct(val[selected['sae']])}, dense {f_pct(val[selected['dense']])} "+
             f"(SAE − dense {h2_val_gap*100:+.2f} points).",
             f"Answerable-test mean per-dataset macro-F1: SAE {f_pct(comp_mean[selected['sae']])}, " +
             f"dense {f_pct(comp_mean[selected['dense']])} (SAE − dense {h2_test_gap*100:+.2f} points).",
             "Selection used validation; test estimates are evaluation only. This is a comparison of independently "
             "selected layers, not a paired same-layer feature ablation.", "",
             "## H3: index-hole unsafe answers", "",
             "Wrong answers per total hole questions (weighted across NQ, TriviaQA and HotpotQA):",
             f"- Three-way SAE: {f_pct(h3_aggs[primary]['wrong_per_hole_question'])}, "
             f"coverage {f_pct(h3_aggs[primary]['coverage'])}, "
             f"wrong among attempted {f_pct(h3_aggs[primary]['wrong_among_attempted'])}.",
             f"- Two-way SAE: {f_pct(h3_aggs[primary+'__2way']['wrong_per_hole_question'])}, "
             f"coverage {f_pct(h3_aggs[primary+'__2way']['coverage'])}, "
             f"wrong among attempted {f_pct(h3_aggs[primary+'__2way']['wrong_among_attempted'])}.",
             "The per-question unsafe-answer rate includes abstentions as safe; the separate conditional "
             "error rate above uses only attempted answers. Report both, along with coverage.", "",
             "## Existing latency evidence", "",
    ]
    if len(routed):
        for rid in (SEED_ROUTER, "b4_verbalised", "b8_tfidf_lr", f"{TAG}__{PRIMARY}__sae16k_last@L17__l2",
                    f"{TAG}__{PRIMARY}__resid_last@L17__l2", primary):
            if rid in lat_rows:
                rr = lat_rows[rid]
                lines.append(f"- `{rid}`: p50={rr.p50_ms:.1f} ms, p95={rr.p95_ms:.1f} ms, "
                             f"BM25 calls={f_pct(rr.retrieval_calls)} (n={int(rr.n)}).")
    lines += ["", "## Regression and diagnostic evidence", ""]
    if anomaly is not None:
        a = anomaly[(anomaly.labels == PRIMARY) & (anomaly.model == "sae16k_last@L17")]
        for b in (1000, 2000, 4000):
            sub = a[a.budget == b]
            if not sub.empty:
                r = sub.iloc[0]
                lines.append(f"- SAE L17, {b} labels: macro-F1={r.macro_f1:.3f}, "
                             f"inner-CV={r.inner_cv_f1:.3f}, seeds={int(r.seeds)}.")
        lines.append("The original 1,000→2,000 degradation is not reproduced in these later seeds; "
                     "the original cause remains unverified.")
    lines += ["", "## Audit limitations and open items", "",
              "1. Re-run `scripts/analysis/bundle_analysis.py` on the machine holding raw router "
              "probabilities: earlier paired bootstrap confidence intervals were invalid because duplicate "
              "bootstrap sample IDs were collapsed by dictionaries.",
              "2. Time the **selected SAE layer** at batch size 1 before claiming H4 for the locked primary router.",
              "3. The authoritative handcheck is `sheet_strict.csv` / `flagged_strict.json`. The lenient "
              "adjudication does not pass Gate 1; report that sensitivity and the confidence interval.",
              "4. Explicitly disclose lexical top-100 index holes and retrieval-derived gold stand-ins.",
              "5. Include reproducibility metadata: Gemma checkpoint, SAE site/width/L0, corpus revision, "
              "data splits, seeds, GPU, decoding and matcher revision.",
              "6. Verify availability of raw outputs (`data/`, `runs/`) on SageMaker. The repository ZIP "
              "only contains the smaller exported summaries.", "",
              "## Handoff readiness", "",
              f"Raw data directory included in snapshot: {source['raw_data_present']}; router probabilities included: "
              f"{source['router_probabilities_present']}. Neither is needed to inspect exported CSV results, "
              "but both are needed to fully reproduce experiments and recompute bootstrap intervals.", ""]
    main_report = "\n".join(lines) + "\n"
    draft = "\n".join([
        "# SAE-Gated RAG: Deciding to Answer, Retrieve, or Abstain Before Generation",
        "", "## Abstract (draft)",
        "We study a three-way pre-generation controller using frozen Gemma 3 4B activations and "
        "Gemma Scope 2 sparse autoencoder codes, comparing against a dense probe on the same model. "
        "On 2018 Wikipedia retrieval over 21,015,324 passages, questions from Natural Questions, TriviaQA, "
        "PopQA, HotpotQA and SelfAware are evaluated under closed-book, BM25 and diagnostic gold conditions. "
        "The primary label rule is `miss_as_retrieve`. The quantitative conclusions below are computed from "
        "the saved score exports, not a new training run.",
        "", "## 1. Research questions", "",
        "H1: Accuracy within two points of always-retrieve with at least 30% fewer retrieval calls. "
        "H2: SAE versus dense macro-F1 at full data and in data-limited/transfer settings. "
        "H3: Three-way abstention reduces unsafe answers on lexical index holes. "
        "H4: Controller-based latency compared with always-retrieve and generation-time triggers.",
        "", "## 2. Data and protocol", "",
        "Question-only pre-generation activations are taken from frozen Gemma 3 4B. Dense probes "
        "operate on residual-stream vectors; SAE probes use pretrained Gemma Scope 2 features. "
        "Models are trained on NQ and TriviaQA training sets, with layer selection on their "
        "validation splits. The final test set consists of 1,000 questions each from NQ, TriviaQA, "
        "PopQA and HotpotQA, plus 500 unanswerable SelfAware questions.",
        "", "## 3. Experimental implementation", "",
        "The 2018 DPR-style Wikipedia corpus has 21,015,324 100-word passages. "
        "BM25 returns top three. Lexical index-hole cases are simulated by removing detected "
        "answer-bearing passages from the top 100 BM25 candidates at query time. "
        "This is not physical document deletion. Gold passages are often BM25-derived stand-ins, "
        "not independently annotated supporting documents. The primary three-way label "
        "rule assigns ANSWER to a correct closed-book answer, RETRIEVE to a correct BM25 "
        "answer or a gold-correct retriever miss, and ABSTAIN otherwise, with SelfAware "
        "marked unanswerable.",
        "", "## 4. Results", "", main_report.split("## Gate summary", 1)[1].split("## Audit limitations", 1)[0].join(["## Gate summary", ""]),
        "## 5. Threats to validity", "",
        "String matching in question answering has false positives and false negatives; "
        "the strict human handcheck estimates an error rate but its uncertainty should be retained. "
        "Index-hole filtering is lexical, gold evidence is a stand-in, selected models are "
        "validation-chosen, and the 200 timed questions may not represent all workloads. "
        "Raw end-to-end answer accuracy on SelfAware is zero by metric definition; "
        "abstention recall is the appropriate SelfAware measure.",
        "", "## 6. Conclusion", "",
        "Use the gate-by-gate evidence above to state supported and unsupported hypotheses. "
        "Do not claim universal SAE superiority or faster latency at an unmeasured layer.",
        "", "## References", "",
        "Reuse the vetted bibliography from `report/report_mid.typ` and add precise citations "
        "for Gemma 3, Gemma Scope 2, the datasets, BM25 and the probe methodology.", ""])
    return gates, main_report, draft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, help="relative to repository root; default results/final")
    args = parser.parse_args()
    repo = args.root.resolve()
    dest = (repo / args.out if args.out else repo / "results/final")
    dest.mkdir(parents=True, exist_ok=True)
    scores = _scores(repo)
    summary = _probe_summary(repo)
    selected, selection = select_models(summary)
    candidate_ids = [SEED_ROUTER, "b2_never_retrieve", "b4_verbalised", "b8_tfidf_lr",
                     _router_id(selected["sae"]), _router_id(selected["dense"])]
    agg = pd.DataFrame([aggregate(scores, rid) for rid in candidate_ids])
    base = agg[agg.router == SEED_ROUTER].iloc[0]
    agg["accuracy_delta_vs_b1"] = agg.accuracy - base.accuracy
    agg["relative_retrieval_reduction"] = 1 - agg.retrieval_call_rate
    agg["H1_pass"] = (agg.accuracy_delta_vs_b1 >= -.02) & (agg.relative_retrieval_reduction >= .30)
    g2_rows = []
    for file, df in scores.items():
        if file == "selfaware_test":
            continue
        base_row = _all_row(df, SEED_ROUTER)
        for rid in candidate_ids:
            r = _all_row(df, rid)
            g2_rows.append({"dataset": file, "router": rid, "n": int(r.n),
                            "accuracy": float(r.accuracy), "retrieval_calls": float(r.retrieval_call_rate),
                            "b1_accuracy": float(base_row.accuracy),
                            "delta_accuracy": float(r.accuracy - base_row.accuracy),
                            "H1_pass": bool(r.accuracy >= base_row.accuracy - .02 and
                                            r.retrieval_call_rate <= .70)})
    gate2_by_dataset = pd.DataFrame(g2_rows)
    h2 = _h2(summary, selected)
    h3 = _h3(scores, _router_id(selected["sae"]))
    pil, pil_src = _pilot(repo)
    strict, lenient = _handcheck(repo)
    anomaly = _anomaly(repo)
    presence = _check_handoff(repo)
    gates, report, draft = _report(repo, dest, selected, selection, scores, agg, h2, h3,
                                   pil, pil_src, strict, lenient, anomaly, presence)
    (dest / "gate_summary.json").write_text(json.dumps(gates, indent=2, allow_nan=False) + "\n")
    (dest / "final_summary.md").write_text(report)
    (dest / "report_draft.md").write_text(draft)
    for name, df in (("gate2_aggregate.csv", agg), ("gate2_by_dataset.csv", gate2_by_dataset),
                     ("model_selection.csv", selection), ("h2_label_efficiency.csv", h2),
                     ("h3_hole_risk.csv", h3)):
        df.to_csv(dest / name, index=False)
    if anomaly is not None:
        anomaly.to_csv(dest / "anomaly_summary.csv", index=False)
    print("Primary:", selected)
    for label, obj in gates.items():
        if label.startswith("gate") or label in ("H2", "H3"):
            print(label, json.dumps(obj, ensure_ascii=False)[:250])
    print("WROTE", dest)


if __name__ == "__main__":
    main()

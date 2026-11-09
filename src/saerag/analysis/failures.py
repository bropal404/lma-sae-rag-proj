"""Failure analysis for the report. Owner: Person 2. CPU only.

    uv run python -m saerag.analysis.failures [--variant miss_as_retrieve] [--probe-dir DIR ...]

Sections, all on the test splits:
  1. failure types: why each question gets its label (retrieval fixes it, the reader ignores a
     passage that has the answer, the retriever misses, the model fails even with gold, ...)
  2. retrieval that hurts: closed-book right, BM25-RAG wrong
  3. PopQA popularity bins (s_pop quartiles; note the split keeps only s_pop < 100)
  4. HotpotQA by level and question type
  5. answer-matching near misses (token F1 >= 0.5 but not counted correct): how many labels
     would change if they counted, an automatic stand-in for label noise next to Gate 1
  6. with --probe-dir (runs/routers/{probe_id}, from probes/train.py): probe confusion matrix,
     error rate per failure type, hole and popularity bin, and a sample of misroutes
Writes runs/analysis/{tag}/failures.md, one CSV per table, figs/*.png and
examples_for_handcheck.csv (empty hand_category / notes columns to fill in by hand).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import load_config, path
from ..labels import label_rule
from ..schema import ACTIONS
from .frame import TEST_FILES, add_probe, available_files, load_all, shares
from .plots import grouped_bars, save
from .tables import md_table

FAILURE_TYPES = {
    "known, RAG keeps it": "closed right, BM25-RAG right",
    "RAG hurts": "closed right, BM25-RAG wrong",
    "RAG fixes": "closed wrong, BM25-RAG right",
    "reader failure": "closed and RAG wrong, but the BM25 passages contain the answer",
    "retriever miss": "closed and RAG wrong, passages lack the answer, gold passage works",
    "beyond model": "wrong even with the gold passage",
    "no gold passage": "closed and RAG wrong, passages lack the answer, no gold stand-in",
    "unanswerable": "Self-Aware: no answer exists",
}
NEAR_MISS_F1 = 0.5


def failure_type(r) -> str:
    if r.unanswerable:
        return "unanswerable"
    closed, rag = bool(r.ok_closed), bool(r.ok_bm25)
    if closed:
        return "known, RAG keeps it" if rag else "RAG hurts"
    if rag:
        return "RAG fixes"
    if pd.notna(r.bm25_has_answer) and bool(r.bm25_has_answer):
        return "reader failure"
    if pd.isna(r.ok_gold):
        return "no gold passage"
    return "retriever miss" if r.ok_gold else "beyond model"


def bootstrap_ci(x: np.ndarray, rng, n: int = 2000) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    means = rng.choice(x, size=(n, len(x))).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def type_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all test", df)] + [(d, g) for d, g in df.groupby("dataset", sort=False)]
    groups += [(f"{d} hole={h}", g) for (d, h), g in df.groupby(["dataset", "hole"], sort=False)
               if df[df.dataset == d].hole.any()]
    for name, g in groups:
        c = g.ftype.value_counts(normalize=True)
        rows.append({"subset": name, "n": len(g), **{t: float(c.get(t, 0.0)) for t in FAILURE_TYPES}})
    t = pd.DataFrame(rows)
    return t.loc[:, (t != 0).any(axis=0)]


def hurt_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, h), g in df[~df.unanswerable].groupby(["dataset", "hole"], sort=False):
        known = g[g.ok_closed.astype(bool)]
        hurt = known[~known.ok_bm25.astype(bool)]
        rows.append({"dataset": d, "hole": h, "n": len(g), "closed right": len(known),
                     "RAG hurts": len(hurt), "hurt share of known": len(hurt) / len(known) if len(known) else np.nan,
                     "hurt with answer in passages": int((hurt.bm25_has_answer == True).sum())})
    return pd.DataFrame(rows)


def popqa_table(df: pd.DataFrame, variant: str, rng) -> tuple[pd.DataFrame, float]:
    p = df[(df.dataset == "popqa") & df.s_pop.notna()].copy()
    if p.empty:
        return pd.DataFrame(), np.nan
    p["bin"] = pd.qcut(p.s_pop.astype(float), 4, labels=["Q1 (rarest)", "Q2", "Q3", "Q4 (most viewed)"])
    rows = []
    for b, g in p.groupby("bin", observed=True):
        row = {"bin": str(b), "s_pop range": f"{int(g.s_pop.min())}-{int(g.s_pop.max())}", "n": len(g)}
        for c in ("closed", "bm25", "gold"):
            ok = g[f"ok_{c}"].dropna().astype(float).to_numpy()
            lo, hi = bootstrap_ci(ok, rng)
            row.update({c: ok.mean() if len(ok) else np.nan, f"{c} lo": lo, f"{c} hi": hi})
        row.update(shares(g[f"label_{variant}"]))
        rows.append(row)
    rho = p.s_pop.astype(float).corr(p.ok_closed.astype(float), method="spearman")
    return pd.DataFrame(rows), rho


def popqa_figure(t: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    conds = ("closed", "bm25", "gold")
    grouped_bars(ax, list(t["bin"]), {c: list(t[c]) for c in conds},
                 errors={c: list(zip(t[f"{c} lo"], t[f"{c} hi"])) for c in conds})
    ax.set_ylabel("accuracy")
    ax.set_xlabel("PopQA popularity quartile (s_pop, monthly page views; split keeps s_pop < 100)")
    ax.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, 1.18), fontsize=8)
    save(fig, out)


def hole_figure(df: pd.DataFrame, out: Path) -> None:
    ds = [d for d in ("nq", "triviaqa", "hotpotqa") if d in set(df.dataset)]
    groups, series = [], {c: [] for c in ("closed", "bm25", "gold")}
    for d in ds:
        for h in (False, True):
            g = df[(df.dataset == d) & (df.hole == h)]
            groups.append(f"{d}\n{'hole' if h else 'non-hole'}")
            for c, vals in series.items():
                ok = g[f"ok_{c}"].dropna().astype(float)
                vals.append(ok.mean() if len(ok) else np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    grouped_bars(ax, groups, series)
    ax.set_ylabel("accuracy")
    ax.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, 1.18), fontsize=8)
    save(fig, out)


def hotpot_table(df: pd.DataFrame) -> pd.DataFrame:
    h = df[df.dataset == "hotpotqa"]
    rows = []
    for col in ("level", "qtype"):
        for v, g in h.groupby(col):
            rows.append({"by": col, "value": v, "n": len(g),
                         **{c: g[f"ok_{c}"].dropna().astype(float).mean() for c in ("closed", "bm25", "gold")}})
    return pd.DataFrame(rows)


def near_miss_table(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Relabel with near misses counted correct and count the labels that change."""
    rows = []
    for f, g in df.groupby("file", sort=False):
        loose = {}
        for c in ("closed", "bm25", "gold"):
            ok = g[f"ok_{c}"]
            loose[c] = ok.where(ok.isna(), ok.fillna(False) | (g[f"f1_{c}"] >= NEAR_MISS_F1))
        new = [label_rule(bool(a), bool(b), None if pd.isna(c) else bool(c), u, variant)[0]
               for a, b, c, u in zip(loose["closed"], loose["bm25"], loose["gold"], g.unanswerable)]
        changed = (np.array(new) != g[f"label_{variant}"].to_numpy()).mean()
        row = {"split": f, "n": len(g)}
        for c in ("closed", "bm25", "gold"):
            near = (g[f"ok_{c}"] == False) & (g[f"f1_{c}"] >= NEAR_MISS_F1)
            row[f"near misses {c}"] = int(near.sum())
        row["labels that would change"] = changed
        rows.append(row)
    return pd.DataFrame(rows)


def probe_tables(df: pd.DataFrame, variant: str) -> dict[str, pd.DataFrame]:
    p = df[df.probe_pred.notna()].copy()
    if p.empty:
        return {}
    y = p[f"label_{variant}"]
    p["probe_right"] = p.probe_pred == y
    conf = pd.crosstab(y, p.probe_pred, rownames=["label"], colnames=["probe"]).reindex(
        index=list(ACTIONS), columns=list(ACTIONS), fill_value=0).reset_index()
    by_type = (p.groupby("ftype").probe_right.agg(["size", "mean"])
               .rename(columns={"size": "n", "mean": "probe accuracy"}).reset_index())
    by_ds = (p.groupby(["dataset", "hole"]).probe_right.agg(["size", "mean"])
             .rename(columns={"size": "n", "mean": "probe accuracy"}).reset_index())
    costly = pd.DataFrame([{
        "wrong answer given (pred ANSWER, closed wrong)":
            ((p.probe_pred == "ANSWER") & ~p.ok_closed.astype(bool)).sum() / len(p),
        "false refusal (pred ABSTAIN, label ANSWER/RETRIEVE)":
            ((p.probe_pred == "ABSTAIN") & (y != "ABSTAIN")).sum() / len(p),
        "wasted retrieval (pred RETRIEVE, label ANSWER)":
            ((p.probe_pred == "RETRIEVE") & (y == "ANSWER")).sum() / len(p),
        "n": len(p)}])
    return {"probe_confusion": conf, "probe_by_type": by_type, "probe_by_dataset": by_ds,
            "probe_costly": costly}


def sample_examples(df: pd.DataFrame, variant: str, k: int, seed: int) -> pd.DataFrame:
    parts = []
    for t, g in df[df.ftype != "unanswerable"].groupby("ftype"):
        parts.append(g.sample(min(k, len(g)), random_state=seed))
    if "probe_pred" in df:
        wrong = df[df.probe_pred.notna() & (df.probe_pred != df[f"label_{variant}"])]
        parts.append(wrong.sample(min(2 * k, len(wrong)), random_state=seed).assign(ftype="probe misroute"))
    s = pd.concat(parts)
    cols = ["qid", "dataset", "hole", "s_pop", "ftype", f"label_{variant}", "question", "answers",
            "out_closed", "out_bm25", "out_gold", "bm25_has_answer", "bm25_titles"]
    cols += [c for c in ("probe_pred", "probe_conf") if c in s]
    s = s[cols].copy()
    s["answers"] = s.answers.map(lambda a: " | ".join(a[:5]))
    s["hand_category"], s["notes"] = "", ""
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--variant", default="miss_as_retrieve")
    ap.add_argument("--probe-dir", nargs="*", default=[],
                    help="runs/routers/{probe_id} directories with probs_{file}.jsonl")
    ap.add_argument("--examples", type=int, default=12, help="hand-check examples per failure type")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["seed"])
    out = path(cfg, "results", "analysis", cfg["model"]["tag"])
    out.mkdir(parents=True, exist_ok=True)

    df = load_all(cfg, available_files(cfg, TEST_FILES))
    df["ftype"] = [failure_type(r) for r in df.itertuples()]

    tables: dict[str, tuple[str, pd.DataFrame]] = {
        "failure_types": ("Failure types (share of questions)", type_table(df)),
        "rag_hurts": ("When retrieval hurts (closed right, BM25-RAG wrong)", hurt_table(df)),
    }
    pop, rho = popqa_table(df, args.variant, rng)
    if not pop.empty:
        title = ("PopQA popularity quartiles (Spearman s_pop vs closed-book correct: "
                 f"{rho:+.3f}; 95% bootstrap CIs)")
        tables["popqa_bins"] = (title, pop)
        popqa_figure(pop, out / "figs" / "popqa_bins.png")
    tables["hotpotqa"] = ("HotpotQA by level and question type", hotpot_table(df))
    title = (f"Answer-matching near misses (token F1 >= {NEAR_MISS_F1}, not counted correct) "
             f"and the {args.variant} labels they would change")
    tables["near_misses"] = (title,
                             near_miss_table(df, args.variant))
    for d in args.probe_dir:
        name = Path(d).name
        dfp = pd.concat([add_probe(df[df.file == f], Path(d), f) for f in df.file.unique()])
        if "probe_pred" not in dfp or dfp.probe_pred.isna().all():
            print(f"no probs_<file>.jsonl for the test files in {d}, skipped")
            continue
        for k, t in probe_tables(dfp, args.variant).items():
            tables[f"{k}__{name}"] = (f"{k.replace('_', ' ')}: {name}", t)
        df = dfp  # the last probe's misroutes go into the example sheet
    hole_figure(df, out / "figs" / "accuracy_hole.png")

    lines = [f"# Failure analysis ({cfg['model']['tag']}, rule: {args.variant})", "",
             "Failure types:", ""] + [f"- **{k}**: {v}" for k, v in FAILURE_TYPES.items()] + [""]
    for key, (title, t) in tables.items():
        t.round(4).to_csv(out / f"{key}.csv", index=False)
        lines += [f"## {title}", "", *md_table(t), "", f"CSV: `{key}.csv`", ""]
    lines += ["![accuracy by hole](figs/accuracy_hole.png)", ""]
    if not pop.empty:
        lines += ["![PopQA bins](figs/popqa_bins.png)", ""]
    ex = sample_examples(df, args.variant, args.examples, cfg["seed"])
    ex.to_csv(out / "examples_for_handcheck.csv", index=False)
    lines += [f"{len(ex)} examples for hand categorisation: `examples_for_handcheck.csv`", ""]
    (out / "failures.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {out}/failures.md")


if __name__ == "__main__":
    main()

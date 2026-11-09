"""Tables for the report sections on data, the label rule and Gate 0. Owner: Person 2. CPU only.

    uv run python -m saerag.analysis.tables [--variant miss_as_retrieve]

Reads questions, retrieval, generations and both label variants; writes
runs/analysis/{tag}/tables.md, one CSV per table (Typst reads them with csv()) and
figs/label_shares.png. --variant is the rule the project uses; the other rule is reported
beside it.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from ..config import load_config, path
from ..schema import ACTIONS
from .frame import TEST_FILES, available_files, load_all, shares
from .plots import grouped_bars, save


def pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.1%}"


def md_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(pct(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return rows


def data_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per split: size, holes, unanswerable, gold stand-in coverage, BM25 top-k answer recall."""
    rows = []
    for file, g in df.groupby("file", sort=False):
        nh, h = g[~g.hole], g[g.hole]
        rows.append({
            "split": file, "n": len(g), "hole": g.hole.mean(), "unanswerable": g.unanswerable.mean(),
            "gold passage": g.gold_available.mean(),
            "BM25 has answer (non-hole)": nh.bm25_has_answer.astype(float).mean() if len(nh) else float("nan"),
            "BM25 has answer (hole)": h.bm25_has_answer.astype(float).mean() if len(h) else float("nan"),
        })
    return pd.DataFrame(rows)


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for file, g in df.groupby("file", sort=False):
        row = {"split": file}
        for c in ("closed", "bm25", "gold"):
            ok = g[f"ok_{c}"].dropna().astype(float)
            row[c] = ok.mean() if len(ok) else float("nan")
            row[f"n {c}"] = len(ok)
        rows.append(row)
    return pd.DataFrame(rows)


def label_table(df: pd.DataFrame, variants: list[str], gate: float) -> pd.DataFrame:
    """Class shares per split and rule, with the classes under the gate named."""
    rows = []
    for file, g in df.groupby("file", sort=False):
        for v in variants:
            s = shares(g[f"label_{v}"])
            low = [a for a in ACTIONS if s[a] < gate]
            rows.append({"split": file, "rule": v, "n": int(g[f"label_{v}"].notna().sum()), **s,
                         f"under {gate:.0%}": ", ".join(low) or "-"})
    return pd.DataFrame(rows)


def rule_change_table(df: pd.DataFrame) -> pd.DataFrame:
    """How the second rule moves questions: only default ABSTAIN (retriever_miss) can change."""
    rows = []
    for file, g in df.groupby("file", sort=False):
        moved = (g.label_default != g.label_miss_as_retrieve).sum()
        reasons = g.reason_default.value_counts()
        rows.append({"split": file, "n": len(g), "moved to RETRIEVE": int(moved),
                     "moved share": moved / len(g),
                     "ABSTAIN beyond_model": int(reasons.get("beyond_model", 0)),
                     "ABSTAIN retriever_miss": int(reasons.get("retriever_miss", 0)),
                     "ABSTAIN unanswerable": int(reasons.get("unanswerable", 0))})
    return pd.DataFrame(rows)


def gate0_table(df: pd.DataFrame, variants: list[str], gate: float) -> pd.DataFrame:
    """Gate 0 on the pilot and on the probe training set, overall and per dataset."""
    rows = []
    sets = {"pilot": df[df.file == "pilot"],
            "train (nq+triviaqa)": df[df.file.isin(["nq_train", "triviaqa_train"])]}
    for name, g in sets.items():
        if g.empty:
            continue
        for sub, gg in [("all", g)] + list(g.groupby("dataset")):
            for v in variants:
                s = shares(gg[f"label_{v}"])
                rows.append({"set": name, "subset": sub, "rule": v, "n": len(gg), **s,
                             "gate": "PASS" if min(s.values()) >= gate else "FAIL"})
    return pd.DataFrame(rows)


def label_figure(df: pd.DataFrame, variants: list[str], gate: float, out) -> None:
    test = df[df.file.isin(TEST_FILES) & ~df.unanswerable]
    groups = [f for f in TEST_FILES if f in set(test.file) and f != "selfaware_test"]
    fig, axes = plt.subplots(1, len(variants), figsize=(3.4 * len(variants), 2.6), sharey=True)
    axes = [axes] if len(variants) == 1 else axes
    for ax, v in zip(axes, variants):
        per = {f: shares(test[test.file == f][f"label_{v}"]) for f in groups}
        grouped_bars(ax, [f.replace("_test", "") for f in groups],
                     {a: [per[f][a] for f in groups] for a in ACTIONS})
        ax.axhline(gate, color="#52514e", linewidth=0.8, linestyle="--")
        ax.set_title(f"rule: {v}", fontsize=9)
    axes[0].set_ylabel("share of questions")
    handles, names = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#52514e", linewidth=0.8, linestyle="--"))
    names.append(f"Gate 0 ({gate:.0%})")
    axes[0].legend(handles, names, ncols=4, loc="upper left", bbox_to_anchor=(0, 1.22), fontsize=8)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--variant", default="miss_as_retrieve", help="the label rule the project uses")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gate = cfg["pilot"]["gate_min_class_share"]
    variants = [args.variant] + [v for v in cfg["labels"]["variants"] if v != args.variant]
    out = path(cfg, "results", "analysis", cfg["model"]["tag"])
    out.mkdir(parents=True, exist_ok=True)

    df = load_all(cfg, available_files(cfg))
    tables = {
        "data": ("Data: splits, holes and retrieval coverage", data_table(df)),
        "accuracy": ("Accuracy per condition (alias containment)", accuracy_table(df)),
        "labels": (f"Label shares under both rules (main rule: {args.variant})",
                   label_table(df, variants, gate)),
        "rule_change": ("What the rule change moves", rule_change_table(df)),
        "gate0": (f"Gate 0: each class >= {gate:.0%}", gate0_table(df, variants, gate)),
    }
    lines = [f"# Data, label rule and Gate 0 ({cfg['model']['tag']})", ""]
    for key, (title, t) in tables.items():
        t.round(4).to_csv(out / f"{key}.csv", index=False)
        lines += [f"## {title}", "", *md_table(t), "", f"CSV: `{key}.csv`", ""]
    label_figure(df, variants, gate, out / "figs" / "label_shares.png")
    lines += ["![label shares](figs/label_shares.png)", ""]
    (out / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {out}/tables.md")


if __name__ == "__main__":
    main()

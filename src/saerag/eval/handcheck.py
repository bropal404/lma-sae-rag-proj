"""The 200-label hand check (Gate 1: label error < 10%). Owner: Person 4.

    uv run python -m saerag.eval.handcheck sample --files nq_train triviaqa_train popqa_test ...
    # fill the verdict_* columns in runs/handcheck/sheet.csv (protocol: docs/HANDCHECK.md)
    uv run python -m saerag.eval.handcheck score

`sample` draws a simple random sample (so the error rate is an unbiased estimate) and
writes one row per question with every output, the aliases and the automatic verdicts.
`score` recomputes each label with the hand verdicts through labels.label_rule and
reports the label error rate, the match error rate, and the flagged qids
(runs/handcheck/flagged.json) so probe results can be reported with and without them.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ..answers import normalize
from ..config import load_config, path
from ..labels import label_rule
from ..schema import Generation, Label, Question, read_jsonl

CONDITIONS = ("closed", "bm25", "gold")
VERDICTS = {"y": True, "n": False}


def short_alias(aliases: list[str], max_chars: int = 3) -> bool:
    """Aliases like "AR" or "PL" that can match by accident (D6 known risk)."""
    return any(0 < len(normalize(a).replace(" ", "")) <= max_chars for a in aliases)


def build_sheet(questions: list[Question], labels: dict[str, Label],
                gens: dict[str, dict[str, Generation]], n: int, seed: int) -> pd.DataFrame:
    pool = sorted(q.qid for q in questions if q.qid in labels)
    picked = set(random.Random(seed).sample(pool, min(n, len(pool))))
    rows = []
    for q in questions:
        if q.qid not in picked:
            continue
        lab = labels[q.qid]
        row = {"qid": q.qid, "dataset": q.dataset, "hole": q.hole, "unanswerable": q.unanswerable,
               "question": q.question, "aliases": " | ".join(q.answers),
               "short_alias": short_alias(q.answers), "label": lab.label,
               "abstain_reason": lab.abstain_reason}
        auto = {"closed": lab.closed_correct, "bm25": lab.rag_correct, "gold": lab.gold_correct}
        for c in CONDITIONS:
            g = gens[c].get(q.qid)
            row[f"{c}_output"] = g.output if g else ""
            row[f"{c}_auto"] = auto[c]
            row[f"verdict_{c}"] = ""  # y / n; blank = agree with auto
        row["notes"] = ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values("qid").reset_index(drop=True)


def _hand(row, c: str):
    v = str(row[f"verdict_{c}"]).strip().lower()
    auto = row[f"{c}_auto"]
    auto = None if pd.isna(auto) or auto == "" else (str(auto).lower() == "true")
    return VERDICTS.get(v[:1], auto) if v not in ("", "nan") else auto


def score_sheet(df: pd.DataFrame, variant: str = "default") -> dict:
    flagged, match_errors, checks = [], 0, 0
    for _, row in df.iterrows():
        hand = {c: _hand(row, c) for c in CONDITIONS}
        for c in CONDITIONS:
            auto = row[f"{c}_auto"]
            if not (pd.isna(auto) or auto == "") and row[f"{c}_output"] != "":
                checks += 1
                match_errors += hand[c] != (str(auto).lower() == "true")
        new, _ = label_rule(bool(hand["closed"]), bool(hand["bm25"]), hand["gold"],
                            unanswerable=str(row["unanswerable"]).lower() == "true", variant=variant)
        if new != row["label"]:
            flagged.append({"qid": row["qid"], "auto": row["label"], "hand": new})
    n = len(df)
    return {"n": n, "label_errors": len(flagged), "label_error_rate": len(flagged) / n if n else 0,
            "match_checks": checks, "match_errors": match_errors,
            "match_error_rate": match_errors / checks if checks else 0,
            "gate1_pass": n > 0 and len(flagged) / n < 0.10, "flagged": flagged}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--files", nargs="+", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--variant", default="default")
    s.add_argument("--out")
    sc = sub.add_parser("score")
    sc.add_argument("--sheet")
    sc.add_argument("--variant", default="default")
    args = ap.parse_args()
    cfg = load_config(args.config)
    default_sheet = path(cfg, "results", "handcheck", "sheet.csv")

    if args.cmd == "sample":
        questions, labels, gens = [], {}, {c: {} for c in CONDITIONS}
        for f in args.files:
            questions += read_jsonl(path(cfg, "questions", f"{f}.jsonl"), Question)
            labels |= {r.qid: r for r in read_jsonl(path(cfg, "labels", args.variant, f"{f}.jsonl"),
                                                    Label)}
            for c in CONDITIONS:
                p = path(cfg, "generations", c, f"{f}.jsonl")
                if p.exists():
                    gens[c] |= {g.qid: g for g in read_jsonl(p, Generation)}
        df = build_sheet(questions, labels, gens, args.n, cfg["seed"])
        out = args.out or default_sheet
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")  # opens cleanly in Excel / Sheets
        print(f"wrote {len(df)} rows to {out}; {int(df.short_alias.sum())} have short aliases")
        print(df.label.value_counts().to_string())
    else:
        df = pd.read_csv(args.sheet or default_sheet, encoding="utf-8-sig", keep_default_na=False)
        res = score_sheet(df, args.variant)
        out = path(cfg, "results", "handcheck", "flagged.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        print(f"label error {res['label_errors']}/{res['n']} = {res['label_error_rate']:.1%} "
              f"(gate 1 needs < 10%: {'PASS' if res['gate1_pass'] else 'FAIL'}); "
              f"match error {res['match_errors']}/{res['match_checks']}; details -> {out}")


if __name__ == "__main__":
    main()

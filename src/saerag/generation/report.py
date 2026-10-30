"""Gate 0 report: accuracy, class balance and speed for a split. Owner: Person 2.

    uv run python -m saerag.generation.report --files pilot

Reads the generations, labels and {file}.stats.json from earlier runs and writes
runs/reports/{tag}/{file}.md. Exits with status 1 if a class in the default labels is under
pilot.gate_min_class_share.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from ..config import load_config, path
from ..schema import ACTIONS, Generation, Label, Question, read_jsonl


def accuracy_rows(cfg: dict, file: str, qs: dict[str, Question]) -> list[str]:
    rows = ["| condition | n | accuracy | non-hole | hole | sec / 1k questions |", "|---|---|---|---|---|---|"]
    for condition in ("closed", "bm25", "gold", "dense"):
        p = path(cfg, "generations", condition, f"{file}.jsonl")
        if not p.exists():
            continue
        gens = list(read_jsonl(p, Generation))

        def acc(subset: list[Generation]) -> str:
            return f"{sum(g.correct for g in subset) / len(subset):.1%}" if subset else "-"

        s = p.with_suffix(".stats.json")
        speed = json.loads(s.read_text())["sec_per_1k"] if s.exists() else "-"
        rows.append(f"| {condition} | {len(gens)} | {acc(gens)} "
                    f"| {acc([g for g in gens if not qs[g.qid].hole])} "
                    f"| {acc([g for g in gens if qs[g.qid].hole])} | {speed} |")
    return rows


def balance_rows(labels: list[Label], qs: dict[str, Question], gate: float) -> tuple[list[str], bool]:
    rows = ["| subset | n | " + " | ".join(ACTIONS) + " |", "|---|---|---|---|---|"]
    groups = {"all": labels, "non-hole": [x for x in labels if not x.hole],
              "hole": [x for x in labels if x.hole]}
    for ds in sorted({qs[x.qid].dataset for x in labels}):
        groups[ds] = [x for x in labels if qs[x.qid].dataset == ds]
    passed = True
    for name, sub in groups.items():
        if not sub:
            continue
        c = Counter(x.label for x in sub)
        rows.append(f"| {name} | {len(sub)} | " + " | ".join(f"{c[a] / len(sub):.1%}" for a in ACTIONS) + " |")
        if name == "all":
            passed = all(c[a] / len(sub) >= gate for a in ACTIONS)
    return rows, passed


def report(cfg: dict, file: str) -> tuple[str, bool]:
    gate = cfg["pilot"]["gate_min_class_share"]
    qs = {q.qid: q for q in read_jsonl(path(cfg, "questions", f"{file}.jsonl"), Question)}
    lines = [f"# Generation report: {file} ({cfg['model']['tag']})", "", "## Accuracy per condition", ""]
    lines += accuracy_rows(cfg, file, qs)
    passed = True
    for variant in cfg["labels"]["variants"]:
        labels = list(read_jsonl(path(cfg, "labels", variant, f"{file}.jsonl"), Label))
        rows, ok = balance_rows(labels, qs, gate)
        if variant == "default":
            passed = ok
        reasons = Counter(x.abstain_reason for x in labels if x.label == "ABSTAIN")
        lines += ["", f"## Labels: {variant}", ""] + rows + ["", f"Abstain reasons: {dict(reasons)}"]
    lines += ["", f"**Gate (each class >= {gate:.0%} of the default labels): "
                  f"{'PASS' if passed else 'FAIL'}**"]
    return "\n".join(lines) + "\n", passed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    all_passed = True
    for file in args.files:
        text, passed = report(cfg, file)
        out = path(cfg, "results", "reports", cfg["model"]["tag"], f"{file}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(text)
        print(f"wrote {out}")
        all_passed &= passed
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

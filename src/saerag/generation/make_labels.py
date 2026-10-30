"""Join the generation runs into labels. Owner: Person 2.

    uv run python -m saerag.generation.make_labels --files pilot
    uv run python -m saerag.generation.make_labels --files nq_test --rag dense   # shift-test labels

Needs data/generations/{closed,<rag>}/{file}.jsonl; gold is optional (missing -> gold_correct
None). Writes data/labels/{variant}/{file}.jsonl for every variant in labels.VARIANTS (suffixed
_{rag} when rag != bm25) and prints the class balance against the Phase 0 gate.
"""

from __future__ import annotations

import argparse
from collections import Counter

from ..config import load_config, path
from ..labels import VARIANTS, make_label
from ..schema import ACTIONS, Generation, Question, read_jsonl, write_jsonl


def load_gens(cfg: dict, condition: str, file: str, required: bool) -> dict[str, bool]:
    p = path(cfg, "generations", condition, f"{file}.jsonl")
    if not p.exists():
        if required:
            raise FileNotFoundError(f"{p} missing: run generation.run --condition {condition}")
        print(f"note: {p} missing, gold_correct will be None")
        return {}
    return {g.qid: g.correct for g in read_jsonl(p, Generation)}


def balance(labels, gate: float) -> str:
    n = len(labels)
    c = Counter(lab.label for lab in labels)
    parts = [f"{a}={c[a] / n:.1%}" + ("" if c[a] / n >= gate else " (<gate)") for a in ACTIONS]
    return f"n={n}  " + "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--closed", default="closed", help="closed-book generation dir")
    ap.add_argument("--rag", default="bm25", help="retrieval condition used for RETRIEVE")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gate = cfg["pilot"]["gate_min_class_share"]

    for file in args.files:
        qs = list(read_jsonl(path(cfg, "questions", f"{file}.jsonl"), Question))
        closed = load_gens(cfg, args.closed, file, required=True)
        rag = load_gens(cfg, args.rag, file, required=True)
        gold = load_gens(cfg, "gold", file, required=False)

        missing = [q.qid for q in qs if not q.unanswerable and (q.qid not in closed or q.qid not in rag)]
        if missing:
            print(f"{file}: {len(missing)} questions lack a closed or {args.rag} generation, skipped")
        skip = set(missing)
        qs = [q for q in qs if q.unanswerable or q.qid not in skip]

        for variant in VARIANTS:
            labels = [make_label(q.qid, closed.get(q.qid, False), rag.get(q.qid, False),
                                 gold.get(q.qid), unanswerable=q.unanswerable, hole=q.hole,
                                 variant=variant) for q in qs]
            out = variant if args.rag == "bm25" else f"{variant}_{args.rag}"
            write_jsonl(path(cfg, "labels", out, f"{file}.jsonl"), labels)
            print(f"[{file} / {out}] all:      {balance(labels, gate)}")
            for name, sub in [("hole", [x for x in labels if x.hole]),
                              ("non-hole", [x for x in labels if not x.hole])]:
                if sub:
                    print(f"[{file} / {out}] {name:9s} {balance(sub, gate)}")
            reasons = Counter(x.abstain_reason for x in labels if x.label == "ABSTAIN")
            print(f"[{file} / {out}] abstain reasons: {dict(reasons)}")


if __name__ == "__main__":
    main()

"""Recompute `correct` in existing generation files with the current answers.is_correct.

No model is run: only the stored outputs are re-matched. The old files are copied to
data/generations_prev/ first (once), and each *.stats.json gets its accuracy updated.
Run make_labels afterwards to rebuild the labels.

    uv run python scripts/rescore_generations.py
    uv run python -m saerag.generation.make_labels --files pilot nq_train ...
"""

from __future__ import annotations

import json
import shutil
from collections import Counter

from saerag.answers import is_correct
from saerag.config import load_config, path
from saerag.schema import Generation, Question, read_jsonl, write_jsonl


def main() -> None:
    cfg = load_config(None)
    gen_root = path(cfg, "generations")
    backup = gen_root.parent / "generations_prev"
    if not backup.exists():
        shutil.copytree(gen_root, backup)
        print(f"old generations copied to {backup}")

    total = Counter()
    for f in sorted(gen_root.glob("*/*.jsonl")):
        cond, file = f.parent.name, f.stem
        qpath = path(cfg, "questions", f"{file}.jsonl")
        if not qpath.exists():
            print(f"{f}: no question file, skipped")
            continue
        qs = {q.qid: q for q in read_jsonl(qpath, Question)}
        gens = list(read_jsonl(f, Generation))
        changes = Counter()
        for g in gens:
            new = is_correct(g.output, qs[g.qid].answers)
            if new != g.correct:
                changes["F->T" if new else "T->F"] += 1
                g.correct = new
        write_jsonl(f, gens)

        stats_path = f.with_suffix(".stats.json")
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            stats["accuracy"] = round(sum(g.correct for g in gens) / max(1, len(gens)), 4)
            hole = [g for g in gens if qs[g.qid].hole]
            if hole:
                stats["accuracy_hole"] = round(sum(g.correct for g in hole) / len(hole), 4)
            stats["rescored"] = "answers.is_correct v2 (Gate 1 matcher fix)"
            stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        total += changes
        print(f"{cond:7s} {file:15s} n={len(gens):5d}  {dict(changes) or 'no change'}")
    print("total:", dict(total))


if __name__ == "__main__":
    main()

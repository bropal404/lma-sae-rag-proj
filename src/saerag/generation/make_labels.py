"""Join the three generation runs into labels. Owner: Person 2.

    uv run python -m saerag.generation.make_labels --files pilot

Writes data/labels/{variant}/{file}.jsonl for every variant in labels.VARIANTS and
prints the class balance (the Phase 0 gate needs each class >= 15%).
"""

from __future__ import annotations

import argparse

from ..config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    # TODO(P2): read questions + closed/bm25/gold generations by qid, call labels.make_label.
    # Questions with no gold generation get gold_correct=None.
    raise NotImplementedError(cfg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate canonical files on SageMaker/ADA without loading large SAE arrays.

Usage: python scripts/verify_raw_outputs.py --config configs/sagemaker.yaml
Read-only. Requires the machine that has data/labels, data/features and runs/routers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from saerag.config import load_config, path
from saerag.labels import label_rule
from saerag.schema import Question, Label, read_jsonl

EXPECTED = {
    "nq_train": 4000, "nq_val": 500, "nq_test": 1000,
    "triviaqa_train": 4000, "triviaqa_val": 500, "triviaqa_test": 1000,
    "popqa_test": 1000, "hotpotqa_test": 1000, "selfaware_test": 500,
    "pilot": 500,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sagemaker.yaml")
    ap.add_argument("--variant", default="miss_as_retrieve")
    ap.add_argument("--layer", type=int, default=29)
    ap.add_argument("--sae-width", default="16k")
    ap.add_argument("--check-probs", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tag = cfg["model"]["tag"]
    label_count = 0
    for name, expected_n in EXPECTED.items():
        question_file = path(cfg, "questions", name + ".jsonl")
        label_file = path(cfg, "labels", args.variant, name + ".jsonl")
        if not question_file.exists() or not label_file.exists():
            raise FileNotFoundError(f"Missing {question_file} or {label_file}")
        qs = list(read_jsonl(question_file, Question))
        ls = list(read_jsonl(label_file, Label))
        if len(qs) != expected_n or len(ls) != expected_n:
            raise ValueError(f"{name}: questions {len(qs)}, labels {len(ls)}, expected {expected_n}")
        qids = {q.qid for q in qs}
        label_ids = {l.qid for l in ls}
        if len(qids) != len(qs) or label_ids != qids:
            raise ValueError(f"{name}: duplicate or mismatched IDs")
        by_qid = {q.qid: q for q in qs}
        for lab in ls:
            q = by_qid[lab.qid]
            action, reason = label_rule(lab.closed_correct, lab.rag_correct, lab.gold_correct,
                                        unanswerable=q.unanswerable, variant=args.variant)
            if lab.label != action or lab.abstain_reason != reason or lab.hole != q.hole:
                raise ValueError(f"{name}: bad label rule or hole flag at {lab.qid}")
        feat_dir = path(cfg, "features", tag)
        id_file = feat_dir / f"qids_{name}.json"
        if not id_file.exists():
            raise FileNotFoundError(f"Missing feature IDs: {id_file}")
        fq = json.loads(id_file.read_text())
        if len(fq) != expected_n or set(fq) != qids or len(set(fq)) != expected_n:
            raise ValueError(f"Feature qids mismatch for {name}")
        for kind in ("resid_last", f"sae{args.sae_width}_last"):
            ext = "npy" if kind.startswith("resid") else "npz"
            f = feat_dir / f"L{args.layer}" / f"{kind}_{name}.{ext}"
            if not f.is_file() or f.stat().st_size < 128:
                raise FileNotFoundError(f"Missing / empty features {f}")
        if args.check_probs and name not in ("pilot", "nq_train", "triviaqa_train"):
            rid = f"{tag}__{args.variant}__sae{args.sae_width}_last@L{args.layer}__l2"
            p = path(cfg, "results", "routers", rid, f"probs_{name}.jsonl")
            if not p.exists():
                raise FileNotFoundError(f"Missing router: {p}")
            rows = list(read_jsonl(p))
            if len(rows) != expected_n or set(r["qid"] for r in rows) != qids:
                raise ValueError(f"router probabilities mismatch: {name}")
            for r in rows:
                ps = r["p"]
                if set(ps) != {"ANSWER", "RETRIEVE", "ABSTAIN"} or abs(sum(ps.values()) - 1) > 1e-4:
                    raise ValueError(f"bad probabilities in {name}: {r['qid']}")
        label_count += len(ls)
        print(f"PASS {name}: {expected_n} matched questions/labels/feature-qids", flush=True)
    print(f"RAW VALIDATION PASSED: {label_count} records including overlapping 500 pilot")


if __name__ == "__main__":
    main()

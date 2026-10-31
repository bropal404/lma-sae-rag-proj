"""Write probability files for baselines B1-B4 and B8. Owner: Person 4.

    uv run python -m saerag.baselines.run --routers b1 b2 b3 --files pilot
    uv run python -m saerag.baselines.run --routers b8 --files nq_test triviaqa_test \
        --train-files nq_train triviaqa_train --val-files nq_val triviaqa_val
    uv run python -m saerag.baselines.run --routers b4 --files pilot      # GPU node, vLLM
    uv run python -m saerag.baselines.run --routers b4 --files pilot --backend hf   # Kaggle

Reads data/questions/{file}.jsonl and data/labels/{variant}/{file}.jsonl and writes
    runs/routers/{router_id}/probs_{file}.jsonl   {"qid", "p": {...}, "latency_ms"?}
B4 also keeps its raw replies in runs/routers/b4_verbalised/raw_{file}.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time

from ..config import load_config, path
from ..prompts import verbalised_router_messages
from ..schema import Label, Question, read_jsonl, write_jsonl
from . import routers

ROUTER_IDS = {"b1": "b1_always_retrieve", "b2": "b2_never_retrieve", "b3": "b3_oracle",
              "b4": "b4_verbalised", "b8": "b8_tfidf_lr"}


def load_questions(cfg, file: str) -> list[Question]:
    return list(read_jsonl(path(cfg, "questions", f"{file}.jsonl"), Question))


def load_labels(cfg, file: str, variant: str) -> dict[str, Label]:
    return {r.qid: r for r in read_jsonl(path(cfg, "labels", variant, f"{file}.jsonl"), Label)}


def write_probs(cfg, router_id: str, file: str, probs: dict, latency_ms: dict | None = None):
    rows = [{"qid": q, "p": p, **({"latency_ms": latency_ms[q]} if latency_ms else {})}
            for q, p in probs.items()]
    write_jsonl(path(cfg, "results", "routers", router_id, f"probs_{file}.jsonl"), rows)


def _texts_and_labels(cfg, files: list[str], variant: str):
    qs, ys = [], []
    for f in files:
        labels = load_labels(cfg, f, variant)
        for q in load_questions(cfg, f):
            if q.qid in labels:
                qs.append(q.question)
                ys.append(labels[q.qid].label)
    return qs, ys


def run_b4(cfg, files: list[str], backend: str | None = None) -> None:
    """Reuses Person 2's generation backends, so B4 sees the same model setup as the labels."""
    from ..generation.run import generate_hf, generate_vllm
    from ..hf import first_line

    g = {**cfg["generation"], "max_new_tokens": 8}  # one word is enough
    name, backend = cfg["model"]["generator"], backend or cfg["generation"]["backend"]
    if backend == "hf":
        from ..hf import load_model
        tok, model, in_device = load_model(name, g.get("dtype", "auto"))
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
    rid = ROUTER_IDS["b4"]
    for f in files:
        qs = load_questions(cfg, f)
        prompts = [tok.apply_chat_template(verbalised_router_messages(q.question), tokenize=False,
                                           add_generation_prompt=True) for q in qs]
        t0 = time.perf_counter()
        texts = (generate_vllm(name, tok, prompts, g) if backend == "vllm"
                 else generate_hf(model, tok, in_device, prompts, g))
        per_q_ms = (time.perf_counter() - t0) * 1000 / max(len(qs), 1)  # batched; not the p50
        outputs = {q.qid: first_line(t) for q, t in zip(qs, texts, strict=True)}
        probs, fail = routers.verbalised(outputs)
        write_jsonl(path(cfg, "results", "routers", rid, f"raw_{f}.jsonl"),
                    [{"qid": q, "output": o} for q, o in outputs.items()])
        write_probs(cfg, rid, f, probs)
        print(f"b4 {f}: {len(qs)} questions, parse failures {fail:.1%}, "
              f"{per_q_ms:.1f} ms/question batched")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--routers", nargs="+", choices=list(ROUTER_IDS), required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--backend", choices=["vllm", "hf"], help="B4 only; default from config")
    ap.add_argument("--train-files", nargs="+", default=["nq_train", "triviaqa_train"])
    ap.add_argument("--val-files", nargs="+", default=["nq_val", "triviaqa_val"])
    args = ap.parse_args()
    cfg = load_config(args.config)

    for r in args.routers:
        rid = ROUTER_IDS[r]
        if r == "b4":
            run_b4(cfg, args.files, args.backend)
            continue
        if r == "b8":
            tq, ty = _texts_and_labels(cfg, args.train_files, args.variant)
            vq, vy = _texts_and_labels(cfg, args.val_files, args.variant)
            clf = routers.tune_text_classifier(tq, ty, vq, vy, seed=cfg["seed"])
            print(f"b8 C={clf.steps[-1][1].C} trained on {len(tq)} questions")
            print(json.dumps(routers.top_terms(clf, 10), indent=1))
        for f in args.files:
            qs = load_questions(cfg, f)
            qids = [q.qid for q in qs]
            if r == "b1":
                probs = routers.always_retrieve(qids)
            elif r == "b2":
                probs = routers.never_retrieve(qids)
            elif r == "b3":
                probs = routers.oracle({q: l.label for q, l in
                                        load_labels(cfg, f, args.variant).items()})
            else:
                t0 = time.perf_counter()
                probs = routers.predict_probs(clf, qids, [q.question for q in qs])
                ms = (time.perf_counter() - t0) * 1000 / max(len(qs), 1)
                print(f"b8 {f}: {ms:.3f} ms/question")
            write_probs(cfg, rid, f, probs)
            print(f"{rid} {f}: wrote {len(probs)}")


if __name__ == "__main__":
    main()

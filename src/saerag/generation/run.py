"""Generation runs for labelling: closed-book, BM25-RAG, gold (and dense). Owner: Person 2.

    uv run python -m saerag.generation.run --condition closed --files pilot
    uv run python -m saerag.generation.run --condition bm25 gold --files pilot
    uv run python -m saerag.generation.run --condition closed --files pilot --backend hf

Reads data/questions/{file}.jsonl (+ data/retrieval/{condition}/{file}.jsonl for RAG) and
writes data/generations/{condition}/{file}.jsonl as Generation records, scored with
answers.is_correct on the first line of the output. Greedy decoding, chat template.
Gold retrieval is sparse, so questions with no gold record are skipped. BM25 and dense must
cover every question, with hole flags that match the question file (a mismatch means the two
files come from different question builds). Self-Aware questions are generated too (their
label is ABSTAIN regardless). Finished files are skipped on a rerun unless --overwrite is set;
each run also writes {file}.stats.json next to its output for generation.report.

Backends: vllm (default; fast, wants a bf16 GPU) or hf (transformers, works on any GPU
in fp32 across several cards, and on CPU).
"""

from __future__ import annotations

import argparse
import json
import time

from tqdm import tqdm

from ..answers import is_correct
from ..config import load_config, path
from ..hf import first_line, pick_dtype
from ..prompts import closed_book_messages, rag_messages
from ..schema import Generation, Question, Retrieval, read_jsonl, write_jsonl


def build_messages(cfg: dict, condition: str, file: str) -> tuple[list[Question], list[list[dict]]]:
    qs = list(read_jsonl(path(cfg, "questions", f"{file}.jsonl"), Question))
    if condition == "closed":
        return qs, [closed_book_messages(q.question) for q in qs]
    rpath = path(cfg, "retrieval", condition, f"{file}.jsonl")
    ret = {r.qid: r for r in read_jsonl(rpath, Retrieval)}
    kept = [q for q in qs if q.qid in ret]
    if bad := [q.qid for q in kept if ret[q.qid].hole != q.hole]:
        raise ValueError(f"{file}/{condition}: {len(bad)} hole flags disagree with the question "
                         f"file, e.g. {bad[0]}. Questions and retrieval are from different builds.")
    if len(kept) < len(qs):
        if condition != "gold":
            raise ValueError(f"{file}/{condition}: {len(qs) - len(kept)} of {len(qs)} questions "
                             "have no retrieval record. Questions and retrieval are from different builds.")
        print(f"{file}/{condition}: {len(qs) - len(kept)} questions have no retrieval record, skipped")
    k = cfg["retrieval"]["top_k"]
    return kept, [rag_messages(q.question, ret[q.qid].passages[:k]) for q in kept]


def generate_vllm(model_name: str, tok, prompts: list[str], g: dict) -> list[str]:
    import torch
    from vllm import LLM, SamplingParams

    llm = generate_vllm.llm = getattr(generate_vllm, "llm", None) or LLM(
        model=model_name, dtype=str(pick_dtype(g.get("dtype", "auto"))).removeprefix("torch."),
        max_model_len=g["max_model_len"], tensor_parallel_size=max(1, torch.cuda.device_count()),
        gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0}, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=g["max_new_tokens"])
    # Pass token ids: the templated string already contains <bos>.
    ids = [{"prompt_token_ids": tok(p, add_special_tokens=False).input_ids} for p in prompts]
    return [o.outputs[0].text for o in llm.generate(ids, sp)]


def generate_hf(model, tok, in_device, prompts: list[str], g: dict) -> list[str]:
    import torch

    outs = []
    bs = g["batch_size"]
    for i in tqdm(range(0, len(prompts), bs), desc="hf generate"):
        enc = tok(prompts[i : i + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(in_device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=g["max_new_tokens"], do_sample=False)
        outs += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--condition", nargs="+", choices=["closed", "bm25", "gold", "dense"],
                    required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"])
    ap.add_argument("--limit", type=int, help="first N questions only (smoke tests)")
    ap.add_argument("--overwrite", action="store_true", help="redo files that already exist")
    args = ap.parse_args()
    cfg = load_config(args.config)
    g, name = cfg["generation"], cfg["model"]["generator"]
    backend = args.backend or g["backend"]

    if backend == "hf":
        from ..hf import load_model
        tok, model, in_device = load_model(name, g.get("dtype", "auto"))
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)

    for condition in args.condition:
        for file in args.files:
            out = path(cfg, "generations", condition, f"{file}.jsonl")
            if out.exists() and not args.overwrite:
                print(f"{out} exists, skipped")
                continue
            qs, msgs = build_messages(cfg, condition, file)
            qs, msgs = qs[: args.limit], msgs[: args.limit]
            if not qs:                  # e.g. Self-Aware has no gold passages
                print(f"{file}/{condition}: nothing to generate")
                continue
            prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in msgs]
            t0 = time.perf_counter()
            texts = (generate_vllm(name, tok, prompts, g) if backend == "vllm"
                     else generate_hf(model, tok, in_device, prompts, g))
            elapsed = time.perf_counter() - t0
            gens = []
            for q, t in zip(qs, texts):
                t = first_line(t)
                gens.append(Generation(q.qid, condition, t, is_correct(t, q.answers),
                                       latency_ms=1000 * elapsed / len(qs)))
            write_jsonl(out, gens)
            acc = sum(x.correct for x in gens) / max(1, len(gens))
            hole = [x for x, q in zip(gens, qs) if q.hole]
            stats = {"file": file, "condition": condition, "backend": backend, "model": name,
                     "n": len(gens), "accuracy": round(acc, 4),
                     "accuracy_hole": round(sum(x.correct for x in hole) / len(hole), 4)
                     if hole else None,
                     "seconds": round(elapsed, 1),
                     "sec_per_1k": round(1000 * elapsed / max(1, len(gens)), 1)}
            out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
            print(json.dumps(stats))


if __name__ == "__main__":
    main()

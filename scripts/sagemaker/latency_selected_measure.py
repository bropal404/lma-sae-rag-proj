"""Measure the *validation-selected* SAE L29 and dense L22 controller costs.

This is the GPU half of the final Gate 3/H4 experiment. It times each
controller independently (forward pass, optional SAE encoding, CPU transfer,
feature scaling and probe prediction), plus closed-book/RAG/B4 generation.

The probes are re-fitted ONLY on NQ/TriviaQA training records using the
project's original 3-fold inner-CV training routine; routing decisions are
NOT taken from these re-fits. The composed-latency script uses the existing
validation-selected test probabilities and validation-tuned thresholds.

Example:
  PYTHONPATH=src .venv/bin/python scripts/sagemaker/latency_selected_measure.py \\
    --config configs/sagemaker.yaml --preflight
  PYTHONPATH=src .venv/bin/python scripts/sagemaker/latency_selected_measure.py \\
    --config configs/sagemaker.yaml --out-name l29
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from saerag.config import load_config, path
from saerag.schema import Question, Retrieval, read_jsonl


def required_paths(cfg: dict, files: list[str], variant: str, sae_layer: int,
                   dense_layer: int) -> list[Path]:
    """Preflight without importing torch or loading the 4B model."""
    tag = cfg["model"]["tag"]
    feat = path(cfg, "features", tag)
    required = []
    for f in ("nq_train", "triviaqa_train"):
        required += [path(cfg, "questions", f"{f}.jsonl"),
                     path(cfg, "labels", variant, f"{f}.jsonl"),
                     feat / f"qids_{f}.json",
                     feat / f"L{sae_layer}" / f"sae16k_last_{f}.npz",
                     feat / f"L{dense_layer}" / f"resid_last_{f}.npy"]
    tagid = cfg["model"]["tag"]
    sae_router = f"{tagid}__{variant}__sae16k_last@L{sae_layer}__l2"
    dense_router = f"{tagid}__{variant}__resid_last@L{dense_layer}__l2"
    routers = ["b1_always_retrieve", "b2_never_retrieve", "b3_oracle",
               "b4_verbalised", "b8_tfidf_lr", sae_router, dense_router]
    for f in files:
        required += [path(cfg, "questions", f"{f}.jsonl"),
                     path(cfg, "retrieval", "bm25", f"{f}.jsonl")]
        val = "triviaqa_val" if f == "triviaqa_test" else "nq_val"
        for rid in routers:
            required += [path(cfg, "results", "routers", rid, f"probs_{f}.jsonl"),
                         path(cfg, "results", "routers", rid, f"probs_{val}.jsonl")]
    return required


def check_inputs(cfg: dict, files: list[str], variant: str,
                 sae_layer: int, dense_layer: int) -> None:
    needed = required_paths(cfg, files, variant, sae_layer, dense_layer)
    missing = [p for p in needed if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing required SageMaker inputs:\n" +
                                "\n".join(str(p) for p in missing))
    bm25 = path(cfg, "results", "latency", "bm25.jsonl")
    if not bm25.is_file():
        other = Path(__file__).resolve().parents[2] / "results/sagemaker/latency/bm25.jsonl"
        if not other.is_file():
            raise FileNotFoundError(f"Missing BM25 stage timings: {bm25}. "
                                    "Restore the existing bm25.jsonl; don't estimate zeros.")
    print(f"PREFLIGHT OK: {len(needed)} input files, BM25 timings present", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sae-layer", type=int, default=29)
    ap.add_argument("--dense-layer", type=int, default=22)
    ap.add_argument("--variant", default="miss_as_retrieve")
    ap.add_argument("--files", nargs="+", default=["nq_test", "triviaqa_test"])
    ap.add_argument("--n", type=int, default=100, help="per file, matching prior BM25 timing")
    ap.add_argument("--out-name", default="l29")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--cv-jobs", type=int, default=2, help="parallel fits for original probe CV")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    check_inputs(cfg, a.files, a.variant, a.sae_layer, a.dense_layer)
    if a.preflight:
        return
    if a.n <= 0 or a.cv_jobs < 1:
        ap.error("--n and --cv-jobs must be positive")

    import torch
    from saerag.features.sae import JumpReLUSAE, sae_id
    from saerag.hf import chat_prompt, load_model
    from saerag.probes.train import load_block, make_model
    from saerag.prompts import (closed_book_messages, rag_messages,
                                verbalised_router_messages)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for hardware timing; this is not the ADA login node")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("GPU must support native bf16, as in the original L4 run")

    out = path(cfg, "results", "latency")
    out.mkdir(parents=True, exist_ok=True)
    stages_path = out / f"stages_{a.out_name}.jsonl"
    summary_path = out / f"stages_{a.out_name}_summary.csv"
    if stages_path.exists() and not a.overwrite:
        raise FileExistsError(f"Refusing to overwrite {stages_path}; use --overwrite deliberately")

    # Fit the *same class of probe* used in the scored experiments: sparse
    # MaxAbsScaler+LR versus dense StandardScaler+LR, hyperparameters chosen by
    # train-only inner CV. The measured actions come from exported probabilities.
    train = ["nq_train", "triviaqa_train"]
    sae_qids, sae_y, _, sae_blocks = load_block(cfg, train, a.variant,
                                                [a.sae_layer], ["sae16k_last"])
    dense_qids, dense_y, _, dense_blocks = load_block(cfg, train, a.variant,
                                                    [a.dense_layer], ["resid_last"])
    if sae_qids != dense_qids or not np.array_equal(sae_y, dense_y):
        raise ValueError("Dense and SAE probe training rows/labels are not aligned")
    sae_name = f"sae16k_last@L{a.sae_layer}"
    dense_name = f"resid_last@L{a.dense_layer}"
    trained = {}
    for name, X in ((sae_name, sae_blocks[sae_name]),
                    (dense_name, dense_blocks[dense_name])):
        fitted = make_model(name, "l2", False, seed=0)
        fitted.n_jobs = a.cv_jobs
        print(f"Fitting {name} with original train-only inner CV ...", flush=True)
        fitted.fit(X, sae_y)
        trained[name] = fitted
        print(f"Fitted {name}; C={fitted.best_params_['logisticregression__C']}", flush=True)

    tok, model, dev = load_model(cfg["model"]["generator"],
                                 cfg["generation"].get("dtype", "auto"))
    sae = JumpReLUSAE.load(cfg["model"]["sae_repo"],
                           sae_id(cfg["model"]["sae_site"], a.sae_layer,
                                  "16k", cfg["model"]["sae_l0"]), dev, decoder=False)
    k = cfg["retrieval"]["top_k"]
    tokens = cfg["generation"]["max_new_tokens"]

    def inputs(messages):
        return tok(chat_prompt(tok, messages), return_tensors="pt",
                   add_special_tokens=False).to(dev)

    def timed(action):
        torch.cuda.synchronize()
        t = time.perf_counter()
        action()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000.0

    def controller(question: str, kind: str):
        layer = a.sae_layer if kind == "sae" else a.dense_layer
        with torch.inference_mode():
            h = model(**inputs(closed_book_messages(question)),
                      output_hidden_states=True).hidden_states[layer + 1][:, -1]
            if kind == "sae":
                features = sae.encode(h.float()).cpu().numpy()
                trained[sae_name].predict_proba(features)
            else:
                features = h.float().cpu().numpy()
                trained[dense_name].predict_proba(features)

    def generate(messages, max_new_tokens):
        with torch.inference_mode():
            model.generate(**inputs(messages), max_new_tokens=max_new_tokens,
                           do_sample=False)

    rows = []
    for f in a.files:
        qs = list(read_jsonl(path(cfg, "questions", f"{f}.jsonl"), Question))[:a.n]
        passages = {r.qid: r for r in read_jsonl(
            path(cfg, "retrieval", "bm25", f"{f}.jsonl"), Retrieval)}
        if not qs:
            raise ValueError(f"No questions found for {f}")
        missing = [q.qid for q in qs if q.qid not in passages]
        if missing:
            raise ValueError(f"BM25 retrieval records missing for {f}: {missing[:5]}")

        def rag_msg(q):
            return rag_messages(q.question, passages[q.qid].passages[:k])

        for q in qs[:min(3, len(qs))]:
            controller(q.question, "sae")
            controller(q.question, "dense")
            generate(closed_book_messages(q.question), 4)
            generate(rag_msg(q), 4)
            generate(verbalised_router_messages(q.question), 4)
        for j, q in enumerate(qs):
            # Alternate timing order to avoid systematically favoring the
            # second controller due to GPU clock/cache warm-up.
            if j % 2:
                dense_ms = timed(lambda: controller(q.question, "dense"))
                sae_ms = timed(lambda: controller(q.question, "sae"))
            else:
                sae_ms = timed(lambda: controller(q.question, "sae"))
                dense_ms = timed(lambda: controller(q.question, "dense"))
            rows.append({
                "qid": q.qid, "file": f,
                "sae_controller_ms": sae_ms,
                "dense_controller_ms": dense_ms,
                "closed_ms": timed(lambda: generate(closed_book_messages(q.question), tokens)),
                "rag_ms": timed(lambda: generate(rag_msg(q), tokens)),
                "b4_ms": timed(lambda: generate(verbalised_router_messages(q.question), 8)),
            })
        print(f"{f}: timed {len(qs)} questions", flush=True)
    if len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate QIDs in latency measurement")

    with stages_path.open("w") as fp:
        fp.writelines(json.dumps(r) + "\n" for r in rows)
    cols = ["sae_controller_ms", "dense_controller_ms", "closed_ms", "rag_ms", "b4_ms"]
    df = pd.DataFrame(rows)
    summary = pd.DataFrame({"p50": df[cols].median(),
                            "p95": df[cols].quantile(0.95),
                            "mean": df[cols].mean()}).round(3)
    summary.to_csv(summary_path)
    meta = {"gpu": torch.cuda.get_device_name(0), "model": cfg["model"]["generator"],
            "sae_layer": a.sae_layer, "dense_layer": a.dense_layer,
            "variant": a.variant, "files": a.files, "n_per_file": a.n,
            "probe_training": "train-only 3-fold CV, L2, same preprocessing as probes.train",
            "generation_max_new_tokens": tokens,
            "timing": "batch size 1; independently measured controllers and generations"}
    (out / f"stages_{a.out_name}_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(), flush=True)
    print(f"WROTE {stages_path}, {summary_path}", flush=True)


if __name__ == "__main__":
    main()

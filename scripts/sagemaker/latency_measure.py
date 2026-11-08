"""Batch-size-1 timings on this GPU for questions from nq_test and triviaqa_test.

Per question: controller (forward pass + SAE encode + probe), closed-book generation,
BM25-RAG generation (generation only; BM25 search time is measured on ADA) and the B4
verbalised router. The probe here has a fixed C: its prediction cost does not depend on C.
Writes runs/latency/stages.jsonl and runs/latency/stages_summary.csv.
"""
import argparse, json, time

import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from saerag.config import load_config, path
from saerag.features.sae import JumpReLUSAE, sae_id
from saerag.hf import chat_prompt, load_model
from saerag.probes.train import load_block
from saerag.prompts import closed_book_messages, rag_messages, verbalised_router_messages
from saerag.schema import Question, Retrieval, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--layer", type=int, default=17)
ap.add_argument("--variant", default="miss_as_retrieve")
ap.add_argument("--files", nargs="+", default=["nq_test", "triviaqa_test"])
ap.add_argument("--n", type=int, default=100, help="questions per file")
args = ap.parse_args()
cfg = load_config(args.config)
m, g, L = cfg["model"], cfg["generation"], args.layer

name = f"sae16k_last@L{L}"
_, y, _, X = load_block(cfg, ["nq_train", "triviaqa_train"], args.variant, [L], ["sae16k_last"])
probe = make_pipeline(MaxAbsScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)).fit(X[name], y)

tok, model, dev = load_model(m["generator"], g.get("dtype", "auto"))
sae = JumpReLUSAE.load(m["sae_repo"], sae_id(m["sae_site"], L, "16k", m["sae_l0"]), dev, decoder=False)
k = cfg["retrieval"]["top_k"]

def enc(msgs):
    return tok(chat_prompt(tok, msgs), return_tensors="pt", add_special_tokens=False).to(dev)

def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000

def controller(q):
    with torch.no_grad():
        h = model(**enc(closed_book_messages(q.question)), output_hidden_states=True).hidden_states[L + 1][:, -1]
        codes = sae.encode(h.float()).cpu().numpy()
    probe.predict_proba(codes)

def generate(msgs, n_tokens):
    with torch.no_grad():
        model.generate(**enc(msgs), max_new_tokens=n_tokens, do_sample=False)

rows = []
for f in args.files:
    qs = list(read_jsonl(path(cfg, "questions", f"{f}.jsonl"), Question))[: args.n]
    ret = {r.qid: r for r in read_jsonl(path(cfg, "retrieval", "bm25", f"{f}.jsonl"), Retrieval)}
    for q in qs[:3]:                                   # warm-up
        controller(q); generate(closed_book_messages(q.question), 4)
    for q in qs:
        rows.append({"qid": q.qid, "file": f,
                     "controller_ms": timed(lambda: controller(q)),
                     "closed_ms": timed(lambda: generate(closed_book_messages(q.question), g["max_new_tokens"])),
                     "rag_ms": timed(lambda: generate(rag_messages(q.question, ret[q.qid].passages[:k]), g["max_new_tokens"])),
                     "b4_ms": timed(lambda: generate(verbalised_router_messages(q.question), 8))})
    print(f"{f}: timed {len(qs)} questions", flush=True)

out = path(cfg, "results", "latency")
out.mkdir(parents=True, exist_ok=True)
with open(out / "stages.jsonl", "w") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in rows)
df = pd.DataFrame(rows)
cols = ["controller_ms", "closed_ms", "rag_ms", "b4_ms"]
summary = pd.DataFrame({"p50": df[cols].median(), "p95": df[cols].quantile(0.95), "mean": df[cols].mean()}).round(1)
summary.to_csv(out / "stages_summary.csv")
print(summary.to_string())

"""Query-time activations and SAE features (D5). Owner: Person 3.

    uv run python -m saerag.features.extract --files pilot
    uv run python -m saerag.features.extract --files pilot --closed-book   # + HF closed-book run
    uv run python -m saerag.features.extract --config configs/cpu_1b.yaml --files pilot --limit 64

One forward pass per question over the templated closed-book prompt (with the generation
prompt). For every layer in `dense_layers` it stores the residual stream at the last token
and its mean over the question's tokens; for every layer in `sae_layers` it also stores the
SAE codes of both. Layout: features/store.py. Only questions are needed, so this runs
before any label exists.

--closed-book also greedily generates the closed-book answer and writes
data/generations/closed_hf/{file}.jsonl. It lets the probes run before Person 2's vLLM
labels land; the two runs should agree and are worth diffing.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from ..answers import is_correct
from ..config import load_config, path
from ..hf import chat_prompt, first_line, load_model
from ..prompts import closed_book_messages
from ..schema import Generation, Question, read_jsonl, write_jsonl
from . import store
from .sae import JumpReLUSAE, sae_id


def question_mask(prompts: list[str], questions: list[str], offsets: torch.Tensor) -> torch.Tensor:
    """[batch, seq] True on the tokens that fall inside the question text."""
    mask = torch.zeros(offsets.shape[:2], dtype=torch.bool)
    for i, (p, q) in enumerate(zip(prompts, questions)):
        q = q.strip()           # the chat template trims the message
        start = p.rfind(q)
        assert start >= 0, f"question not found in its prompt: {q!r}"
        end = start + len(q)
        s, e = offsets[i, :, 0], offsets[i, :, 1]
        mask[i] = (s >= start) & (e <= end) & (e > s)
    return mask


def masked_mean_codes(sae: JumpReLUSAE, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of per-token SAE codes over masked tokens: [batch, W]."""
    rows = mask.nonzero()[:, 0]
    codes = sae.encode(h[mask])
    out = torch.zeros(h.shape[0], codes.shape[1], device=codes.device)
    out.index_add_(0, rows, codes)
    return out / mask.sum(1, keepdim=True).clamp(min=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--closed-book", action="store_true")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--limit", type=int, help="first N questions only (smoke tests)")
    ap.add_argument("--sae-layers", nargs="*", type=int, help="override model.sae_layers")
    ap.add_argument("--widths", nargs="*", help="override model.sae_widths, e.g. 16k or 262k")
    ap.add_argument("--dense-layers", nargs="*", type=int, help="override model.dense_layers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    m, g = cfg["model"], cfg["generation"]
    for key, val in [("sae_layers", args.sae_layers), ("sae_widths", args.widths),
                     ("dense_layers", args.dense_layers)]:
        if val is not None:
            m[key] = val
    bs = args.batch_size or g["batch_size"]

    tok, model, in_device = load_model(m["generator"], g.get("dtype", "auto"))
    # SAEs and pooling live on the last GPU (device_map="auto" fills the first ones first).
    device = f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else "cpu"
    n_layers = model.config.get_text_config().num_hidden_layers

    layers = sorted(set(m["dense_layers"]) | set(m["sae_layers"]))
    assert max(layers) < n_layers - 1, "the last hidden state is post-norm; pick an earlier layer"
    # Wide SAEs (262k: 2.7 GB per matrix in fp32) skip the decoder, so no FVU for them.
    saes = {(L, w): JumpReLUSAE.load(m["sae_repo"], sae_id(m["sae_site"], L, w, m["sae_l0"]), device,
                                     decoder=w not in ("262k", "1m"))
            for L in m["sae_layers"] for w in m["sae_widths"]}

    for file in args.files:
        qs = list(read_jsonl(path(cfg, "questions", f"{file}.jsonl"), Question))[: args.limit]
        feats: dict[tuple[int, str], list] = defaultdict(list)
        stats: dict[str, list[float]] = defaultdict(list)
        gens: list[Generation] = []
        t0 = time.perf_counter()

        for i in tqdm(range(0, len(qs), bs), desc=file):
            batch = qs[i : i + bs]
            prompts = [chat_prompt(tok, closed_book_messages(q.question)) for q in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                      return_offsets_mapping=True)
            qmask = question_mask(prompts, [q.question for q in batch], enc.pop("offset_mapping"))
            enc = enc.to(in_device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            assert len(hs) == n_layers + 1
            qmask = qmask.to(device)

            for L in layers:
                h = hs[L + 1].to(device).float()  # output of block L = Gemma Scope's hook point
                last = h[:, -1]                # left padding: final position is the last token
                mean = (h * qmask[..., None]).sum(1) / qmask.sum(1, keepdim=True).clamp(min=1)
                assert last.abs().max() < 65504, "activation overflows the float16 store"
                feats[(L, "resid_last")].append(last.half().cpu().numpy())
                feats[(L, "resid_mean")].append(mean.half().cpu().numpy())
                for w in m["sae_widths"]:
                    if (sae := saes.get((L, w))) is None:
                        continue
                    codes = sae.encode(last)
                    stats[f"L{L}_{w}_l0"].append((codes > 0).sum(1).float().mean().item())
                    if sae.w_dec is not None and len(batch) > 1:
                        recon = sae.decode(codes)
                        stats[f"L{L}_{w}_fvu"].append(((last - recon) ** 2).sum().item()
                                                      / ((last - last.mean(0)) ** 2).sum().item())
                    feats[(L, f"sae{w}_last")].append(sp.csr_matrix(codes.cpu().numpy()))
                    feats[(L, f"sae{w}_mean")].append(
                        sp.csr_matrix(masked_mean_codes(sae, h, qmask).cpu().numpy()))
            del hs

            if args.closed_book:
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=g["max_new_tokens"], do_sample=False)
                texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for q, t in zip(batch, texts):
                    t = first_line(t)
                    gens.append(Generation(q.qid, "closed", t, is_correct(t, q.answers)))

        elapsed = time.perf_counter() - t0
        store.save_qids(cfg, file, [q.qid for q in qs])
        for (L, kind), parts in feats.items():
            X = sp.vstack(parts).tocsr() if sp.issparse(parts[0]) else np.concatenate(parts)
            store.save(cfg, file, L, kind, X)
        summary = {k: float(np.mean(v)) for k, v in stats.items()}
        summary |= {"n": len(qs), "seconds": elapsed, "dtype": str(model.dtype),
                    "closed_book": args.closed_book}
        # Wide SAEs run as separate passes, so keep what earlier passes recorded.
        stats_file = store.feature_dir(cfg) / f"stats_{file}.json"
        if stats_file.exists():
            summary = json.loads(stats_file.read_text()) | summary
        stats_file.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        if gens:
            write_jsonl(path(cfg, "generations", "closed_hf", f"{file}.jsonl"), gens)
            acc = sum(x.correct for x in gens) / len(gens)
            print(f"closed-book accuracy on {file}: {acc:.3f} ({len(gens)} questions)")


if __name__ == "__main__":
    main()

# Decisions

These are the choices the code assumes. **Decided** means build on it now. **Phase 0** means the pilot confirms or changes it. **Team** means it needs a group call this week.
If you change a decision, edit this file and `configs/default.yaml` in the same commit.

| ID | Decision | Status |
|----|----------|--------|
| D1 | Generator **Gemma 3 4B-IT** + **Gemma Scope 2** residual SAEs | Phase 0 (P3, by day 2) |
| D2 | BM25 = Pyserini prebuilt `wikipedia-dpr-100w` (2018 DPR dump, 21M passages) | Decided |
| D3 | Index holes = query-time filter, 30% of NQ/TQA/HotpotQA by qid hash | Decided, see caveat |
| D4 | Two generation prompts, no abstention instruction, greedy, 32 new tokens | Decided, frozen at pilot |
| D5 | Router reads the last token of the templated closed-book prompt | Decided |
| D6 | Correct = token-level normalised alias containment, all datasets | Decided |
| D7 | Cost-ordered label rule + `miss_as_retrieve` variant | Decided (from proposal) |
| D8 | Dataset sources and split construction | Decided |
| D9 | JSONL per stage keyed by `qid`; features as `.npy` + qid order | Decided |
| D10 | Gold condition: HotpotQA gold paragraphs; others = top answer-bearing BM25 passage | Team |
| D11 | Large data lives outside git under `SAERAG_DATA` | Team (where?) |
| D12 | Dense retriever for the shift test: e5-base-v2 over a 2M subsample, FAISS flat | Decided, week 3 |
| D13 | Person 1's question build is canonical | Decided |

## D1. Model and SAE
The labels depend on the generator, so **every full labelling run waits on this decision**. The pilot can start on Gemma 3 4B-IT right away.
Person 3 checks on day 1–2 that a Gemma Scope 2 release covers this model with residual-stream SAEs at about six layers and two widths, and that the SAEs load in `sae_lens`. Then they fill in `model.sae_release`, `sae_widths` and `layers` in the config.
The SAEs must match the model we generate with. If only PT-model SAEs exist, check reconstruction error on IT activations before using them.
Fallback chain (from the proposal): Gemma 2 2B or 9B with Gemma Scope, then Llama with Llama Scope.

## D2. BM25
The prebuilt index means nobody builds or hosts an index. It needs Java 21+ and about 10 GB of disk (`PYSERINI_CACHE`). DPR passages are 100 words. Top-3 matches SEAKR and DRAGIN.

## D3. Index holes
A hole question fetches the BM25 top-100, drops every passage that contains an answer alias or comes from a HotpotQA supporting article, and keeps the next 3. The removed pids are logged.
Hole questions are a disjoint 30% of every split; the same question never appears twice. They are chosen by `sha1(seed:qid)`, so the choice does not depend on how the files are built.
**Caveat for H3 (team should read):** the probe sees only the question, so it cannot observe whether a question is in a hole. On hole questions it can only learn "unknown to the model and hard to retrieve". We expect H3's gain to come from that. Holes then add label noise the probe cannot remove. Report hole and non-hole questions separately. Also consider an extra ablation that trains without holes and tests with them.

## D4. Prompts
`src/saerag/prompts.py`. No prompt allows "I don't know". Deciding to abstain is the router's job, which keeps B1 and B2 honest.
Prompts are frozen once the pilot starts. Changing them means relabelling.

## D5. Activation position
We read the last token of the closed-book prompt after `apply_chat_template(..., add_generation_prompt=True)`. That is the position from which ANSWER would start generating, so the router's forward pass can be reused (prefix cache) when it picks ANSWER. That reuse is part of the latency argument.
Mean pooling over the question tokens is stored as the pooling ablation.

## D6. Correctness
Normalisation is SQuAD-style. Matching uses token spans, not substrings, so "1" does not match "1990". Before scoring, keep only the first line of the output.
For HotpotQA, EM and F1 are also reported but not used for labels.
Known risk: PopQA has very short aliases (e.g. "AR", "PL"). Watch for them in the 200-label hand check.

## D8. Splits
- NQ-open (`google-research-datasets/nq_open`) and TriviaQA (`mandarjoshi/trivia_qa`, rc.nocontext): train and val are disjoint samples from the source train split; test comes from the source validation split.
- PopQA: long-tail subset (`s_pop < 100`), 1,000 sampled.
- HotpotQA: distractor dev, 1,000 sampled.
- Self-Aware: 500 of its 1,032 unanswerable questions.
- Pilot: 250 NQ + 250 TQA, drawn from train so the labels are reused later.
- Seed 13. `qid = "{dataset}:{source_id}"`. Rebuild with `uv run python -m saerag.data.build`; the output is deterministic.

## D10. Gold condition (team)
NQ and TQA have no gold passage ids in the DPR corpus without DPR's own preprocessed files. The proposed proxy is the highest-ranked BM25 top-100 passage that contains the answer. If there is none, `gold_correct = None` and the abstain reason is `beyond_model`.
The alternative is DPR's `positive_ctxs` files. They are more faithful but cover only part of our sample and need a mapping.

## D13. Canonical questions
`.gitignore` used to hide `src/saerag/data/`, so Person 1 wrote their own `data/build.py`. Its splits share almost no qids with the first build, and its hole hash differs. All retrieval output is keyed to Person 1's files, so that build is canonical. It also has no duplicate TriviaQA questions. `uv run python -m saerag.data.build` reproduces the handoff's question files byte for byte. `generation.run` now refuses retrieval whose qids or hole flags disagree with the question file.

## D11. Shared storage (team)
Where do `data/` and the Pyserini cache live so everyone reads the same labels? A shared cluster directory is preferred, with a Drive mirror for files under 1 GB.

// Compile from this folder: typst compile report_mid.typst report_mid.pdf

#let ink = luma(30)
#let muted = luma(105)
#let rule = luma(215)
#let wash = luma(243)
#let accent = luma(30)
#let sans = ("Fira Sans", "Liberation Sans")

#set document(title: "SAE-Gated RAG: Mid-Evaluation Report", author: (
  "Gopal Kataria",
  "Nidhi Vaidya",
  "Manya Jain",
  "Salvik Krishna",
))
#set page(paper: "a4", margin: (x: 1.6cm, y: 1.7cm), columns: 2, numbering: none, header: none, footer: none)
#set columns(gutter: 0.65cm)
#set text(font: "Libertinus Serif", size: 10pt, fill: ink, lang: "en")
#set par(justify: true, leading: 0.58em, spacing: 0.8em)
#set list(indent: 0.2em, body-indent: 0.5em, spacing: 0.55em, marker: text(fill: accent)[•])
#set enum(indent: 0.1em, body-indent: 0.5em, spacing: 0.55em)

#show heading.where(level: 1): it => block(above: 1.25em, below: 0.6em)[
  #text(font: sans, size: 11.5pt, weight: "semibold", fill: accent, it.body)
]
#show heading.where(level: 2): it => block(above: 0.95em, below: 0.45em)[
  #text(font: sans, size: 9.5pt, weight: "semibold", fill: ink, it.body)
]
#show figure.caption: it => {
  set text(size: 8.5pt, fill: muted)
  set par(justify: false, leading: 0.5em)
  [#text(font: sans, weight: "medium", fill: ink)[#it.supplement #context it.counter.display()] #h(0.3em) #it.body]
}
#show figure: set block(above: 1em, below: 1em)
#show raw: set text(font: "JetBrains Mono", size: 7.8pt)
#show table.cell.where(y: 0): set text(font: sans, weight: "medium", size: 8pt)

#let c(n) = text(fill: muted, size: 8.5pt)[[#n]]
#let wide(path, cap) = figure(image(path, width: 100%), caption: cap, placement: top, scope: "parent")

#place(top, scope: "parent", float: true, clearance: 1.1em)[
  #text(font: sans, size: 21pt, weight: "semibold", fill: ink)[SAE-Gated RAG]
  #v(-0.45em)
  #text(
    font: sans,
    size: 11.5pt,
    fill: accent,
  )[Deciding to answer, retrieve or abstain before generation · Mid-evaluation report]
  #v(0.15em)
  #text(font: sans, size: 8.8pt, fill: muted)[
    Gopal Kataria (2023112006) · Nidhi Vaidya (2023114005) · Manya Jain (2024102068) · Salvik Krishna (2023113015)
  ]

]
#text(font: sans, weight: "semibold")[Current status]

*Completed:* BM25 retrieval for all 13,500 questions; Gemma 3 4B generations and labels on the benchmark splits; a two-way probe run on a separate sample; metrics, baselines and scoring code; a 40-question end-to-end demo on Gemma 3 1B.

*Preliminary:* on the two-way label, test macro-F1 is 0.71 for the dense probe, 0.67 for the SAE probe, 0.63 for question text and 0.40 for majority.

*Unexpected:* the default label rule fails Gate 0, and the SAE probe scores lower on 2,000 labels than on 1,000. We have not explained the second result yet.

*Pending:* three-way probes on 4B, routing results, latency, the label hand check, and the full probe grid.

= 1. Problem Statement

RAG systems usually retrieve for every question, or decide during generation. Always retrieving costs latency and can mislead a model that would have answered correctly #c(10). FLARE and DRAGIN need a draft or token probabilities first #c(13)#c(14). SEAKR samples about 20 generations to estimate uncertainty #c(10). Search tools also return their top passages even when the index has no answer #c(15).

This project tests whether a frozen LLM's activation at the end of the question can be used to choose ANSWER, RETRIEVE or ABSTAIN before retrieval or generation, using sparse autoencoder (SAE) features. We compare this with a dense probe on the same activation.

The proposal's hypotheses are:

- *H1:* at most 2 points below always-retrieve accuracy, with at least 30% fewer retrieval calls.
- *H2:* the SAE probe is within 2 points of dense macro-F1 at full labels, and better on at least one of few labels, transfer or retriever shift.
- *H3:* a three-way router has lower hallucination on index-hole questions than a two-way router.
- *H4:* median latency is below always-retrieve and generation-time triggers.

= 2. Sparse Autoencoders

An SAE maps an activation $x$ to a much wider vector with few non-zero entries, then reconstructs $x$ from it (Fig. 1). Each entry is a feature. Features are often easier to interpret than raw activation dimensions #c(1).

#figure(
  image("figs/sae.webp", width: 58%),
  caption: [SAE structure. The decoder's rows are the feature vectors.],
)

We use the JumpReLU SAEs from Gemma Scope 2 #c(2):
$ f = "JumpReLU"_theta (W_"enc" x + b_"enc"), quad x' = f W_"dec" + b_"dec" $
A feature is zero unless its pre-activation exceeds its learned threshold $theta$. For Gemma 3 4B, $x$ has 2,560 dimensions. We use the 16k SAEs ($f$ has 16,384 dimensions) at the medium sparsity setting.

The SAEs are pretrained and we did not retrain them. On our questions, the mean number of active features is:

#align(center)[
  #set text(font: sans, size: 8pt)
  #table(
    columns: (auto, auto, auto, auto, auto),
    stroke: none,
    inset: (x: 6pt, y: 3pt),
    align: (left, right, right, right, right),
    table.hline(stroke: 0.6pt + ink),
    [Layer], [9], [17], [22], [29],
    table.hline(stroke: 0.4pt + rule),
    [Target L0], [53], [60], [60], [60],
    [Measured L0], [34], [59], [60], [82],
    [Error / energy], [0.001], [0.003], [0.009], [0.047],
    table.hline(stroke: 0.6pt + ink),
  )
]

Layers 17 and 22 match the target. Layers 9 and 29 do not, and we have not looked into why.

= 3. Implementation

== 3.1 Routing pipeline

#let node(body, fill: wash, stroke: none) = box(
  width: 100%,
  inset: (x: 6pt, y: 5pt),
  radius: 2.5pt,
  fill: fill,
  stroke: stroke,
  align(center, {
    set par(justify: false)
    text(font: sans, size: 8pt, body)
  }),
)
#let down = align(center, block(above: 2.5pt, below: 2.5pt, text(fill: muted, size: 8pt)[↓]))
#figure(
  block(width: 100%, inset: (y: 2pt))[
    #node[Question only]
    #down
    #node[Frozen Gemma 3 4B: one forward pass]
    #down
    #grid(
      columns: (1fr, 1fr),
      gutter: 5pt,
      node[Last-token activation], node(stroke: 0.6pt + rule, fill: white)[Dense path (comparison arm)],
    )
    #down
    #node[Frozen SAE: sparse 16k features]
    #down
    #node(fill: accent)[#text(fill: white)[Logistic-regression probe (only trained part)]]
    #down
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 5pt,
      node[*ANSWER*\ model's own knowledge],
      node[*RETRIEVE*\ BM25, top-3 passages],
      node[*ABSTAIN*\ decline or hand off],
    )
  ],
  caption: [Routing pipeline. Only the probe is trained.],
)

The question goes through the model once, using the closed-book prompt with the chat template. We read the residual stream at the last prompt token at layer $L$. In Hugging Face this is `hidden_states[L + 1]`, the output of block $L$. The SAE encodes this vector, and the probe picks an action.

== 3.2 Data and retrieval

Questions come from NQ-open and TriviaQA (4,000 train, 500 val, 1,000 test each), long-tail PopQA (1,000), HotpotQA (1,000) and Self-Aware unanswerable questions (500), with seed 13. The pilot is 250 NQ and 250 TriviaQA questions taken from train.

Retrieval is BM25 on a Lucene index of the 2018 Wikipedia DPR passages (21,015,324 passages of 100 words). We fetch the top 100 and keep the top 3.

About 30% of NQ, TriviaQA and HotpotQA questions are flagged as index holes #c(15). For these, passages containing an answer alias, or HotpotQA's supporting titles, are removed from the top 100 at query time. On the pilot this removed 1,800 passages from 157 questions, and all of them still had 3 passages left.

The gold condition uses the highest-ranked top-100 passage containing the answer. HotpotQA uses its supporting paragraphs. Questions with no such passage have no gold run.

== 3.3 Generation and labels

Gemma 3 overflows in fp16 and the T4 has no bf16, so 4B runs in fp32 split across two T4s. Decoding is greedy with 32 new tokens. An output is correct if an alias appears as a token span in the first line, after SQuAD-style normalisation.

Each question has three runs: closed-book, BM25 top-3 and gold. Labels follow this rule:

$
  "label" = cases(
    "ANSWER" & "if closed-book correct",
    "RETRIEVE" & "else if BM25 correct",
    "ABSTAIN" & "otherwise"
  )
$

The variant `miss_as_retrieve` gives RETRIEVE to questions that only the gold run answered. Self-Aware questions are always ABSTAIN.

== 3.4 Probe training

The first probe run predicts a two-way label: ANSWER if the closed-book answer is correct, otherwise NOT_ANSWER. It used a Kaggle notebook on a new Hugging Face sample of 2,000 train, 400 val, 600 test and 300 PopQA questions.

All probes are class-weighted logistic regression, with C from {0.001, 0.01, 0.1, 1, 10} chosen by 3-fold CV on macro-F1. Dense features use `StandardScaler` and SAE features use `MaxAbsScaler`. The text baseline is TF-IDF on 1- and 2-grams. Layers 9, 17, 22 and 29 were compared on validation. Intervals are 95% bootstrap with 1,000 resamples. For the label budgets we trained on 50 to 1,000 labels with five seeds each.

== 3.5 Evaluation code

`eval/metrics.py` implements the proposal's metrics: accuracy, selective accuracy, coverage, hallucination rate, retrieval-call rate, RETRIEVE precision and recall, abstention recall at 10% false refusal, risk-coverage and AURC, macro-F1, ECE, and latency by stage. `eval/score.py` scores any router from one file format, split by hole, non-hole and dataset. `eval/handcheck.py` draws the 200-label sheet.

= 4. Related Work

Probing-RAG predicts retrieval need with a probe on intermediate hidden states #c(9). It uses a dense probe and has no abstain action.

Xin et al. find SAE latents linked to context-versus-memory and answer-versus-reject decisions #c(5). SpARE steers knowledge selection with SAE features #c(6). RAGLens and SAFE use them for faithfulness and hallucination #c(7)#c(8). None of these trains a pre-generation router or measures retrieval calls and latency.

SEAKR reads uncertainty from the middle layer #c(10) and CtrlA finds intermediate layers best for steering #c(11), so we expected middle layers to do well. Adaptive-RAG classifies the question text #c(12), and our B8 baseline is close to that.

Evidence on SAE versus dense probes is mixed. Gallifant et al. report SAE features beating hidden-state baselines, with transfer across model sizes #c(4). Kantamneni et al. find SAE probes do not consistently beat strong baselines under scarcity, imbalance, noise or shift #c(3).

= 5. Code Status

#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: none,
    inset: (x: 3pt, y: 3.4pt),
    align: (left, left, left),
    table.hline(stroke: 0.6pt + ink),
    [Module], [Contents], [State],
    table.hline(stroke: 0.4pt + rule),
    [`data/`], [Splits, hole flags], [Complete],
    [`retrieval/`], [BM25, hole filter, gold proxy], [Run, all splits],
    [`generation/`], [Runs, labels, gate report], [Run on 4B],
    [`features/`], [Activations, SAE codes], [Run on sample],
    [`probes/`], [Training, analysis], [Two-way only],
    [`baselines/`], [B1–B4, B8], [1B demo only],
    [`eval/`], [Metrics, scoring, hand check], [1B demo only],
    [`scripts/`], [ADA jobs, probe grid], [Not run],
    table.hline(stroke: 0.6pt + ink),
  ),
  kind: table,
  supplement: none,
  numbering: none,
)

The code is merged on branch `p4-eval`, where all 45 tests pass. Closed-book generation took about 140 s per 1,000 questions and BM25 about 1,250 s, on 2×T4.

= 6. Preliminary Results

Sections 6.1 to 6.3 use the canonical splits on 4B. Sections 6.4 and 6.5 use the separate probe sample. Closed-book accuracy on that sample is within 4 points of the canonical run. Section 6.6 is the 1B demo.

== 6.1 Generator accuracy by condition

Fig. 3 shows accuracy relative to closed-book. On the test splits, BM25 adds 0.9 points on NQ and 0.3 on TriviaQA. It adds 16.9 on PopQA and 2.9 on HotpotQA. These numbers include hole questions. The gold run adds 33 to 53 points.

#figure(
  image("figs/03_bm25_gain_vs_headroom.pdf", width: 100%),
  caption: [Accuracy change from closed-book for BM25 and gold passages.],
)

== 6.2 Accuracy on index-hole questions

On hole questions, BM25 accuracy is below closed-book on every split (Fig. 4): 52% to 16% on TriviaQA test, 20% to 5% on NQ test, and 20% to 12% on HotpotQA. The filter removes answer text by design, so some drop is expected. On TriviaQA the model gets about half of these right with no passages, and most of those answers are lost once it sees unrelated text.

#figure(image("figs/02_hole_accuracy.pdf", width: 100%), caption: [Accuracy on index-hole questions.])

== 6.3 Label distribution and Gate 0

#wide("figs/04_label_distribution.pdf", [Label shares under both rules.])

Gate 0 needs each class at 15% or more of pilot labels. It fails with the default rule: RETRIEVE is 11.2%. With `miss_as_retrieve` the pilot passes at 35 / 36 / 28 (ANSWER / RETRIEVE / ABSTAIN), but ABSTAIN is 13% on TriviaQA train and test.

ABSTAIN is about 70% of NQ under the default rule. We have not hand-checked these labels yet. The team still has to choose a rule. We lean towards `miss_as_retrieve`.

== 6.4 Probe comparison (two-way label)

#wide(
  "figs/11_probe_results.pdf",
  [Macro-F1 at layer 17 with 95% bootstrap intervals and AUROC. Test is 600 NQ + TriviaQA questions. PopQA is 300 questions.],
)

Layer 17 had the best validation macro-F1 for both probes (SAE 0.74, dense 0.76). Layer 9 was the worst (0.66, 0.63). Dense AUROC peaks slightly later, at layer 22 (0.845 against 0.840).

Test results are in Fig. 6. The dense interval just clears question text, and the SAE interval overlaps it. On PopQA, text drops to 0.52 while SAE and dense stay at 0.72 and 0.75. So the activation seems to contain information about whether the model knows the answer that the question text does not.

The SAE probe is 3.9 points below dense on test and 2.5 below on PopQA. Both gaps are outside H2's 2-point margin, although the intervals overlap.

== 6.5 Label efficiency

The SAE probe does not beat dense at any budget (Fig. 7). At 50 labels it is about 0.66 against 0.68, and at 200 about 0.67 against 0.71. At 1,000 they are level. On this label, H2 is not supported so far.

*Unexplained result.* SAE reaches about 0.71 on 1,000-label subsamples but 0.673 on all 2,000 labels. Dense stays flat. We do not know why yet. Until we do, the 3.9-point gap in Section 6.4 may be wrong.

We have the top-weighted SAE features at layer 17, but nobody has reviewed them yet.

#figure(
  image("figs/12_label_efficiency.pdf", width: 88%),
  caption: [Test macro-F1 against training labels, mean of five seeds.],
)

== 6.6 End-to-end demo on Gemma 3 1B

This run checked that every stage works. It used 40 pilot questions (20 NQ, 20 TriviaQA, 13 holes) with Gemma 3 1B on a laptop CPU. Table 1 is plumbing output and should not be read as a result. Generator accuracy was 15.0% closed-book, 15.0% BM25 and 51.4% gold (35 questions).

#figure(
  {
    set text(font: sans, size: 7.6pt)
    table(
      columns: (1fr, auto, auto, auto, auto, auto),
      stroke: none,
      inset: (x: 2.5pt, y: 3pt),
      align: (left, right, right, right, right, right),
      table.hline(stroke: 0.6pt + ink),
      [Router], [Acc.], [Halluc.], [Calls], [Abst. rec.], [False ref.],
      table.hline(stroke: 0.4pt + rule),
      [B1 always retrieve], [0.150], [0.850], [100%], [0.00], [0.00],
      [B2 never retrieve], [0.150], [0.850], [0%], [0.00], [0.00],
      [B3 oracle], [0.275], [0.000], [12.5%], [1.00], [0.00],
      [B4 verbalised], [0.175], [0.816], [80%], [0.03], [0.09],
      [B8 question text], [0.050], [0.857], [20%], [0.59], [0.82],
      [Dense probe L17], [0.000], [1.000], [2.5%], [0.93], [1.00],
      [SAE probe L17], [0.000], [none], [0%], [1.00], [1.00],
      table.hline(stroke: 0.6pt + ink),
    )
  },
  kind: table,
  supplement: [Table],
  caption: [1B demo, 40 questions, argmax routing. B8 and the probes use 3-fold CV on the same questions.],
)

With 72.5% of labels ABSTAIN, both probes predict ABSTAIN for almost everything. B4 picked RETRIEVE 80% of the time and rarely abstained. B1's hallucination rate was 0.815 on non-hole and 0.923 on hole questions.

= 7. Limitations

- The 4B probe results use a two-way proxy label. H1, H3 and H4 have not been tested on 4B, and H2 only on this label.
- Probe results and labels come from different question samples, so they cannot be joined yet.
- The gold passage is chosen because it contains the answer, except on HotpotQA. The gold accuracy is therefore an upper bound.
- Holes are filtered at query time from the top 100 by lexical match. Evidence is not removed from the index, and other passages may still support the answer.
- A probe that reads only the question cannot see whether the index has a hole. We will report hole and non-hole questions separately for H3.
- String matching may mark some correct answers as wrong, which would inflate ABSTAIN. Short PopQA aliases such as "AR" and "PL" are a known risk.
- BM25 has no per-question latency yet because the run was batched.

= 8. Remaining Work

#block(breakable: false, {
  set text(font: sans, size: 8pt)
  set par(justify: false)
  table(
    columns: (auto, 1fr, 1.15fr),
    stroke: none,
    inset: (x: 3pt, y: 3.2pt),
    table.hline(stroke: 0.6pt + ink),
    [Gate], [Requirement], [Status],
    table.hline(stroke: 0.4pt + rule),
    [0], [Each class ≥ 15% of pilot], [Default fails; variant passes],
    [1], [Label error < 10%], [Tool ready; check not run],
    [2], [Fewer calls, ≤ 2 points lost], [Scoring ready; needs probes],
    [3], [Latency for every method], [Metric ready; needs timings],
    table.hline(stroke: 0.6pt + ink),
  )
})

== Final SAE runs

+ Find the cause of the 1,000 versus 2,000 label result.
+ Decide the label rule.
+ Extract activations and SAE codes on the canonical splits: dense at layers 4, 9, 13, 17, 22, 29, SAEs at 9, 17, 22, 29, widths 16k and 262k.
+ Train three-way probes on the 4B labels.
+ Run `scripts/ada_grid.sbatch`: layers × widths × budgets (50 to 5,000) × seeds, plus L1, binarised and mean-pooling ablations.

== Comparisons

+ *H2:* SAE and dense on the same activation and labels, across label budget, PopQA and HotpotQA transfer, calibration, and BM25-to-dense retriever shift. The dense retriever is not built yet.
+ *H1:* score B1–B4, B8 and the probes on 4B with `eval/score.py`.
+ *H3:* two-way and three-way routers, on hallucination for hole questions and abstention recall at 10% false refusal.
+ *H4:* p50 and p95 at batch size 1, split by controller, retrieval and generation.

B5, B6 and B9 are still unassigned and depend on time.

== Other tasks

- Run the 200-label hand check, and report probes with and without flagged questions.
- Review the sampled hole passages by hand.
- Check top features by hand on 30 examples, and measure the F1 drop after removing the top-k features.
- Failure analysis, including PopQA popularity bins.

= References

#set text(size: 8.3pt)
#set par(justify: false, leading: 0.46em, spacing: 0.36em)
#let refs = (
  [Huben et al. (2024). Sparse Autoencoders Find Highly Interpretable Features in Language Models.],
  [Google DeepMind. Gemma Scope 2: SAEs and transcoders for the Gemma 3 family.],
  [Kantamneni et al. (2025). Are Sparse Autoencoders Useful? A Case Study in Sparse Probing.],
  [Gallifant et al. (2025). Sparse Autoencoder Features for Classifications and Transferability.],
  [Xin et al. (2025). Sparse Latents Steer Retrieval-Augmented Generation. ACL.],
  [Zhao et al. (2025). Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering (SpARE). NAACL.],
  [Xiong et al. (2026). Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders (RAGLens). ICLR.],
  [Abdaljalil et al. (2025). SAFE: A Sparse Autoencoder-Based Framework for Robust Query Enrichment and Hallucination Mitigation in LLMs.],
  [Baek et al. (2024). Probing-RAG: Self-Probing to Guide Language Models in Selective Document Retrieval.],
  [Yao et al. (2024). SEAKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation.],
  [Liu et al. (2024). CtrlA: Adaptive Retrieval-Augmented Generation via Inherent Control.],
  [Jeong et al. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.],
  [Jiang et al. (2023). Active Retrieval Augmented Generation (FLARE).],
  [Su et al. (2024). DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of LLMs.],
  [Chandradevan et al. (2026). Search Engines Never Say No: How Frozen Agents React When the Retrieval Tool Refuses.],
)
#for (i, r) in refs.enumerate() [
  #grid(
    columns: (1.5em, 1fr),
    text(fill: muted)[#(i + 1)], r,
  )
]

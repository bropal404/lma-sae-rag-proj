// Compile from this folder: typst compile report_final.typ report_final.pdf

#let ink = luma(30)
#let muted = luma(105)
#let rule = luma(215)
#let wash = luma(243)
#let accent = luma(30)
#let sans = ("Fira Sans", "Segoe UI", "Liberation Sans", "Arial")

#set document(title: "SAE-Gated RAG: Final Report", author: (
  "Gopal Kataria",
  "Nidhi Vaidya",
  "Manya Jain",
  "Salvik Krishna",
))
#set page(paper: "a4", margin: (x: 1.6cm, y: 1.7cm), columns: 2, numbering: "1", number-align: center)
#set columns(gutter: 0.65cm)
#set text(font: "Libertinus Serif", size: 10pt, fill: ink, lang: "en")
#set par(justify: true, leading: 0.55em, spacing: 0.7em)
#set list(indent: 0.2em, body-indent: 0.5em, spacing: 0.55em, marker: text(fill: accent)[•])
#set enum(indent: 0.1em, body-indent: 0.5em, spacing: 0.55em)

#show heading.where(level: 1): it => block(above: 1.2em, below: 0.6em)[
  #text(font: sans, size: 11.5pt, weight: "semibold", fill: accent, it.body)
]
#show heading.where(level: 2): it => block(above: 0.9em, below: 0.45em)[
  #text(font: sans, size: 9.5pt, weight: "semibold", fill: ink, it.body)
]
#show figure.caption: it => {
  set text(size: 8.5pt, fill: muted)
  set par(justify: false, leading: 0.5em)
  [#text(font: sans, weight: "medium", fill: ink)[#it.supplement #context it.counter.display()] #h(0.3em) #it.body]
}
#show figure: set block(above: 0.9em, below: 0.9em)
#show raw: set text(font: "DejaVu Sans Mono", size: 7.6pt)
#show table.cell.where(y: 0): set text(font: sans, weight: "medium", size: 7.8pt)

#let c(..n) = text(fill: muted, size: 8.5pt)[[#n.pos().map(str).join(", ")]]
#let wide(path, cap) = figure(image(path, width: 100%), caption: cap, placement: top, scope: "parent")
#let tbl(cols, aligns, ..rows) = {
  set text(font: sans, size: 7.6pt)
  set par(justify: false)
  table(
    columns: cols,
    stroke: none,
    inset: (x: 3pt, y: 2.6pt),
    align: aligns,
    table.hline(stroke: 0.6pt + ink),
    ..rows.pos().slice(0, 1).flatten(),
    table.hline(stroke: 0.4pt + rule),
    ..rows.pos().slice(1).flatten(),
    table.hline(stroke: 0.6pt + ink),
  )
}

#place(top, scope: "parent", float: true, clearance: 1.2em)[
  #text(font: sans, size: 21pt, weight: "semibold", fill: ink)[SAE-Gated RAG]
  #v(-0.45em)
  #text(font: sans, size: 11.5pt, fill: accent)[Deciding to answer, retrieve or abstain before generation — Final report]
  #v(0.15em)
  #text(font: sans, size: 8.8pt, fill: muted)[
    Gopal Kataria (2023112006); Manya Jain (2024102068); Nidhi Vaidya (2023114005); Salvik Krishna (2023113015)
  ]
  #v(0.1em)
  #text(font: sans, size: 8.8pt)[Repository: #link("https://github.com/bropal404/lma-sae-rag-proj")[github.com/bropal404/lma-sae-rag-proj]]
  #v(0.5em)
  #block(fill: wash, inset: (x: 10pt, y: 8pt), radius: 3pt, width: 100%)[
    #set par(leading: 0.52em)
    #text(font: sans, weight: "semibold", size: 9pt)[Abstract] #h(0.4em)
    Retrieval-augmented generation (RAG) usually retrieves for every question, which costs latency and can distract a model that already knows the answer. We study a *pre-generation controller* that reads one forward pass of a frozen Gemma 3 4B over the question alone and chooses ANSWER, RETRIEVE or ABSTAIN before any retrieval or decoding. The controller is a logistic-regression probe on pretrained Gemma Scope 2 sparse autoencoder (SAE) features, compared against a probe on the dense residual stream and against five baselines. We build three-way labels for 13,500 questions from NQ, TriviaQA, PopQA, HotpotQA and SelfAware by running closed-book, BM25 (21M-passage 2018 Wikipedia) and gold-passage generations, and hand-check 200 labels (7.5% error). The validation-selected SAE router (16k features, layer 29) matches always-retrieve accuracy within 0.35 points while making *50.8% fewer* retrieval calls on 4,000 answerable test questions. Three-way routing lowers wrong answers on simulated index-hole questions from 82.2% to 61.0%. SAE probes abstain far better on unanswerable SelfAware questions (83% recall versus 26% for dense), while the dense probe is 1.2–2.4 macro-F1 points ahead on answerable data. The controller adds 57 ms and the routed pipeline is faster than always-retrieve at the median.
  ]
]

= 1. Introduction

Retrieval-augmented generation grounds a language model in external text, but the usual policy of retrieving for every question has three costs. It adds retrieval and longer-prompt latency to every query, it can mislead a model that would have answered correctly from memory #c(10, 16), and search tools return their top passages even when the index holds no answer, which invites confident wrong answers #c(15). Adaptive methods such as FLARE and DRAGIN decide during generation, so they must decode a draft first #c(13, 14). SEAKR samples about 20 generations to estimate uncertainty #c(10). All of these pay the generation cost before deciding.

We ask whether the decision can be made *before* generation and retrieval, from a single forward pass over the question. Our controller reads the residual-stream activation at the last prompt token of a frozen Gemma 3 4B #c(20), encodes it with a pretrained sparse autoencoder from Gemma Scope 2 #c(2), and a linear probe selects one of three actions:

- *ANSWER* — generate closed-book from the model's own knowledge;
- *RETRIEVE* — fetch the BM25 top-3 passages and generate with them;
- *ABSTAIN* — decline, for questions neither path will answer correctly.

SAE features are sparse and often interpretable #c(1, 22), so a router built on them could be easier to inspect and could transfer better than one on raw activations. We test this against a dense probe on the same activations. The project's four hypotheses are:

- *H1:* routed accuracy at most 2 points below always-retrieve, with at least 30% fewer retrieval calls.
- *H2:* the SAE probe is within 2 macro-F1 points of the dense probe at full labels, and better on at least one of: few labels, transfer, or abstention.
- *H3:* a three-way router gives fewer wrong answers on index-hole questions than a two-way (ANSWER / RETRIEVE) router.
- *H4:* median end-to-end latency is below always-retrieve and below a generation-based self-router.

Four gates guard the experiment: *Gate 0* (every label class at least 15% of the pilot), *Gate 1* (label error below 10% on a hand check), *Gate 2* (the H1 criterion) and *Gate 3* (latency measured for every method).

*Contributions.* (i) A labelled dataset of 13,500 questions with closed-book, BM25 and gold generations, three-way labels, index-hole flags and a 200-question human audit. (ii) A full grid of SAE and dense probes over four layers, two SAE widths, seven label budgets, three seeds and four ablations. (iii) An end-to-end evaluation against five baselines on accuracy, retrieval calls, unsafe answers on index holes, abstention and batch-size-1 latency. All four gates pass. H1 and H3 are supported, and H2 is supported on abstention but not on answerable-question macro-F1.

= 2. Literature Review

*Adaptive retrieval.* Self-RAG trains the generator to emit reflection tokens that trigger retrieval #c(17). FLARE retrieves when an upcoming token has low probability #c(13), and DRAGIN uses token-level information need #c(14). These work during decoding, so a draft must be generated first. SEAKR reads self-aware uncertainty from internal states across sampled generations #c(10). Adaptive-RAG classifies question complexity from the text alone #c(12); our B8 baseline is close to it. Mallen et al. show that retrieval mainly helps on long-tail entities and can hurt on popular ones, and they introduced PopQA for this reason #c(16).

*Probing for knowledge boundaries.* Probing-RAG trains a probe on intermediate hidden states to predict whether retrieval is needed #c(9). It uses a dense probe and has no abstain action. CtrlA finds intermediate layers best for steering honesty and confidence #c(11). SelfAware tests whether models recognise unanswerable questions #c(18), and we use it as our abstention slice.

*Sparse autoencoders.* SAEs decompose activations into a wider, sparse feature basis that is often monosemantic #c(1, 22). Gemma Scope 2 provides JumpReLU SAEs #c(21) for every layer of Gemma 3 #c(2). In RAG, Xin et al. find SAE latents tied to context-versus-memory and answer-versus-reject behaviour #c(5). SpARE steers knowledge selection with SAE features #c(6), and RAGLens and SAFE use them for faithfulness and hallucination #c(7, 8). None of these trains a pre-generation router or measures retrieval calls and latency.

*SAE versus dense probes.* The evidence is mixed. Gallifant et al. report SAE features beating hidden-state baselines, with transfer across model sizes #c(4). Kantamneni et al. find that SAE probes do not consistently beat strong dense baselines under data scarcity, class imbalance, label noise or shift #c(3). Our study adds a three-way routing task, an end-to-end cost evaluation and an abstention test to this comparison.

= 3. Dataset Created and Used

== 3.1 Questions and corpus

#figure(
  tbl(
    (1fr, auto, auto, auto, auto),
    (left, right, right, right, right),
    ([Source], [Train], [Val], [Test], [Holes (test)]),
    ([NQ-open #c(23)], [4,000], [500], [1,000], [309]),
    ([TriviaQA #c(24)], [4,000], [500], [1,000], [283]),
    ([PopQA (pop. < 100) #c(16)], [—], [—], [1,000], [—]),
    ([HotpotQA distractor #c(25)], [—], [—], [1,000], [302]),
    ([SelfAware unanswerable #c(18)], [—], [—], [500], [—]),
    ([*Total: 13,500*], [8,000], [1,000], [4,500], [894]),
  ),
  kind: table,
  supplement: [Table],
  caption: [Question splits (seed 13). The 500-question pilot (250 NQ + 250 TriviaQA) is drawn from train. PopQA and HotpotQA are out-of-distribution test sets.],
) <tab:data>

NQ and TriviaQA (`rc.nocontext`) provide in-distribution train, validation and test splits (Table 1). PopQA is restricted to long-tail entities with popularity below 100. HotpotQA tests multi-hop transfer, and 500 of SelfAware's 1,032 unanswerable questions form the abstention slice.

The retrieval corpus is the 2018 Wikipedia DPR passage set #c(26): *21,015,324 passages* of 100 words. We use BM25 #c(19) over Pyserini's Lucene index of this corpus #c(27), retrieve the top 100 passages for all 13,500 questions and keep the top 3 for generation.

== 3.2 Index holes and gold passages

To test behaviour when the index lacks the answer, about 30% of NQ, TriviaQA and HotpotQA questions are flagged as *index holes* #c(15). For these, every top-100 passage that contains an answer alias (for HotpotQA, its supporting titles) is removed at query time. A validation script confirmed that no lexical answer leaks remain. The *gold* condition uses the highest-ranked top-100 passage that contains the answer; HotpotQA uses its supporting paragraphs. Questions without such a passage have no gold run (194 NQ, 73 TriviaQA and 372 PopQA test questions).

== 3.3 Generations and labels

Every question is answered three times by Gemma 3 4B-it with greedy decoding (32 new tokens): closed-book, with BM25 top-3 and with the gold passage. An output is correct if any alias appears as a token span in its first line after SQuAD-style normalisation. Table 2 shows the generator accuracy.

#figure(
  tbl(
    (1fr, auto, auto, auto, auto),
    (left, right, right, right, right),
    ([Test split], [Closed], [BM25], [Gold], [BM25 on holes]),
    ([NQ], [18.2], [20.0], [63.4], [6.5]),
    ([TriviaQA], [50.8], [51.0], [84.0], [15.6]),
    ([PopQA], [17.2], [34.0], [70.4], [—]),
    ([HotpotQA], [19.3], [22.5], [63.6], [13.3]),
  ),
  kind: table,
  supplement: [Table],
  caption: [Gemma 3 4B accuracy (%) by condition. Gold is computed on questions with a gold passage only.],
)

BM25 helps most on long-tail PopQA (+16.8 points) and little on NQ and TriviaQA. On hole questions BM25 accuracy collapses below closed-book (TriviaQA: 51.2% closed versus 15.6% with BM25), which is the failure the ABSTAIN and ANSWER actions should avoid.

Labels follow the `miss_as_retrieve` rule:
$
  "label" = cases(
    "ANSWER" & "if closed-book correct",
    "RETRIEVE" & "else if BM25 or gold correct",
    "ABSTAIN" & "otherwise"
  )
$
SelfAware questions are always ABSTAIN. The rule treats a gold-only success as a retriever miss that a better retriever could fix. The stricter default rule (RETRIEVE only on a BM25 success) left RETRIEVE at 11.2% of the pilot and failed Gate 0.

== 3.4 Label quality: Gates 0 and 1

*Gate 0 passes:* the pilot labels are ANSWER 35.4%, RETRIEVE 36.6% and ABSTAIN 28.0% (Fig. 1).

#wide("figs/04_label_distribution.png", [Label shares by split under the default rule (left) and the `miss_as_retrieve` rule (right).])

For *Gate 1* we drew a simple random sample of 200 labelled questions (seed 13) across train and test and judged each of its 566 generations by hand, following a written protocol. The hand judgements were pushed through the same label rule. Under the strict protocol, 15 of 200 labels change (*7.5%*, Wilson 95% CI 4.6–12.0%), so Gate 1 passes. Of these, 8 move ABSTAIN→RETRIEVE, 6 ABSTAIN→ANSWER and 1 RETRIEVE→ANSWER. The matcher's per-generation error is 4.8% (27/566). Every label change moves a question *up* (towards ANSWER or RETRIEVE), so the matcher errs on the side of false negatives and slightly inflates ABSTAIN. A lenient protocol, which also accepts loosely phrased answers, gives 22/200 (11%); we report it as a sensitivity check.

= 4. Methodology

== 4.1 Controller

#let node(body, fill: wash, stroke: none) = box(
  width: 100%,
  inset: (x: 6pt, y: 4.5pt),
  radius: 2.5pt,
  fill: fill,
  stroke: stroke,
  align(center, {
    set par(justify: false)
    text(font: sans, size: 7.8pt, body)
  }),
)
#let down = align(center, block(above: 2pt, below: 2pt, text(fill: muted, size: 8pt)[↓]))
#figure(
  block(width: 100%, inset: (y: 2pt))[
    #node[Question only (closed-book prompt, chat template)]
    #down
    #node[Frozen Gemma 3 4B: one forward pass, last-token residual at layer $L$]
    #down
    #grid(
      columns: (1fr, 1fr),
      gutter: 5pt,
      node[Frozen JumpReLU SAE → 16k sparse features], node(stroke: 0.6pt + rule, fill: white)[Dense path: 2,560-d residual],
    )
    #down
    #node(fill: accent)[#text(fill: white)[Logistic-regression probe (only trained part) + validation-tuned ABSTAIN threshold]]
    #down
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 5pt,
      node[*ANSWER*\ closed-book],
      node[*RETRIEVE*\ BM25 top-3],
      node[*ABSTAIN*\ decline],
    )
  ],
  caption: [The routing pipeline. Only the probe is trained.],
)

The question passes once through the model with the closed-book prompt. We take the residual stream at the last prompt token after block $L$ (`hidden_states[L+1]` in Hugging Face). The SAE encodes it as
$ f = "JumpReLU"_theta (W_"enc" x + b_"enc"), quad hat(x) = f W_"dec" + b_"dec", $
where a feature is zero unless its pre-activation exceeds a learned threshold $theta$ #c(21). We use Gemma Scope 2 residual SAEs at medium sparsity, with width 16,384 at layers 9, 17, 22 and 29 and width 262,144 at layers 17 and 22. Measured L0 on our questions is 34–82 active features. The SAEs are not retrained.

== 4.2 Probe training and selection

Probes are multinomial logistic regression with balanced class weights. $C in {10^(-3), ..., 10}$ is chosen by 3-fold cross-validation on macro-F1 within the training set. SAE codes use `MaxAbsScaler`, which keeps them sparse, and dense vectors use `StandardScaler`. Training uses NQ + TriviaQA train (8,000 labels). The SAE and dense layers are chosen *independently* by full-budget macro-F1 on NQ + TriviaQA validation. Test sets are never used for selection.

*Routing rule.* ABSTAIN is chosen when $p("ABSTAIN") >= tau$; otherwise the router takes the likelier of ANSWER and RETRIEVE. $tau$ is the lowest threshold whose false-refusal rate on answerable *validation* questions is at most 10%. This sets the operating point without looking at test data.

== 4.3 Baselines

- *B1 always retrieve* — standard RAG; the accuracy reference.
- *B2 never retrieve* — closed-book only.
- *B3 oracle* — follows the gold label; an upper bound.
- *B4 verbalised self-router* — the model is prompted to output ANSWER, RETRIEVE or ABSTAIN, which needs a generation step.
- *B8 question text* — TF-IDF (1–2-grams) + logistic regression on the question, similar to Adaptive-RAG #c(12).

B5 (generation-time trigger), B6 (sampled uncertainty) and B9/B10 from the proposal were not pursued. B4 represents the generation-based family.

== 4.4 Metrics

`eval/metrics.py` implements accuracy (an abstention counts as not correct), selective accuracy, coverage, hallucination rate, retrieval-call rate, RETRIEVE precision and recall, abstention recall at 10% false refusal, AURC, macro-F1, ECE and stage-wise latency. `eval/score.py` scores every router from one probability-file format, split into all / hole / non-hole questions and by dataset. Confidence intervals are paired percentile bootstraps (2,000 draws) that resample questions within each dataset.

= 5. Experiment Runs

*Compute.* Retrieval ran on the ADA cluster (Lucene index, 21M passages). The first generations ran on Kaggle (2×T4, fp32 because Gemma 3 overflows in fp16), and the final generations, feature extraction, probes and latency ran on AWS SageMaker on a single NVIDIA L4. Generation costs about 18–30 s per 1,000 questions closed-book and 163–189 s with BM25 passages on the L4.

*Runs completed.*
+ BM25 top-100 retrieval for all 13,500 questions, with hole filtering and gold selection.
+ 3 × 14,000 generations (closed, BM25, gold; the pilot overlaps train), rescored after the Gate 1 matcher fix, and labels for all splits under both rules.
+ Dense activations at layers 4, 9, 13, 17, 22 and 29, and SAE codes at 9/17/22/29 (16k) and 17/22 (262k), for all 13,500 questions.
+ Three-way and two-way probes for every feature set, plus the label-budget grid (50, 100, 200, 500, 1k, 2k, 5k and full) × 3 seeds for the selected SAE and dense models.
+ Ablations: L1 penalty, binarised codes, mean pooling over question tokens, a wider C grid, and extra dense layers.
+ Baselines B1–B4 and B8, scored with the same script as the probes.
+ Batch-size-1 latency on 200 questions (100 NQ + 100 TriviaQA): BM25, closed and RAG generation, controller and B4.
+ A 200-label hand check, paired bootstraps, confusion matrices, a routing-error sample, PopQA popularity bins and Neuronpedia feature lookups.

*Layer selection.* Fig. 3 shows validation macro-F1 by layer. Dense peaks at layer 22 (53.1%). The SAE probe keeps improving with depth to layer 29 (51.8%), and SAE layer 9 is below the question-text baseline. Following the protocol, the main comparison is *SAE 16k @ L29* against *dense @ L22*.

#figure(image("figs_final/f6_layers.png", width: 100%), caption: [Validation macro-F1 (NQ + TriviaQA) by layer, three-way labels, all 8,000 training labels.])

*The mid-evaluation anomaly.* The mid-term SAE probe scored lower with 2,000 labels than with 1,000. With three seeds per budget on the canonical labels, macro-F1 rises monotonically (L17: 0.533 → 0.544 → 0.563 at 1k/2k/4k), so the drop does not reproduce. It came from the earlier single-seed run on a separate sample.

= 6. Ablations

Table 3 compares feature sets and training variants. It reports macro-F1 on validation and the unweighted mean over the four answerable test sets.

#figure(
  tbl(
    (1fr, 1.3fr, 1.3fr, 1fr),
    (left, right, right, right),
    ([Feature set / variant], [Validation], [Test (mean of 4)], [SelfAware]),
    ([Majority class], [16.4], [13.3], [0.0]),
    ([Question text TF-IDF], [44.3], [34.7], [26.6]),
    ([Dense L9 / L17 / *L22* / L29], [48.3 / 51.8 / *53.1* / 49.5], [41.1 / 46.8 / *48.2* / 48.0], [19 / 27 / 27 / 31]),
    ([Dense L4 / L13], [42.4 / 49.7], [32.6 / 44.3], [19 / 25]),
    ([SAE 16k L9 / L17 / L22 / *L29*], [40.8 / 49.3 / 50.5 / *51.8*], [31.9 / 47.1 / 50.1 / *45.8*], [25 / 31 / 33 / 33]),
    ([SAE 262k L17 / L22], [46.8 / 50.1], [43.1 / 46.3], [32 / 32]),
    ([SAE 16k, L1 penalty L17 / L22], [47.6 / 49.8], [45.7 / 49.6], [33 / 32]),
    ([SAE 16k, binarised L17 / L22], [48.5 / 50.6], [46.2 / 48.0], [33 / 33]),
    ([SAE 16k, mean-pooled L17 / L22], [49.9 / 47.8], [40.5 / 41.1], [26 / 24]),
    ([Dense, mean-pooled L17 / L22], [50.1 / 50.3], [41.1 / 39.9], [26 / 29]),
    ([Dense, wide C grid L17 / L22], [50.8 / 50.2], [48.0 / 49.7], [32 / 33]),
  ),
  kind: table,
  supplement: [Table],
  caption: [Ablations: three-way macro-F1 (%) with all training labels, L2 penalty unless stated. SelfAware macro-F1 is over the predicted actions on unanswerable questions.],
  placement: top,
  scope: "parent",
) <tab:abl>

*Layer.* Middle-to-late layers are best for both representations, which agrees with SEAKR and CtrlA #c(10, 11). At *matched* layers the SAE probe is level with or ahead of dense on test: at L22 it scores 50.1 versus 48.2 (paired bootstrap per dataset: NQ +1.0, TriviaQA +2.3, PopQA +0.6, HotpotQA −0.5; none significant). The SAE deficit in the headline comparison comes from L29 transferring worse to PopQA, not from a general loss of information.

*SAE width.* The 262k SAEs do not improve on 16k (L22: 46.3 versus 50.1 test). With 16× more features and the same 8,000 labels, the probe has more noise to fit, and the narrower dictionary is the better choice for routing.

*Sparsity and binarisation.* An L1 penalty gives similar scores with far fewer active weights. At L22 it is slightly below L2 on test (49.6 versus 50.1). Binarising the codes (feature fires or not) costs only 0–2 points, so most of the routing signal is *which* features fire, not how strongly. This supports an interpretable reading of the router.

*Pooling.* Mean-pooling over question tokens hurts both representations by 6–10 test points relative to the last token. The last token, where the model has read the whole question, carries the decision.

*Label efficiency.* Fig. 4 shows test macro-F1 against the label budget. Both probes improve steadily, and dense stays ahead at every budget. At 5k labels the gap narrows to 3 points (45.5 versus 48.4). So H2's few-label advantage is not supported for the L29/L22 pair.

#figure(image("figs_final/f2_label_eff.png", width: 100%), caption: [Label efficiency: mean macro-F1 over four answerable test sets, 3 seeds per budget ("all" = 8,000 labels, 1 seed).])

*Feature necessity (drop top-k).* We rank SAE features by probe weight per class at L22 and remove the top $k$ per class (Fig. 5). Zeroing only the top 10 per class (30 of 16,384 features) drops macro-F1 from 53.5 to 44.5. Zeroing 72 features drops it to 26.2. Retraining without them recovers part of the loss (51.0 at 30 removed), so the information is concentrated in a few features with partial redundancy elsewhere.

#figure(image("figs_final/f5_dropk.png", width: 100%), caption: [Removing the highest-weight SAE features (layer 22).])

*What the features are.* Neuronpedia auto-interpretations of the top L22 features include, for ABSTAIN, "phrase or proverb", "epoch counting" and "as cameo"; for ANSWER, "locations including US states"; and for RETRIEVE, "proper nouns following prepositions", "past time references" and "academic citations". RETRIEVE features thus fire on specific named entities and dated facts, which is where BM25 helps, and ANSWER features fire on common geographic knowledge and famous entities (Table 4). Several top features have no clean description, so we treat these labels as suggestive.

#figure(
  tbl(
    (auto, auto, 1fr, 1.1fr),
    (left, right, left, left),
    ([Class], [Feature], [Neuronpedia description], [Top promoted tokens]),
    ([ABSTAIN], [6503], [phrase or proverb], [slogan, motto, phrase, quote]),
    ([ABSTAIN], [12221], [as cameo], [cameo, cameos, contributions]),
    ([ABSTAIN], [2989], [numbers followed by options or chapter], [7, 9, ninth, 8, seventeen]),
    ([ANSWER], [685], [(none)], [famously, revered, iconic]),
    ([ANSWER], [804], [locations including US states], [Illinois, Florida, California, Utah]),
    ([RETRIEVE], [12215], [proper nouns after prepositions or status words], [WrestleMania, Olympics, Arizona, Haryana]),
    ([RETRIEVE], [894], [past time references], [during, durante, "in the year"]),
    ([RETRIEVE], [5104], [academic citations], [weekly, ongoing]),
  ),
  kind: table,
  supplement: [Table],
  caption: [Top-weighted SAE 16k features at layer 22 with Neuronpedia auto-interpretations.],
)

*Two-way versus three-way.* Removing the ABSTAIN class forces every question to be answered. This is evaluated under H3 below.

= 7. Evaluations

== 7.1 Gate summary

#figure(
  tbl(
    (auto, 1fr, auto),
    (left, left, left),
    ([Gate], [Requirement → result], [Status]),
    ([0], [Each class ≥ 15% → 35.4 / 36.6 / 28.0%], [Pass]),
    ([1], [Label error < 10% → 7.5% (15/200)], [Pass]),
    ([2], [≤ 2 pt loss, ≥ 30% fewer calls → −0.35 pt, −50.8%], [Pass]),
    ([3], [Latency for every method at batch size 1], [Pass]),
  ),
  kind: table,
  supplement: [Table],
  caption: [Project gates.],
)

== 7.2 H1: accuracy versus retrieval calls

#figure(image("figs_final/f1_tradeoff.png", width: 100%), caption: [Accuracy against retrieval calls on 4,000 answerable test questions (NQ, TriviaQA, PopQA, HotpotQA). The shaded region satisfies Gate 2.])

On the combined answerable test set (Fig. 6), the selected SAE router reaches 31.5% accuracy against 31.9% for always-retrieve while calling BM25 on only 49.2% of questions. *H1 holds.* The dense router is more accurate (34.0%, +2.1 points over B1) with 39.6% fewer calls. B4 also passes (−0.6 points, 35% fewer calls) but needs a generation step to decide. B8, which reads only the question text, saves calls but loses 4.1 points.

#figure(
  tbl(
    (1fr, auto, auto, auto, auto),
    (left, right, right, right, right),
    ([Router], [NQ], [TriviaQA], [PopQA], [HotpotQA]),
    ([B1 always retrieve], [20.0 (100)], [51.0 (100)], [34.0 (100)], [22.5 (100)]),
    ([B4 verbalised], [21.0 (68)], [53.7 (34)], [30.2 (83)], [20.1 (74)]),
    ([B8 question text], [20.2 (56)], [50.1 (30)], [20.6 (28)], [20.3 (39)]),
    ([*SAE 16k @ L29*], [*22.3* (47)], [*53.0* (32)], [28.6 (58)], [22.2 (60)]),
    ([Dense @ L22], [23.0 (57)], [55.0 (33)], [34.5 (83)], [23.4 (68)]),
  ),
  kind: table,
  supplement: [Table],
  caption: [Accuracy % (retrieval-call % in brackets) per test set.],
)

Per dataset, the SAE router *beats* always-retrieve on NQ (+2.3) and TriviaQA (+2.0) while retrieving for under half the questions: answering from memory avoids being distracted by irrelevant passages. It matches B1 on HotpotQA (−0.3 points at 60% of the calls). PopQA is the hardest case. Long-tail entities need retrieval (+16.8 points from BM25), and the dense router learns to retrieve more there (83% of calls against 58%).

== 7.3 H2: SAE versus dense

On validation, the selected SAE trails the selected dense probe by 1.2 macro-F1 points, within the 2-point margin. On answerable test data the gap is 2.4 points (45.8 versus 48.2), and the paired bootstrap over routed decisions gives −1.9 points [−3.5, −0.3]. Most of this gap comes from PopQA (−7.7 points); NQ, TriviaQA and HotpotQA are within ±1 point with intervals covering zero. At matched layers (Section 6) the SAE probe is level with or ahead of dense.

#figure(image("figs_final/f7_selfaware.png", width: 100%), caption: [Abstention recall on 500 SelfAware unanswerable questions (thresholds tuned on validation at 10% false refusal).])

On *abstention*, SAE probes are much better (Fig. 7). The selected SAE router abstains on 83% of SelfAware's unanswerable questions, against 26% for the selected dense router, 21% for B4 and 13% for B8. The SAE advantage in SelfAware macro-F1 is significant at all three tested layers (+10 to +13 points, 95% CIs excluding zero). Over all test data, the SAE router correctly abstains on 497 of 1,887 ABSTAIN-labelled questions at L22, against 307 for dense. H2 is therefore *supported on abstention and transfer to unanswerable questions*, while dense remains slightly stronger on answerable-question macro-F1.

== 7.4 H3: unsafe answers on index holes

#figure(image("figs_final/f3_holes.png", width: 100%), caption: [Wrong answers per hole question (abstentions count as safe). Number of hole questions in brackets.])

On the 894 hole questions, the three-way SAE router gives a wrong answer on 61.0% of them, against 82.2% for the same probe trained two-way (Fig. 8). It answers 84.3% of hole questions, and among those it attempts the error rate is still lower (72.3% versus 82.2%). The router cannot see the index, but it learns to send questions the model knows to ANSWER and to abstain on those that neither path solves. *H3 holds* on all three datasets (Table 7).

#figure(
  tbl(
    (1fr, auto, auto, auto, auto),
    (left, right, right, right, right),
    ([Hole questions], [n], [Coverage], [Wrong / attempted], [Wrong / all]),
    ([NQ, three-way], [309], [78.3], [79.8], [*62.5*]),
    ([NQ, two-way], [309], [100], [89.6], [89.6]),
    ([TriviaQA, three-way], [283], [84.8], [55.0], [*46.6*]),
    ([TriviaQA, two-way], [283], [100], [70.7], [70.7]),
    ([HotpotQA, three-way], [302], [90.1], [80.9], [*72.8*]),
    ([HotpotQA, two-way], [302], [100], [85.4], [85.4]),
  ),
  kind: table,
  supplement: [Table],
  caption: [Index-hole risk (%) for the SAE L29 probe trained three-way versus two-way.],
)

== 7.5 H4: latency

#figure(image("figs_final/f4_latency.png", width: 100%), caption: [End-to-end latency at batch size 1 on an NVIDIA L4, 200 questions. Each router's actual decisions are timed.])

#figure(
  tbl(
    (1fr, auto, auto),
    (left, right, right),
    ([Stage], [p50 (ms)], [p95 (ms)]),
    ([Controller: forward pass + SAE + probe], [56.5], [58.5]),
    ([BM25 top-3 (top-100 for holes)], [19.1 (95.1)], [108.3 (264.3)]),
    ([Closed-book generation], [201.1], [399.9]),
    ([RAG generation (3 passages)], [229.9], [513.3]),
    ([B4 verbalised decision], [249.1], [257.7]),
  ),
  kind: table,
  supplement: [Table],
  caption: [Per-stage latency, batch size 1.],
)

The controller costs 57 ms, about a quarter of one generation, and it saves both retrieval and the longer RAG prompt when it routes to ANSWER or ABSTAIN. The SAE router's median end-to-end latency is 212 ms against 251 ms for always-retrieve (−16%), with a lower p95 (480 versus 541 ms). B4 needs a full generation to decide and has a median of 466 ms, more than twice the SAE router. *H4 holds:* the pre-generation controller is faster than both always-retrieve and the generation-based self-router. These timings use the layer-17 routers. Selected-layer (L29/L22) timing scripts are prepared, and the controller cost is expected to be similar because it is one forward pass to a different depth.

== 7.6 Error analysis

The most common mistake is sending ABSTAIN-labelled questions to RETRIEVE (1,004 of 1,887 for SAE L22). This is the safe direction, because it only costs a retrieval call. Within the PopQA long tail, closed-book accuracy does not rise with popularity (27% in the rarest quartile, 11% in the most popular), and both probes retrieve more as popularity rises (SAE: 65% → 87% of calls; Table 9).

#figure(
  tbl(
    (1fr, auto, auto, auto, auto, auto),
    (left, right, right, right, right, right),
    ([PopQA quartile], [n], [Closed], [BM25], [SAE acc.], [SAE calls]),
    ([Q1 (rarest)], [254], [27.2], [35.8], [39.8], [65.0]),
    ([Q2], [250], [18.0], [38.8], [38.4], [80.8]),
    ([Q3], [250], [12.0], [26.4], [26.8], [88.4]),
    ([Q4 (most popular)], [246], [11.4], [35.0], [35.8], [87.0]),
  ),
  kind: table,
  supplement: [Table],
  caption: [PopQA by popularity quartile (%), SAE 16k L22 router.],
) A hand-read sample of routing errors shows many disagreements on questions with time-dependent or ambiguous gold answers (e.g. "where was War for the Planet of the Apes filmed").

= 8. Limitations

- *Index holes are simulated.* Answer-bearing passages are filtered from the BM25 top 100 by lexical match at query time; evidence is not deleted from the corpus, and paraphrased evidence may remain.
- *Gold passages are stand-ins.* Outside HotpotQA, gold is the best answer-bearing BM25 passage, so gold accuracy is an upper bound, and 372 PopQA questions have no gold run.
- *String matching.* Gate 1 passes by point estimate (7.5%), but the 95% interval reaches 12.0% and the lenient protocol gives 11%. Errors mostly inflate ABSTAIN.
- *Scope.* One generator (Gemma 3 4B), one retriever (BM25) and English open-domain QA. Retriever shift (e.g. a dense e5 retriever) and the sampling-based B6 baseline remain future work.
- *Latency* was timed on 200 NQ + TriviaQA questions with the layer-17 routers; selected-layer timing is pending.
- *SelfAware* has no answerable questions, so answer accuracy there is zero by definition and we report abstention recall instead.

= 9. Conclusions

A single forward pass over the question carries enough information to decide, before generating, whether a model should answer from memory, retrieve or abstain. A linear probe on pretrained Gemma Scope 2 SAE features keeps always-retrieve accuracy (−0.35 points) with half the retrieval calls, beats always-retrieve on NQ and TriviaQA, cuts wrong answers on index-hole questions by a quarter, and lowers median latency by 16% with a 57 ms controller. All four project gates pass, and H1, H3 and H4 are supported.

The SAE-versus-dense comparison is nuanced. Dense probes are slightly stronger on answerable data, mainly on long-tail PopQA. SAE probes are level at matched layers and much better at recognising unanswerable questions, which is the behaviour that matters most for safe RAG. Binarisation and drop-top-k results show that the router relies on a small number of features, some of them interpretable, which makes the controller inspectable. Natural next steps are a dense retriever for the retriever-shift test, combining SAE and dense features, and checking whether the abstention features transfer to larger Gemma 3 models.

= References

#set text(size: 8pt)
#set par(justify: false, leading: 0.44em, spacing: 0.34em)
#let refs = (
  [Huben, Cunningham et al. (2024). Sparse Autoencoders Find Highly Interpretable Features in Language Models. ICLR.],
  [Google DeepMind (2025). Gemma Scope 2: SAEs and transcoders for the Gemma 3 family.],
  [Kantamneni et al. (2025). Are Sparse Autoencoders Useful? A Case Study in Sparse Probing. ICML.],
  [Gallifant et al. (2025). Sparse Autoencoder Features for Classifications and Transferability.],
  [Xin et al. (2025). Sparse Latents Steer Retrieval-Augmented Generation. ACL.],
  [Zhao et al. (2025). Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering (SpARE). NAACL.],
  [Xiong et al. (2026). Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders (RAGLens). ICLR.],
  [Abdaljalil et al. (2025). SAFE: A Sparse Autoencoder-Based Framework for Robust Query Enrichment and Hallucination Mitigation in LLMs.],
  [Baek et al. (2024). Probing-RAG: Self-Probing to Guide Language Models in Selective Document Retrieval.],
  [Yao et al. (2024). SEAKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation.],
  [Liu et al. (2024). CtrlA: Adaptive Retrieval-Augmented Generation via Inherent Control.],
  [Jeong et al. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity. NAACL.],
  [Jiang et al. (2023). Active Retrieval Augmented Generation (FLARE). EMNLP.],
  [Su et al. (2024). DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of LLMs. ACL.],
  [Chandradevan et al. (2026). Search Engines Never Say No: How Frozen Agents React When the Retrieval Tool Refuses.],
  [Mallen et al. (2023). When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories (PopQA). ACL.],
  [Asai et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR.],
  [Yin et al. (2023). Do Large Language Models Know What They Don't Know? (SelfAware). Findings of ACL.],
  [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond. FnTIR.],
  [Gemma Team (2025). Gemma 3 Technical Report.],
  [Rajamanoharan et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders.],
  [Bricken et al. (2023). Towards Monosemanticity: Decomposing Language Models with Dictionary Learning. Anthropic.],
  [Kwiatkowski et al. (2019). Natural Questions: A Benchmark for Question Answering Research. TACL.],
  [Joshi et al. (2017). TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. ACL.],
  [Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP.],
  [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.],
  [Lin et al. (2021). Pyserini: A Python Toolkit for Reproducible Information Retrieval Research. SIGIR.],
)
#for (i, r) in refs.enumerate() [
  #grid(
    columns: (1.6em, 1fr),
    text(fill: muted)[#(i + 1)], r,
  )
]

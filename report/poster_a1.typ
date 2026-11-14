// A1 portrait poster. Compile from this folder: typst compile poster_a1.typ poster_a1.pdf

#let ink = rgb("#0b0b0b")
#let ink2 = rgb("#52514e")
#let muted = rgb("#7a7974")
#let rule = rgb("#dcdbd6")
#let wash = rgb("#f3f2ee")
#let blue = rgb("#2a78d6")
#let orange = rgb("#eb6834")
#let navy = rgb("#123a6b")
#let sans = ("Fira Sans", "Segoe UI", "Liberation Sans", "Arial")

#set document(title: "SAE-Gated RAG: Poster", author: ("Gopal Kataria", "Nidhi Vaidya", "Manya Jain", "Salvik Krishna"))
#set page(paper: "a1", margin: (x: 2.6cm, top: 2.2cm, bottom: 2.2cm), fill: white)
#set text(font: sans, size: 22pt, fill: ink, lang: "en")
#set par(justify: false, leading: 0.5em, spacing: 0.75em)
#set list(indent: 0.3em, body-indent: 0.5em, spacing: 0.6em, marker: text(fill: blue)[●])
#show raw: set text(font: "DejaVu Sans Mono", size: 21pt)

#let panel(title, body) = block(
  width: 100%,
  breakable: false,
  below: 0.8cm,
  {
    block(below: 0.55cm, {
      text(size: 31pt, weight: "bold", fill: navy, title)
      v(-0.35cm)
      line(length: 100%, stroke: 3pt + blue)
    })
    body
  },
)
#let fig(path, cap) = block(width: 100%, above: 0.5cm, below: 0.6cm, {
  align(center, image(path, width: 88%))
  v(-0.15cm)
  text(size: 17pt, fill: ink2, cap)
})
#let stat(big, small, col: blue) = box(
  width: 100%,
  fill: wash,
  radius: 10pt,
  inset: (x: 0.5cm, y: 0.45cm),
  align(center, {
    text(size: 58pt, weight: "bold", fill: col, big)
    linebreak()
    v(-0.1cm)
    text(size: 21pt, fill: ink2, small)
  }),
)
#let node(body, fill: wash, fg: ink) = box(
  width: 100%,
  inset: (x: 0.4cm, y: 0.32cm),
  radius: 8pt,
  fill: fill,
  align(center, text(size: 19pt, fill: fg, body)),
)
#let down = align(center, block(above: 0.12cm, below: 0.12cm, text(fill: muted, size: 24pt)[↓]))
#let tbl(cols, aligns, ..rows) = {
  set text(size: 17.5pt)
  table(
    columns: cols,
    stroke: none,
    inset: (x: 0.2cm, y: 0.14cm),
    align: aligns,
    table.hline(stroke: 1.6pt + ink),
    ..rows.pos().slice(0, 1).flatten().map(c => text(weight: "bold", c)),
    table.hline(stroke: 1pt + rule),
    ..rows.pos().slice(1).flatten(),
    table.hline(stroke: 1.6pt + ink),
  )
}

// ---------------- Header ----------------
#block(width: 100%, fill: navy, radius: 14pt, inset: (x: 1.4cm, y: 0.9cm), below: 0.8cm, {
  set text(fill: white)
  text(size: 78pt, weight: "bold")[SAE-Gated RAG]
  v(-0.6cm)
  text(size: 36pt, weight: "medium")[Deciding to Answer, Retrieve or Abstain Before Generation]
  v(0.2cm)
  text(size: 25pt)[Gopal Kataria (2023112006) · Manya Jain (2024102068) · Nidhi Vaidya (2023114005) · Salvik Krishna (2023113015)]
  v(-0.1cm)
  text(size: 21pt, fill: rgb("#c9d8ee"))[Code and results: github.com/bropal404/lma-sae-rag-proj]
})

// ---------------- Hero numbers ----------------
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 0.8cm,
  stat[−50.8%][retrieval calls vs. always-retrieve],
  stat[−0.35 pt][accuracy vs. always-retrieve \ (Gate 2: ≤ 2 pt)],
  stat[83%][abstention on unanswerable \ questions (dense probe: 26%)],
  stat[57 ms][controller cost: one forward \ pass + sparse probe],
)
#v(0.8cm)

// ---------------- Three columns ----------------
#grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 1.4cm,

  // ===== Column 1 =====
  [
    #panel[Motivation][
      - Standard RAG *retrieves for every question*: slow, and irrelevant passages can mislead a model that already knew the answer.
      - Adaptive methods (FLARE, DRAGIN, SEAKR) decide *during generation*, so they pay the decoding cost first.
      - When the index lacks the answer, search still returns passages, and the model answers wrongly.

      *Can the model's activation on the question alone decide, before generating, whether to answer, retrieve or abstain?*
    ]

    #panel[Method][
      #node[Question only (closed-book prompt)]
      #down
      #node[Frozen *Gemma 3 4B*: one forward pass, \ last-token residual stream at layer $L$]
      #down
      #grid(
        columns: (1fr, 1fr),
        gutter: 0.35cm,
        node(fill: rgb("#dce9f8"))[*Gemma Scope 2 SAE* \ 16k sparse features],
        node(fill: rgb("#fbe3d8"))[*Dense* residual \ 2,560-d (comparison)],
      )
      #down
      #node(fill: navy, fg: white)[*Logistic-regression probe* (the only trained part) \ + ABSTAIN threshold tuned on validation]
      #down
      #grid(
        columns: (1fr, 1fr, 1fr),
        gutter: 0.35cm,
        node[*ANSWER* \ closed-book],
        node[*RETRIEVE* \ BM25 top-3],
        node[*ABSTAIN* \ decline],
      )
      Layers and SAE width are selected on *validation only*. Test sets are never used for selection.
    ]

    #panel[Data and Labels][
      #tbl(
        (1fr, auto, auto),
        (left, right, right),
        ([Source], [Questions], [Role]),
        ([NQ-open, TriviaQA], [11,000], [train / val / test]),
        ([PopQA (long tail)], [1,000], [transfer]),
        ([HotpotQA], [1,000], [multi-hop]),
        ([SelfAware], [500], [unanswerable]),
      )
      - *21,015,324* Wikipedia passages, BM25 retrieval.
      - Each question is answered 3 ways: *closed-book*, *BM25* and *gold passage*.
      - *Label:* closed-book correct → ANSWER; else BM25 or gold correct → RETRIEVE; else ABSTAIN.
      - *Index holes:* about 30% of questions have answer passages removed, simulating a search that cannot help.
      - *Label audit:* 200 labels checked by hand, *7.5% error* (Gate 1 ✓). Class balance 35 / 37 / 28% (Gate 0 ✓).
    ]
  ],

  // ===== Column 2 =====
  [
    #panel[H1: Half the retrieval, same accuracy][
      #fig("figs_poster/f1_tradeoff.png")[4,000 answerable test questions (NQ, TriviaQA, PopQA, HotpotQA). The shaded region meets Gate 2.]
      #tbl(
        (1fr, auto, auto, auto),
        (left, right, right, right),
        ([Accuracy % (calls %)], [NQ], [TriviaQA], [HotpotQA]),
        ([B1 always retrieve], [20.0 (100)], [51.0 (100)], [22.5 (100)]),
        ([*SAE router*], [*22.3* (47)], [*53.0* (32)], [22.2 (60)]),
      )
      The SAE router *beats always-retrieve* on NQ and TriviaQA while retrieving for under half the questions. Answering from memory avoids distracting passages.
    ]

    #panel[H3: Fewer wrong answers on index holes][
      #fig("figs_poster/f3_holes.png")[Wrong answers per hole question (abstaining counts as safe), 894 questions.]
      Adding ABSTAIN cuts unsafe answers from *82.2% to 61.0%*. Among questions it still answers, the error is also lower (72% vs 82%).
    ]

    #panel[H4: Faster than always-retrieve][
      #fig("figs_poster/f4_latency.png")[Batch size 1, NVIDIA L4, 200 questions; each router's real decisions are timed.]
      The median is *212 ms vs 251 ms* for always-retrieve, and less than half the latency of the generation-based self-router B4 (466 ms).
    ]
  ],

  // ===== Column 3 =====
  [
    #panel[H2: SAE vs dense probe][
      #fig("figs_poster/f7_selfaware.png")[Abstention recall on 500 unanswerable SelfAware questions, at a 10% false-refusal budget.]
      - SAE probes *recognise unanswerable questions far better*: 83% vs 26% recall, significant at every layer.
      - On answerable questions, dense is ahead by 1.2 pt (val) and 2.4 pt (test), mostly on long-tail PopQA. At a matched layer (L22), SAE is level or ahead.
    ]

    #panel[Interpretability][
      #fig("figs_poster/f5_dropk.png")[Removing the highest-weight SAE features (layer 22).]
      Zeroing just *30 of 16,384 features* drops macro-F1 from 53.5 to 44.5, so the decision lives in a few features. Binarised codes lose only 0–2 pt, so what matters is *which* features fire.
      #v(0.2cm)
      #tbl(
        (auto, 1fr),
        (left, left),
        ([Routes to], [Example feature (Neuronpedia)]),
        ([RETRIEVE], ["past time references"]),
        ([RETRIEVE], ["proper nouns after prepositions"]),
        ([ANSWER], ["locations including US states"]),
        ([ABSTAIN], ["phrase or proverb"]),
      )
    ]

    #panel[Conclusions][
      #tbl(
        (auto, 1fr, auto),
        (left, left, center),
        ([Gate], [Criterion], [Result]),
        ([0], [each class ≥ 15%], [✓]),
        ([1], [label error < 10% (7.5%)], [✓]),
        ([2], [≤ 2 pt loss, ≥ 30% fewer calls], [✓]),
        ([3], [latency for every method], [✓]),
      )
      - A *single forward pass* over the question is enough to route RAG before generation.
      - Three-way routing makes RAG *safer* when the index lacks the answer.
      - SAE features give an *inspectable* router that excels at abstention.
    ]
  ],
)

// ---------------- Footer ----------------
#v(1fr)
#line(length: 100%, stroke: 1pt + rule)
#text(size: 16pt, fill: muted)[
  *References.* Gemma Team (2025) Gemma 3 · Google DeepMind (2025) Gemma Scope 2 · Rajamanoharan et al. (2024) JumpReLU SAEs · Jiang et al. (2023) FLARE · Su et al. (2024) DRAGIN · Yao et al. (2024) SEAKR · Baek et al. (2024) Probing-RAG · Kantamneni et al. (2025) Sparse probing · Mallen et al. (2023) PopQA · Yin et al. (2023) SelfAware · Kwiatkowski et al. (2019) NQ · Joshi et al. (2017) TriviaQA · Yang et al. (2018) HotpotQA · Karpukhin et al. (2020) DPR.
]

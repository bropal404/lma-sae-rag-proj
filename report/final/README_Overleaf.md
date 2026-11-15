# SAE-Gated RAG — final report source

This folder is a self-contained Overleaf project. Upload the ZIP through **Overleaf → New Project → Upload Project**, then select **pdfLaTeX** as the compiler (Menu → Settings). The bibliography uses **biblatex + biber** and is compiled automatically by Overleaf.

Main source: `main.tex`; bibliography: `references.bib`; vector figures: `figures/*.pdf`. The chart generator `generate_figures.py` is included for optional local regeneration from the repository's exported results (`python report/final/generate_figures.py --results-root .`). It is not needed to compile the report in Overleaf. All figure PDFs are included. Figure 1 (the pre-generation SAE-versus-dense flowchart) is editable TikZ in `main.tex` and appears in the Introduction. The title is monochrome; plots and the diagram retain color.

The article is A4, two-column, 11-pt body text and less than the eight-page limit. All mandatory TA headings are included. The authors and registration numbers were taken from the prior team mid-evaluation report; verify them before uploading.

Repository cited: https://github.com/bropal404/lma-sae-rag-proj/tree/final-evaluation

## Chart provenance

All chart numbers come from the final branch's exported CSVs:

- `results/final/gate2_aggregate.csv` and `gate2_by_dataset.csv`: routed accuracy and retrieval calls.
- `results/final/model_selection.csv`: SAE/dense validation layers.
- `results/final/h2_label_efficiency.csv`: label-budget curves.
- `results/final/h3_hole_risk.csv`: wrong answer and coverage on lexical holes.
- `results/sagemaker/probes/...`: layer-22 ablations and feature removal.
- `results/sagemaker/latency/routed_l29.csv`: p50 and p95 composed latency.
- `results/analysis/selected_model_bootstrap.csv`: selected-model paired bootstrap CI.
- `results/final/gate_summary.json`: Gate 0-3 / H1-H4 audit.

**Important interpretation notes:** The 7.5% strict hand-check point estimate does not guarantee <10% population error (95% Wilson upper bound ~12%). H2's *probe-grid* macro-F1 and the separate *threshold-routed* bootstrap macro-F1 are not identical estimands. H4 does not pass the originally specified p50 criterion despite lower mean and p95. Latency is composed from measured stages and not a single timed serving request.

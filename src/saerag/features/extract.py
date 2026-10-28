"""Query-time activations and SAE features (D5). Owner: Person 3.

    uv run python -m saerag.features.extract --files pilot nq_train ...

Runs the closed-book prompt (prompts.closed_book_messages, chat template applied, with
the generation prompt) through the frozen generator once per question and saves, for
each configured layer:
    data/features/{tag}/L{layer}/resid_last_{file}.npy    [n, d_model] float16
    data/features/{tag}/L{layer}/resid_mean_{file}.npy    mean over the question tokens
    data/features/{tag}/L{layer}/sae{width}_last_{file}.npy  sparse codes (scipy .npz if large)
    data/features/{tag}/qids_{file}.json                  row order
Only questions are needed, so this runs before any label exists.
"""

from __future__ import annotations

import argparse

from ..config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    # TODO(P3): transformers forward hooks on the residual stream at cfg.model.layers;
    # "last token" = final token of the templated prompt. Encode with sae_lens SAE.load
    # from cfg.model.sae_release. Check that the SAE's hook point matches the layer.
    raise NotImplementedError(cfg)


if __name__ == "__main__":
    main()

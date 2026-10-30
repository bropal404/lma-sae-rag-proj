"""Download the generator and SAE weights into HF_HOME. Run on a node with internet
(the login node) before submitting GPU jobs, since compute nodes may be offline.

    uv run python -m saerag.prefetch                  # model + 16k SAEs
    uv run python -m saerag.prefetch --widths 16k 262k
"""

from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download, snapshot_download

from .config import load_config
from .features.sae import sae_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--widths", nargs="*", help="default: model.sae_widths")
    args = ap.parse_args()
    m = load_config(args.config)["model"]
    print("model:", snapshot_download(m["generator"], allow_patterns=["*.json", "*.jinja", "*.safetensors",
                                                                      "tokenizer*", "*.model"]))
    for L in m["sae_layers"]:
        for w in args.widths or m["sae_widths"]:
            sid = sae_id(m["sae_site"], L, w, m["sae_l0"])
            for f in ("config.json", "params.safetensors"):
                hf_hub_download(m["sae_repo"], f"{sid}/{f}")
            print("sae:", sid)


if __name__ == "__main__":
    main()

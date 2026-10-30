"""Minimal JumpReLU SAE loader for Gemma Scope 2 (no sae-lens registry needed).

Each SAE folder holds config.json and params.safetensors with w_enc [d, W], b_enc [W],
threshold [W], w_dec [W, d], b_dec [d]. The hook point is the output of decoder block
`layer`, which is hidden_states[layer + 1] in a HF forward pass.
"""

from __future__ import annotations

import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def sae_id(site: str, layer: int, width: str, l0: str) -> str:
    return f"{site}/layer_{layer}_width_{width}_l0_{l0}"


class JumpReLUSAE:
    def __init__(self, params: dict[str, torch.Tensor], config: dict):
        self.w_enc, self.b_enc = params["w_enc"], params["b_enc"]
        self.w_dec, self.b_dec = params.get("w_dec"), params["b_dec"]
        self.threshold = params["threshold"]
        self.config = config

    @classmethod
    def load(cls, repo: str, sid: str, device: str | torch.device = "cpu",
             decoder: bool = True) -> JumpReLUSAE:
        """decoder=False drops w_dec (encode-only), halving memory for wide SAEs."""
        with open(hf_hub_download(repo, f"{sid}/config.json")) as f:
            config = json.load(f)
        params = load_file(hf_hub_download(repo, f"{sid}/params.safetensors"))
        if not decoder:
            params.pop("w_dec")
        return cls({k: v.float().to(device) for k, v in params.items()}, config)

    @property
    def width(self) -> int:
        return self.w_enc.shape[1]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x.float() @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes @ self.w_dec + self.b_dec

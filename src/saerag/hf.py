"""Loading the generator with transformers, shared by feature extraction and HF generation.

dtype "auto": bf16 on GPUs that support it (Ampere+: A100, 3090, L4). Otherwise fp32, because
Gemma 3 overflows in fp16; on 2080 Ti / T4 nodes fp32 4B (~17 GB) is split across the visible
GPUs with device_map="auto", so request 2+ GPUs there.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_dtype(dtype: str = "auto") -> torch.dtype:
    if dtype != "auto":
        return getattr(torch, dtype)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float32


def load_model(name: str, dtype: str = "auto"):
    """Returns (tokenizer, model, input_device). Tokenizer pads on the left."""
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    td = pick_dtype(dtype)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(name, dtype=td, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=td)
    model.eval()
    print(f"loaded {name} as {td} on {getattr(model, 'hf_device_map', None) or model.device}")
    return tok, model, model.device


def chat_prompt(tok, messages: list[dict]) -> str:
    """Templated prompt string (starts with <bos>; tokenize it with add_special_tokens=False)."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def first_line(text: str) -> str:
    return text.strip().split("\n")[0].strip()

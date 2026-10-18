"""Shared model loading + prompt formatting for training and evaluation.

SiliconLM's own stages load Qwen2.5-Coder-1.5B in 4-bit and `merge_and_unload()` each previous
adapter (DAPT -> SFT -> DPO) before training the next one. `load_causal_lm` reproduces that chain,
so a SiliconLM checkpoint is `adapters=[dapt, sft, dpo]` and the base model is `adapters=[]`.
"""

from __future__ import annotations

import os

DEFAULT_BASE = "Qwen/Qwen2.5-Coder-1.5B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_causal_lm(base: str = DEFAULT_BASE, adapters: list[str] | None = None,
                   load_4bit: bool = True, trainable_adapter: str | None = None):
    """-> (tokenizer, model). `adapters` are merged in order; `trainable_adapter` stays a live LoRA."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # one process per GPU under `accelerate launch` (DDP): each rank keeps its own full 4-bit copy
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    kwargs: dict = {"device_map": {"": local_rank} if torch.cuda.is_available() else None}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    for path in adapters or []:
        model = PeftModel.from_pretrained(model, path, is_trainable=False).merge_and_unload()
    if trainable_adapter:
        model = PeftModel.from_pretrained(model, trainable_adapter, is_trainable=False)
    model.eval()
    return tok, model


def format_prompt(tok, spec: str) -> str:
    """Chat-template the spec like SiliconLM's SFT did; plain text for tokenizers without a template."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": spec}],
                                       tokenize=False, add_generation_prompt=True)
    return spec + "\n"

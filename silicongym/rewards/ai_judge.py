"""LLM-as-judge reward: asks a model to grade a Verilog module against its spec, returns a score in [0, 1].

Backends (config `judge:` section):
  openai_compat  any OpenAI-style /chat/completions endpoint (Groq, Mistral, OpenAI, a local vLLM server).
                 Needs `base_url`, `model`, and an API key in the env var named by `api_key_env`.
  local          a Hugging Face causal LM on the GPU (`model`), greedy decoding.
The judge never sees the testbench or simulator output, which is the point: it is a learned-style
proxy whose disagreements with the simulator are measured in eval/agreement.py.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import urllib.request
import json
from typing import Callable

PROMPT = """You are a strict hardware design reviewer. Decide whether the Verilog module correctly implements the specification.

Specification:
{spec}

Verilog module:
{code}

Rate functional correctness from 0 (completely wrong or empty) to 10 (certainly correct). Reply with only the integer."""

_INT = re.compile(r"\b(10|[0-9])\b")


def parse_score(text: str) -> float | None:
    m = _INT.search(text or "")
    return int(m.group(1)) / 10 if m else None


class Judge:
    def __init__(self, complete: Callable[[str], str], min_interval_s: float = 0.0):
        self.complete, self.min_interval_s = complete, min_interval_s
        self._cache: dict[str, float | None] = {}
        self._last = 0.0

    def score(self, spec: str, code: str) -> float | None:
        prompt = PROMPT.format(spec=spec, code=code)
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in self._cache:
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                self._cache[key] = parse_score(self.complete(prompt))
            except Exception:  # network / quota / model errors must not kill a training run
                self._cache[key] = None
        return self._cache[key]

    @classmethod
    def from_config(cls, conf: dict) -> "Judge":
        backend = conf.get("backend", "openai_compat")
        if backend == "openai_compat":
            return cls(_openai_compat(conf), conf.get("min_interval_s", 2.0))
        if backend == "local":
            return cls(_local_hf(conf))
        raise ValueError(f"unknown judge backend {backend!r}")


def _openai_compat(conf: dict) -> Callable[[str], str]:
    key = os.environ.get(conf.get("api_key_env", "JUDGE_API_KEY"), "")
    if not key:
        raise RuntimeError(f"set ${conf.get('api_key_env', 'JUDGE_API_KEY')} for the AI judge")
    url = conf["base_url"].rstrip("/") + "/chat/completions"

    def complete(prompt: str) -> str:
        body = json.dumps({"model": conf["model"], "temperature": 0, "max_tokens": 8,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    return complete


def _local_hf(conf: dict) -> Callable[[str], str]:
    import torch
    from silicongym.train.model import load_causal_lm

    tok, model = load_causal_lm(conf["model"], [], conf.get("load_4bit", True))

    def complete(prompt: str) -> str:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, do_sample=False, max_new_tokens=6, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return complete

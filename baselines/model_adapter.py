"""Model adapter for ProtStructQA baselines.

Three backend modes:
  - "vllm-http"     : POST to a running vLLM server (recommended; matches
                       SBE's existing setup)
  - "hf-transformers": local in-process HF model (slow, fallback only)
  - "openai-api"    : frontier API sanity check

The adapter exposes ONE method:
    generate(prompt: str, max_tokens: int, temperature: float,
              stop: list[str] | None) -> dict

Returning {"text": str, "n_input_tokens": int, "n_output_tokens": int,
            "stop_reason": str, "elapsed_s": float}.

Used by run_baseline.py.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


# --------------------------- vLLM HTTP --------------------------- #


@dataclass
class VLLMHTTPAdapter:
    """Minimal client for a vLLM /generate or /v1/completions endpoint.

    Configure via $PROTSTRUCTQA_VLLM_URL (default: http://localhost:8000) +
    $PROTSTRUCTQA_VLLM_MODEL (model name as registered with vLLM).
    """
    url: str = ""
    model: str = ""
    timeout_s: float = 120.0

    def __post_init__(self):
        if not self.url:
            self.url = os.environ.get("PROTSTRUCTQA_VLLM_URL",
                                          "http://localhost:8000")
        if not self.model:
            self.model = os.environ.get("PROTSTRUCTQA_VLLM_MODEL", "")

    def generate(self, prompt: str, max_tokens: int = 512,
                  temperature: float = 0.0,
                  stop: list[str] | None = None,
                  guided_grammar: str | None = None) -> dict:
        payload = {
            "model": self.model or "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        if guided_grammar is not None:
            # vLLM accepts GBNF-format grammars under this key. The
            # sampler will filter token candidates to only those that
            # keep the partial output parseable.
            payload["guided_grammar"] = guided_grammar
        endpoint = self.url.rstrip("/") + "/v1/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint, data=data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as e:
            return {"text": "", "n_input_tokens": 0, "n_output_tokens": 0,
                      "stop_reason": "error", "elapsed_s": time.perf_counter() - t0,
                      "error": str(e)}
        dt = time.perf_counter() - t0
        try:
            d = json.loads(body)
        except json.JSONDecodeError:
            return {"text": "", "n_input_tokens": 0, "n_output_tokens": 0,
                      "stop_reason": "error", "elapsed_s": dt,
                      "error": "json_decode_failed"}
        # OpenAI-style response: choices[0].text
        choice = (d.get("choices") or [{}])[0]
        usage = d.get("usage", {})
        return {
            "text": choice.get("text", ""),
            "n_input_tokens": usage.get("prompt_tokens", 0),
            "n_output_tokens": usage.get("completion_tokens", 0),
            "stop_reason": choice.get("finish_reason", ""),
            "elapsed_s": dt,
        }


# --------------------------- HF transformers (in-process) --------- #


@dataclass
class HFTransformersAdapter:
    """In-process generation via Hugging Face transformers. Slow but
    self-contained: useful when no vLLM server is up.

    Lazy-loads the model on first generate() call.
    """
    model_path: str = ""
    device: str = "auto"
    dtype: str = "auto"

    _tokenizer: Any = None
    _model: Any = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        if self.dtype == "auto":
            torch_dtype = "auto"
        elif self.dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif self.dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = "auto"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch_dtype, device_map=self.device,
        )
        self._model.eval()

    def generate(self, prompt: str, max_tokens: int = 512,
                  temperature: float = 0.0,
                  stop: list[str] | None = None) -> dict:
        self._ensure_loaded()
        import torch
        t0 = time.perf_counter()
        ids = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **ids,
                max_new_tokens=max_tokens,
                do_sample=temperature > 1e-6,
                temperature=max(temperature, 1e-6),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0, ids.input_ids.shape[1]:]
        text = self._tokenizer.decode(gen, skip_special_tokens=True)
        # Manual stop-sequence trimming
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
                    break
        dt = time.perf_counter() - t0
        return {
            "text": text,
            "n_input_tokens": int(ids.input_ids.shape[1]),
            "n_output_tokens": int(gen.shape[0]),
            "stop_reason": "stop" if stop else "length",
            "elapsed_s": dt,
        }


# --------------------------- factory ---------------------------- #


def make_adapter(backend: str = "vllm-http", **kwargs):
    if backend == "vllm-http":
        return VLLMHTTPAdapter(**kwargs)
    if backend == "hf-transformers":
        return HFTransformersAdapter(**kwargs)
    raise ValueError(f"unknown backend: {backend!r}; "
                     "use 'vllm-http' or 'hf-transformers'")

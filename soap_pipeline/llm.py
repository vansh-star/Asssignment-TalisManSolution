"""Thin LLM client layer.

``ollama``    - default; local model (e.g. llama3.1:8b, mistral, qwen2.5) -> PHI stays on-prem.
``openai``    - any OpenAI-compatible endpoint (OpenAI, Azure, vLLM, LM Studio, llama.cpp server).
``anthropic`` - Claude API, for machines without a GPU.

Every client exposes ``chat_json(system, messages) -> dict`` and runs with
temperature 0 for reproducible notes. Only the standard library + ``requests``
are used so no vendor SDK is required.
"""
from __future__ import annotations

import json
import os
import re

import requests


def _parse_json(text: str) -> dict:
    """Parse model output, tolerating stray ```json fences or leading prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


class LLMClient:
    def __init__(self, provider: str = "ollama", model: str | None = None, timeout: int = 600):
        self.provider = provider
        self.timeout = timeout
        defaults = {"ollama": "llama3.1:8b", "openai": "gpt-4o-mini", "anthropic": "claude-sonnet-5-5"}
        if provider not in defaults:
            raise ValueError(f"Unknown LLM provider {provider!r}")
        self.model = model or os.getenv("SOAP_LLM_MODEL") or defaults[provider]

    # ------------------------------------------------------------------ #
    def chat_json(self, system: str, messages: list[dict]) -> dict:
        return _parse_json(getattr(self, f"_{self.provider}")(system, messages))

    def _ollama(self, system: str, messages: list[dict]) -> str:
        url = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
        r = requests.post(url, timeout=self.timeout, json={
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "format": "json",            # grammar-constrained JSON decoding
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 16384},
        })
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _openai(self, system: str, messages: list[dict]) -> str:
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        r = requests.post(f"{base}/chat/completions", timeout=self.timeout,
                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                          json={
                              "model": self.model,
                              "temperature": 0,
                              "response_format": {"type": "json_object"},
                              "messages": [{"role": "system", "content": system}, *messages],
                          })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _anthropic(self, system: str, messages: list[dict]) -> str:
        body = {"model": self.model, "max_tokens": 4096, "system": system, "messages": messages}
        # Newer Claude models are deterministic-by-default and reject `temperature`.
        if not self.model.startswith(("claude-sonnet-5", "claude-opus-5", "claude-haiku-5")):
            body["temperature"] = 0
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=self.timeout,
                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                   "anthropic-version": "2023-06-01"},
                          json=body)
        r.raise_for_status()
        return "".join(b["text"] for b in r.json()["content"] if b["type"] == "text")

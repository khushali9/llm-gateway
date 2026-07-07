# inference/llamacpp_client.py
#
# Client for llama.cpp server (CPU inference, GGUF model).
# Same OpenAI-compatible API as vLLM/SGLang — just a different port.
# This is the CPU/edge backend: slower, but no GPU required.

import time
import httpx
import json
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

LLAMACPP_URL = "http://localhost:8003"


class LlamaCppClient:

    def __init__(self, base_url: str = LLAMACPP_URL):
        self.base_url = base_url

    async def generate(
        self,
        prompt:        str,
        max_tokens:    int   = 256,
        temperature:   float = 0.7,
        model:         str   = "mistral-7b-cpu",
        system_prompt: str   = None,
    ) -> dict:
        start = time.perf_counter()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=300.0) as client:  # long timeout: CPU is slow
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                }
            )
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "text":              data["choices"][0]["message"]["content"],
            "prompt_tokens":     data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "latency_ms":        round(latency_ms, 2),
            "model":             "mistral-7b-cpu",
        }

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False

    async def close(self):
        pass

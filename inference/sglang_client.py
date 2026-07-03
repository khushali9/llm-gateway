# inference/sglang_client.py

import time
import httpx
import json
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

SGLANG_URL = "http://localhost:8002"


class SGLangClient:

    def __init__(self, base_url: str = SGLANG_URL):
        self.base_url = base_url

    async def generate(
        self,
        prompt:        str,
        max_tokens:    int   = 256,
        temperature:   float = 0.7,
        model:         str   = "Mistral-7B-Instruct-v0.3",
        system_prompt: str   = None,
    ) -> dict:
        start = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model":       model,
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
            "model":             data["model"],
        }

    async def generate_stream(
        self,
        prompt:        str,
        max_tokens:    int   = 256,
        temperature:   float = 0.7,
        model:         str   = "Mistral-7B-Instruct-v0.3",
        system_prompt: str   = None,
    ) -> AsyncGenerator[str, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model":       model,
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                    "stream":      True,
                },
            ) as response:
                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        pass

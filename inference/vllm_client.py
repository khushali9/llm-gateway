# inference/vllm_client.py
#
# Client for vLLM OpenAI-compatible API.
# Used by the router to forward requests to vLLM backend.

import time
import httpx
import logging
from typing import Optional, AsyncGenerator

logger = logging.getLogger(__name__)

VLLM_URL = "http://localhost:8001"


class VLLMClient:
    """
    Async client for vLLM's OpenAI-compatible API.
    Supports both streaming and non-streaming inference.
    """

    def __init__(self, base_url: str = VLLM_URL):
        self.base_url = base_url
        self.client   = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        prompt:     str,
        max_tokens: int          = 256,
        temperature: float       = 0.7,
        model:      str          = "mistral-7b",
    ) -> dict:
        """
        Non-streaming generation.
        Returns complete response with latency metrics.
        """
        start = time.perf_counter()

        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model":       model,
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  max_tokens,
                "temperature": temperature,
            }
        )
        response.raise_for_status()
        data = response.json()

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "text":             data["choices"][0]["message"]["content"],
            "prompt_tokens":    data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "latency_ms":       round(latency_ms, 2),
            "model":            data["model"],
        }

    async def generate_stream(
        self,
        prompt:     str,
        max_tokens: int    = 256,
        temperature: float = 0.7,
        model:      str    = "mistral-7b",
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation — yields tokens as they arrive.
        Used for real-time token streaming to the user.
        """
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json={
                "model":       model,
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  max_tokens,
                "temperature": temperature,
                "stream":      True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    import json
                    data = json.loads(line[6:])
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]

    async def health(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self.client.aclose()

from __future__ import annotations

import logging
import json
from typing import AsyncGenerator

logger = logging.getLogger("nanobot.llm.client")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    logger.warning("aiohttp not available — using urllib fallback")


class LlamaClient:
    def __init__(self, config: dict):
        self.server_url = config.get("server_url", "http://127.0.0.1:8080")
        self.max_tokens = config.get("max_tokens", 150)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.repeat_penalty = config.get("repeat_penalty", 1.1)

    async def stream_completion(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """
        Stream tokens from llama.cpp /v1/chat/completions endpoint.
        Yields individual token strings as they arrive.
        """
        url = f"{self.server_url}/v1/chat/completions"
        payload = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "stream": True,
        }

        if HAS_AIOHTTP:
            async for token in self._stream_aiohttp(url, payload):
                yield token
        else:
            # Fallback: non-streaming request
            for token in self._sync_request(url, payload):
                yield token

    async def _stream_aiohttp(self, url: str, payload: dict) -> AsyncGenerator[str, None]:
        """Stream using aiohttp for proper async SSE handling."""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"LLM server error {resp.status}: {error}")
                    return

                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def _sync_request(self, url: str, payload: dict) -> list[str]:
        """Fallback: non-streaming request using urllib."""
        import urllib.request

        payload["stream"] = False
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode())
                content = result["choices"][0]["message"]["content"]
                return [content]
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return ["I'm having trouble thinking right now."]

    async def health_check(self) -> bool:
        """Check if llama.cpp server is running."""
        try:
            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.server_url}/health") as resp:
                        return resp.status == 200
            else:
                import urllib.request
                with urllib.request.urlopen(f"{self.server_url}/health", timeout=5):
                    return True
        except Exception:
            return False

"""OpenAI-compatible chat client for build.nvidia.com, with retries and token counts."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    def add(self, usage: Any) -> None:
        self.llm_calls += 1
        if usage is not None:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0


class ChatClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict], usage: Usage, **kw: Any) -> Any:
        """Return a chat completion message (with .content and .tool_calls)."""


class OpenAIChat:
    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        base_url: str = NVIDIA_BASE_URL,
        api_key: str | None = None,
    ):
        from openai import OpenAI

        key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise SystemExit("Set NVIDIA_API_KEY (see README).")
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=180, max_retries=0)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, messages, tools, usage, **kw):
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        delay = 5.0
        for attempt in range(8):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **kw,
                )
                usage.add(resp.usage)
                return resp.choices[0].message
            except (APIConnectionError, APITimeoutError):
                pass
            except APIStatusError as e:
                if e.status_code not in (429, 500, 502, 503, 504):
                    raise
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 120)
        raise RuntimeError("LLM call failed after retries")

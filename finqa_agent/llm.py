"""OpenAI-compatible chat client for build.nvidia.com, with retries and token counts."""

from __future__ import annotations

import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    retries: int = 0

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
        rpm: float | None = None,
    ):
        from openai import OpenAI

        self._interval = 60.0 / rpm if rpm else 0.0
        self._next = 0.0
        self._lock = threading.Lock()
        key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise SystemExit("Set NVIDIA_API_KEY (see README).")
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=180, max_retries=0)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _pace(self) -> None:
        """Space requests at least 60/rpm seconds apart across all threads."""
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        time.sleep(max(0.0, slot - now))

    def chat(self, messages, tools, usage, **kw):
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        delay = 5.0
        for attempt in range(8):
            self._pace()
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
            except (APIConnectionError, APITimeoutError) as e:
                reason = type(e).__name__
            except APIStatusError as e:
                # build.nvidia.com answers 400 "DEGRADED function cannot be invoked" during outages.
                degraded = e.status_code == 400 and "DEGRADED" in str(e)
                if e.status_code not in (429, 500, 502, 503, 504) and not degraded:
                    raise
                reason = f"HTTP {e.status_code}"
            usage.retries += 1
            print(f"retry {attempt + 1}/8 after {reason}", file=sys.stderr, flush=True)
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 120)
        raise RuntimeError("LLM call failed after retries")

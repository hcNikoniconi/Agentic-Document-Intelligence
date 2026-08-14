"""OpenAI-compatible model client used by the extraction pipeline.

The rest of the application should not need to know whether the model is a
self-hosted vLLM server or an external API provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class ModelSettings:
    model_name: str
    base_url: str
    api_key: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "ModelSettings":
        return cls(
            model_name=os.getenv("MODEL_NAME", "qwen3-8b"),
            base_url=os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("MODEL_API_KEY", "EMPTY"),
            timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "120")),
        )


@lru_cache(maxsize=1)
def get_model_settings() -> ModelSettings:
    return ModelSettings.from_env()


@lru_cache(maxsize=1)
def get_model_client() -> OpenAI:
    settings = get_model_settings()
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
    )


def create_chat_completion(
    *,
    messages: list[dict[str, Any]],
    temperature: float = 0,
    max_tokens: int = 1536,
    model: str | None = None,
    extra_body: dict[str, Any] | None = None,
):
    settings = get_model_settings()
    request: dict[str, Any] = {
        "model": model or settings.model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if extra_body is not None:
        request["extra_body"] = extra_body

    return get_model_client().chat.completions.create(**request)


def chat_completion_content(**kwargs: Any) -> str:
    response = create_chat_completion(**kwargs)
    return response.choices[0].message.content.strip()

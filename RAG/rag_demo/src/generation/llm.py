from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from langchain_google_genai import ChatGoogleGenerativeAI


class LLM(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class GoogleChatLLM:
    api_key: str
    model: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        object.__setattr__(self, "_client", ChatGoogleGenerativeAI(model=self.model, api_key=self.api_key))

    @property
    def name(self) -> str:
        return f"google:{self.model}"

    def generate(self, prompt: str) -> str:
        resp = self._client.invoke(prompt)
        return (resp.content or "").strip()


@dataclass(frozen=True)
class FallbackLLM:
    """
    Offline fallback:
    - Returns a concise "answer" by extracting the most relevant sentences from the context.
    - Keeps the pipeline runnable without an API key.
    """

    @property
    def name(self) -> str:
        return "fallback:extractive"

    def generate(self, prompt: str) -> str:
        # The prompt includes a "Context:" section; extract a simple summary.
        ctx_marker = "Context:"
        if ctx_marker not in prompt:
            return prompt[:800]
        ctx = prompt.split(ctx_marker, 1)[1].strip()
        # naive: take first few non-empty lines/sentences
        lines = [l.strip() for l in ctx.splitlines() if l.strip()]
        excerpt = "\n".join(lines[:10])
        return (
            "No `GOOGLE_API_KEY` configured. Showing an extractive answer from retrieved context.\n\n"
            + excerpt[:1500]
        ).strip()


def make_llm(google_api_key: Optional[str], model: str) -> LLM:
    if google_api_key:
        return GoogleChatLLM(api_key=google_api_key, model=model)
    return FallbackLLM()


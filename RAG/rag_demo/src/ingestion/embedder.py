from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np

from langchain_google_genai import GoogleGenerativeAIEmbeddings

import hashlib


class Embedder(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...

    @property
    def name(self) -> str: ...


@dataclass(frozen=True)
class GoogleEmbedder:
    api_key: str
    model: str = "text-embedding-004"

    def __post_init__(self) -> None:
        object.__setattr__(self, "_client", GoogleGenerativeAIEmbeddings(model=self.model, api_key=self.api_key))

    @property
    def name(self) -> str:
        return f"google:{self.model}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)


@dataclass(frozen=True)
class HashEmbedder:
    """
    Local fallback embedder (no API key required).

    This is not a semantic embedding model; it exists so the demo stays runnable offline.
    The retrieval quality is lower, but the pipeline layers remain demonstrable.
    """

    # Match common Google embedding dimensionality so the pgvector schema stays fixed.
    dim: int = 3072

    @property
    def name(self) -> str:
        return f"hash:{self.dim}"

    def _hash_vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            # Deterministic across processes (unlike Python's built-in hash()).
            h_bytes = hashlib.md5(tok.encode("utf-8", errors="ignore")).digest()
            h_int = int.from_bytes(h_bytes[:8], byteorder="big", signed=False)
            idx = h_int % self.dim
            v[idx] += 1.0
        # Normalize to unit length
        norm = float(np.linalg.norm(v) + 1e-8)
        v = v / norm
        return v

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vec(t).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def make_embedder(google_api_key: Optional[str], model: str) -> Embedder:
    if google_api_key:
        return GoogleEmbedder(api_key=google_api_key, model=model)
    return HashEmbedder()


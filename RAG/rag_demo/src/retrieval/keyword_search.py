from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from rank_bm25 import BM25Okapi

from ..models import ChunkRecord, RetrievedChunk


def _tokenize(text: str) -> List[str]:
    # Simple tokenizer for demo purposes.
    return [t for t in "".join([c.lower() if c.isalnum() else " " for c in text]).split() if t]


@dataclass
class KeywordSearchResult:
    retrieved: List[RetrievedChunk]
    debug: Dict


class BM25KeywordSearcher:
    """
    Keyword retrieval over active chunks (already access-filtered upstream).

    For teaching:
    - We build BM25 over the current chunk set.
    - This is fine for small local demos; production would use an inverted index.
    """

    def search(self, query: str, chunks: Sequence[ChunkRecord], top_k: int) -> KeywordSearchResult:
        if not chunks:
            return KeywordSearchResult(retrieved=[], debug={"reason": "no_chunks"})

        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokenize(query))

        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:top_k]
        retrieved: List[RetrievedChunk] = []
        for rank_idx, (i, score) in enumerate(ranked, start=1):
            retrieved.append(
                RetrievedChunk(
                    chunk=chunks[i],
                    score=float(score),
                    rank=rank_idx,
                    source="keyword",
                    debug={"bm25_score": float(score)},
                )
            )

        return KeywordSearchResult(
            retrieved=retrieved,
            debug={
                "corpus_size": len(chunks),
                "top_k": top_k,
            },
        )


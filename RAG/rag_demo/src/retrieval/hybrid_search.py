from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..models import RetrievedChunk


@dataclass
class HybridSearchResult:
    retrieved: List[RetrievedChunk]
    debug: Dict


def reciprocal_rank_fusion(
    vector_results: Sequence[RetrievedChunk],
    keyword_results: Sequence[RetrievedChunk],
    top_k: int,
    k: int = 60,
) -> HybridSearchResult:
    """
    Simple Reciprocal Rank Fusion (RRF).

    Score per system: 1 / (k + rank)
    Combined score = sum over systems.
    """
    scores: Dict[str, float] = {}
    sources: Dict[str, Dict] = {}

    def add(system: str, results: Sequence[RetrievedChunk]) -> None:
        for r in results:
            uid = r.chunk.chunk_uid
            scores[uid] = scores.get(uid, 0.0) + 1.0 / float(k + r.rank)
            sources.setdefault(uid, {})[system] = {"rank": r.rank, "score": r.score}

    add("vector", vector_results)
    add("keyword", keyword_results)

    # pick representative chunk object: prefer vector (usually has better ordering), else keyword
    by_uid: Dict[str, RetrievedChunk] = {}
    for r in list(vector_results) + list(keyword_results):
        by_uid.setdefault(r.chunk.chunk_uid, r)

    ranked_uids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out: List[RetrievedChunk] = []
    for new_rank, (uid, fused_score) in enumerate(ranked_uids, start=1):
        base = by_uid[uid]
        out.append(
            RetrievedChunk(
                chunk=base.chunk,
                score=float(fused_score),
                rank=new_rank,
                source="hybrid",
                debug={"rrf_components": sources.get(uid, {})},
            )
        )

    return HybridSearchResult(
        retrieved=out,
        debug={
            "fusion": "rrf",
            "k": k,
            "vector_in": len(vector_results),
            "keyword_in": len(keyword_results),
            "top_k": top_k,
        },
    )


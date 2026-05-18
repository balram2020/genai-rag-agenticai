from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..ingestion.embedder import Embedder
from ..ingestion.vector_store import PgVectorStore
from ..models import ChunkRecord, RetrievedChunk


@dataclass
class SemanticSearchResult:
    retrieved: List[RetrievedChunk]
    debug: Dict


class SemanticSearcher:
    def __init__(self, vector_store: PgVectorStore, embedder: Embedder) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int,
        where: Optional[Dict] = None,
        chunk_lookup: Optional[Dict[str, ChunkRecord]] = None,
    ) -> SemanticSearchResult:
        """
        Perform vector search in Chroma.

        Note:
        - Chroma returns cosine distance for "cosine" space; smaller is closer.
        - We convert distance into a similarity-like score via (1 - distance) for readability.
        """
        emb = self.embedder.embed_query(query)
        res = self.vector_store.query(query_embeddings=[emb], top_k=top_k, where=where)

        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        docs = res.get("documents", [[]])[0]

        retrieved: List[RetrievedChunk] = []
        for idx, (cid, dist, meta, doc_text) in enumerate(zip(ids, dists, metas, docs), start=1):
            score = float(1.0 - float(dist)) if dist is not None else 0.0
            if not chunk_lookup or str(cid) not in chunk_lookup:
                # The demo pipeline expects registry as the source of truth for chunk records.
                # If a vector exists without a registry record, we skip it (should be rare).
                continue
            chunk = chunk_lookup[str(cid)]

            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    rank=idx,
                    source="vector",
                    debug={"distance": float(dist) if dist is not None else None},
                )
            )

        return SemanticSearchResult(
            retrieved=retrieved,
            debug={
                "top_k": top_k,
                "where": where or {},
                "embedder": self.embedder.name,
            },
        )


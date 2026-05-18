from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import AppConfig
from ..ingestion.embedder import make_embedder
from ..ingestion.record_manager import RecordManager
from ..ingestion.vector_store import PgVectorStore
from ..models import (
    AccessLevel,
    RetrievalDebugInfo,
    RetrievalMode,
    RetrievedChunk,
    ROLE_ALLOWED_LEVELS,
    UserRole,
)
from ..utils.logging_utils import EventLogger
from .access_control import allowed_levels_for_role, filter_chunks_by_role
from .hybrid_search import reciprocal_rank_fusion
from .keyword_search import BM25KeywordSearcher
from .semantic_search import SemanticSearcher


@dataclass
class RetrievalResult:
    retrieved: List[RetrievedChunk]
    debug: RetrievalDebugInfo
    per_layer_debug: Dict[str, Any]


class Retriever:
    def __init__(self, cfg: AppConfig, logger: EventLogger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.record_manager = RecordManager(cfg.sqlite_path)
        self.vector_store = PgVectorStore(cfg.pg_dsn, table_name=cfg.pgvector_table)
        self.embedder = make_embedder(cfg.google_api_key, cfg.google_embedding_model)
        self.semantic = SemanticSearcher(self.vector_store, self.embedder)
        self.keyword = BM25KeywordSearcher()

    def close(self) -> None:
        self.record_manager.close()

    def retrieve(
        self,
        query: str,
        role: UserRole,
        mode: RetrievalMode,
        vector_top_k: int = 6,
        keyword_top_k: int = 8,
        final_top_k: int = 6,
        extracted_filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        extracted_filters = extracted_filters or {}

        # Access control gate for registry-based keyword retrieval.
        allowed_levels = list(allowed_levels_for_role(role))
        all_allowed_chunks = self.record_manager.get_active_chunks(allowed_access_levels=allowed_levels)

        filtered_out_count = 0
        filtered_out_examples: List[Dict] = []
        # (Included for teaching even though registry query already filters by access level)
        all_allowed_chunks, filtered_out_count, filtered_out_examples = filter_chunks_by_role(role, all_allowed_chunks)

        chunk_lookup = {c.chunk_uid: c for c in all_allowed_chunks}

        # Build Chroma where-filter. Chroma filter enforces access control at retrieval-time too.
        chroma_where: Dict[str, Any] = {"access_level": {"$in": [l.value for l in allowed_levels]}}
        if "document" in extracted_filters:
            # allow either document_id or document_name match
            chroma_where["document_id"] = str(extracted_filters["document"])
        if "version" in extracted_filters:
            try:
                chroma_where["version"] = int(extracted_filters["version"])
            except Exception:
                pass

        per_layer_debug: Dict[str, Any] = {}

        vec_results: List[RetrievedChunk] = []
        kw_results: List[RetrievedChunk] = []
        hybrid_results: List[RetrievedChunk] = []

        if mode in (RetrievalMode.vector_only, RetrievalMode.hybrid):
            sem = self.semantic.search(query=query, top_k=vector_top_k, where=chroma_where, chunk_lookup=chunk_lookup)
            vec_results = sem.retrieved
            per_layer_debug["semantic"] = sem.debug
            self.logger.info("Semantic retrieval complete", {"returned": len(vec_results), "where": chroma_where})

        if mode in (RetrievalMode.keyword_only, RetrievalMode.hybrid):
            kw = self.keyword.search(query=query, chunks=all_allowed_chunks, top_k=keyword_top_k)
            kw_results = kw.retrieved
            per_layer_debug["keyword"] = kw.debug
            self.logger.info("Keyword retrieval complete", {"returned": len(kw_results)})

        if mode == RetrievalMode.vector_only:
            final = vec_results[:final_top_k]
        elif mode == RetrievalMode.keyword_only:
            final = kw_results[:final_top_k]
        else:
            fused = reciprocal_rank_fusion(vec_results, kw_results, top_k=final_top_k)
            hybrid_results = fused.retrieved
            per_layer_debug["hybrid"] = fused.debug
            self.logger.info("Hybrid fusion complete", {"returned": len(hybrid_results)})
            final = hybrid_results

        dbg = RetrievalDebugInfo(
            mode=mode,
            user_role=role,
            allowed_access_levels=allowed_levels,
            vector_top_k=vector_top_k,
            keyword_top_k=keyword_top_k,
            filtered_out_count=filtered_out_count,
            filtered_out_examples=filtered_out_examples,
            notes=[],
        )

        return RetrievalResult(retrieved=final, debug=dbg, per_layer_debug=per_layer_debug)


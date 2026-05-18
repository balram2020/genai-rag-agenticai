from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class AccessLevel(str, Enum):
    public = "public"
    internal = "internal"
    restricted = "restricted"


class UserRole(str, Enum):
    guest = "guest"
    employee = "employee"
    admin = "admin"


ROLE_ALLOWED_LEVELS: Dict[UserRole, Tuple[AccessLevel, ...]] = {
    UserRole.guest: (AccessLevel.public,),
    UserRole.employee: (AccessLevel.public, AccessLevel.internal),
    UserRole.admin: (AccessLevel.public, AccessLevel.internal, AccessLevel.restricted),
}


class QueryType(str, Enum):
    factual_lookup = "factual_lookup"
    summarization = "summarization"
    comparison = "comparison"
    explain_concept = "explain_concept"
    other = "other"


class QueryUnderstandingOutput(BaseModel):
    raw_query: str
    normalized_query: str
    expanded_terms: List[str] = Field(default_factory=list)
    query_type: QueryType = QueryType.other
    extracted_filters: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class DocumentIngestRequest(BaseModel):
    source_path: str
    document_name: str
    access_level: AccessLevel = AccessLevel.public


class DocumentRecord(BaseModel):
    document_id: str
    document_name: str
    source_path: str
    version: int
    access_level: AccessLevel
    updated_at: datetime
    content_hash: str
    is_active: bool = True


class ChunkRecord(BaseModel):
    chunk_uid: str  # stable unique id for vector store + registry
    document_id: str
    document_name: str
    source_path: str
    chunk_id: int
    version: int
    section_title: Optional[str] = None
    access_level: AccessLevel
    updated_at: datetime
    content_hash: str
    text: str
    is_active: bool = True


class RetrievalMode(str, Enum):
    vector_only = "vector_only"
    keyword_only = "keyword_only"
    hybrid = "hybrid"


class RetrievedChunk(BaseModel):
    chunk: ChunkRecord
    score: float
    rank: int
    source: str  # "vector" | "keyword" | "hybrid"
    debug: Dict[str, Any] = Field(default_factory=dict)


class RetrievalDebugInfo(BaseModel):
    mode: RetrievalMode
    user_role: UserRole
    allowed_access_levels: List[AccessLevel]
    vector_top_k: int
    keyword_top_k: int
    filtered_out_count: int
    filtered_out_examples: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class AnswerResult(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    context_used: str = ""
    used_chunks: List[RetrievedChunk] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)


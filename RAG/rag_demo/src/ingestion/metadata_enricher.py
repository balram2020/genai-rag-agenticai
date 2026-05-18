from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..models import AccessLevel, ChunkRecord
from ..utils.hashing import sha256_text


@dataclass(frozen=True)
class EnrichmentInput:
    document_id: str
    document_name: str
    source_path: Path
    version: int
    access_level: AccessLevel
    updated_at: datetime


def build_chunk_uid(document_id: str, version: int, chunk_id: int, content_hash: str) -> str:
    """
    Chunk UID must be unique and allow easy cleanup of stale chunks.
    We include version + content hash prefix to avoid accidental collisions.
    """
    return f"{document_id}::v{version}::c{chunk_id}::{content_hash[:12]}"


def enrich_chunks(
    inp: EnrichmentInput,
    chunks: List[dict],
) -> List[ChunkRecord]:
    enriched: List[ChunkRecord] = []
    for c in chunks:
        text: str = c["text"]
        section_title: Optional[str] = c.get("section_title")
        chunk_id: int = int(c["chunk_id"])
        content_hash = sha256_text(text)
        chunk_uid = build_chunk_uid(inp.document_id, inp.version, chunk_id, content_hash)
        enriched.append(
            ChunkRecord(
                chunk_uid=chunk_uid,
                document_id=inp.document_id,
                document_name=inp.document_name,
                source_path=str(inp.source_path),
                chunk_id=chunk_id,
                version=inp.version,
                section_title=section_title,
                access_level=inp.access_level,
                updated_at=inp.updated_at,
                content_hash=content_hash,
                text=text,
                is_active=True,
            )
        )
    return enriched


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig
from ..models import AccessLevel, ChunkRecord, DocumentIngestRequest, DocumentRecord
from ..utils.hashing import sha256_file_bytes, sha256_text
from ..utils.logging_utils import EventLogger
from .chunker import chunk_text
from .embedder import Embedder, make_embedder
from .metadata_enricher import EnrichmentInput, enrich_chunks, now_utc
from .pdf_loader import load_pdf
from .record_manager import ChunkDiff, RecordManager
from .vector_store import PgVectorStore


@dataclass(frozen=True)
class IngestResult:
    document: DocumentRecord
    diff: ChunkDiff
    embedder_name: str
    vector_store_count: int
    notes: List[str]


def _stable_document_id(source_path: str) -> str:
    # Stable across runs for same path (so updates map to same doc).
    return sha256_text(source_path)[:16]


class IngestPipeline:
    def __init__(self, cfg: AppConfig, logger: EventLogger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.record_manager = RecordManager(cfg.sqlite_path)
        self.vector_store = PgVectorStore(cfg.pg_dsn, table_name=cfg.pgvector_table)
        self.embedder: Embedder = make_embedder(cfg.google_api_key, cfg.google_embedding_model)

    def close(self) -> None:
        self.record_manager.close()

    def ingest_pdf(self, req: DocumentIngestRequest) -> IngestResult:
        path = Path(req.source_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        self.logger.info("Loading PDF", {"path": str(path)})
        loaded = load_pdf(path, document_name=req.document_name)
        pdf_bytes = path.read_bytes()
        file_hash = sha256_file_bytes(pdf_bytes)

        prev_doc = self.record_manager.get_document_by_source(str(path))
        if prev_doc is None:
            version = 1
        else:
            version = prev_doc.version + 1 if prev_doc.content_hash != file_hash else prev_doc.version

        document_id = prev_doc.document_id if prev_doc else _stable_document_id(str(path))
        updated_at = now_utc()

        doc_record = DocumentRecord(
            document_id=document_id,
            document_name=loaded.document_name,
            source_path=str(path),
            version=version,
            access_level=req.access_level,
            updated_at=updated_at,
            content_hash=file_hash,
            is_active=True,
        )

        self.logger.info("Chunking document", {"document_id": document_id, "version": version})
        raw_chunks = chunk_text(
            loaded.full_text,
            target_chars=self.cfg.chunk_target_chars,
            overlap_chars=self.cfg.chunk_overlap_chars,
        )
        self.logger.info("Chunking complete", {"chunks": len(raw_chunks)})

        enrich_in = EnrichmentInput(
            document_id=document_id,
            document_name=loaded.document_name,
            source_path=path,
            version=version,
            access_level=req.access_level,
            updated_at=updated_at,
        )
        new_chunks: List[ChunkRecord] = enrich_chunks(
            enrich_in,
            [
                {"chunk_id": c.chunk_id, "text": c.text, "section_title": c.section_title}
                for c in raw_chunks
            ],
        )

        diff = self.record_manager.compute_diff(document_id=document_id, new_chunks=new_chunks)
        self.logger.info(
            "Record manager diff",
            {
                "new": len(diff.new_chunks),
                "changed": len(diff.changed_chunks),
                "unchanged": len(diff.unchanged_chunks),
                "previous_active": len(diff.previous_active_chunk_uids),
                "deleted": len(diff.deleted_chunk_uids),
            },
        )

        notes: List[str] = []

        if prev_doc and prev_doc.content_hash == file_hash:
            notes.append("Document content hash unchanged; version remains the same.")
        elif prev_doc:
            notes.append("Document changed; new version created and stale chunks will be deactivated/removed.")
        else:
            notes.append("New document ingested.")

        # If we are creating a new version (or first ingest), we replace active chunks for the doc.
        # For simplicity (and teachability), we deactivate all previous active chunks and then insert new ones.
        if not prev_doc or prev_doc.content_hash != file_hash:
            deactivated = self.record_manager.deactivate_previous_chunks(document_id=document_id)
            if deactivated:
                self.logger.warn("Deactivated previous active chunks", {"count": deactivated, "document_id": document_id})

            # Remove ALL previous active chunk uids from vector store, then upsert new version chunks.
            if diff.previous_active_chunk_uids:
                self.vector_store.delete_ids(diff.previous_active_chunk_uids)
                self.logger.warn("Deleted stale vectors for previous version", {"count": len(diff.previous_active_chunk_uids)})

            # Upsert vectors for all new chunks in this version.
            ids = [c.chunk_uid for c in new_chunks]
            docs = [c.text for c in new_chunks]
            metas: List[Dict[str, Any]] = [
                {
                    "document_id": c.document_id,
                    "document_name": c.document_name,
                    "source_path": c.source_path,
                    "chunk_id": c.chunk_id,
                    "version": c.version,
                    "section_title": c.section_title or "",
                    "access_level": c.access_level.value,
                    "updated_at": c.updated_at.isoformat(),
                    "content_hash": c.content_hash,
                }
                for c in new_chunks
            ]

            self.logger.info("Embedding chunks", {"embedder": self.embedder.name, "count": len(new_chunks)})
            embeddings = self.embedder.embed_documents(docs)
            self.vector_store.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
            self.logger.info("Vector upsert complete", {"vectors_upserted": len(ids)})

            # Update registry
            self.record_manager.upsert_document(doc_record)
            self.record_manager.insert_chunks(new_chunks)
        else:
            # No change: ensure registry contains the doc record (idempotent) and do not re-embed.
            self.record_manager.upsert_document(doc_record)
            notes.append("Skipped re-embedding because document is unchanged.")

        stats = self.vector_store.stats()
        self.logger.info("Vector store stats", {"count": stats.count})

        return IngestResult(
            document=doc_record,
            diff=diff,
            embedder_name=self.embedder.name,
            vector_store_count=stats.count,
            notes=notes,
        )


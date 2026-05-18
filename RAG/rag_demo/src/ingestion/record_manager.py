from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import AccessLevel, ChunkRecord, DocumentRecord


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
  document_id TEXT PRIMARY KEY,
  document_name TEXT NOT NULL,
  source_path TEXT NOT NULL,
  version INTEGER NOT NULL,
  access_level TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  is_active INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_uid TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  document_name TEXT NOT NULL,
  source_path TEXT NOT NULL,
  chunk_id INTEGER NOT NULL,
  version INTEGER NOT NULL,
  section_title TEXT,
  access_level TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  text TEXT NOT NULL,
  is_active INTEGER NOT NULL,
  FOREIGN KEY(document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_docid_active ON chunks(document_id, is_active);
CREATE INDEX IF NOT EXISTS idx_chunks_access_active ON chunks(access_level, is_active);
"""


@dataclass(frozen=True)
class ChunkDiff:
    new_chunks: List[ChunkRecord]
    changed_chunks: List[ChunkRecord]
    unchanged_chunks: List[ChunkRecord]
    deleted_chunk_uids: List[str]
    previous_active_chunk_uids: List[str]


class RecordManager:
    """
    SQLite-backed registry for documents + chunks.

    Teachable concept:
    - Every ingestion run performs a "diff" against the registry.
    - New/changed chunks are upserted; deleted chunks are deactivated and removed from vector store.
    - This mirrors production change management patterns.
    """

    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def list_documents(self, include_inactive: bool = False) -> List[DocumentRecord]:
        q = "SELECT * FROM documents"
        if not include_inactive:
            q += " WHERE is_active=1"
        q += " ORDER BY updated_at DESC"
        rows = self._conn.execute(q).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def get_document_by_source(self, source_path: str) -> Optional[DocumentRecord]:
        row = self._conn.execute("SELECT * FROM documents WHERE source_path=? LIMIT 1", (source_path,)).fetchone()
        return self._row_to_doc(row) if row else None

    def get_active_chunks(self, allowed_access_levels: Sequence[AccessLevel]) -> List[ChunkRecord]:
        levels = tuple(l.value for l in allowed_access_levels)
        placeholders = ",".join(["?"] * len(levels))
        q = f"SELECT * FROM chunks WHERE is_active=1 AND access_level IN ({placeholders}) ORDER BY document_id, chunk_id"
        rows = self._conn.execute(q, levels).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_active_chunks_for_document(self, document_id: str) -> List[ChunkRecord]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id=? AND is_active=1 ORDER BY chunk_id",
            (document_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_active_chunk_uids_for_document(self, document_id: str) -> List[str]:
        rows = self._conn.execute(
            "SELECT chunk_uid FROM chunks WHERE document_id=? AND is_active=1 ORDER BY chunk_id",
            (document_id,),
        ).fetchall()
        return [str(r["chunk_uid"]) for r in rows]

    def compute_diff(self, document_id: str, new_chunks: List[ChunkRecord]) -> ChunkDiff:
        prev_active = self.get_active_chunks_for_document(document_id=document_id)
        prev_by_key: Dict[int, ChunkRecord] = {c.chunk_id: c for c in prev_active}
        prev_uids = [c.chunk_uid for c in prev_active]

        new_list: List[ChunkRecord] = []
        changed_list: List[ChunkRecord] = []
        unchanged_list: List[ChunkRecord] = []

        seen_chunk_ids = set()
        for c in new_chunks:
            seen_chunk_ids.add(c.chunk_id)
            prev = prev_by_key.get(c.chunk_id)
            if prev is None:
                new_list.append(c)
            elif prev.content_hash != c.content_hash or prev.text != c.text:
                changed_list.append(c)
            else:
                unchanged_list.append(c)

        deleted_uids: List[str] = []
        for prev in prev_active:
            if prev.chunk_id not in seen_chunk_ids:
                deleted_uids.append(prev.chunk_uid)

        # Also treat "changed" as deleting the previous chunk_uid (because UID encodes hash/version)
        # We'll deactivate all previous active chunks and then activate the new version's chunks.
        return ChunkDiff(
            new_chunks=new_list,
            changed_chunks=changed_list,
            unchanged_chunks=unchanged_list,
            deleted_chunk_uids=deleted_uids,
            previous_active_chunk_uids=prev_uids,
        )

    def upsert_document(self, doc: DocumentRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO documents(document_id, document_name, source_path, version, access_level, updated_at, content_hash, is_active)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(document_id) DO UPDATE SET
              document_name=excluded.document_name,
              source_path=excluded.source_path,
              version=excluded.version,
              access_level=excluded.access_level,
              updated_at=excluded.updated_at,
              content_hash=excluded.content_hash,
              is_active=excluded.is_active
            """,
            (
                doc.document_id,
                doc.document_name,
                doc.source_path,
                doc.version,
                doc.access_level.value,
                doc.updated_at.isoformat(),
                doc.content_hash,
                1 if doc.is_active else 0,
            ),
        )
        self._conn.commit()

    def deactivate_previous_chunks(self, document_id: str) -> int:
        cur = self._conn.execute("UPDATE chunks SET is_active=0 WHERE document_id=? AND is_active=1", (document_id,))
        self._conn.commit()
        return int(cur.rowcount or 0)

    def insert_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        self._conn.executemany(
            """
            INSERT INTO chunks(
              chunk_uid, document_id, document_name, source_path, chunk_id, version, section_title,
              access_level, updated_at, content_hash, text, is_active
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    c.chunk_uid,
                    c.document_id,
                    c.document_name,
                    c.source_path,
                    c.chunk_id,
                    c.version,
                    c.section_title,
                    c.access_level.value,
                    c.updated_at.isoformat(),
                    c.content_hash,
                    c.text,
                    1 if c.is_active else 0,
                )
                for c in chunks
            ],
        )
        self._conn.commit()

    def _row_to_doc(self, r: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=str(r["document_id"]),
            document_name=str(r["document_name"]),
            source_path=str(r["source_path"]),
            version=int(r["version"]),
            access_level=AccessLevel(str(r["access_level"])),
            updated_at=datetime.fromisoformat(str(r["updated_at"])),
            content_hash=str(r["content_hash"]),
            is_active=bool(int(r["is_active"])),
        )

    def _row_to_chunk(self, r: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_uid=str(r["chunk_uid"]),
            document_id=str(r["document_id"]),
            document_name=str(r["document_name"]),
            source_path=str(r["source_path"]),
            chunk_id=int(r["chunk_id"]),
            version=int(r["version"]),
            section_title=str(r["section_title"]) if r["section_title"] is not None else None,
            access_level=AccessLevel(str(r["access_level"])),
            updated_at=datetime.fromisoformat(str(r["updated_at"])),
            content_hash=str(r["content_hash"]),
            text=str(r["text"]),
            is_active=bool(int(r["is_active"])),
        )


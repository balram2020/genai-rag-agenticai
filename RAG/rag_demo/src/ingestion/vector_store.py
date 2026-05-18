from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg
from pgvector.psycopg import register_vector


@dataclass
class VectorStoreStats:
    table_name: str
    count: int


class PgVectorStore:
    """
    Vector store using Postgres + pgvector (typically running in Docker).

    We store:
    - chunk_uid (PK)
    - embedding (vector)
    - text + metadata columns (for explainability + filtering)
    """

    def __init__(self, dsn: str, table_name: str = "rag_chunks") -> None:
        self.dsn = dsn
        self.table_name = table_name

        # Ensure extension exists (idempotent) and vector support is registered.
        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                # Table schema is also created by docker init script, but we keep this as a safety net.
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
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
                      embedding vector(1536) NOT NULL
                    );
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_access ON {self.table_name}(access_level);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_doc ON {self.table_name}(document_id);"
                )
            conn.commit()

    def stats(self) -> VectorStoreStats:
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                n = int(cur.fetchone()[0])
        return VectorStoreStats(table_name=self.table_name, count=n)

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        if not ids:
            return
        if len(ids) != len(documents) or len(ids) != len(metadatas) or len(ids) != len(embeddings):
            raise ValueError("ids/documents/metadatas/embeddings length mismatch")

        rows = []
        for cid, text, meta, emb in zip(ids, documents, metadatas, embeddings):
            rows.append(
                (
                    cid,
                    meta["document_id"],
                    meta["document_name"],
                    meta["source_path"],
                    int(meta["chunk_id"]),
                    int(meta["version"]),
                    meta.get("section_title") or None,
                    meta["access_level"],
                    meta["updated_at"],
                    meta["content_hash"],
                    text,
                    emb,
                )
            )

        sql = f"""
        INSERT INTO {self.table_name}(
          chunk_uid, document_id, document_name, source_path, chunk_id, version, section_title,
          access_level, updated_at, content_hash, text, embedding
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT(chunk_uid) DO UPDATE SET
          document_id=EXCLUDED.document_id,
          document_name=EXCLUDED.document_name,
          source_path=EXCLUDED.source_path,
          chunk_id=EXCLUDED.chunk_id,
          version=EXCLUDED.version,
          section_title=EXCLUDED.section_title,
          access_level=EXCLUDED.access_level,
          updated_at=EXCLUDED.updated_at,
          content_hash=EXCLUDED.content_hash,
          text=EXCLUDED.text,
          embedding=EXCLUDED.embedding;
        """

        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()

    def delete_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE chunk_uid = ANY(%s);",
                    (ids,),
                )
            conn.commit()

    def query(
        self,
        query_embeddings: List[List[float]],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        if not query_embeddings:
            raise ValueError("query_embeddings required")
        emb = query_embeddings[0]
        where = where or {}

        clauses: List[str] = []
        params: List[Any] = []

        # Access control filter (expected shape from retriever): {"access_level": {"$in": ["public", ...]}}
        if "access_level" in where and isinstance(where["access_level"], dict) and "$in" in where["access_level"]:
            allowed = list(where["access_level"]["$in"])
            clauses.append("access_level = ANY(%s)")
            params.append(allowed)

        if "document_id" in where:
            clauses.append("document_id = %s")
            params.append(str(where["document_id"]))

        if "version" in where:
            clauses.append("version = %s")
            params.append(int(where["version"]))

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        # pgvector: embedding <=> query = cosine distance (smaller is better)
        sql = f"""
        SELECT
          chunk_uid,
          document_id,
          document_name,
          source_path,
          chunk_id,
          version,
          section_title,
          access_level,
          updated_at,
          content_hash,
          text,
          (embedding <=> %s::vector) AS distance
        FROM {self.table_name}
        {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT {int(top_k)};
        """

        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(sql, [emb] + params + [emb])
                rows = cur.fetchall()

        ids = [r[0] for r in rows]
        metas = [
            {
                "document_id": r[1],
                "document_name": r[2],
                "source_path": r[3],
                "chunk_id": r[4],
                "version": r[5],
                "section_title": r[6] or "",
                "access_level": r[7],
                "updated_at": r[8],
                "content_hash": r[9],
            }
            for r in rows
        ]
        docs = [r[10] for r in rows]
        dists = [float(r[11]) for r in rows]

        return {"ids": [ids], "metadatas": [metas], "documents": [docs], "distances": [dists]}


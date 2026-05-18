CREATE EXTENSION IF NOT EXISTS vector;

-- Table used by the demo vector store (pgvector).
CREATE TABLE IF NOT EXISTS rag_chunks (
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
  embedding vector(3072) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_access ON rag_chunks(access_level);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(document_id);

-- Optional IVFFLAT index (requires ANALYZE and enough rows). For small classroom demos,
-- sequential scan is fine, but this shows a production-like knob.
-- CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_ivfflat
--   ON rag_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

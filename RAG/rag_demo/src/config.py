from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    db_dir: Path
    sqlite_path: Path
    pg_dsn: str
    pgvector_table: str

    google_api_key: Optional[str]
    google_embedding_model: str
    google_chat_model: str

    chunk_target_chars: int
    chunk_overlap_chars: int


def load_config(project_root: Path) -> AppConfig:
    """
    Load configuration from environment variables with safe local defaults.
    """
    load_dotenv(project_root / ".env", override=False)

    data_dir = Path(os.getenv("RAG_DEMO_DATA_DIR", str(project_root / "data"))).resolve()
    db_dir = Path(os.getenv("RAG_DEMO_DB_DIR", str(project_root / "db"))).resolve()
    sqlite_path = Path(os.getenv("RAG_DEMO_SQLITE_PATH", str(db_dir / "registry.sqlite"))).resolve()

    pg_dsn = os.getenv("RAG_DEMO_PG_DSN", "postgresql://rag:rag@localhost:5432/rag_demo")
    pgvector_table = os.getenv("RAG_DEMO_PGVECTOR_TABLE", "rag_chunks")

    google_api_key = os.getenv("GOOGLE_API_KEY") or None
    google_embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
    google_chat_model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")

    chunk_target_chars = int(os.getenv("RAG_DEMO_CHUNK_TARGET_CHARS", "1400"))
    chunk_overlap_chars = int(os.getenv("RAG_DEMO_CHUNK_OVERLAP_CHARS", "200"))

    return AppConfig(
        data_dir=data_dir,
        db_dir=db_dir,
        sqlite_path=sqlite_path,
        pg_dsn=pg_dsn,
        pgvector_table=pgvector_table,
        google_api_key=google_api_key,
        google_embedding_model=google_embedding_model,
        google_chat_model=google_chat_model,
        chunk_target_chars=chunk_target_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )


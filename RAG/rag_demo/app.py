from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

import sys

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.generation.answer_generator import AnswerGenerator
from src.generation.llm import make_llm
from src.ingestion.ingest_pipeline import IngestPipeline
from src.ingestion.record_manager import RecordManager
from src.models import (
    AccessLevel,
    DocumentIngestRequest,
    RetrievalMode,
    UserRole,
)
from src.retrieval.query_understanding import make_query_understanding
from src.retrieval.retriever import Retriever
from src.utils.file_utils import ensure_dir, list_pdfs
from src.utils.logging_utils import EventLogger


def _get_logger() -> EventLogger:
    if "event_logger" not in st.session_state:
        st.session_state.event_logger = EventLogger()
    return st.session_state.event_logger


def _clear_logs() -> None:
    if "event_logger" in st.session_state:
        st.session_state.event_logger.clear()


def _save_uploaded_pdf(data_dir: Path) -> Optional[Path]:
    uploaded = st.file_uploader("Upload a PDF into `data/`", type=["pdf"], accept_multiple_files=False)
    if not uploaded:
        return None
    ensure_dir(data_dir)
    out_path = data_dir / uploaded.name
    out_path.write_bytes(uploaded.getvalue())
    return out_path


def _doc_registry_table(registry_docs):
    rows = []
    for d in registry_docs:
        rows.append(
            {
                "document_name": d.document_name,
                "document_id": d.document_id,
                "version": d.version,
                "access_level": d.access_level.value,
                "updated_at": d.updated_at.isoformat(),
                "source_path": d.source_path,
                "content_hash": d.content_hash[:12],
            }
        )
    return rows


def main():
    st.set_page_config(page_title="RAG Demo (Production-Inspired)", layout="wide")

    cfg = load_config(PROJECT_ROOT)
    logger = _get_logger()

    st.title("Retrieval-Augmented Generation (RAG) Demo")
    st.caption("Local-first, production-inspired layers: query understanding, access control, hybrid search, and change management.")

    with st.sidebar:
        st.subheader("Runtime config")
        st.write({"data_dir": str(cfg.data_dir), "db_dir": str(cfg.db_dir)})
        st.write({"sqlite": str(cfg.sqlite_path), "pg_dsn": cfg.pg_dsn, "pgvector_table": cfg.pgvector_table})
        st.write({"embeddings": cfg.google_embedding_model, "llm": cfg.google_chat_model})
        st.write({"google_api_key_configured": bool(cfg.google_api_key)})

        st.divider()
        if st.button("Clear debug logs"):
            _clear_logs()

        st.divider()
        st.subheader("Role & retrieval mode")
        role = st.selectbox("User role", options=[r.value for r in UserRole], index=0)
        mode = st.selectbox("Retrieval mode", options=[m.value for m in RetrievalMode], index=2)

        st.divider()
        st.subheader("Retrieval knobs")
        vector_top_k = st.slider("Vector top_k", 2, 20, 6, 1)
        keyword_top_k = st.slider("Keyword top_k", 2, 30, 8, 1)
        final_top_k = st.slider("Final top_k", 2, 12, 6, 1)

    tabs = st.tabs(["1) Ingest & Manage Documents", "2) Ask Questions (RAG)", "3) Debug / Observability"])

    # --- Tab 1: Ingestion ---
    with tabs[0]:
        st.subheader("Document ingestion + change management (SQLite + Chroma)")
        st.write(
            "This tab demonstrates: PDF loading, chunking, metadata enrichment, embedding, vector storage, and the record manager diff "
            "(new/changed/unchanged/deleted chunks)."
        )

        ensure_dir(cfg.data_dir)
        ensure_dir(cfg.db_dir)

        colA, colB = st.columns([2, 1], gap="large")
        with colA:
            st.markdown("**1) Add PDFs to the `data/` folder**")
            saved = _save_uploaded_pdf(cfg.data_dir)
            if saved:
                st.success(f"Saved: {saved.name}")

            pdfs = list_pdfs(cfg.data_dir)
            if not pdfs:
                st.info("No PDFs found in `data/` yet.")
                st.stop()

            selected = st.selectbox("Select a PDF to ingest", options=[p.name for p in pdfs], index=0)
            pdf_path = next(p for p in pdfs if p.name == selected)

            access = st.selectbox("Assign access_level to this document", options=[a.value for a in AccessLevel], index=0)
            ingest_req = DocumentIngestRequest(
                source_path=str(pdf_path),
                document_name=pdf_path.name,
                access_level=AccessLevel(access),
            )

            st.markdown("**2) Ingest (or re-ingest after updates)**")
            if st.button("Ingest PDF now", type="primary"):
                logger.info("Starting ingestion", {"pdf": pdf_path.name, "access_level": access})
                pipeline = IngestPipeline(cfg, logger)
                try:
                    res = pipeline.ingest_pdf(ingest_req)
                    st.success("Ingestion complete.")
                    st.json(
                        {
                            "document_id": res.document.document_id,
                            "version": res.document.version,
                            "embedder": res.embedder_name,
                            "vector_store_count": res.vector_store_count,
                            "diff": {
                                "new": len(res.diff.new_chunks),
                                "changed": len(res.diff.changed_chunks),
                                "unchanged": len(res.diff.unchanged_chunks),
                                "previous_active": len(res.diff.previous_active_chunk_uids),
                                "deleted": len(res.diff.deleted_chunk_uids),
                            },
                            "notes": res.notes,
                        }
                    )
                finally:
                    pipeline.close()

        with colB:
            st.markdown("**Registry (SQLite) — ingested docs**")
            rm = RecordManager(cfg.sqlite_path)
            try:
                docs = rm.list_documents(include_inactive=True)
            finally:
                rm.close()

            if docs:
                st.dataframe(_doc_registry_table(docs), use_container_width=True, height=320)
                st.caption("Tip: re-upload/replace a PDF with the same filename, then re-ingest to demonstrate versioning + stale chunk cleanup.")
            else:
                st.info("No documents ingested yet.")

    # --- Tab 2: Q&A ---
    with tabs[1]:
        st.subheader("Query understanding → retrieval (access control + hybrid) → answer generation")

        st.markdown("**Enter a question**")
        default_q = "What is BERT and what are the two pre-training tasks?"
        raw_query = st.text_input("Question", value=st.session_state.get("last_query", default_q))
        st.session_state["last_query"] = raw_query

        if st.button("Run RAG", type="primary", disabled=not raw_query.strip()):
            logger.info("Starting RAG request", {"mode": mode, "role": role})

            qu = make_query_understanding(cfg.google_api_key, cfg.google_chat_model)
            qu_out = qu.understand(raw_query)
            st.markdown("**Query understanding output**")
            st.json(qu_out.model_dump())

            # Retrieval
            retriever = Retriever(cfg, logger)
            try:
                rr = retriever.retrieve(
                    query=" ".join([qu_out.normalized_query] + qu_out.expanded_terms).strip(),
                    role=UserRole(role),
                    mode=RetrievalMode(mode),
                    vector_top_k=vector_top_k,
                    keyword_top_k=keyword_top_k,
                    final_top_k=final_top_k,
                    extracted_filters=qu_out.extracted_filters,
                )
            finally:
                retriever.close()

            st.markdown("**Retrieval debug**")
            st.json(rr.debug.model_dump())
            st.markdown("**Per-layer retrieval debug**")
            st.json(rr.per_layer_debug)

            st.markdown("**Retrieved chunks**")
            for r in rr.retrieved:
                c = r.chunk
                with st.expander(f"{r.rank}. {c.document_name} | v{c.version} | chunk {c.chunk_id} | {c.access_level.value} | score={r.score:.4f} | {r.source}"):
                    st.write(
                        {
                            "chunk_uid": c.chunk_uid,
                            "document_id": c.document_id,
                            "section_title": c.section_title,
                            "access_level": c.access_level.value,
                            "content_hash": c.content_hash[:12],
                            "source_path": c.source_path,
                        }
                    )
                    st.text(c.text[:2500])
                    if r.debug:
                        st.caption("Debug")
                        st.json(r.debug)

            # Generation
            llm = make_llm(cfg.google_api_key, cfg.google_chat_model)
            gen = AnswerGenerator(llm=llm, logger=logger)
            ans = gen.generate(user_question=raw_query, retrieved=rr.retrieved)

            st.markdown("**Final answer**")
            st.write(ans.answer)

            st.markdown("**Citations (chunk metadata)**")
            st.json(ans.citations)

            st.markdown("**Context actually used (assembled)**")
            st.text(ans.context_used[:8000])

    # --- Tab 3: Observability ---
    with tabs[2]:
        st.subheader("Event log (teaching-friendly observability)")
        st.write("These are structured events emitted by ingestion/retrieval/generation.")
        if not logger.events:
            st.info("No events yet. Run ingestion or RAG to populate logs.")
        else:
            st.dataframe(
                [{"ts": e.ts, "level": e.level, "message": e.message, "data": e.data} for e in logger.events],
                use_container_width=True,
                height=420,
            )


if __name__ == "__main__":
    main()


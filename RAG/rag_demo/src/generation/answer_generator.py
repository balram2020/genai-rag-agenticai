from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..models import AnswerResult, RetrievedChunk
from ..utils.logging_utils import EventLogger
from .llm import LLM


def _format_citations(chunks: List[RetrievedChunk]) -> List[Dict]:
    out: List[Dict] = []
    for r in chunks:
        c = r.chunk
        out.append(
            {
                "chunk_uid": c.chunk_uid,
                "document_name": c.document_name,
                "document_id": c.document_id,
                "version": c.version,
                "chunk_id": c.chunk_id,
                "section_title": c.section_title,
                "access_level": c.access_level.value,
                "score": r.score,
                "source": r.source,
            }
        )
    return out


def _assemble_context(chunks: List[RetrievedChunk], max_chars: int = 6000) -> str:
    parts: List[str] = []
    total = 0
    for r in chunks:
        c = r.chunk
        header = f"[{c.document_name} | v{c.version} | chunk {c.chunk_id} | {c.section_title or 'no_section'}]"
        body = c.text.strip()
        piece = header + "\n" + body
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(parts).strip()


@dataclass
class AnswerGenerator:
    llm: LLM
    logger: EventLogger

    def generate(self, user_question: str, retrieved: List[RetrievedChunk]) -> AnswerResult:
        context = _assemble_context(retrieved)
        self.logger.info("Assembled context", {"chunks_used": len(retrieved), "context_chars": len(context)})

        prompt = (
            "You are a helpful assistant for a RAG demo.\n"
            "Answer the question using ONLY the provided context.\n"
            "If the context does not contain the answer, say you don't know.\n"
            "Be concise, and include a short bullet list of citations (chunk ids) at the end.\n\n"
            f"Question: {user_question}\n\n"
            f"Context:\n{context}\n"
        )

        answer_text = self.llm.generate(prompt)
        citations = _format_citations(retrieved)
        return AnswerResult(
            answer=answer_text,
            citations=citations,
            context_used=context,
            used_chunks=retrieved,
            debug={"llm": self.llm.name},
        )


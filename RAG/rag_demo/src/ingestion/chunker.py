from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    text: str
    section_title: Optional[str]


_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*)\s+([A-Z][^\n]{2,120})\s*$")


def _normalize_whitespace(text: str) -> str:
    # Keep newlines for heading detection, but collapse excessive whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Heuristic section splitter.
    - Detect lines that look like: "1 Introduction" or "2.1 Model Architecture"
    - Everything until the next heading is considered that section.
    """
    lines = text.split("\n")
    sections: List[Tuple[Optional[str], List[str]]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    def flush():
        nonlocal current_lines, current_title
        if current_lines:
            sections.append((current_title, current_lines))
        current_lines = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            current_title = f"{m.group(1)} {m.group(3)}".strip()
            continue
        current_lines.append(line)

    flush()
    return [(title, "\n".join(ls).strip()) for title, ls in sections if "\n".join(ls).strip()]


def chunk_text(
    text: str,
    target_chars: int = 1400,
    overlap_chars: int = 200,
) -> List[TextChunk]:
    """
    Chunk text into semi-semantic chunks while preserving (best-effort) section titles.

    Strategy:
    - Split into heuristic sections by headings.
    - Within each section, chunk by paragraphs to approach target_chars.
    - Add a small overlap to stabilize retrieval.
    """
    text = _normalize_whitespace(text)
    sections = _split_into_sections(text)
    if not sections:
        sections = [(None, text)]

    chunks: List[TextChunk] = []
    chunk_id = 0

    for section_title, section_text in sections:
        paras = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        buf: List[str] = []
        buf_len = 0

        def emit():
            nonlocal chunk_id, buf, buf_len
            if not buf:
                return
            chunk_text_val = "\n\n".join(buf).strip()
            if chunk_text_val:
                chunks.append(TextChunk(chunk_id=chunk_id, text=chunk_text_val, section_title=section_title))
                chunk_id += 1
            buf = []
            buf_len = 0

        for p in paras:
            if buf_len + len(p) + 2 <= target_chars:
                buf.append(p)
                buf_len += len(p) + 2
                continue

            # If buffer already has content, emit it.
            if buf:
                emit()

            # If paragraph alone is huge, hard-split it.
            if len(p) > target_chars:
                start = 0
                while start < len(p):
                    end = min(len(p), start + target_chars)
                    part = p[start:end].strip()
                    if part:
                        chunks.append(TextChunk(chunk_id=chunk_id, text=part, section_title=section_title))
                        chunk_id += 1
                    start = max(end - overlap_chars, end) if overlap_chars > 0 else end
                continue

            # Otherwise start buffer with this paragraph.
            buf = [p]
            buf_len = len(p)

        emit()

    # Add lightweight overlaps between consecutive chunks
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped: List[TextChunk] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
                continue
            prev = chunks[i - 1].text
            overlap = prev[-overlap_chars:].strip()
            text_with_overlap = (overlap + "\n\n" + ch.text).strip() if overlap else ch.text
            overlapped.append(TextChunk(chunk_id=ch.chunk_id, text=text_with_overlap, section_title=ch.section_title))
        chunks = overlapped

    return chunks


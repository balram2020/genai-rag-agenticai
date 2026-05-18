from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader


@dataclass(frozen=True)
class PDFPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class LoadedPDF:
    source_path: Path
    document_name: str
    pages: List[PDFPage]
    full_text: str


def load_pdf(path: Path, document_name: Optional[str] = None) -> LoadedPDF:
    """
    Load a PDF and extract per-page text.

    Notes for teaching:
    - PDF extraction is noisy. We keep per-page text so we can debug issues.
    - In production you might use a more robust pipeline (OCR, layout-aware parsing).
    """
    reader = PdfReader(str(path))
    pages: List[PDFPage] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(PDFPage(page_number=i + 1, text=text))

    full_text = "\n\n".join(p.text for p in pages).strip()
    return LoadedPDF(
        source_path=path,
        document_name=document_name or path.name,
        pages=pages,
        full_text=full_text,
    )


from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def list_pdfs(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def chunk_iterable(items: Iterable, size: int):
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


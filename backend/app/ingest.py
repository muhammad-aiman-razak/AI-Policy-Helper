import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _read_text_file(path: str) -> str:
    """Read a text file and return its contents."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _clean_title(filename: str) -> str:
    """Strip extension and replace underscores with spaces.

    Example: 'Returns_and_Refunds.md' -> 'Returns_and_Refunds'
    """
    return Path(filename).stem


def _md_sections(text: str) -> List[Tuple[str, str]]:
    """Split markdown text into (heading, body) pairs by heading lines.

    Sections that contain only a heading with no body content are
    skipped to avoid creating empty chunks.
    """
    parts = re.split(r"\n(?=#+\s)", text)
    out: List[Tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        first = lines[0]
        if first.startswith("#"):
            heading = first.lstrip("#").strip()
            body_lines = [
                ln for ln in lines[1:] if ln.strip()
            ]
            if not body_lines:
                continue
        else:
            heading = "Introduction"
        out.append((heading, part))
    return out or [("Introduction", text)]


def chunk_text(
    text: str, chunk_size: int, overlap: int
) -> List[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: The text to chunk.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.
    """
    tokens = text.split()
    if not tokens:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    return chunks


def load_documents(data_dir: str) -> List[Dict[str, str]]:
    """Load markdown/text documents and split into per-section records.

    Each record has keys: title (filename stem), section, text.
    """
    docs: List[Dict[str, str]] = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(data_dir, fname)
        text = _read_text_file(path)
        title = _clean_title(fname)
        for section, body in _md_sections(text):
            docs.append(
                {"title": title, "section": section, "text": body}
            )
    logger.info(
        "Loaded %d sections from %s", len(docs), data_dir
    )
    return docs


def doc_hash(text: str) -> str:
    """Return a SHA-256 hex digest for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

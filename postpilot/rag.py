"""Knowledge base and post history in pgvector: chunk, embed, store, search."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from postpilot.config import get_settings
from postpilot.db import connect
from postpilot.llm import get_embeddings

log = logging.getLogger(__name__)

KNOWLEDGE_SUFFIXES = {".md", ".markdown", ".txt"}


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Pack paragraphs into ~`size`-char chunks; hard-split paragraphs that are too long."""
    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        while len(para) > size:
            pieces.append(para[:size])
            para = para[size - overlap :]
        if para:
            pieces.append(para)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + piece if overlap else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def add_document(kind: str, source: str, text: str, platform: str | None = None) -> int:
    """Store `text` under `source`, replacing older content from that source.

    Returns the number of chunks written (0 when the content is unchanged).
    """
    text = text.strip()
    if not text:
        return 0
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    with connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM documents WHERE source = %s AND content_hash = %s LIMIT 1",
            (source, content_hash),
        ).fetchone()
        if exists:
            return 0

    chunks = chunk_text(text)
    vectors = get_embeddings().embed_documents(chunks)
    with connect() as conn:
        conn.execute("DELETE FROM documents WHERE source = %s", (source,))
        for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True)):
            conn.execute(
                "INSERT INTO documents (kind, platform, source, content_hash, chunk_index, "
                "content, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (kind, platform, source, content_hash, i, chunk, np.array(vec)),
            )
    log.info("indexed %s (%d chunks)", source, len(chunks))
    return len(chunks)


def sync_knowledge_dir(path: Path | None = None) -> int:
    """Index every .md/.txt file under the knowledge folder. Unchanged files are skipped."""
    root = path or get_settings().knowledge_dir
    if not root.exists():
        return 0
    written = 0
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in KNOWLEDGE_SUFFIXES and file.name != "README.md":
            written += add_document("knowledge", f"file:{file.relative_to(root)}", file.read_text())
    return written


def search(
    query: str,
    kinds: tuple[str, ...] = ("knowledge", "post"),
    platform: str | None = None,
    k: int | None = None,
) -> list[dict]:
    """Cosine-similarity search. `platform` limits to that platform plus untagged docs."""
    k = k or get_settings().rag_top_k
    vec = np.array(get_embeddings().embed_query(query))
    with connect() as conn:
        return conn.execute(
            "SELECT content, source, kind, platform, 1 - (embedding <=> %(v)s) AS score "
            "FROM documents WHERE kind = ANY(%(kinds)s) "
            "AND (%(p)s::text IS NULL OR platform IS NULL OR platform = %(p)s) "
            "ORDER BY embedding <=> %(v)s LIMIT %(k)s",
            {"v": vec, "kinds": list(kinds), "p": platform, "k": k},
        ).fetchall()

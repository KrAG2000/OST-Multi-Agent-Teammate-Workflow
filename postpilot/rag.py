"""Knowledge base and post history in pgvector: chunk, embed, store, search."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from postpilot import quota
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


def _embed_documents(texts: list[str]) -> list[list[float]]:
    quota.charge("embed_requests")
    return get_embeddings().embed_documents(texts)


def _embed_query(text: str) -> list[float]:
    quota.charge("embed_requests")
    return get_embeddings().embed_query(text)


def add_document(user_id: int, kind: str, source: str, text: str, platform: str | None = None) -> int:
    """Store `text` under the user's `source`, replacing older content from that source.

    Returns the number of chunks written (0 when the content is unchanged).
    """
    text = text.strip()
    if not text:
        return 0
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    with connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM documents WHERE user_id = %s AND source = %s AND content_hash = %s LIMIT 1",
            (user_id, source, content_hash),
        ).fetchone()
        if exists:
            return 0

    chunks = chunk_text(text)
    vectors = _embed_documents(chunks)
    with connect() as conn:
        conn.execute("DELETE FROM documents WHERE user_id = %s AND source = %s", (user_id, source))
        for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True)):
            conn.execute(
                "INSERT INTO documents (user_id, kind, platform, source, content_hash, chunk_index, "
                "content, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (user_id, kind, platform, source, content_hash, i, chunk, np.array(vec)),
            )
    log.info("indexed %s (%d chunks)", source, len(chunks))
    return len(chunks)


def knowledge_files(path: Path | None = None) -> list[Path]:
    root = path or get_settings().knowledge_dir
    if not root.exists():
        return []
    return [
        f for f in sorted(root.rglob("*"))
        if f.is_file() and f.suffix.lower() in KNOWLEDGE_SUFFIXES and f.name != "README.md"
    ]


def sync_knowledge_dir(user_id: int, path: Path | None = None) -> int:
    """Index every .md/.txt file under the knowledge folder for a user (the admin).

    Unchanged files are skipped.
    """
    root = path or get_settings().knowledge_dir
    return sum(
        add_document(user_id, "knowledge", f"file:{f.relative_to(root)}", f.read_text())
        for f in knowledge_files(root)
    )


def search(
    user_id: int,
    query: str,
    kinds: tuple[str, ...] = ("knowledge", "post"),
    platform: str | None = None,
    k: int | None = None,
) -> list[dict]:
    """Cosine-similarity search. `platform` limits to that platform plus untagged docs."""
    k = k or get_settings().rag_top_k
    vec = np.array(_embed_query(query))
    with connect() as conn:
        return conn.execute(
            "SELECT content, source, kind, platform, 1 - (embedding <=> %(v)s) AS score "
            "FROM documents WHERE user_id = %(u)s AND kind = ANY(%(kinds)s) "
            "AND (%(p)s::text IS NULL OR platform IS NULL OR platform = %(p)s) "
            "ORDER BY embedding <=> %(v)s LIMIT %(k)s",
            {"u": user_id, "v": vec, "kinds": list(kinds), "p": platform, "k": k},
        ).fetchall()

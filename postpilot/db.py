"""Postgres + pgvector connection and schema."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from postpilot.config import get_settings

log = logging.getLogger(__name__)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('knowledge', 'post')),
    platform      TEXT,
    source        TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    chunk_index   INT  NOT NULL,
    content       TEXT NOT NULL,
    embedding     vector({dim}) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, content_hash, chunk_index)
);
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS ideas (
    id          BIGSERIAL PRIMARY KEY,
    platform    TEXT NOT NULL,
    title       TEXT NOT NULL,
    angle       TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL CHECK (source IN ('ai', 'user')),
    status      TEXT NOT NULL DEFAULT 'suggested'
                CHECK (status IN ('suggested', 'queued', 'used', 'dropped')),
    priority    INT  NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    used_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ideas_open_idx ON ideas (platform, status);

CREATE TABLE IF NOT EXISTS steering (
    id          BIGSERIAL PRIMARY KEY,
    platform    TEXT NOT NULL,  -- platform id or 'all'
    directive   TEXT NOT NULL,
    active      BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS drafts (
    id                    BIGSERIAL PRIMARY KEY,
    run_date              DATE NOT NULL,
    platform              TEXT NOT NULL,
    idea_id               BIGINT REFERENCES ideas(id),
    parent_id             BIGINT REFERENCES drafts(id),
    title                 TEXT NOT NULL DEFAULT '',
    body                  TEXT NOT NULL,
    hashtags              TEXT[] NOT NULL DEFAULT '{{}}',
    image                 JSONB,
    critic_score          INT,
    critic_notes          TEXT NOT NULL DEFAULT '',
    feedback              TEXT,
    status                TEXT NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending', 'approved', 'rejected', 'superseded')),
    telegram_notified_at  TIMESTAMPTZ,
    emailed_at            TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS drafts_status_idx ON drafts (status, run_date);

CREATE TABLE IF NOT EXISTS telegram_messages (
    chat_id     BIGINT NOT NULL,
    message_id  BIGINT NOT NULL,
    draft_id    BIGINT NOT NULL REFERENCES drafts(id),
    PRIMARY KEY (chat_id, message_id)
);
"""


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Connection with dict rows and pgvector types; commits on clean exit."""
    with psycopg.connect(get_settings().database_url, row_factory=dict_row) as conn:
        register_vector(conn)
        yield conn


def init_db() -> None:
    """Create tables (idempotent), check the embedding size, and set up the checkpointer."""
    s = get_settings()
    # The scheduler and the bot both start by calling this; serialize them.
    with psycopg.connect(s.database_url, autocommit=True) as lock_conn:
        lock_conn.execute("SELECT pg_advisory_lock(7310)")
        try:
            _init_db_locked()
        finally:
            lock_conn.execute("SELECT pg_advisory_unlock(7310)")
    log.info("database ready")


def _init_db_locked() -> None:
    s = get_settings()
    with psycopg.connect(s.database_url) as conn:
        conn.execute(SCHEMA.format(dim=s.embedding_dim))
        (dim,) = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'documents'::regclass AND attname = 'embedding'"
        ).fetchone()
    if dim != s.embedding_dim:
        raise RuntimeError(
            f"documents.embedding is vector({dim}) but EMBEDDING_DIM={s.embedding_dim}. "
            "Switching embedding models requires re-ingesting: drop the documents table "
            "and run `postpilot ingest` again."
        )

    from postpilot.graph.pipeline import checkpointer

    with checkpointer() as cp:
        cp.setup()

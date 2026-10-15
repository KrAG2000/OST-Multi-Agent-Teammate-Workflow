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

CREATE TABLE IF NOT EXISTS users (
    id                BIGSERIAL PRIMARY KEY,
    telegram_chat_id  BIGINT NOT NULL UNIQUE,
    name              TEXT NOT NULL DEFAULT '',
    is_admin          BOOLEAN NOT NULL DEFAULT false,
    email_to          TEXT,
    timezone          TEXT NOT NULL,
    generate_time     TEXT NOT NULL,  -- 'HH:MM' in the user's timezone
    paused            BOOLEAN NOT NULL DEFAULT false,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_platforms (
    user_id  BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name     TEXT NOT NULL,
    config   JSONB NOT NULL,
    PRIMARY KEY (user_id, name)
);

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('knowledge', 'post')),
    platform      TEXT,
    source        TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    chunk_index   INT  NOT NULL,
    content       TEXT NOT NULL,
    embedding     vector({dim}) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
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
    status                TEXT NOT NULL DEFAULT 'pending',
    telegram_notified_at  TIMESTAMPTZ,
    emailed_at            TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS telegram_messages (
    chat_id     BIGINT NOT NULL,
    message_id  BIGINT NOT NULL,
    draft_id    BIGINT NOT NULL REFERENCES drafts(id),
    PRIMARY KEY (chat_id, message_id)
);

-- The single work queue. A job reserves its worst-case usage when queued.
CREATE TABLE IF NOT EXISTS jobs (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,
    payload      JSONB NOT NULL DEFAULT '{{}}',
    status       TEXT NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'running', 'done', 'failed', 'refused')),
    quota_day    DATE NOT NULL,
    reserved     JSONB NOT NULL DEFAULT '{{}}',
    used         JSONB NOT NULL DEFAULT '{{}}',
    result       JSONB,
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs (status, id);
CREATE INDEX IF NOT EXISTS jobs_user_idx ON jobs (user_id, kind, created_at);

-- Shared daily counters per resource (UTC day): used + reserved never exceeds the limit.
CREATE TABLE IF NOT EXISTS usage_daily (
    day       DATE NOT NULL,
    resource  TEXT NOT NULL,
    used      BIGINT NOT NULL DEFAULT 0,
    reserved  BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (day, resource)
);

-- Messages from the worker to users, delivered by the bot.
CREATE TABLE IF NOT EXISTS outbox (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    sent_at     TIMESTAMPTZ
);

-- Upgrades for databases created by earlier versions.
ALTER TABLE drafts ADD COLUMN IF NOT EXISTS posted_at TIMESTAMPTZ;
ALTER TABLE drafts ADD COLUMN IF NOT EXISTS reject_reason TEXT;
ALTER TABLE drafts DROP CONSTRAINT IF EXISTS drafts_status_check;
ALTER TABLE drafts ADD CONSTRAINT drafts_status_check
    CHECK (status IN ('pending', 'approved', 'posted', 'rejected', 'superseded'));
-- 'draft' = shows a draft (replies are rewrite feedback); 'reject_reason' = asks why it was rejected
ALTER TABLE telegram_messages ADD COLUMN IF NOT EXISTS purpose TEXT NOT NULL DEFAULT 'draft';

-- Multi-user: every row belongs to a user. Older rows are assigned to the first admin.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE ideas     ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE steering  ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE drafts    ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_source_content_hash_chunk_index_key;
CREATE UNIQUE INDEX IF NOT EXISTS documents_user_source_idx
    ON documents (user_id, source, content_hash, chunk_index);
DROP INDEX IF EXISTS ideas_open_idx;
CREATE INDEX IF NOT EXISTS ideas_user_open_idx ON ideas (user_id, platform, status);
DROP INDEX IF EXISTS drafts_status_idx;
CREATE INDEX IF NOT EXISTS drafts_user_status_idx ON drafts (user_id, status, run_date);
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
    # The worker and the bot both start by calling this; serialize them.
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

    from postpilot import store

    admins = [store.ensure_user(chat_id, is_admin=True) for chat_id in sorted(s.admin_chat_ids)]
    if admins:
        with connect() as conn:
            for table in ("documents", "ideas", "steering", "drafts"):
                conn.execute(f"UPDATE {table} SET user_id = %s WHERE user_id IS NULL", (admins[0]["id"],))

    from postpilot.graph.pipeline import checkpointer

    with checkpointer() as cp:
        cp.setup()

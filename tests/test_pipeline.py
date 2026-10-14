"""End-to-end pipeline against a real Postgres+pgvector, with fake LLM and embeddings.

Needs the compose database: `docker compose up -d db`. Skipped if unreachable.
"""

import hashlib
import os

import psycopg
import pytest

ADMIN_URL = os.environ.get("TEST_ADMIN_DATABASE_URL", "postgresql://postpilot:postpilot@localhost:5433/postpilot")
TEST_DB = "postpilot_test"


class FakeEmbeddings:
    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [(digest[i % 32] - 128) / 128 for i in range(768)]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def fake_ask_structured(schema, system, user, temperature=None):
    from postpilot.graph import agents

    if schema is agents.IdeaBatch:
        return agents.IdeaBatch(
            ideas=[agents.IdeaOut(title=f"Idea {i}", angle="an angle") for i in range(5)]
        )
    if schema is agents.DraftOut:
        return agents.DraftOut(
            title="Hook line",
            body="word " * 130,
            hashtags=["#one", "#two", "#three", "#four"],
            image=agents.ImageOut(size_label="square_1_1", prompt="a scene", alt_text="alt"),
        )
    if schema is agents.Critique:
        return agents.Critique(score=8, approved=True, issues=[])
    raise AssertionError(schema)


@pytest.fixture
def db(monkeypatch):
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("test database not reachable")
    admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.execute(f"CREATE DATABASE {TEST_DB}")
    url = ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"

    from postpilot import config, rag
    from postpilot.graph import agents

    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    config.get_settings.cache_clear()
    monkeypatch.setattr(rag, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(agents, "ask_structured", fake_ask_structured)

    from postpilot.db import init_db

    init_db()
    yield url
    config.get_settings.cache_clear()
    admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.close()


def test_daily_run_regen_and_resume(db):
    from postpilot import rag, store
    from postpilot.graph import pipeline

    assert rag.add_document("knowledge", "file:a.md", "I built an agent framework.\n\nIt uses Postgres.") == 1
    assert rag.add_document("knowledge", "file:a.md", "I built an agent framework.\n\nIt uses Postgres.") == 0
    store.add_ideas("linkedin", [{"title": "My own idea", "angle": "mine"}], source="user")
    store.add_steering("all", "keep it practical")

    drafts = pipeline.run_daily()
    by_platform = {d["platform"]: d for d in drafts}
    assert set(by_platform) == {"instagram", "linkedin", "medium"}

    li = store.get_draft(by_platform["linkedin"]["id"])
    assert li["title"] == "Hook line"
    assert store.get_idea(li["idea_id"])["title"] == "My own idea"  # user idea goes first
    assert li["image"]["width"] == 1080 and li["image"]["size_label"] == "square_1_1"

    # Medium needs 800+ words, so the critic loop revises until MAX_REVISIONS, then saves with notes.
    medium = store.get_draft(by_platform["medium"]["id"])
    assert "at least 800" in medium["critic_notes"]
    assert medium["image"]["size_label"] == "cover_16_9"  # square isn't allowed on Medium

    # Same-day scheduled run is idempotent; force makes a new set.
    assert {d["id"] for d in pipeline.run_daily()} == {d["id"] for d in drafts}
    assert len(pipeline.run_daily(force=True)) == 3

    new = pipeline.regenerate(li["id"], "shorter please")
    assert store.get_draft(li["id"])["status"] == "superseded"
    regen = store.get_draft(new["id"])
    assert regen["parent_id"] == li["id"] and regen["feedback"] == "shorter please"

    hits = rag.search("agent framework", kinds=("knowledge",))
    assert hits and hits[0]["source"] == "file:a.md"


def test_failed_run_resumes_only_unfinished_platforms(db, monkeypatch):
    from postpilot import store
    from postpilot.graph import pipeline

    real_save = store.save_draft
    calls = {"medium": 0}

    def flaky_save(**kw):
        if kw["platform"] == "medium":
            calls["medium"] += 1
            if calls["medium"] == 1:
                raise RuntimeError("simulated crash")
        return real_save(**kw)

    monkeypatch.setattr(store, "save_draft", flaky_save)
    with pytest.raises(RuntimeError, match="simulated crash"):
        pipeline.run_daily()
    assert {d["platform"] for d in store.pending_drafts()} == {"instagram", "linkedin"}

    drafts = pipeline.run_daily()  # resumes the same thread
    assert sorted(d["platform"] for d in drafts) == ["instagram", "linkedin", "medium"]
    assert len(store.pending_drafts()) == 3  # finished platforms were not redone

"""End-to-end tests against a real Postgres+pgvector, with fake LLM and embeddings.

Needs the compose database: `docker compose up -d db`. Skipped if unreachable.
"""

import hashlib
import os
import threading

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


class FakeRaw:
    usage_metadata = {"total_tokens": 1000}


class FakeModel:
    """Stands in for a chat model: returns canned structured output per schema."""

    def with_structured_output(self, schema, include_raw=False):
        from postpilot.graph import agents

        model = self

        class Runner:
            def invoke(self, messages):
                model.calls += 1
                if schema is agents.IdeaBatch:
                    parsed = agents.IdeaBatch(
                        ideas=[agents.IdeaOut(title=f"Idea {model.calls}-{i}", angle="an angle") for i in range(5)]
                    )
                elif schema is agents.DraftOut:
                    parsed = agents.DraftOut(
                        title="Hook line",
                        body="word " * 130,
                        hashtags=["#one", "#two", "#three", "#four"],
                        image=agents.ImageOut(size_label="square_1_1", prompt="a scene", alt_text="alt"),
                    )
                else:
                    parsed = agents.Critique(score=8, approved=True, issues=[])
                return {"raw": FakeRaw(), "parsed": parsed, "parsing_error": None}

        return Runner()

    calls = 0


@pytest.fixture
def db(monkeypatch):
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("test database not reachable")
    admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.execute(f"CREATE DATABASE {TEST_DB}")
    url = ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"

    from postpilot import config, llm, rag

    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_IDS", "1001")
    monkeypatch.setenv("SMTP_USER", "")  # never send real email from tests
    monkeypatch.setenv("SMTP_HOST", "")
    config.get_settings.cache_clear()
    model = FakeModel()
    monkeypatch.setattr(rag, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(llm, "get_chat_model", lambda temperature=None: model)

    from postpilot.db import init_db

    init_db()
    yield model
    config.get_settings.cache_clear()
    admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.close()


def _limits(monkeypatch, **env):
    from postpilot import config

    for k, v in env.items():
        monkeypatch.setenv(k.upper(), str(v))
    config.get_settings.cache_clear()


def _drain():
    """Run queued jobs the way the worker does, one at a time."""
    from postpilot import jobs

    while job := jobs.claim_next():
        jobs.run_job(job)


def test_daily_run_regen_and_resume(db):
    from postpilot import rag, store
    from postpilot.graph import pipeline

    user = store.user_by_chat(1001)
    assert user["is_admin"]
    uid = user["id"]
    assert rag.add_document(uid, "knowledge", "file:a.md", "I built an agent framework.\n\nIt uses Postgres.") == 1
    assert rag.add_document(uid, "knowledge", "file:a.md", "I built an agent framework.\n\nIt uses Postgres.") == 0
    store.add_ideas(uid, "linkedin", [{"title": "My own idea", "angle": "mine"}], source="user")
    store.add_steering(uid, "all", "keep it practical")

    drafts = pipeline.run_daily(user)
    by_platform = {d["platform"]: d for d in drafts}
    assert set(by_platform) == {"instagram", "linkedin", "medium"}

    li = store.get_draft(uid, by_platform["linkedin"]["id"])
    assert store.get_idea(uid, li["idea_id"])["title"] == "My own idea"  # user idea goes first
    assert li["image"]["width"] == 1080 and li["image"]["size_label"] == "square_1_1"

    medium = store.get_draft(uid, by_platform["medium"]["id"])
    assert "at least 800" in medium["critic_notes"]  # revised until MAX_REVISIONS, then saved with notes
    assert medium["image"]["size_label"] == "cover_16_9"

    assert {d["id"] for d in pipeline.run_daily(user)} == {d["id"] for d in drafts}  # idempotent per day
    assert len(pipeline.run_daily(user, force=True)) == 3

    new = pipeline.regenerate(user, li["id"], "shorter please")
    assert store.get_draft(uid, li["id"])["status"] == "superseded"
    assert store.get_draft(uid, new["id"])["parent_id"] == li["id"]

    hits = rag.search(uid, "agent framework", kinds=("knowledge",))
    assert hits and hits[0]["source"] == "file:a.md"


def test_failed_run_resumes_only_unfinished_platforms(db, monkeypatch):
    from postpilot import store
    from postpilot.graph import pipeline

    user = store.user_by_chat(1001)
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
        pipeline.run_daily(user)
    assert {d["platform"] for d in store.pending_drafts(user["id"])} == {"instagram", "linkedin"}
    assert sorted(d["platform"] for d in pipeline.run_daily(user)) == ["instagram", "linkedin", "medium"]
    assert len(store.pending_drafts(user["id"])) == 3


def test_users_are_isolated(db):
    from postpilot import rag, store

    a = store.user_by_chat(1001)
    b = store.ensure_user(2002, "Bea")
    assert not b["is_admin"] and set(store.load_platforms(b["id"])) == set(store.load_platforms(a["id"]))
    store.update_platform(b["id"], "medium", enabled=False, tone="dry")
    assert "medium" not in store.enabled_platforms(b["id"])
    assert "medium" in store.enabled_platforms(a["id"])

    rag.add_document(a["id"], "knowledge", "note", "Alice's secret notes")
    assert rag.search(b["id"], "secret") == []
    store.add_ideas(a["id"], "linkedin", [{"title": "A's idea"}], source="user")
    assert store.open_ideas(b["id"]) == []
    with pytest.raises(ValueError):
        store.update_platform(b["id"], "nope", enabled=True)


def test_worker_runs_jobs_and_releases_unused_reservations(db, monkeypatch):
    from postpilot import jobs, quota, store

    _limits(monkeypatch, quota_llm_requests_per_day=200)
    user = store.user_by_chat(1001)
    job_id, position = jobs.enqueue(user, "generate", {"platforms": ["linkedin"]})
    assert position == 1
    reserved = quota.usage_today()["llm_requests"]["reserved"]
    assert reserved == 1 + 2 * 3 + 2  # ideas + write/critique rounds + retry headroom

    _drain()
    usage = quota.usage_today()["llm_requests"]
    assert usage["reserved"] == 0  # unused reservation released
    assert usage["used"] == db.calls == 3  # ideas, write, critique (approved first time)
    assert quota.usage_today()["llm_tokens"]["used"] == 3000  # real token counts recorded
    assert quota.usage_today()["jobs"] == {"used": 1, "reserved": 0, "limit": 60}
    assert len(store.pending_drafts(user["id"])) == 1
    assert any("new draft" in m["text"] for m in store.unsent_messages())


def test_requests_refused_when_capacity_is_full(db, monkeypatch):
    from postpilot import jobs, quota, store
    from postpilot.quota import AlreadyQueued, CapacityError

    _limits(monkeypatch, quota_llm_requests_per_day=12, quota_jobs_per_user_per_day=0)
    user = store.user_by_chat(1001)
    jobs.enqueue(user, "generate", {"platforms": ["linkedin"]})  # reserves 9 of 12
    with pytest.raises(AlreadyQueued):
        jobs.enqueue(user, "generate", {"platforms": ["instagram"]})
    with pytest.raises(CapacityError, match="try again in"):
        jobs.enqueue(user, "regen", {"draft_id": 1})  # needs 8 more, only 3 left
    jobs.enqueue(user, "refill", {"platform": "linkedin"})  # needs 2: fits
    assert quota.usage_today()["llm_requests"]["reserved"] == 11


def test_concurrent_enqueues_never_overbook(db, monkeypatch):
    from postpilot import jobs, quota, store
    from postpilot.quota import CapacityError

    _limits(monkeypatch, quota_llm_requests_per_day=20, quota_jobs_per_user_per_day=0)
    users = [store.ensure_user(3000 + i) for i in range(10)]
    accepted, refused = [], []
    barrier = threading.Barrier(len(users))

    def attempt(u):
        barrier.wait()
        try:
            accepted.append(jobs.enqueue(u, "refill", {"platform": "linkedin"}))
        except CapacityError:
            refused.append(u)

    threads = [threading.Thread(target=attempt, args=(u,)) for u in users]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(accepted) == 10 and not refused  # 10 x 2 = 20 fits exactly
    with pytest.raises(CapacityError):
        jobs.enqueue(store.ensure_user(4000), "refill", {"platform": "linkedin"})
    assert quota.usage_today()["llm_requests"]["reserved"] == 20


def test_running_job_stops_at_the_hard_limit(db, monkeypatch):
    from postpilot import jobs, quota, store
    from postpilot.db import connect

    # Each fake call really uses 1000 tokens but only 1 is reserved per call, so the job must
    # borrow from today's headroom; once that is gone it is stopped instead of overrunning.
    _limits(monkeypatch, quota_llm_tokens_per_day=1500, est_tokens_per_llm_request=1)
    user = store.user_by_chat(1001)
    job_id, _ = jobs.enqueue(user, "generate", {"platforms": ["linkedin"]})
    _drain()

    with connect() as conn:
        job = conn.execute("SELECT status, error FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert job["status"] == "failed" and "capacity" in job["error"]
    assert quota.usage_today()["llm_requests"]["used"] == 2  # ideas ran, write ran, critique never started
    assert quota.usage_today()["llm_tokens"]["reserved"] == 0
    assert store.pending_drafts(user["id"]) == []
    assert any("stopped" in m["text"] for m in store.unsent_messages())
    # And nothing new is accepted today.
    from postpilot.quota import CapacityError

    with pytest.raises(CapacityError):
        jobs.enqueue(user, "refill", {"platform": "linkedin"})


def test_daily_schedule_enqueues_once_and_reports_refusals(db, monkeypatch):
    from postpilot import jobs, store

    user = store.user_by_chat(1001)
    store.update_user(user["id"], generate_time="00:00")
    jobs.enqueue_due_daily()
    jobs.enqueue_due_daily()  # second tick the same day does nothing
    assert jobs.queue_length() == 1

    late = store.ensure_user(5005)
    store.update_user(late["id"], generate_time="00:00")
    _limits(monkeypatch, quota_jobs_per_day=1)
    jobs.enqueue_due_daily()
    assert any("skipped" in m["text"] for m in store.unsent_messages() if m["user_id"] == late["id"])
    jobs.enqueue_due_daily()  # refusal is recorded, so no retry spam
    assert sum("skipped" in m["text"] for m in store.unsent_messages()) == 1


def test_draft_lifecycle_and_message_purposes(db):
    from postpilot import store
    from postpilot.graph import pipeline

    user = store.user_by_chat(1001)
    draft_id = pipeline.run_now(user, ["linkedin"])[0]["id"]
    store.update_draft(user["id"], draft_id, status="posted")
    d = store.get_draft(user["id"], draft_id)
    assert d["status"] == "posted" and d["posted_at"] is not None
    assert store.get_draft(store.ensure_user(6006)["id"], draft_id) is None  # not visible to others

    store.mark_notified(draft_id, [(1, 100)])
    store.remember_message(1, 101, draft_id, "reject_reason")
    assert store.draft_for_message(1, 100) == {"draft_id": draft_id, "purpose": "draft"}
    assert store.draft_for_message(1, 101) == {"draft_id": draft_id, "purpose": "reject_reason"}

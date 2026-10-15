"""The single work queue.

Anything that calls an LLM, embeds text or sends email is a job. The bot and the daily
scheduler only *enqueue*; one worker process runs jobs strictly one at a time, oldest
first. Enqueueing reserves the job's worst-case usage (see quota.py) in the same
transaction that inserts it, so the queue can never hold more work than today's limits.
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from psycopg.types.json import Jsonb

from postpilot import quota, rag, store
from postpilot.config import get_settings
from postpilot.db import connect
from postpilot.notify import email
from postpilot.quota import AlreadyQueued, CapacityError

log = logging.getLogger(__name__)

# Jobs that use the LLM: they count toward the per-user daily cap, and a user can
# have only one of each kind waiting at a time.
LLM_KINDS = {"generate", "regen", "refill"}
LABELS = {
    "generate": "draft generation",
    "regen": "rewrite",
    "refill": "idea refill",
    "ingest": "note",
    "index_post": "post history update",
    "sync_knowledge": "knowledge sync",
}


def estimate(user: dict, kind: str, payload: dict) -> dict[str, int]:
    """Worst-case usage of a job, reserved up front and released when it finishes."""
    s = get_settings()
    rounds = 2 * (1 + s.max_revisions)  # write + critique per attempt
    llm = embed = emails = 0
    if kind == "generate":
        n = len(payload.get("platforms") or store.enabled_platforms(user["id"]))
        llm = n * (1 + rounds + 2)  # idea refill + rounds + retry headroom
        embed = n * 4  # idea grounding + two retrieval searches + headroom
        if payload.get("daily"):
            if user["is_admin"]:
                embed += len(rag.knowledge_files())
            if s.smtp_configured:
                emails = len(email.recipients(user))
    elif kind == "regen":
        llm, embed = rounds + 2, 3
    elif kind == "refill":
        llm, embed = 2, 1
    elif kind in ("ingest", "index_post"):
        embed = 1
    elif kind == "sync_knowledge":
        embed = len(rag.knowledge_files())
    needs = {
        "llm_requests": llm,
        "llm_tokens": llm * s.est_tokens_per_llm_request,
        "embed_requests": embed,
        "emails": emails,
        "jobs": 1 if kind in LLM_KINDS else 0,
    }
    return {k: v for k, v in needs.items() if v}


def enqueue(user: dict, kind: str, payload: dict | None = None) -> tuple[int, int]:
    """Queue a job or raise CapacityError. Returns (job_id, position in queue)."""
    payload = payload or {}
    needs = estimate(user, kind, payload)
    s = get_settings()
    with connect() as conn:  # one transaction: checks, reservation and insert succeed or fail together
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (quota.QUOTA_LOCK,))
        if kind in LLM_KINDS:
            waiting = conn.execute(
                "SELECT 1 FROM jobs WHERE user_id = %s AND kind = %s AND status IN ('queued', 'running')",
                (user["id"], kind),
            ).fetchone()
            if waiting:
                raise AlreadyQueued(f"You already have a {LABELS[kind]} in the queue. Please wait for it to finish.")
            cap = s.quota_jobs_per_user_per_day
            done_today = conn.execute(
                "SELECT count(*) AS n FROM jobs WHERE user_id = %s AND quota_day = %s "
                "AND kind = ANY(%s) AND status <> 'refused'",
                (user["id"], quota.quota_day(), list(LLM_KINDS)),
            ).fetchone()["n"]
            if cap and done_today >= cap:
                raise CapacityError(
                    f"You've used your {cap} requests for today. Please try again in {quota.resets_in()}."
                )
        day = quota.reserve(conn, needs)
        job_id = conn.execute(
            "INSERT INTO jobs (user_id, kind, payload, quota_day, reserved) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (user["id"], kind, Jsonb(payload), day, Jsonb(needs)),
        ).fetchone()["id"]
        position = conn.execute(
            "SELECT count(*) AS n FROM jobs WHERE status IN ('queued', 'running') AND id <= %s", (job_id,)
        ).fetchone()["n"]
    log.info("queued job %d (%s) for user %d, reserved %s", job_id, kind, user["id"], needs)
    return job_id, position


def record_refusal(user: dict, kind: str, payload: dict, reason: str) -> None:
    """Keep a record of a refused request (also stops the scheduler from retrying it every minute)."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (user_id, kind, payload, status, quota_day, error, finished_at) "
            "VALUES (%s, %s, %s, 'refused', %s, %s, now())",
            (user["id"], kind, Jsonb(payload), quota.quota_day(), reason),
        )


def claim_next() -> dict | None:
    with connect() as conn:
        return conn.execute(
            "UPDATE jobs SET status = 'running', started_at = now() WHERE id = ("
            "  SELECT id FROM jobs WHERE status = 'queued' ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1"
            ") RETURNING *"
        ).fetchone()


def recover_running() -> int:
    """After a crash, jobs left 'running' go back to the queue (runs resume from checkpoints)."""
    with connect() as conn:
        return conn.execute("UPDATE jobs SET status = 'queued' WHERE status = 'running'").rowcount


def finish(job_id: int, status: str, result: dict | None = None, error: str | None = None) -> None:
    with connect() as conn:
        quota.release(conn, job_id)
        conn.execute(
            "UPDATE jobs SET status = %s, result = %s, error = %s, finished_at = now() WHERE id = %s",
            (status, Jsonb(result) if result is not None else None, error, job_id),
        )


def queue_length() -> int:
    with connect() as conn:
        return conn.execute("SELECT count(*) AS n FROM jobs WHERE status IN ('queued', 'running')").fetchone()["n"]


# --- running jobs --------------------------------------------------------------


def _generate(user: dict, payload: dict) -> dict:
    from postpilot.graph import pipeline

    if payload.get("daily"):
        if user["is_admin"]:
            rag.sync_knowledge_dir(user["id"])
        drafts = pipeline.run_daily(user)
        try:
            emailed = email.send_digest(user)
        except CapacityError:
            raise
        except Exception as exc:  # drafts are fine; just report the email problem
            log.exception("email failed for user %d", user["id"])
            store.notify(user["id"], f"⚠️ Your drafts are ready, but the email failed: {exc}")
            emailed = 0
        return {"drafts": len(drafts), "emailed": emailed}
    drafts = pipeline.run_now(user, payload.get("platforms"))
    store.notify(user["id"], f"✅ {len(drafts)} new draft(s) ready.")
    return {"drafts": len(drafts)}


def _regen(user: dict, payload: dict) -> dict:
    from postpilot.graph import pipeline

    new = pipeline.regenerate(user, payload["draft_id"], payload.get("feedback", ""))
    return {"draft_id": new["id"]}


def _refill(user: dict, payload: dict) -> dict:
    from postpilot.graph import pipeline

    ids = pipeline.refill_ideas(user["id"], payload["platform"])
    store.notify(user["id"], f"💡 {len(ids)} new ideas for {payload['platform']}. /ideas {payload['platform']} to see them.")
    return {"ideas": ids}


def _ingest(user: dict, payload: dict) -> dict:
    n = rag.add_document(user["id"], "knowledge", payload["source"], payload["text"])
    store.notify(user["id"], f"🧠 Saved to your knowledge base ({n} chunk{'s' if n != 1 else ''}).")
    return {"chunks": n}


def _index_post(user: dict, payload: dict) -> dict:
    d = store.get_draft(user["id"], payload["draft_id"])
    from postpilot import render

    n = rag.add_document(user["id"], "post", f"draft:{d['id']}", render.post_text(d), d["platform"])
    return {"chunks": n}


def _sync_knowledge(user: dict, payload: dict) -> dict:
    return {"chunks": rag.sync_knowledge_dir(user["id"])}


HANDLERS = {
    "generate": _generate,
    "regen": _regen,
    "refill": _refill,
    "ingest": _ingest,
    "index_post": _index_post,
    "sync_knowledge": _sync_knowledge,
}


def run_job(job: dict) -> None:
    user = store.get_user(job["user_id"])
    log.info("running job %d (%s) for user %d", job["id"], job["kind"], job["user_id"])
    with quota.job_context(job["id"]):
        try:
            if job["kind"] in LLM_KINDS:
                quota.charge("jobs")  # converts the job's reservation into usage
            result = HANDLERS[job["kind"]](user, job["payload"])
        except CapacityError as exc:
            finish(job["id"], "failed", error=str(exc))
            store.notify(user["id"], f"⏸ Your {LABELS[job['kind']]} stopped: {exc}")
        except Exception as exc:
            log.exception("job %d failed", job["id"])
            finish(job["id"], "failed", error=f"{type(exc).__name__}: {exc}")
            store.notify(user["id"], f"⚠️ Your {LABELS[job['kind']]} failed: {type(exc).__name__}: {exc}")
        else:
            finish(job["id"], "done", result=result)


# --- daily schedule --------------------------------------------------------------


def enqueue_due_daily() -> None:
    """Queue each user's daily run once their local generate_time has passed today.

    Using "has passed" rather than "is exactly now" means a machine that was asleep
    at the scheduled time still runs it when it wakes up.
    """
    for user in store.active_users():
        now = datetime.now(ZoneInfo(user["timezone"]))
        hh, mm = (int(x) for x in user["generate_time"].split(":"))
        if (now.hour, now.minute) < (hh, mm) or not store.enabled_platforms(user["id"]):
            continue
        payload = {"daily": True, "run_date": now.date().isoformat()}
        with connect() as conn:
            seen = conn.execute(
                "SELECT 1 FROM jobs WHERE user_id = %s AND kind = 'generate' "
                "AND payload->>'daily' = 'true' AND payload->>'run_date' = %s",
                (user["id"], payload["run_date"]),
            ).fetchone()
        if seen:
            continue
        try:
            enqueue(user, "generate", payload)
        except AlreadyQueued:
            continue  # a manual run is waiting; try again next minute
        except CapacityError as exc:
            record_refusal(user, "generate", payload, str(exc))
            store.notify(user["id"], f"⏸ Today's drafts were skipped. {exc} Use /run once capacity is back.")

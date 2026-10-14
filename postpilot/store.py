"""Queries for ideas, steering directives and drafts."""

from __future__ import annotations

from datetime import date
from typing import Any

from psycopg.types.json import Jsonb

from postpilot.db import connect

# --- ideas -------------------------------------------------------------------


def add_ideas(platform: str, ideas: list[dict], source: str = "ai") -> list[int]:
    status = "queued" if source == "user" else "suggested"
    priority = 10 if source == "user" else 0
    with connect() as conn:
        return [
            conn.execute(
                "INSERT INTO ideas (platform, title, angle, source, status, priority) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (platform, i["title"], i.get("angle", ""), source, status, priority),
            ).fetchone()["id"]
            for i in ideas
        ]


def open_ideas(platform: str | None = None) -> list[dict]:
    """Backlog in pick order: user-queued first, then by priority and age."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM ideas WHERE status IN ('queued', 'suggested') "
            "AND (%(p)s::text IS NULL OR platform = %(p)s) "
            "ORDER BY platform, status = 'queued' DESC, priority DESC, created_at",
            {"p": platform},
        ).fetchall()


def next_idea(platform: str) -> dict | None:
    ideas = open_ideas(platform)
    return ideas[0] if ideas else None


def get_idea(idea_id: int) -> dict | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM ideas WHERE id = %s", (idea_id,)).fetchone()


def update_idea(idea_id: int, **fields: Any) -> bool:
    allowed = {"title", "angle", "status", "priority"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return False
    sets = ", ".join(f"{k} = %({k})s" for k in fields)
    used = ", used_at = now()" if fields.get("status") == "used" else ""
    with connect() as conn:
        cur = conn.execute(f"UPDATE ideas SET {sets}{used} WHERE id = %(id)s", {**fields, "id": idea_id})
        return cur.rowcount == 1


def recent_titles(platform: str, limit: int = 30) -> list[str]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT title FROM ideas WHERE platform = %s ORDER BY created_at DESC LIMIT %s",
            (platform, limit),
        ).fetchall()
    return [r["title"] for r in rows]


# --- steering ----------------------------------------------------------------


def add_steering(platform: str, directive: str) -> int:
    with connect() as conn:
        return conn.execute(
            "INSERT INTO steering (platform, directive) VALUES (%s, %s) RETURNING id",
            (platform, directive),
        ).fetchone()["id"]


def active_steering(platform: str | None = None) -> list[dict]:
    """Directives for one platform (plus global 'all' ones), or every active one."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM steering WHERE active "
            "AND (%(p)s::text IS NULL OR platform IN (%(p)s, 'all')) ORDER BY created_at",
            {"p": platform},
        ).fetchall()


def deactivate_steering(steering_id: int) -> bool:
    with connect() as conn:
        return conn.execute(
            "UPDATE steering SET active = false WHERE id = %s", (steering_id,)
        ).rowcount == 1


# --- drafts ------------------------------------------------------------------


def save_draft(
    *,
    run_date: date | str,
    platform: str,
    idea_id: int | None,
    draft: dict,
    critique: dict | None,
    parent_id: int | None = None,
    feedback: str | None = None,
) -> int:
    with connect() as conn:
        draft_id = conn.execute(
            "INSERT INTO drafts (run_date, platform, idea_id, parent_id, title, body, hashtags, "
            "image, critic_score, critic_notes, feedback) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (
                run_date,
                platform,
                idea_id,
                parent_id,
                draft.get("title", ""),
                draft["body"],
                draft.get("hashtags", []),
                Jsonb(draft["image"]) if draft.get("image") else None,
                (critique or {}).get("score"),
                "\n".join((critique or {}).get("issues", [])),
                feedback,
            ),
        ).fetchone()["id"]
        if idea_id:
            conn.execute(
                "UPDATE ideas SET status = 'used', used_at = now() WHERE id = %s", (idea_id,)
            )
        if parent_id:
            conn.execute(
                "UPDATE drafts SET status = 'superseded', updated_at = now() WHERE id = %s",
                (parent_id,),
            )
    return draft_id


def get_draft(draft_id: int) -> dict | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM drafts WHERE id = %s", (draft_id,)).fetchone()


def update_draft(draft_id: int, **fields: Any) -> bool:
    allowed = {"body", "title", "status"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return False
    sets = ", ".join(f"{k} = %({k})s" for k in fields)
    with connect() as conn:
        return conn.execute(
            f"UPDATE drafts SET {sets}, updated_at = now() WHERE id = %(id)s",
            {**fields, "id": draft_id},
        ).rowcount == 1


def pending_drafts(run_date: date | None = None) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM drafts WHERE status = 'pending' "
            "AND (%(d)s::date IS NULL OR run_date = %(d)s) ORDER BY run_date, platform, id",
            {"d": run_date},
        ).fetchall()


def drafts_to_notify() -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM drafts WHERE status = 'pending' AND telegram_notified_at IS NULL "
            "ORDER BY id"
        ).fetchall()


def drafts_to_email() -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM drafts WHERE status = 'pending' AND emailed_at IS NULL "
            "ORDER BY run_date, platform, id"
        ).fetchall()


def mark_notified(draft_id: int, messages: list[tuple[int, int]]) -> None:
    """Record delivery and remember which Telegram messages show this draft."""
    with connect() as conn:
        conn.execute(
            "UPDATE drafts SET telegram_notified_at = now() WHERE id = %s", (draft_id,)
        )
        for chat_id, message_id in messages:
            conn.execute(
                "INSERT INTO telegram_messages (chat_id, message_id, draft_id) "
                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (chat_id, message_id, draft_id),
            )


def draft_for_message(chat_id: int, message_id: int) -> int | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT draft_id FROM telegram_messages WHERE chat_id = %s AND message_id = %s",
            (chat_id, message_id),
        ).fetchone()
    return row["draft_id"] if row else None


def mark_emailed(draft_ids: list[int]) -> None:
    with connect() as conn:
        conn.execute("UPDATE drafts SET emailed_at = now() WHERE id = ANY(%s)", (draft_ids,))

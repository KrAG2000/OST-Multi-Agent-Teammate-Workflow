"""Queries for users, platforms, ideas, steering, drafts and the outbox.

Everything a user owns is filtered by user_id, so users never see each other's data.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from psycopg.types.json import Jsonb

from postpilot.config import Platform, get_settings, load_platform_templates
from postpilot.db import connect

# --- users -------------------------------------------------------------------


def get_user(user_id: int) -> dict | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()


def user_by_chat(chat_id: int) -> dict | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE telegram_chat_id = %s", (chat_id,)).fetchone()


def count_users() -> int:
    with connect() as conn:
        return conn.execute("SELECT count(*) AS n FROM users").fetchone()["n"]


def ensure_user(chat_id: int, name: str = "", is_admin: bool = False) -> dict:
    """Get or create the user for a Telegram chat; new users get the template platforms."""
    s = get_settings()
    with connect() as conn:
        user = conn.execute(
            "INSERT INTO users (telegram_chat_id, name, is_admin, email_to, timezone, generate_time) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (telegram_chat_id) DO NOTHING RETURNING *",
            (chat_id, name, is_admin, (s.email_to or None) if is_admin else None, s.timezone, s.generate_time),
        ).fetchone()
        if user is None:
            if is_admin:
                conn.execute("UPDATE users SET is_admin = true WHERE telegram_chat_id = %s", (chat_id,))
            return conn.execute("SELECT * FROM users WHERE telegram_chat_id = %s", (chat_id,)).fetchone()
        for p in load_platform_templates().values():
            conn.execute(
                "INSERT INTO user_platforms (user_id, name, config) VALUES (%s, %s, %s)",
                (user["id"], p.name, Jsonb(p.model_dump(exclude={"name"}))),
            )
        return user


def update_user(user_id: int, **fields: Any) -> bool:
    allowed = {"email_to", "timezone", "generate_time", "paused", "name"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return False
    sets = ", ".join(f"{k} = %({k})s" for k in fields)
    with connect() as conn:
        return conn.execute(f"UPDATE users SET {sets} WHERE id = %(id)s", {**fields, "id": user_id}).rowcount == 1


def active_users() -> list[dict]:
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE NOT paused ORDER BY id").fetchall()


# --- per-user platforms --------------------------------------------------------


def load_platforms(user_id: int) -> dict[str, Platform]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT name, config FROM user_platforms WHERE user_id = %s ORDER BY name", (user_id,)
        ).fetchall()
    return {r["name"]: Platform(name=r["name"], **r["config"]) for r in rows}


def enabled_platforms(user_id: int) -> dict[str, Platform]:
    return {k: v for k, v in load_platforms(user_id).items() if v.enabled}


def update_platform(user_id: int, name: str, **changes: Any) -> Platform:
    """Change fields of a user's platform; validated against the Platform model."""
    current = load_platforms(user_id).get(name)
    if current is None:
        raise ValueError(f"unknown platform '{name}'")
    updated = Platform.model_validate({**current.model_dump(), **changes})
    with connect() as conn:
        conn.execute(
            "UPDATE user_platforms SET config = %s WHERE user_id = %s AND name = %s",
            (Jsonb(updated.model_dump(exclude={"name"})), user_id, name),
        )
    return updated


# --- ideas -------------------------------------------------------------------


def add_ideas(user_id: int, platform: str, ideas: list[dict], source: str = "ai") -> list[int]:
    status = "queued" if source == "user" else "suggested"
    priority = 10 if source == "user" else 0
    with connect() as conn:
        return [
            conn.execute(
                "INSERT INTO ideas (user_id, platform, title, angle, source, status, priority) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (user_id, platform, i["title"], i.get("angle", ""), source, status, priority),
            ).fetchone()["id"]
            for i in ideas
        ]


def open_ideas(user_id: int, platform: str | None = None) -> list[dict]:
    """Backlog in pick order: user-queued first, then by priority and age."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM ideas WHERE user_id = %(u)s AND status IN ('queued', 'suggested') "
            "AND (%(p)s::text IS NULL OR platform = %(p)s) "
            "ORDER BY platform, status = 'queued' DESC, priority DESC, created_at",
            {"u": user_id, "p": platform},
        ).fetchall()


def next_idea(user_id: int, platform: str) -> dict | None:
    ideas = open_ideas(user_id, platform)
    return ideas[0] if ideas else None


def get_idea(user_id: int, idea_id: int) -> dict | None:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM ideas WHERE id = %s AND user_id = %s", (idea_id, user_id)
        ).fetchone()


def update_idea(user_id: int, idea_id: int, **fields: Any) -> bool:
    allowed = {"title", "angle", "status", "priority"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return False
    sets = ", ".join(f"{k} = %({k})s" for k in fields)
    used = ", used_at = now()" if fields.get("status") == "used" else ""
    with connect() as conn:
        cur = conn.execute(
            f"UPDATE ideas SET {sets}{used} WHERE id = %(id)s AND user_id = %(u)s",
            {**fields, "id": idea_id, "u": user_id},
        )
        return cur.rowcount == 1


def recent_titles(user_id: int, platform: str, limit: int = 30) -> list[str]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT title FROM ideas WHERE user_id = %s AND platform = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (user_id, platform, limit),
        ).fetchall()
    return [r["title"] for r in rows]


# --- steering ----------------------------------------------------------------


def add_steering(user_id: int, platform: str, directive: str) -> int:
    with connect() as conn:
        return conn.execute(
            "INSERT INTO steering (user_id, platform, directive) VALUES (%s, %s, %s) RETURNING id",
            (user_id, platform, directive),
        ).fetchone()["id"]


def active_steering(user_id: int, platform: str | None = None) -> list[dict]:
    """Directives for one platform (plus 'all' ones), or every active one."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM steering WHERE user_id = %(u)s AND active "
            "AND (%(p)s::text IS NULL OR platform IN (%(p)s, 'all')) ORDER BY created_at",
            {"u": user_id, "p": platform},
        ).fetchall()


def deactivate_steering(user_id: int, steering_id: int) -> bool:
    with connect() as conn:
        return conn.execute(
            "UPDATE steering SET active = false WHERE id = %s AND user_id = %s", (steering_id, user_id)
        ).rowcount == 1


# --- drafts ------------------------------------------------------------------


def save_draft(
    *,
    user_id: int,
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
            "INSERT INTO drafts (user_id, run_date, platform, idea_id, parent_id, title, body, "
            "hashtags, image, critic_score, critic_notes, feedback) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (
                user_id,
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
                "UPDATE ideas SET status = 'used', used_at = now() WHERE id = %s AND user_id = %s",
                (idea_id, user_id),
            )
        if parent_id:
            conn.execute(
                "UPDATE drafts SET status = 'superseded', updated_at = now() "
                "WHERE id = %s AND user_id = %s",
                (parent_id, user_id),
            )
    return draft_id


def get_draft(user_id: int, draft_id: int) -> dict | None:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM drafts WHERE id = %s AND user_id = %s", (draft_id, user_id)
        ).fetchone()


def update_draft(user_id: int, draft_id: int, **fields: Any) -> bool:
    allowed = {"body", "title", "status", "reject_reason"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return False
    sets = ", ".join(f"{k} = %({k})s" for k in fields)
    posted = ", posted_at = now()" if fields.get("status") == "posted" else ""
    with connect() as conn:
        return conn.execute(
            f"UPDATE drafts SET {sets}{posted}, updated_at = now() WHERE id = %(id)s AND user_id = %(u)s",
            {**fields, "id": draft_id, "u": user_id},
        ).rowcount == 1


def pending_drafts(user_id: int, run_date: date | None = None) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM drafts WHERE user_id = %(u)s AND status = 'pending' "
            "AND (%(d)s::date IS NULL OR run_date = %(d)s) ORDER BY run_date, platform, id",
            {"u": user_id, "d": run_date},
        ).fetchall()


def drafts_to_notify() -> list[dict]:
    """Pending drafts not yet sent to Telegram, across all users, with their chat id."""
    with connect() as conn:
        return conn.execute(
            "SELECT d.*, u.telegram_chat_id FROM drafts d JOIN users u ON u.id = d.user_id "
            "WHERE d.status = 'pending' AND d.telegram_notified_at IS NULL ORDER BY d.id"
        ).fetchall()


def drafts_to_email(user_id: int) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM drafts WHERE user_id = %s AND status = 'pending' AND emailed_at IS NULL "
            "ORDER BY run_date, platform, id",
            (user_id,),
        ).fetchall()


def mark_notified(draft_id: int, messages: list[tuple[int, int]]) -> None:
    """Record delivery and remember which Telegram messages show this draft."""
    with connect() as conn:
        conn.execute("UPDATE drafts SET telegram_notified_at = now() WHERE id = %s", (draft_id,))
        for chat_id, message_id in messages:
            conn.execute(
                "INSERT INTO telegram_messages (chat_id, message_id, draft_id) "
                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (chat_id, message_id, draft_id),
            )


def remember_message(chat_id: int, message_id: int, draft_id: int, purpose: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO telegram_messages (chat_id, message_id, draft_id, purpose) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (chat_id, message_id, draft_id, purpose),
        )


def draft_for_message(chat_id: int, message_id: int) -> dict | None:
    """Which draft a Telegram message belongs to, and why it was sent ({draft_id, purpose})."""
    with connect() as conn:
        return conn.execute(
            "SELECT draft_id, purpose FROM telegram_messages WHERE chat_id = %s AND message_id = %s",
            (chat_id, message_id),
        ).fetchone()


def mark_emailed(draft_ids: list[int]) -> None:
    with connect() as conn:
        conn.execute("UPDATE drafts SET emailed_at = now() WHERE id = ANY(%s)", (draft_ids,))


# --- outbox (worker -> user messages, delivered by the bot) -------------------


def notify(user_id: int, text: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO outbox (user_id, text) VALUES (%s, %s)", (user_id, text))


def unsent_messages() -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT o.*, u.telegram_chat_id FROM outbox o JOIN users u ON u.id = o.user_id "
            "WHERE o.sent_at IS NULL ORDER BY o.id"
        ).fetchall()


def mark_message_sent(message_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE outbox SET sent_at = now() WHERE id = %s", (message_id,))

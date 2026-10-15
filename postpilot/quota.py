"""Shared daily limits for every external resource, enforced transactionally.

Two layers keep usage inside the free tiers no matter how many users there are:

1. Reservation: a job reserves its worst-case usage when it is queued. The check and
   the reservation happen in one transaction under a lock, so concurrent requests
   can't both squeeze into the last bit of capacity. If it doesn't fit, it's refused.
2. Metering: while a job runs, every LLM / embedding / email call is charged against
   the job's reservation first. If a job needs more than it reserved, it may borrow
   only what is still free today; otherwise the call is refused and the job stops.

Days are UTC days, matching how most providers reset their daily quotas.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime, timedelta

from psycopg.types.json import Jsonb

from postpilot.config import get_settings
from postpilot.db import connect

log = logging.getLogger(__name__)

QUOTA_LOCK = 7311
_current_job: ContextVar[int | None] = ContextVar("current_job", default=None)
_throttle_lock = threading.Lock()
_last_llm_call = 0.0


class CapacityError(Exception):
    """A request can't be accepted or continued within today's limits. Message is user-facing."""


class AlreadyQueued(CapacityError):
    """The user already has the same kind of job waiting."""


def quota_day() -> date:
    return datetime.now(UTC).date()


def resets_in() -> str:
    now = datetime.now(UTC)
    left = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), UTC) - now
    hours, rem = divmod(int(left.total_seconds()), 3600)
    return f"{hours}h {rem // 60}m"


def _full(resource: str) -> CapacityError:
    return CapacityError(
        f"Today's free capacity is used up ({resource.replace('_', ' ')}). "
        f"Please try again in {resets_in()}, after the daily reset (00:00 UTC)."
    )


def reserve(conn, needs: dict[str, int]) -> date:
    """Reserve `needs` for today inside the caller's transaction, or raise CapacityError.

    The caller must hold the QUOTA_LOCK transaction lock.
    """
    day = quota_day()
    limits = get_settings().quota_limits
    rows = {
        r["resource"]: r
        for r in conn.execute(
            "SELECT resource, used, reserved FROM usage_daily WHERE day = %s FOR UPDATE", (day,)
        ).fetchall()
    }
    for resource, need in needs.items():
        limit = limits.get(resource, 0)
        row = rows.get(resource) or {"used": 0, "reserved": 0}
        if limit and row["used"] + row["reserved"] + need > limit:
            raise _full(resource)
    for resource, need in needs.items():
        conn.execute(
            "INSERT INTO usage_daily (day, resource, reserved) VALUES (%s, %s, %s) "
            "ON CONFLICT (day, resource) DO UPDATE SET reserved = usage_daily.reserved + EXCLUDED.reserved",
            (day, resource, need),
        )
    return day


@contextmanager
def job_context(job_id: int) -> Iterator[None]:
    token = _current_job.set(job_id)
    try:
        yield
    finally:
        _current_job.reset(token)


def charge(resource: str, amount: int = 1, *, after_the_fact: bool = False) -> None:
    """Meter usage for the running job.

    Normal calls are checked *before* the call is made and raise if there's no room.
    `after_the_fact` records usage that already happened (e.g. actual token counts):
    it is always recorded, and raises afterwards if it pushed past the limit.
    Outside a queued job (tests, local dev) nothing is metered.
    """
    job_id = _current_job.get()
    if job_id is None or amount <= 0:
        return
    over = False
    with connect() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (QUOTA_LOCK,))
        job = conn.execute(
            "SELECT reserved, used, quota_day FROM jobs WHERE id = %s FOR UPDATE", (job_id,)
        ).fetchone()
        reserved, used = dict(job["reserved"]), dict(job["used"])
        mine_left = reserved.get(resource, 0) - used.get(resource, 0)
        extra = max(0, amount - max(0, mine_left))  # beyond this job's own reservation
        if extra:
            limit = get_settings().quota_limits.get(resource, 0)
            row = conn.execute(
                "SELECT used, reserved FROM usage_daily WHERE day = %s AND resource = %s FOR UPDATE",
                (job["quota_day"], resource),
            ).fetchone() or {"used": 0, "reserved": 0}
            if limit and row["used"] + row["reserved"] + extra > limit:
                if not after_the_fact:
                    raise _full(resource)
                over = True
            reserved[resource] = reserved.get(resource, 0) + extra
            conn.execute(
                "INSERT INTO usage_daily (day, resource, reserved) VALUES (%s, %s, %s) "
                "ON CONFLICT (day, resource) DO UPDATE SET reserved = usage_daily.reserved + EXCLUDED.reserved",
                (job["quota_day"], resource, extra),
            )
        used[resource] = used.get(resource, 0) + amount
        conn.execute(
            "UPDATE usage_daily SET used = used + %s, reserved = GREATEST(reserved - %s, 0) "
            "WHERE day = %s AND resource = %s",
            (amount, amount, job["quota_day"], resource),
        )
        conn.execute(
            "UPDATE jobs SET reserved = %s, used = %s WHERE id = %s", (Jsonb(reserved), Jsonb(used), job_id)
        )
    if over:
        raise _full(resource)


def ensure_room(resource: str, amount: int) -> None:
    """Raise if `amount` more of `resource` couldn't be charged right now (no recording)."""
    job_id = _current_job.get()
    if job_id is None:
        return
    with connect() as conn:
        job = conn.execute("SELECT reserved, used, quota_day FROM jobs WHERE id = %s", (job_id,)).fetchone()
        mine_left = job["reserved"].get(resource, 0) - job["used"].get(resource, 0)
        extra = amount - max(0, mine_left)
        limit = get_settings().quota_limits.get(resource, 0)
        if extra <= 0 or not limit:
            return
        row = conn.execute(
            "SELECT used, reserved FROM usage_daily WHERE day = %s AND resource = %s",
            (job["quota_day"], resource),
        ).fetchone() or {"used": 0, "reserved": 0}
        if row["used"] + row["reserved"] + extra > limit:
            raise _full(resource)


def throttle_llm() -> None:
    """Space LLM calls for providers with per-minute limits (single worker, so in-process is enough)."""
    global _last_llm_call
    rpm = get_settings().llm_requests_per_minute
    if not rpm:
        return
    with _throttle_lock:
        wait = _last_llm_call + 60.0 / rpm - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_llm_call = time.monotonic()


def release(conn, job_id: int) -> None:
    """Give back whatever a finished job reserved but didn't use (caller's transaction)."""
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (QUOTA_LOCK,))
    job = conn.execute("SELECT reserved, used, quota_day FROM jobs WHERE id = %s FOR UPDATE", (job_id,)).fetchone()
    for resource, amount in job["reserved"].items():
        unused = amount - job["used"].get(resource, 0)
        if unused > 0:
            conn.execute(
                "UPDATE usage_daily SET reserved = GREATEST(reserved - %s, 0) WHERE day = %s AND resource = %s",
                (unused, job["quota_day"], resource),
            )
    conn.execute("UPDATE jobs SET reserved = used WHERE id = %s", (job_id,))


def usage_today() -> dict[str, dict]:
    limits = get_settings().quota_limits
    with connect() as conn:
        rows = {
            r["resource"]: r
            for r in conn.execute("SELECT * FROM usage_daily WHERE day = %s", (quota_day(),)).fetchall()
        }
    return {
        res: {"used": rows.get(res, {}).get("used", 0), "reserved": rows.get(res, {}).get("reserved", 0), "limit": limit}
        for res, limit in limits.items()
    }

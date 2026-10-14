"""Cron-style scheduler: daily generation and the email digest."""

from __future__ import annotations

import logging
import time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from postpilot import rag
from postpilot.config import get_settings
from postpilot.db import init_db
from postpilot.graph import pipeline
from postpilot.notify.email import send_digest

log = logging.getLogger(__name__)


def daily_job() -> None:
    rag.sync_knowledge_dir()
    for attempt in range(1, 4):
        try:
            drafts = pipeline.run_daily()
            log.info("daily run produced %d drafts", len(drafts))
            return
        except Exception:
            # The run is checkpointed, so the next attempt resumes only the failed platforms.
            log.exception("daily run failed (attempt %d/3)", attempt)
            time.sleep(120 * attempt)


def main() -> None:
    s = get_settings()
    init_db()
    sched = BlockingScheduler(timezone=s.tz)
    common = {"max_instances": 1, "coalesce": True, "misfire_grace_time": 3600}
    sched.add_job(daily_job, CronTrigger.from_crontab(s.generate_cron, timezone=s.tz), id="generate", **common)
    if s.email_enabled:
        sched.add_job(send_digest, CronTrigger.from_crontab(s.email_cron, timezone=s.tz), id="digest", **common)
    else:
        log.info("SMTP not configured; email digest disabled")
    log.info("scheduled: generate '%s', email '%s' (%s)", s.generate_cron, s.email_cron, s.timezone)
    if s.run_on_startup:
        daily_job()
    sched.start()

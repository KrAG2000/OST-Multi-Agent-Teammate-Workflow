"""The worker: runs queued jobs one at a time and queues each user's daily run."""

from __future__ import annotations

import logging
import time
from datetime import UTC

import psycopg
from apscheduler.schedulers.background import BackgroundScheduler

from postpilot import jobs
from postpilot.config import get_settings
from postpilot.db import init_db

log = logging.getLogger(__name__)

WORKER_LOCK = 7312


def main() -> None:
    init_db()
    # Exactly one worker may run, even if someone starts a second container by mistake.
    lock_conn = psycopg.connect(get_settings().database_url, autocommit=True)
    if not lock_conn.execute("SELECT pg_try_advisory_lock(%s)", (WORKER_LOCK,)).fetchone()[0]:
        raise SystemExit("another worker is already running")

    if n := jobs.recover_running():
        log.info("re-queued %d job(s) interrupted by a restart", n)

    sched = BackgroundScheduler(timezone=UTC)
    sched.add_job(jobs.enqueue_due_daily, "interval", minutes=1, max_instances=1, coalesce=True)
    sched.start()
    jobs.enqueue_due_daily()
    log.info("worker started; processing jobs one at a time")

    while True:
        job = jobs.claim_next()
        if job is None:
            time.sleep(3)
            continue
        jobs.run_job(job)

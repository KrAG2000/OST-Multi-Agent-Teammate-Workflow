"""CLI: `python -m postpilot <command>` (or `postpilot <command>` when installed).

Work-producing commands go through the same queue and limits as Telegram, acting as
the first admin user.
"""

from __future__ import annotations

import argparse
import json
import logging


def _admin() -> dict:
    from postpilot.db import connect

    with connect() as conn:
        user = conn.execute("SELECT * FROM users WHERE is_admin ORDER BY id LIMIT 1").fetchone()
    if user is None:
        raise SystemExit("no admin user: set TELEGRAM_ADMIN_CHAT_IDS and run `postpilot init-db`")
    return user


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="postpilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db", help="create/upgrade tables and admin users")
    sub.add_parser("worker", help="run the queue worker and daily scheduler")
    sub.add_parser("bot", help="run the Telegram bot")
    p = sub.add_parser("run", help="queue a draft run for the admin")
    p.add_argument("platforms", nargs="*")
    sub.add_parser("ingest", help="queue indexing of the knowledge/ folder for the admin")
    sub.add_parser("usage", help="show today's shared usage against the limits")
    sub.add_parser("drafts", help="print the admin's pending drafts")
    args = parser.parse_args()

    from postpilot import jobs, quota, render, store
    from postpilot.db import init_db

    match args.cmd:
        case "init-db":
            init_db()
        case "worker":
            from postpilot import worker

            worker.main()
        case "bot":
            from postpilot.notify import telegram_bot

            init_db()
            telegram_bot.main()
        case "run":
            job_id, position = jobs.enqueue(_admin(), "generate", {"platforms": args.platforms})
            print(f"queued job {job_id} (position {position})")
        case "ingest":
            job_id, position = jobs.enqueue(_admin(), "sync_knowledge", {})
            print(f"queued job {job_id} (position {position})")
        case "usage":
            print(json.dumps(quota.usage_today(), indent=2))
        case "drafts":
            for d in store.pending_drafts(_admin()["id"]):
                print(render.full_text(d), end="\n\n" + "=" * 70 + "\n\n")


if __name__ == "__main__":
    main()

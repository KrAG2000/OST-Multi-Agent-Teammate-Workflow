"""CLI: `python -m postpilot <command>` (or `postpilot <command>` when installed)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="postpilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db", help="create tables and checkpointer schema")
    p = sub.add_parser("ingest", help="index .md/.txt files into the knowledge base")
    p.add_argument("path", nargs="?", type=Path)
    p = sub.add_parser("run", help="generate drafts now")
    p.add_argument("platforms", nargs="*")
    p.add_argument("--daily", action="store_true", help="use (or resume) today's scheduled run")
    p.add_argument("--force", action="store_true", help="with --daily: run again even if done")
    p = sub.add_parser("regen", help="rewrite a draft with feedback")
    p.add_argument("draft_id", type=int)
    p.add_argument("feedback", nargs="*")
    p = sub.add_parser("ideas", help="show the idea backlog")
    p.add_argument("platform", nargs="?")
    p = sub.add_parser("refill", help="generate idea suggestions")
    p.add_argument("platform")
    sub.add_parser("drafts", help="print today's pending drafts")
    sub.add_parser("digest", help="send the email digest now")
    sub.add_parser("scheduler", help="run the cron scheduler (daily generation + email)")
    sub.add_parser("bot", help="run the Telegram review bot")
    args = parser.parse_args()

    from postpilot import rag, render, store
    from postpilot.config import get_settings
    from postpilot.db import init_db

    match args.cmd:
        case "init-db":
            init_db()
        case "ingest":
            print(f"indexed {rag.sync_knowledge_dir(args.path)} new chunks")
        case "run":
            from postpilot.graph import pipeline

            drafts = pipeline.run_daily(args.force) if args.daily else pipeline.run_now(args.platforms or None)
            print(json.dumps(drafts, indent=2))
        case "regen":
            from postpilot.graph import pipeline

            print(json.dumps(pipeline.regenerate(args.draft_id, " ".join(args.feedback)), indent=2))
        case "ideas":
            for i in store.open_ideas(args.platform):
                print(f"{i['id']:>4} {i['status']:<9} [{i['platform']}] {i['title']}  -- {i['angle']}")
        case "refill":
            from postpilot.graph import pipeline

            print(pipeline.refill_ideas(args.platform))
        case "drafts":
            for d in store.pending_drafts(get_settings().today()):
                print(render.full_text(d), end="\n\n" + "=" * 70 + "\n\n")
        case "digest":
            from postpilot.notify.email import send_digest

            print(f"emailed {send_digest()} drafts")
        case "scheduler":
            from postpilot import scheduler

            scheduler.main()
        case "bot":
            from postpilot.notify import telegram_bot

            init_db()
            telegram_bot.main()


if __name__ == "__main__":
    main()

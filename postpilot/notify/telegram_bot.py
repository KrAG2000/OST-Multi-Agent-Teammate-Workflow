"""Telegram bot: sign-up, review drafts, steer ideas, manage settings.

Uses long polling, so it needs no public URL or webhook and runs anywhere. Anyone can
sign up with /start (or /start <code> when INVITE_CODE is set). The bot never calls the
LLM itself: anything heavy goes into the shared queue, which enforces the daily limits.
"""

from __future__ import annotations

import asyncio
import functools
import io
import logging
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from telegram import BotCommand, ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import Forbidden
from telegram.ext import (
    AIORateLimiter,
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from postpilot import jobs, quota, render, store
from postpilot.config import get_settings
from postpilot.graph.pipeline import user_today
from postpilot.quota import CapacityError

log = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4000
MAX_NOTE_CHARS = 50_000
_push_lock = asyncio.Lock()  # the 30s job and command handlers both push; never send twice

HELP = """\
Every day at your chosen time, I draft one post per platform for you to review.
✅ Approve gives copy-ready text; tap 📤 Mark as posted once it's live.
❌ Reject asks why; your answer steers future posts.
🔁 Regenerate rewrites it. Better: reply to a draft with what to change.

Drafts
/run [platform]: generate now
/today: show today's pending drafts again
/regen <id> <feedback>, /edit <id> <text>, /posted <id>

Ideas
/ideas [platform]: upcoming ideas
/idea <platform> <title> | <angle>: add your own (used first)
/queue <id>, /drop <id>, /refill <platform>

Steering
/steer <platform|all> <instruction>, /steering, /unsteer <id>

Knowledge
/note <text>, or send a .md/.txt file

Platforms
/platforms, /enable <p>, /disable <p>
/set <platform> purpose|tone|audience <text>

Settings
/info (or /me): your settings
/email <address> (or /email off)
/time 06:00, /timezone Asia/Kolkata
/pause, /resume: stop or restart daily drafts
/usage: today's shared capacity

Everyone shares one free server, so requests are queued and there are daily limits."""


def user_handler(handler):
    """Resolve the signed-up user for this chat and report errors back to them."""

    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        user = await asyncio.to_thread(store.user_by_chat, chat.id) if chat else None
        if user is None:
            if update.effective_message:
                await update.effective_message.reply_text("Send /start to sign up first.")
            return
        try:
            return await handler(update, context, user)
        except CapacityError as exc:
            await update.effective_message.reply_text(f"⏸ {exc}")
        except Exception as exc:  # report errors to the user instead of failing silently
            log.exception("handler failed")
            await update.effective_message.reply_text(f"⚠️ {type(exc).__name__}: {exc}")

    return wrapper


async def _enqueue(update: Update, user: dict, kind: str, payload: dict, what: str) -> None:
    _, position = await asyncio.to_thread(jobs.enqueue, user, kind, payload)
    ahead = position - 1
    when = "starting now" if ahead == 0 else f"{ahead} request{'s' if ahead != 1 else ''} ahead of you"
    await update.effective_message.reply_text(f"⏳ Queued {what} ({when}). I'll message you when it's done.")


def _args(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
    return context.args or []


def _platform(user: dict, name: str) -> str:
    platforms = store.load_platforms(user["id"])
    if name not in platforms:
        raise ValueError(f"unknown platform '{name}'. Yours: {', '.join(platforms)}")
    return name


def _int_arg(context, usage: str) -> int:
    try:
        return int(_args(context)[0])
    except (IndexError, ValueError):
        raise ValueError(f"usage: {usage}") from None


# --- delivery ------------------------------------------------------------------


def _keyboard(draft_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ Approve", callback_data=f"approve:{draft_id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"reject:{draft_id}"),
            InlineKeyboardButton("🔁 Regenerate", callback_data=f"regen:{draft_id}"),
        ]]
    )


def _posted_keyboard(draft_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("📤 Mark as posted", callback_data=f"posted:{draft_id}")]])


async def send_draft(bot, chat_id: int, d: dict) -> list[int]:
    """Send a draft; long ones (articles) go as a .md file. Returns message ids."""
    text = render.full_text(d)
    if len(text) <= TELEGRAM_LIMIT:
        msg = await bot.send_message(chat_id, text, reply_markup=_keyboard(d["id"]))
        return [msg.message_id]
    summary = "\n\n".join(
        x for x in [render.header(d), d.get("title") or "", render.image_text(d), "Full text attached ⬇️"] if x
    )[:TELEGRAM_LIMIT]
    doc = await bot.send_document(
        chat_id,
        document=io.BytesIO(render.post_text(d).encode()),
        filename=f"draft-{d['id']}-{d['platform']}.md",
    )
    msg = await bot.send_message(chat_id, summary, reply_markup=_keyboard(d["id"]))
    return [doc.message_id, msg.message_id]


async def send_plain(bot, chat_id: int, text: str, filename: str) -> None:
    if len(text) <= TELEGRAM_LIMIT:
        await bot.send_message(chat_id, text)
    else:
        await bot.send_document(chat_id, document=io.BytesIO(text.encode()), filename=filename)


async def push_updates(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Job: deliver new drafts and worker messages to their users."""
    async with _push_lock:
        for d in await asyncio.to_thread(store.drafts_to_notify):
            chat_id = d["telegram_chat_id"]
            try:
                ids = await send_draft(context.bot, chat_id, d)
            except Forbidden:  # user blocked the bot: stop their daily runs
                await asyncio.to_thread(store.update_user, d["user_id"], paused=True)
                ids = []
            await asyncio.to_thread(store.mark_notified, d["id"], [(chat_id, i) for i in ids])
        for m in await asyncio.to_thread(store.unsent_messages):
            try:
                await context.bot.send_message(m["telegram_chat_id"], m["text"])
            except Forbidden:
                await asyncio.to_thread(store.update_user, m["user_id"], paused=True)
            await asyncio.to_thread(store.mark_message_sent, m["id"])


# --- sign-up and settings ---------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    chat = update.effective_chat
    msg = update.effective_message
    existing = await asyncio.to_thread(store.user_by_chat, chat.id)
    if existing:
        await msg.reply_text(HELP)
        return
    is_admin = chat.id in s.admin_chat_ids
    if not is_admin:
        if s.invite_code and _args(context)[:1] != [s.invite_code]:
            await msg.reply_text("This bot is invite-only. Send /start <invite code>.")
            return
        if s.max_users and await asyncio.to_thread(store.count_users) >= s.max_users:
            await msg.reply_text("Sorry, this bot is full right now. Please try again later.")
            return
    name = (update.effective_user.full_name if update.effective_user else "") or ""
    user = await asyncio.to_thread(store.ensure_user, chat.id, name, is_admin)
    platforms = ", ".join(store.enabled_platforms(user["id"]))
    await msg.reply_text(
        f"👋 Welcome! You're signed up.\n\n"
        f"Daily drafts at {user['generate_time']} ({user['timezone']}) for: {platforms}.\n"
        "Set your own with /time and /timezone, add your email with /email, and tell me about "
        "yourself with /note so posts sound like you.\n\n" + HELP
    )


@user_handler
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    await update.effective_message.reply_text(HELP)


@user_handler
async def cmd_me(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    await update.effective_message.reply_text(
        f"Daily drafts: {'paused' if user['paused'] else 'on'} at {user['generate_time']} ({user['timezone']})\n"
        f"Email: {user['email_to'] or 'not set (/email you@example.com)'}\n"
        f"Platforms: {', '.join(store.enabled_platforms(user['id'])) or 'none enabled'}"
        + ("\nRole: admin" if user["is_admin"] else "")
    )


@user_handler
async def cmd_email(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    args = _args(context)
    if not args:
        raise ValueError("usage: /email you@example.com (comma-separate several), or /email off")
    if args[0].lower() == "off":
        await asyncio.to_thread(store.update_user, user["id"], email_to=None)
        await update.effective_message.reply_text("📭 Email turned off.")
        return
    addresses = [a.strip() for a in "".join(args).split(",") if a.strip()]
    bad = [a for a in addresses if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", a)]
    if bad:
        raise ValueError(f"not an email address: {', '.join(bad)}")
    await asyncio.to_thread(store.update_user, user["id"], email_to=", ".join(addresses))
    extra = "" if get_settings().smtp_configured else " (note: email sending isn't set up on this server yet)"
    await update.effective_message.reply_text(f"📬 Daily drafts will also go to {', '.join(addresses)}{extra}.")


@user_handler
async def cmd_time(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    value = (_args(context) or [""])[0]
    m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", value)
    if not m:
        raise ValueError("usage: /time HH:MM (24h), e.g. /time 06:30")
    hhmm = f"{int(m[1]):02d}:{m[2]}"
    await asyncio.to_thread(store.update_user, user["id"], generate_time=hhmm)
    await update.effective_message.reply_text(f"⏰ Daily drafts at {hhmm} ({user['timezone']}).")


@user_handler
async def cmd_timezone(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    value = (_args(context) or [""])[0]
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("usage: /timezone Area/City, e.g. /timezone Asia/Kolkata or Europe/London") from None
    await asyncio.to_thread(store.update_user, user["id"], timezone=value)
    await update.effective_message.reply_text(f"🌍 Timezone set to {value}.")


@user_handler
async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    await asyncio.to_thread(store.update_user, user["id"], paused=True)
    await update.effective_message.reply_text("⏸ Daily drafts paused. /resume to restart.")


@user_handler
async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    await asyncio.to_thread(store.update_user, user["id"], paused=False)
    await update.effective_message.reply_text("▶️ Daily drafts are back on.")


@user_handler
async def cmd_usage(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    usage = await asyncio.to_thread(quota.usage_today)
    lines = [f"Shared capacity today (resets in {quota.resets_in()}):"]
    for res, u in usage.items():
        limit = u["limit"] or "∞"
        lines.append(f"• {res.replace('_', ' ')}: {u['used']} used, {u['reserved']} reserved / {limit}")
    lines.append(f"Queue: {await asyncio.to_thread(jobs.queue_length)} request(s) waiting or running")
    if get_settings().quota_jobs_per_user_per_day:
        lines.append(f"Your limit: {get_settings().quota_jobs_per_user_per_day} requests per day")
    await update.effective_message.reply_text("\n".join(lines))


# --- platforms -------------------------------------------------------------------


@user_handler
async def cmd_platforms(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    platforms = await asyncio.to_thread(store.load_platforms, user["id"])
    lines = [f"{'🟢' if p.enabled else '⚪'} {k}: {p.purpose.strip()[:120]}" for k, p in platforms.items()]
    lines.append("\n/enable <p>, /disable <p>, /set <p> purpose|tone|audience <text>")
    await update.effective_message.reply_text("\n".join(lines))


async def _toggle(update: Update, context, user: dict, enabled: bool):
    name = _platform(user, (_args(context) or [""])[0])
    await asyncio.to_thread(store.update_platform, user["id"], name, enabled=enabled)
    await update.effective_message.reply_text(f"{'🟢 Enabled' if enabled else '⚪ Disabled'} {name}.")


@user_handler
async def cmd_enable(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    await _toggle(update, context, user, True)


@user_handler
async def cmd_disable(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    await _toggle(update, context, user, False)


@user_handler
async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    parts = update.effective_message.text.split(maxsplit=3)
    if len(parts) < 4 or parts[2] not in ("purpose", "tone", "audience"):
        raise ValueError("usage: /set <platform> purpose|tone|audience <text>")
    name = _platform(user, parts[1])
    await asyncio.to_thread(store.update_platform, user["id"], name, **{parts[2]: parts[3]})
    await update.effective_message.reply_text(f"✏️ {name} {parts[2]} updated.")


# --- drafts ----------------------------------------------------------------------


@user_handler
async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    drafts = await asyncio.to_thread(store.pending_drafts, user["id"], user_today(user))
    if not drafts:
        await update.effective_message.reply_text("No pending drafts for today.")
    for d in drafts:
        ids = await send_draft(context.bot, update.effective_chat.id, d)
        await asyncio.to_thread(store.mark_notified, d["id"], [(update.effective_chat.id, i) for i in ids])


@user_handler
async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    platforms = [_platform(user, a) for a in _args(context)]
    if not platforms and not store.enabled_platforms(user["id"]):
        raise ValueError("no platforms enabled. /platforms to see them, /enable <p> to turn one on")
    await _enqueue(update, user, "generate", {"platforms": platforms}, f"drafts for {', '.join(platforms) or 'all your platforms'}")


@user_handler
async def cmd_regen(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    draft_id = _int_arg(context, "/regen <draft_id> <feedback>")
    if not await asyncio.to_thread(store.get_draft, user["id"], draft_id):
        raise ValueError(f"draft #{draft_id} not found")
    feedback = " ".join(_args(context)[1:])
    await _enqueue(update, user, "regen", {"draft_id": draft_id, "feedback": feedback}, f"a rewrite of #{draft_id}")


@user_handler
async def cmd_edit(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    text = update.effective_message.text.split(maxsplit=2)
    if len(text) < 3 or not text[1].isdigit():
        raise ValueError("usage: /edit <draft_id> <new text>")
    ok = await asyncio.to_thread(store.update_draft, user["id"], int(text[1]), body=text[2])
    await update.effective_message.reply_text("✏️ Updated." if ok else "Draft not found.")


async def _mark_posted(update: Update, user: dict, draft_id: int) -> str:
    await asyncio.to_thread(store.update_draft, user["id"], draft_id, status="posted")
    # Posted content becomes RAG context for voice and for avoiding repeat topics.
    try:
        await asyncio.to_thread(jobs.enqueue, user, "index_post", {"draft_id": draft_id})
        return f"📤 Marked #{draft_id} as posted."
    except CapacityError:
        return f"📤 Marked #{draft_id} as posted (adding it to your post history was skipped: capacity full)."


@user_handler
async def cmd_posted(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    draft_id = _int_arg(context, "/posted <draft_id>")
    d = await asyncio.to_thread(store.get_draft, user["id"], draft_id)
    if d is None or d["status"] not in ("pending", "approved"):
        await update.effective_message.reply_text(f"Draft #{draft_id} is {d['status'] if d else 'not found'}.")
        return
    await update.effective_message.reply_text(await _mark_posted(update, user, draft_id))


# --- ideas and steering ------------------------------------------------------------


@user_handler
async def cmd_ideas(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    platform = _platform(user, _args(context)[0]) if _args(context) else None
    ideas = await asyncio.to_thread(store.open_ideas, user["id"], platform)
    if not ideas:
        await update.effective_message.reply_text("Backlog is empty; it refills automatically on the next run.")
        return
    lines = [
        f"{'⭐' if i['status'] == 'queued' else '💡'} {i['id']} [{i['platform']}] {i['title']}"
        + (f"\n    {i['angle']}" if i["angle"] else "")
        for i in ideas
    ]
    await update.effective_message.reply_text("\n".join(lines)[:TELEGRAM_LIMIT])


@user_handler
async def cmd_idea(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    parts = update.effective_message.text.split(maxsplit=2)
    if len(parts) < 3:
        raise ValueError("usage: /idea <platform> <title> | <angle>")
    title, _, angle = parts[2].partition("|")
    platform = _platform(user, parts[1])
    (idea_id,) = await asyncio.to_thread(
        store.add_ideas, user["id"], platform, [{"title": title.strip(), "angle": angle.strip()}], "user"
    )
    await update.effective_message.reply_text(f"⭐ Queued idea {idea_id}; it's next for {platform}.")


@user_handler
async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    idea_id = _int_arg(context, "/queue <idea_id>")
    ok = await asyncio.to_thread(store.update_idea, user["id"], idea_id, status="queued", priority=10)
    await update.effective_message.reply_text("⭐ Moved to the front." if ok else "Idea not found.")


@user_handler
async def cmd_drop(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    idea_id = _int_arg(context, "/drop <idea_id>")
    ok = await asyncio.to_thread(store.update_idea, user["id"], idea_id, status="dropped")
    await update.effective_message.reply_text("🗑 Dropped." if ok else "Idea not found.")


@user_handler
async def cmd_refill(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    if not _args(context):
        raise ValueError("usage: /refill <platform>")
    platform = _platform(user, _args(context)[0])
    await _enqueue(update, user, "refill", {"platform": platform}, f"new ideas for {platform}")


@user_handler
async def cmd_steer(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    parts = update.effective_message.text.split(maxsplit=2)
    if len(parts) < 3:
        raise ValueError("usage: /steer <platform|all> <instruction>")
    platform = parts[1] if parts[1] == "all" else _platform(user, parts[1])
    sid = await asyncio.to_thread(store.add_steering, user["id"], platform, parts[2])
    await update.effective_message.reply_text(f"🧭 Steering {sid} saved for {platform}.")


@user_handler
async def cmd_steering(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    rows = await asyncio.to_thread(store.active_steering, user["id"])
    text = "\n".join(f"{r['id']} [{r['platform']}] {r['directive']}" for r in rows)
    await update.effective_message.reply_text(text[:TELEGRAM_LIMIT] or "No steering instructions yet.")


@user_handler
async def cmd_unsteer(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    sid = _int_arg(context, "/unsteer <id>")
    ok = await asyncio.to_thread(store.deactivate_steering, user["id"], sid)
    await update.effective_message.reply_text("Removed." if ok else "Not found.")


# --- knowledge -------------------------------------------------------------------


@user_handler
async def cmd_note(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    parts = update.effective_message.text.split(maxsplit=1)
    if len(parts) < 2:
        raise ValueError("usage: /note <text>")
    source = f"telegram:{update.effective_message.message_id}"
    await _enqueue(update, user, "ingest", {"source": source, "text": parts[1][:MAX_NOTE_CHARS]}, "your note")


@user_handler
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    doc = update.effective_message.document
    if not (doc.file_name or "").lower().endswith((".md", ".markdown", ".txt")):
        raise ValueError("only .md and .txt files can be added")
    if doc.file_size and doc.file_size > MAX_NOTE_CHARS * 4:
        raise ValueError("file is too large (max ~200 KB)")
    data = await (await doc.get_file()).download_as_bytearray()
    text = data.decode("utf-8", "replace")[:MAX_NOTE_CHARS]
    await _enqueue(update, user, "ingest", {"source": f"telegram-file:{doc.file_name}", "text": text}, doc.file_name)


# --- replies and buttons ----------------------------------------------------------


@user_handler
async def on_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    """Reply to a draft = rewrite feedback. Reply to a "why rejected?" prompt = steering."""
    msg = update.effective_message
    link = await asyncio.to_thread(store.draft_for_message, msg.chat_id, msg.reply_to_message.message_id)
    d = link and await asyncio.to_thread(store.get_draft, user["id"], link["draft_id"])
    if not d:
        await msg.reply_text("Reply to a draft message to give feedback on it. /help for commands.")
        return
    if link["purpose"] == "reject_reason":
        await asyncio.to_thread(store.update_draft, user["id"], d["id"], reject_reason=msg.text)
        directive = f"Avoid what made the owner reject the post '{d['title'] or d['body'][:60]}': {msg.text}"
        sid = await asyncio.to_thread(store.add_steering, user["id"], d["platform"], directive)
        await msg.reply_text(
            f"🧭 Noted. Future {d['platform']} ideas and drafts will take this into account "
            f"(steering {sid}; /unsteer {sid} to undo)."
        )
        return
    await _enqueue(update, user, "regen", {"draft_id": d["id"], "feedback": msg.text}, f"a rewrite of #{d['id']}")


@user_handler
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    query = update.callback_query
    await query.answer()
    action, draft_id = query.data.split(":")
    draft_id = int(draft_id)
    chat_id = query.message.chat_id
    d = await asyncio.to_thread(store.get_draft, user["id"], draft_id)
    allowed = ("pending", "approved") if action == "posted" else ("pending",)
    if d is None or d["status"] not in allowed:
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(f"Draft #{draft_id} is already {d['status'] if d else 'gone'}.")
        return

    if action == "approve":
        await asyncio.to_thread(store.update_draft, user["id"], draft_id, status="approved")
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(f"✅ Approved #{draft_id}. Copy-ready text:")
        await send_plain(context.bot, chat_id, render.post_text(d), f"post-{draft_id}.md")
        await context.bot.send_message(
            chat_id, "Tap once it's live on the platform:", reply_markup=_posted_keyboard(draft_id)
        )
    elif action == "posted":
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(await _mark_posted(update, user, draft_id))
    elif action == "reject":
        await asyncio.to_thread(store.update_draft, user["id"], draft_id, status="rejected")
        await query.edit_message_reply_markup(None)
        ask = await query.message.reply_text(
            f"❌ Rejected #{draft_id}. Why? Reply with a reason and future {d['platform']} posts "
            "will avoid it. Or just ignore this.",
            reply_markup=ForceReply(selective=True, input_field_placeholder="Reason (optional)"),
        )
        await asyncio.to_thread(store.remember_message, chat_id, ask.message_id, draft_id, "reject_reason")
    elif action == "regen":
        await _enqueue(update, user, "regen", {"draft_id": draft_id, "feedback": ""}, f"a rewrite of #{draft_id}")
        await query.edit_message_reply_markup(None)


@user_handler
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE, user: dict):
    if not user["is_admin"]:
        raise ValueError("admins only")
    users = await asyncio.to_thread(store.count_users)
    await update.effective_message.reply_text(f"Users: {users}\nQueue: {await asyncio.to_thread(jobs.queue_length)}")


# The "/" menu users see in Telegram. Published on every start, so it always matches the code
# (it replaces any list set by hand in BotFather).
MENU = [
    ("run", "Generate drafts now: /run [platform]"),
    ("today", "Show today's pending drafts"),
    ("ideas", "Upcoming ideas: /ideas [platform]"),
    ("idea", "Add your idea: /idea <platform> <title> | <angle>"),
    ("refill", "New AI ideas: /refill <platform>"),
    ("steer", "Standing instruction: /steer <platform|all> <text>"),
    ("steering", "List your steering instructions"),
    ("note", "Save a note to your knowledge base"),
    ("regen", "Rewrite a draft: /regen <id> <feedback>"),
    ("edit", "Replace a draft's text: /edit <id> <text>"),
    ("posted", "Mark a draft as posted: /posted <id>"),
    ("platforms", "Your platforms"),
    ("enable", "Turn a platform on: /enable <platform>"),
    ("disable", "Turn a platform off: /disable <platform>"),
    ("set", "Change a platform: /set <platform> purpose|tone|audience <text>"),
    ("email", "Where to email drafts: /email you@x.com or off"),
    ("time", "Daily draft time: /time 06:30"),
    ("timezone", "Your timezone: /timezone Asia/Kolkata"),
    ("pause", "Pause daily drafts"),
    ("resume", "Resume daily drafts"),
    ("usage", "Today's shared capacity and queue"),
    ("info", "Your settings"),
    ("help", "All commands explained"),
    ("start", "Sign up"),
]


async def _publish_menu(app: Application) -> None:
    await app.bot.set_my_commands([BotCommand(name, desc) for name, desc in MENU])


def main() -> None:
    s = get_settings()
    if not s.telegram_bot_token:
        log.warning("TELEGRAM_BOT_TOKEN is not set; bot disabled")
        return

    app = (
        Application.builder()
        .token(s.telegram_bot_token)
        .rate_limiter(AIORateLimiter())  # stays within Telegram's sending limits
        .concurrent_updates(True)
        .post_init(_publish_menu)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    commands = {
        "help": cmd_help, "me": cmd_me, "info": cmd_me, "email": cmd_email, "time": cmd_time, "timezone": cmd_timezone,
        "pause": cmd_pause, "resume": cmd_resume, "usage": cmd_usage,
        "platforms": cmd_platforms, "enable": cmd_enable, "disable": cmd_disable, "set": cmd_set,
        "today": cmd_today, "run": cmd_run, "regen": cmd_regen, "edit": cmd_edit, "posted": cmd_posted,
        "ideas": cmd_ideas, "idea": cmd_idea, "queue": cmd_queue, "drop": cmd_drop, "refill": cmd_refill,
        "steer": cmd_steer, "steering": cmd_steering, "unsteer": cmd_unsteer,
        "note": cmd_note, "stats": cmd_stats,
    }
    for name, handler in commands.items():
        app.add_handler(CommandHandler(name, handler))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.REPLY & filters.TEXT & ~filters.COMMAND, on_reply))
    app.job_queue.run_repeating(push_updates, interval=30, first=5)
    log.info("telegram bot polling")
    app.run_polling()

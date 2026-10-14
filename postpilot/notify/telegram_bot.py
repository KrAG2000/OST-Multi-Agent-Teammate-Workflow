"""Telegram bot: review drafts, steer ideas, feed the knowledge base.

Uses long polling, so it needs no public URL or webhook and runs anywhere.
Only chats listed in TELEGRAM_ALLOWED_CHAT_IDS can use it.
"""

from __future__ import annotations

import asyncio
import functools
import io
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from postpilot import rag, render, store
from postpilot.config import get_settings, load_platforms
from postpilot.graph import pipeline

log = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4000
_push_lock = asyncio.Lock()  # the 60s job and command handlers both push; never send twice

HELP = """\
Drafts arrive here after each daily run. Tap a button, or reply to a draft with feedback to regenerate it.

Drafts
/today: show today's pending drafts again
/run [platform]: generate now (all platforms if none given)
/regen <draft_id> <feedback>: rewrite a draft with your feedback
/edit <draft_id> <new text>: replace a draft's text yourself

Ideas (the backlog each platform pulls from)
/ideas [platform]: list upcoming ideas
/idea <platform> <title> | <angle>: add your own idea (used first)
/queue <idea_id>: promote an AI suggestion to the front
/drop <idea_id>: remove an idea
/refill <platform>: ask the AI for fresh suggestions

Steering (standing instructions for ideas and writing)
/steer <platform|all> <instruction>: e.g. /steer linkedin more system design, fewer career posts
/steering: list active instructions
/unsteer <id>: remove one

Knowledge (RAG)
/note <text>: save a note to the knowledge base
Send a .md or .txt file to index it.
/platforms: list platform ids"""


def authorized(handler):
    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        if chat is None or chat.id not in get_settings().allowed_chat_ids:
            if update.effective_message:
                await update.effective_message.reply_text(
                    f"Not authorized. Add this chat id to TELEGRAM_ALLOWED_CHAT_IDS: {chat.id if chat else '?'}"
                )
            return
        try:
            return await handler(update, context)
        except Exception as exc:  # report errors to the user instead of failing silently
            log.exception("handler failed")
            await update.effective_message.reply_text(f"⚠️ {type(exc).__name__}: {exc}")

    return wrapper


def _keyboard(draft_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ Approve", callback_data=f"approve:{draft_id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"reject:{draft_id}"),
            InlineKeyboardButton("🔁 Regenerate", callback_data=f"regen:{draft_id}"),
        ]]
    )


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


async def push_new_drafts(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Job: deliver drafts created by the scheduler (or anywhere) that weren't sent yet."""
    async with _push_lock:
        for d in await asyncio.to_thread(store.drafts_to_notify):
            sent = []
            for chat_id in get_settings().allowed_chat_ids:
                sent += [(chat_id, m) for m in await send_draft(context.bot, chat_id, d)]
            await asyncio.to_thread(store.mark_notified, d["id"], sent)


def _args(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
    return context.args or []


def _platform(name: str) -> str:
    platforms = load_platforms()
    if name not in platforms:
        raise ValueError(f"unknown platform '{name}'. Known: {', '.join(platforms)}")
    return name


async def _regenerate(update: Update, draft_id: int, feedback: str) -> None:
    msg = update.effective_message
    await msg.reply_text(f"🔁 Rewriting draft #{draft_id}…")
    new = await asyncio.to_thread(pipeline.regenerate, draft_id, feedback)
    await msg.reply_text(f"New draft #{new['id']} is ready; it'll appear here shortly.")


# --- commands ----------------------------------------------------------------


@authorized
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(HELP)


@authorized
async def cmd_platforms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = [f"{'🟢' if p.enabled else '⚪'} {k}: {p.display_name}" for k, p in load_platforms().items()]
    await update.effective_message.reply_text("\n".join(lines))


@authorized
async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    drafts = await asyncio.to_thread(store.pending_drafts, get_settings().today())
    if not drafts:
        await update.effective_message.reply_text("No pending drafts for today.")
    for d in drafts:
        ids = await send_draft(context.bot, update.effective_chat.id, d)
        await asyncio.to_thread(store.mark_notified, d["id"], [(update.effective_chat.id, i) for i in ids])


@authorized
async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    platforms = [_platform(a) for a in _args(context)]
    await update.effective_message.reply_text(
        f"⏳ Generating for {', '.join(platforms) or 'all enabled platforms'}. This can take a few minutes."
    )
    drafts = await asyncio.to_thread(pipeline.run_now, platforms or None)
    await update.effective_message.reply_text(f"Done: {len(drafts)} draft(s). Sending them now.")
    await push_new_drafts(context)


@authorized
async def cmd_regen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = _args(context)
    if not args:
        raise ValueError("usage: /regen <draft_id> <feedback>")
    await _regenerate(update, int(args[0]), " ".join(args[1:]))
    await push_new_drafts(context)


@authorized
async def cmd_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.effective_message.text.split(maxsplit=2)
    if len(text) < 3:
        raise ValueError("usage: /edit <draft_id> <new text>")
    ok = await asyncio.to_thread(store.update_draft, int(text[1]), body=text[2])
    await update.effective_message.reply_text("✏️ Updated." if ok else "Draft not found.")


@authorized
async def cmd_ideas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    platform = _platform(_args(context)[0]) if _args(context) else None
    ideas = await asyncio.to_thread(store.open_ideas, platform)
    if not ideas:
        await update.effective_message.reply_text("Backlog is empty; it refills automatically on the next run.")
        return
    lines = [
        f"{'⭐' if i['status'] == 'queued' else '💡'} {i['id']} [{i['platform']}] {i['title']}"
        + (f"\n    {i['angle']}" if i["angle"] else "")
        for i in ideas
    ]
    await update.effective_message.reply_text("\n".join(lines)[:TELEGRAM_LIMIT])


@authorized
async def cmd_idea(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = update.effective_message.text.split(maxsplit=2)
    if len(parts) < 3:
        raise ValueError("usage: /idea <platform> <title> | <angle>")
    title, _, angle = parts[2].partition("|")
    (idea_id,) = await asyncio.to_thread(
        store.add_ideas, _platform(parts[1]), [{"title": title.strip(), "angle": angle.strip()}], "user"
    )
    await update.effective_message.reply_text(f"⭐ Queued idea {idea_id}; it's next for {parts[1]}.")


@authorized
async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await asyncio.to_thread(store.update_idea, int(_args(context)[0]), status="queued", priority=10)
    await update.effective_message.reply_text("⭐ Moved to the front." if ok else "Idea not found.")


@authorized
async def cmd_drop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await asyncio.to_thread(store.update_idea, int(_args(context)[0]), status="dropped")
    await update.effective_message.reply_text("🗑 Dropped." if ok else "Idea not found.")


@authorized
async def cmd_refill(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _args(context):
        raise ValueError("usage: /refill <platform>")
    platform = _platform(_args(context)[0])
    await update.effective_message.reply_text(f"💡 Thinking up ideas for {platform}…")
    await asyncio.to_thread(pipeline.refill_ideas, platform)
    context.args = [platform]
    await cmd_ideas.__wrapped__(update, context)


@authorized
async def cmd_steer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = update.effective_message.text.split(maxsplit=2)
    if len(parts) < 3:
        raise ValueError("usage: /steer <platform|all> <instruction>")
    platform = parts[1] if parts[1] == "all" else _platform(parts[1])
    sid = await asyncio.to_thread(store.add_steering, platform, parts[2])
    await update.effective_message.reply_text(
        f"🧭 Steering {sid} saved for {platform}. Use /refill {platform if platform != 'all' else '<platform>'} "
        "to regenerate suggestions with it now."
    )


@authorized
async def cmd_steering(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = await asyncio.to_thread(store.active_steering)
    text = "\n".join(f"{r['id']} [{r['platform']}] {r['directive']}" for r in rows)
    await update.effective_message.reply_text(text or "No steering instructions yet.")


@authorized
async def cmd_unsteer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await asyncio.to_thread(store.deactivate_steering, int(_args(context)[0]))
    await update.effective_message.reply_text("Removed." if ok else "Not found.")


@authorized
async def cmd_note(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = update.effective_message.text.split(maxsplit=1)
    if len(parts) < 2:
        raise ValueError("usage: /note <text>")
    source = f"telegram:{update.effective_message.message_id}"
    n = await asyncio.to_thread(rag.add_document, "knowledge", source, parts[1])
    await update.effective_message.reply_text(f"🧠 Saved ({n} chunk{'s' if n != 1 else ''}).")


# --- other updates -----------------------------------------------------------


@authorized
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.effective_message.document
    if not doc.file_name.lower().endswith((".md", ".markdown", ".txt")):
        raise ValueError("only .md and .txt files can be indexed")
    data = await (await doc.get_file()).download_as_bytearray()
    n = await asyncio.to_thread(
        rag.add_document, "knowledge", f"telegram-file:{doc.file_name}", data.decode("utf-8", "replace")
    )
    await update.effective_message.reply_text(f"🧠 Indexed {doc.file_name} ({n} chunks).")


@authorized
async def on_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """A reply to a draft message is treated as regeneration feedback."""
    msg = update.effective_message
    draft_id = await asyncio.to_thread(
        store.draft_for_message, msg.chat_id, msg.reply_to_message.message_id
    )
    if draft_id is None:
        await msg.reply_text("Reply to a draft message to give feedback on it. /help for commands.")
        return
    await _regenerate(update, draft_id, msg.text)
    await push_new_drafts(context)


@authorized
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, draft_id = query.data.split(":")
    draft_id = int(draft_id)
    d = await asyncio.to_thread(store.get_draft, draft_id)
    if d is None or d["status"] != "pending":
        await query.message.reply_text(f"Draft #{draft_id} is already {d['status'] if d else 'gone'}.")
        return

    if action == "approve":
        await asyncio.to_thread(store.update_draft, draft_id, status="approved")
        # Approved posts become RAG context for voice and for avoiding repeats.
        await asyncio.to_thread(
            rag.add_document, "post", f"draft:{draft_id}", render.post_text(d), d["platform"]
        )
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(f"✅ Approved #{draft_id}. Copy-ready text:")
        await send_plain(context.bot, query.message.chat_id, render.post_text(d), f"post-{draft_id}.md")
    elif action == "reject":
        await asyncio.to_thread(store.update_draft, draft_id, status="rejected")
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(f"❌ Rejected #{draft_id}.")
    elif action == "regen":
        await query.edit_message_reply_markup(None)
        await _regenerate(update, draft_id, "")
        await push_new_drafts(context)


async def send_plain(bot, chat_id: int, text: str, filename: str) -> None:
    if len(text) <= TELEGRAM_LIMIT:
        await bot.send_message(chat_id, text)
    else:
        await bot.send_document(chat_id, document=io.BytesIO(text.encode()), filename=filename)


def main() -> None:
    s = get_settings()
    if not s.telegram_bot_token:
        log.warning("TELEGRAM_BOT_TOKEN is not set; bot disabled")
        return
    if not s.allowed_chat_ids:
        log.warning("TELEGRAM_ALLOWED_CHAT_IDS is empty: the bot will only reply with chat ids")

    app = Application.builder().token(s.telegram_bot_token).concurrent_updates(True).build()
    commands = {
        "start": cmd_help, "help": cmd_help, "platforms": cmd_platforms, "today": cmd_today,
        "run": cmd_run, "regen": cmd_regen, "edit": cmd_edit, "ideas": cmd_ideas, "idea": cmd_idea,
        "queue": cmd_queue, "drop": cmd_drop, "refill": cmd_refill, "steer": cmd_steer,
        "steering": cmd_steering, "unsteer": cmd_unsteer, "note": cmd_note,
    }
    for name, handler in commands.items():
        app.add_handler(CommandHandler(name, handler))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.REPLY & filters.TEXT & ~filters.COMMAND, on_reply))
    app.job_queue.run_repeating(push_new_drafts, interval=60, first=5)
    log.info("telegram bot polling")
    app.run_polling()

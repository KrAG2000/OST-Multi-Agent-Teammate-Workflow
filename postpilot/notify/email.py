"""SMTP newsletter: one email with every pending draft that hasn't been mailed yet."""

from __future__ import annotations

import html
import logging
import smtplib
import ssl
from email.message import EmailMessage

from postpilot import render, store
from postpilot.config import get_settings

log = logging.getLogger(__name__)


def _html(drafts: list[dict]) -> str:
    sections = []
    for d in drafts:
        image = render.image_text(d)
        sections.append(
            "<section style='margin:0 0 32px;padding:16px;border:1px solid #ddd;border-radius:8px'>"
            f"<h2 style='margin:0 0 4px;font-size:18px'>{html.escape(d.get('title') or d['platform'])}</h2>"
            f"<p style='margin:0 0 12px;color:#666;font-size:13px'>{html.escape(render.header(d))}</p>"
            "<pre style='white-space:pre-wrap;font-family:inherit;font-size:15px;margin:0'>"
            f"{html.escape(render.post_text(d))}</pre>"
            + (
                "<pre style='white-space:pre-wrap;font-family:inherit;font-size:13px;color:#444;"
                f"background:#f6f6f6;padding:10px;border-radius:6px'>{html.escape(image)}</pre>"
                if image
                else ""
            )
            + "</section>"
        )
    return (
        "<div style='font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:680px;margin:auto'>"
        f"<h1 style='font-size:22px'>Today's drafts ({len(drafts)})</h1>"
        "<p style='color:#666'>Review and approve them in Telegram, then post.</p>"
        + "".join(sections)
        + "</div>"
    )


def send_digest() -> int:
    """Email pending, not-yet-mailed drafts. Returns how many were sent."""
    s = get_settings()
    if not s.email_enabled:
        log.info("email not configured; skipping digest")
        return 0
    drafts = store.drafts_to_email()
    if not drafts:
        log.info("no new drafts to email")
        return 0

    msg = EmailMessage()
    msg["Subject"] = f"Your post drafts for {s.today():%a %d %b} ({len(drafts)})"
    msg["From"] = s.email_from or s.smtp_user
    msg["To"] = ", ".join(s.email_recipients)
    msg.set_content("\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(render.full_text(d) for d in drafts))
    msg.add_alternative(_html(drafts), subtype="html")

    context = ssl.create_default_context()
    if s.smtp_ssl:
        server = smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, context=context, timeout=30)
    else:
        server = smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=30)
    with server:
        if s.smtp_starttls and not s.smtp_ssl:
            server.starttls(context=context)
        if s.smtp_user:
            server.login(s.smtp_user, s.smtp_password)
        server.send_message(msg)

    store.mark_emailed([d["id"] for d in drafts])
    log.info("emailed %d drafts to %s", len(drafts), msg["To"])
    return len(drafts)

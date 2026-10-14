"""Turn a draft row into copy-ready text (Telegram, email, files)."""

from __future__ import annotations

PLATFORM_ICONS = {"instagram": "📸", "linkedin": "💼", "medium": "📝", "x": "🐦"}


def header(d: dict) -> str:
    icon = PLATFORM_ICONS.get(d["platform"], "🗒")
    score = f" · critic {d['critic_score']}/10" if d.get("critic_score") is not None else ""
    return f"{icon} Draft #{d['id']} · {d['platform']} · {d['run_date']}{score}"


def post_text(d: dict) -> str:
    """Exactly what gets pasted into the platform."""
    tags = " ".join(d.get("hashtags") or [])
    return f"{d['body']}\n\n{tags}".strip()


def image_text(d: dict) -> str:
    img = d.get("image")
    if not img:
        return ""
    lines = [
        f"🖼 Image: {img['width']}x{img['height']} px ({img['aspect_ratio']}, {img['size_label']})",
        f"Prompt: {img['prompt']}",
        f"Alt text: {img['alt_text']}",
    ]
    if img.get("text_overlay"):
        lines.append(f"Text overlay: {img['text_overlay']}")
    return "\n".join(lines)


def full_text(d: dict) -> str:
    parts = [header(d), d.get("title") or "", post_text(d), image_text(d)]
    if d.get("critic_notes"):
        parts.append(f"Editor notes:\n{d['critic_notes']}")
    return "\n\n".join(p for p in parts if p)

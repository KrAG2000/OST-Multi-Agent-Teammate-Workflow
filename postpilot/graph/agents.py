"""The three agent roles: ideation strategist, platform writer, and critic.

Each is a structured-output LLM call driven by the platform profile, so adding a
platform in YAML gives it its own team without new code.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from postpilot.config import Platform
from postpilot.llm import ask_structured

# --- output schemas ----------------------------------------------------------


class IdeaOut(BaseModel):
    title: str = Field(description="Short working title for the post")
    angle: str = Field(description="1-2 sentences: the specific take, hook or story")


class IdeaBatch(BaseModel):
    ideas: list[IdeaOut]


class ImageOut(BaseModel):
    size_label: str = Field(description="Exactly one of the allowed image size labels")
    prompt: str = Field(description="Detailed prompt for an image generator")
    alt_text: str = Field(description="Accessible description of the image")
    text_overlay: str = Field(default="", description="Short text to render on the image, if any")


class DraftOut(BaseModel):
    title: str = Field(description="Title or first-line hook")
    body: str = Field(description="The full post text, ready to paste")
    hashtags: list[str] = Field(default_factory=list)
    image: ImageOut | None = None


class Critique(BaseModel):
    # Issues come first so the model reasons before it scores.
    issues: list[str] = Field(default_factory=list, description="Concrete fixes needed, most important first")
    score: int = Field(description="1-10 using the rubric")
    approved: bool = Field(description="True if publishable with at most trivial polish")


# --- prompt building ---------------------------------------------------------

FORMAT_GUIDE = {
    "caption": "A social caption: strong first line, short paragraphs with line breaks.",
    "post": "A feed post: hook in the first two lines, short paragraphs, plain text (no markdown headings).",
    "article": "A long-form markdown article: '# Title', H2 sections, intro, conclusion, fenced code where useful.",
    "thread": "A thread: numbered parts separated by blank lines, each part self-contained.",
}


def _profile(p: Platform) -> str:
    lines = [
        f"Channel: {p.display_name}",
        f"Purpose: {p.purpose.strip()}",
        f"Audience: {p.audience or 'general'}",
        f"Tone: {p.tone or 'natural'}",
        f"Format: {FORMAT_GUIDE[p.format]}",
    ]
    if p.rules:
        lines.append("Channel rules:\n" + "\n".join(f"- {r}" for r in p.rules))
    return "\n".join(lines)


def _length_rule(p: Platform) -> str:
    parts = []
    if p.length.min_words:
        parts.append(f"at least {p.length.min_words} words")
    if p.length.max_words:
        parts.append(f"at most {p.length.max_words} words")
    if p.length.max_chars:
        parts.append(f"under {p.length.max_chars} characters")
    return ", ".join(parts) or "whatever length fits"


def _image_rule(p: Platform) -> str:
    if not p.image:
        return "No image for this channel; set image to null."
    sizes = "\n".join(
        f"- {s.label}: {s.width}x{s.height} px, aspect {s.aspect_ratio()} ({s.use_for})"
        for s in p.image.sizes
    )
    need = "An image is required." if p.image.required else "An image is optional."
    return (
        f"{need} Choose exactly one size_label from:\n{sizes}\n"
        "Pick the size that best fits this content, and compose the image prompt for that "
        "exact resolution and aspect ratio (subject placement, negative space for any text overlay, "
        "style, lighting, colours). Do not ask for text inside the image unless you set text_overlay."
    )


def _bullets(items: list[str], empty: str = "(none)") -> str:
    return "\n".join(f"- {i}" for i in items) if items else empty


def _context(snippets: list[str]) -> str:
    return "\n\n---\n\n".join(snippets) if snippets else "(no stored context matched)"


# --- agents ------------------------------------------------------------------


def generate_ideas(
    p: Platform,
    n: int,
    steering: list[str],
    avoid_titles: list[str],
    knowledge: list[str],
) -> list[IdeaOut]:
    system = (
        "You are the content strategist for one person's channel. You propose post ideas "
        "that fit the channel, follow the owner's steering directives above everything else, "
        "and are grounded in the owner's own knowledge notes when possible.\n\n" + _profile(p)
    )
    user = (
        f"Propose {n} fresh post ideas.\n\n"
        f"Owner's steering directives (highest priority):\n{_bullets(steering)}\n\n"
        f"Topic seeds:\n{_bullets(p.seed_topics)}\n\n"
        f"Already used or queued, do NOT repeat or closely paraphrase:\n{_bullets(avoid_titles)}\n\n"
        f"Owner's knowledge notes:\n{_context(knowledge)}\n\n"
        "Vary the kind of post (story, how-to, opinion, lesson learned, breakdown)."
    )
    return ask_structured(IdeaBatch, system, user, temperature=0.9).ideas[:n]


def write_draft(
    p: Platform,
    idea: dict,
    context: list[str],
    steering: list[str],
    feedback: str = "",
    previous: dict | None = None,
    critique: dict | None = None,
) -> DraftOut:
    system = (
        f"You are the {p.display_name} writer on a small content team that publishes for one "
        "person, in first person, in their voice.\n\n"
        f"{_profile(p)}\n"
        f"Length: {_length_rule(p)}.\n"
        f"Hashtags: between {p.hashtags.min} and {p.hashtags.max}, in the hashtags field only.\n"
        f"Image: {_image_rule(p)}\n\n"
        "Ground facts in the provided context. Never invent personal experiences, employers, "
        "numbers or results that the context does not support; stay general instead."
    )
    user = (
        f"Idea: {idea['title']}\nAngle: {idea.get('angle', '')}\n\n"
        f"Owner's steering directives:\n{_bullets(steering)}\n\n"
        f"Context from the owner's notes and past posts:\n{_context(context)}"
    )
    if previous:
        user += f"\n\nPrevious draft:\n{previous.get('body', '')}"
    if critique and critique.get("issues"):
        user += f"\n\nEditor's required fixes:\n{_bullets(critique['issues'])}"
    if feedback:
        user += f"\n\nOwner's feedback (must be addressed):\n{feedback}"
    return ask_structured(DraftOut, system, user)


def critique_draft(p: Platform, draft: dict, context: list[str]) -> Critique:
    system = (
        f"You are a demanding but fair editor for {p.display_name}. Judge whether a draft can be "
        "published.\n\n"
        f"{_profile(p)}\nLength: {_length_rule(p)}.\n\n"
        "Check: the hook, fit with purpose and tone, clarity, claims not supported by the "
        "context (invented anecdotes, numbers or events), and the channel rules. "
        "Hashtags and the image are stored in separate fields and appended at publish time, "
        "which is correct. List concrete fixes, not praise.\n\n"
        "Score rubric: 9-10 publish as-is; 7-8 good, minor polish only; 5-6 solid but needs "
        "real fixes; 3-4 major problems; 1-2 off-brief. Any invented anecdote or statistic "
        "caps the score at 5. Approve only at 7 or above."
    )
    tags = " ".join(draft.get("hashtags", [])) or "(none)"
    user = (
        f"Context the writer had:\n{_context(context)}\n\n"
        f"Draft title: {draft.get('title', '')}\n\n{draft['body']}\n\n"
        f"Hashtags field: {tags}"
    )
    return ask_structured(Critique, system, user, temperature=0.2)


# --- deterministic checks ----------------------------------------------------


def normalize_draft(p: Platform, d: DraftOut) -> dict:
    """Clean hashtags and pin the image to an allowed size with exact pixel dimensions."""
    # Models often also append hashtags to the body; move trailing hashtag-only lines out.
    lines = d.body.strip().splitlines()
    trailing: list[str] = []
    while lines and (not lines[-1].strip() or all(w.startswith("#") for w in lines[-1].split())):
        trailing = lines.pop().split() + trailing
    body = "\n".join(lines).strip() or d.body.strip()

    tags = []
    for t in [*d.hashtags, *trailing]:
        t = "#" + re.sub(r"[^\w]", "", t.lstrip("#"))
        if len(t) > 1 and t.lower() not in {x.lower() for x in tags}:
            tags.append(t)
    out = {"title": d.title.strip(), "body": body, "hashtags": tags, "image": None}

    if p.image and (d.image or p.image.required):
        img = d.image or ImageOut(size_label="", prompt=f"Illustration for: {d.title}", alt_text=d.title)
        size = p.image.get(img.size_label) or p.image.sizes[0]
        out["image"] = {
            "size_label": size.label,
            "width": size.width,
            "height": size.height,
            "aspect_ratio": size.aspect_ratio(),
            "prompt": f"{img.prompt.strip()} -- {size.width}x{size.height} px, {size.aspect_ratio()} aspect ratio.",
            "alt_text": img.alt_text.strip(),
            "text_overlay": img.text_overlay.strip(),
        }
    return out


def rule_violations(p: Platform, draft: dict) -> list[str]:
    body = draft["body"]
    words = len(body.split())
    issues = []
    if p.length.max_chars and len(body) > p.length.max_chars:
        issues.append(f"Body is {len(body)} characters; limit is {p.length.max_chars}.")
    if p.length.min_words and words < p.length.min_words:
        issues.append(f"Body is {words} words; needs at least {p.length.min_words}.")
    if p.length.max_words and words > p.length.max_words * 1.15:
        issues.append(f"Body is {words} words; aim for at most {p.length.max_words}.")
    n = len(draft["hashtags"])
    if n < p.hashtags.min or n > p.hashtags.max:
        issues.append(f"Use {p.hashtags.min}-{p.hashtags.max} hashtags (got {n}).")
    if p.format == "thread":
        long_parts = [x for x in body.split("\n\n") if len(x) > 280]
        if long_parts:
            issues.append(f"{len(long_parts)} thread part(s) exceed 280 characters.")
    return issues

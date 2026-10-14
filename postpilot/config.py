"""Runtime settings (from env / .env) and per-platform profiles (from YAML)."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postpilot:postpilot@localhost:5432/postpilot"

    # Chat model. Keys for hosted providers are read by LangChain from
    # OPENAI_API_KEY / ANTHROPIC_API_KEY / GROQ_API_KEY.
    llm_provider: Literal["ollama", "openai", "anthropic", "groq"] = "ollama"
    llm_model: str = "qwen3.5:9b"
    llm_temperature: float = 0.7
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192  # Ollama defaults to a tiny context window
    ollama_reasoning: bool = False  # disable "thinking" output on reasoning models

    # Embeddings for RAG. EMBEDDING_DIM must match the model and is baked into the table.
    embeddings_provider: Literal["ollama", "openai"] = "ollama"
    embeddings_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    platforms_file: Path = Path("config/platforms.yaml")
    knowledge_dir: Path = Path("knowledge")

    timezone: str = "UTC"
    generate_cron: str = "0 6 * * *"
    email_cron: str = "0 8 * * *"
    run_on_startup: bool = False

    max_revisions: int = 2
    min_critic_score: int = 7
    idea_backlog_min: int = 3
    idea_batch_size: int = 5
    rag_top_k: int = 5
    max_parallel_platforms: int = 2

    telegram_bot_token: str = ""
    telegram_allowed_chat_ids: str = ""  # comma-separated

    # "gmail" fills in Gmail's server settings; "smtp" uses SMTP_HOST etc. as given.
    email_provider: Literal["gmail", "smtp"] = "smtp"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    smtp_ssl: bool = False
    email_from: str = ""
    email_to: str = ""  # comma-separated

    @model_validator(mode="after")
    def _apply_email_provider(self) -> Settings:
        if self.email_provider == "gmail":
            self.smtp_host, self.smtp_port = "smtp.gmail.com", 587
            self.smtp_starttls, self.smtp_ssl = True, False
        self.email_from = self.email_from or self.smtp_user
        return self

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def today(self) -> date:
        return datetime.now(self.tz).date()

    @property
    def allowed_chat_ids(self) -> set[int]:
        return {int(x) for x in _split(self.telegram_allowed_chat_ids)}

    @property
    def email_recipients(self) -> list[str]:
        return _split(self.email_to)

    @property
    def email_enabled(self) -> bool:
        creds_ok = self.email_provider != "gmail" or bool(self.smtp_user and self.smtp_password)
        return bool(self.smtp_host and self.email_recipients and creds_ok)


def _split(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


# --- Platform profiles -------------------------------------------------------


class ImageSize(BaseModel):
    label: str
    width: int
    height: int
    aspect: str = ""  # e.g. "1.91:1"; derived from width/height when omitted
    use_for: str = ""

    def aspect_ratio(self) -> str:
        if self.aspect:
            return self.aspect
        d = gcd(self.width, self.height)
        return f"{self.width // d}:{self.height // d}"


class ImageRules(BaseModel):
    required: bool = True
    sizes: list[ImageSize] = Field(min_length=1)

    def get(self, label: str) -> ImageSize | None:
        return next((s for s in self.sizes if s.label == label), None)


class LengthRules(BaseModel):
    min_words: int | None = None
    max_words: int | None = None
    max_chars: int | None = None


class HashtagRules(BaseModel):
    min: int = 0
    max: int = 0


class Platform(BaseModel):
    name: str
    display_name: str
    enabled: bool = True
    purpose: str
    audience: str = ""
    tone: str = ""
    format: Literal["caption", "post", "article", "thread"] = "post"
    length: LengthRules = LengthRules()
    hashtags: HashtagRules = HashtagRules()
    image: ImageRules | None = None
    rules: list[str] = []
    seed_topics: list[str] = []


def load_platforms(path: Path | None = None) -> dict[str, Platform]:
    """Read platform profiles. Not cached, so YAML edits apply on the next run."""
    path = path or get_settings().platforms_file
    raw = yaml.safe_load(path.read_text()) or {}
    return {
        name: Platform(name=name, **cfg) for name, cfg in (raw.get("platforms") or {}).items()
    }


def enabled_platforms() -> dict[str, Platform]:
    return {k: v for k, v in load_platforms().items() if v.enabled}

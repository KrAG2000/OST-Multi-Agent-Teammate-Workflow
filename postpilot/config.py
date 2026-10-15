"""Runtime settings (from env / .env) and per-platform profiles (from YAML)."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import AliasChoices, BaseModel, Field, model_validator
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

    # Defaults for new users; each user can change their own with /time and /timezone.
    timezone: str = "UTC"
    generate_time: str = "06:00"

    max_revisions: int = 2
    min_critic_score: int = 7
    idea_backlog_min: int = 3
    idea_batch_size: int = 5
    rag_top_k: int = 5

    telegram_bot_token: str = ""
    # Admins: always allowed, no invite needed, own the knowledge/ folder. Comma-separated.
    telegram_admin_chat_ids: str = Field(
        "", validation_alias=AliasChoices("telegram_admin_chat_ids", "telegram_allowed_chat_ids")
    )
    invite_code: str = ""  # if set, new users must send /start <code>
    max_users: int = 0  # 0 = no cap

    # Daily limits (reset at 00:00 UTC), shared by all users. Set them to what your free
    # tiers allow; 0 = unlimited. Jobs are refused once any limit would be exceeded.
    quota_llm_requests_per_day: int = 300
    quota_llm_tokens_per_day: int = 1_500_000
    quota_embed_requests_per_day: int = 2000
    quota_emails_per_day: int = 400  # Gmail allows ~500/day
    quota_jobs_per_day: int = 60
    quota_jobs_per_user_per_day: int = 6
    llm_requests_per_minute: int = 0  # throttle for per-minute provider limits; 0 = none
    est_tokens_per_llm_request: int = 6000  # reserved per call until the real count is known

    # "gmail" fills in Gmail's server settings; "smtp" uses SMTP_HOST etc. as given.
    email_provider: Literal["gmail", "smtp"] = "smtp"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    smtp_ssl: bool = False
    # Only needed for custom SMTP whose login isn't an email address (e.g. Amazon SES keys).
    email_from: str = ""
    email_to: str = ""  # default email for admins; other users set theirs with /email

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
    def admin_chat_ids(self) -> set[int]:
        return {int(x) for x in _split(self.telegram_admin_chat_ids)}

    @property
    def smtp_configured(self) -> bool:
        creds_ok = self.email_provider != "gmail" or bool(self.smtp_user and self.smtp_password)
        return bool(self.smtp_host and self.email_from and creds_ok)

    @property
    def quota_limits(self) -> dict[str, int]:
        return {
            "llm_requests": self.quota_llm_requests_per_day,
            "llm_tokens": self.quota_llm_tokens_per_day,
            "embed_requests": self.quota_embed_requests_per_day,
            "emails": self.quota_emails_per_day,
            "jobs": self.quota_jobs_per_day,
        }


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


def load_platform_templates(path: Path | None = None) -> dict[str, Platform]:
    """Platform templates from YAML. New users start with a copy; each user then owns theirs."""
    path = path or get_settings().platforms_file
    raw = yaml.safe_load(path.read_text()) or {}
    return {
        name: Platform(name=name, **cfg) for name, cfg in (raw.get("platforms") or {}).items()
    }

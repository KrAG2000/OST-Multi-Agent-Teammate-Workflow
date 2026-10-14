# PostPilot

A multi-agent LangGraph pipeline that writes your daily social posts. Every morning it
drafts one post per platform (Instagram, LinkedIn, Medium, or any platform you add), each
in that channel's voice, with an image prompt sized for the platform. Drafts arrive in
Telegram and in an email newsletter. You review, tweak, approve, and post.

## How it works

```
                    ┌──────────── per platform, in parallel (LangGraph Send) ─────────────┐
cron ──▶ plan ──▶   │ ensure_backlog ─▶ pick_idea ─▶ retrieve ─▶ write ◀──▶ critique ─▶ save │ ──▶ Telegram + email
                    └──────────────────────────────────────────────────────────────────────┘
```

| Agent | Job |
|---|---|
| **Ideation strategist** | Keeps a backlog of ideas per platform, grounded in your knowledge base, avoiding past topics, and following your steering instructions. |
| **Platform writer** | Writes the post for that channel's purpose, tone, format, length and hashtag rules, and picks one allowed image size. The image prompt is composed for that exact resolution. |
| **Critic** | Scores the draft against a rubric, flags invented anecdotes, and runs hard checks (length, character limits, hashtag count). Failing drafts loop back to the writer up to `MAX_REVISIONS` times. |

- **Platforms are config, not code.** Each one is a block in [`config/platforms.yaml`](config/platforms.yaml) with its purpose, tone, length, hashtags, rules and allowed image sizes. Add a block and it gets its own agent team.
- **RAG (pgvector).** Your notes in `knowledge/` (and `/note` in Telegram) ground ideas and drafts. Approved posts are indexed too, so the writer learns your voice and the strategist avoids repeats.
- **You steer, the AI fills the gaps.** By default the AI proposes its own ideas. Your queued ideas always go first, and steering instructions shape all future ideas and writing.
- **Durable runs.** Runs are checkpointed in Postgres. If the model crashes on one platform, the retry resumes only the unfinished work.
- **Any LLM.** Ollama (local, default), OpenAI, Anthropic or Groq, switched with one env var.

## Quick start (local)

Requirements: Docker, plus Ollama on the host (or use the bundled Ollama container, see below).

```bash
cp .env.example .env              # then edit: TIMEZONE, Telegram, SMTP
ollama pull qwen3.5:9b            # or whatever LLM_MODEL you choose
ollama pull nomic-embed-text      # embeddings for RAG
docker compose up -d --build
```

**Let containers reach the host's Ollama.** On Linux, Ollama listens only on `127.0.0.1` by
default, so containers can't reach it. Make it listen on all interfaces:

```bash
sudo systemctl edit ollama        # add the two lines below, save
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

This also exposes Ollama on your network. Block port 11434 in your firewall if that matters to
you (Fedora's default workstation zone allows high ports). Alternatively, set
`COMPOSE_PROFILES=ollama` and `OLLAMA_BASE_URL=http://ollama:11434` to run Ollama in Docker.

**Try a run right away** instead of waiting for the cron schedule:

```bash
docker compose exec scheduler python -m postpilot run              # all enabled platforms
docker compose exec scheduler python -m postpilot run linkedin     # just one
docker compose exec scheduler python -m postpilot drafts           # print today's drafts
```

On an RTX 4060 with `qwen3.5:9b`, a run of all three platforms takes about 4-5 minutes.

## Telegram bot (free)

1. Message [@BotFather](https://t.me/BotFather), send `/newbot`, and put the token in `TELEGRAM_BOT_TOKEN`.
2. `docker compose up -d bot`, then message your bot. It replies with your chat id.
3. Put that id in `TELEGRAM_ALLOWED_CHAT_IDS` and run `docker compose up -d bot` again. Only listed chats can use the bot.

The bot uses long polling, so it needs no domain, webhook or open port.

| You do | It does |
|---|---|
| ✅ / ❌ / 🔁 buttons on a draft | Approve (sends copy-ready text and indexes it as a past post), reject, or rewrite |
| Reply to a draft with feedback | Rewrites it with your feedback |
| `/ideas [platform]` | Shows the upcoming backlog (⭐ = yours, 💡 = AI suggestions) |
| `/idea linkedin Why I wrote my own agent loop \| compare with LangGraph` | Adds your idea; it's used next |
| `/queue 12`, `/drop 12`, `/refill medium` | Promote, remove, or regenerate suggestions |
| `/steer linkedin more system design, fewer career posts` | Standing instruction for that platform's ideas and writing (`all` = every platform) |
| `/steering`, `/unsteer 3` | List or remove instructions |
| `/note ...` or send a `.md`/`.txt` file | Adds to the knowledge base |
| `/run [platform]`, `/today`, `/regen 7 punchier hook`, `/edit 7 <text>` | Generate now, resend today's drafts, rewrite, hand-edit |

## Email newsletter

At `EMAIL_CRON`, every pending draft not yet emailed goes out as a single HTML email to
`EMAIL_TO` (comma-separated). Pick a sender with `EMAIL_PROVIDER`:

| `EMAIL_PROVIDER` | Cost | Set |
|---|---|---|
| `gmail` | Free (about 500 emails/day on a personal account) | `SMTP_USER` = your Gmail address, `SMTP_PASSWORD` = an [App Password](https://myaccount.google.com/apppasswords) (needs 2-Step Verification). Server settings are filled in for you. |
| `smtp` | Depends on your provider | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_STARTTLS` or `SMTP_SSL`, `EMAIL_FROM`. Works with Brevo, Mailgun, Amazon SES, Outlook, or your own server. |

Email is skipped when credentials or recipients are missing. Send one now with
`docker compose exec scheduler python -m postpilot digest`.

## What costs money

The defaults are all free: Ollama, self-hosted Postgres, Telegram and Gmail. Anything paid
only turns on when you supply its credentials: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GROQ_API_KEY` (Groq has a free tier), or a paid SMTP provider.

## Configuration

Everything is in `.env` (see [`.env.example`](.env.example)) and `config/platforms.yaml`.
Both `config/` and `knowledge/` are mounted into the containers, so edits apply on the next run
without a rebuild.

| Want | Set |
|---|---|
| Hosted LLM | `LLM_PROVIDER=groq` + `LLM_MODEL=llama-3.3-70b-versatile` + `GROQ_API_KEY` (or `openai` / `anthropic` with their keys) |
| Different schedule | `GENERATE_CRON`, `EMAIL_CRON`, `TIMEZONE` (cron syntax) |
| New platform | Copy a block in `platforms.yaml`. `format` is one of `caption`, `post`, `article`, `thread` |
| Stricter or looser editing | `MIN_CRITIC_SCORE`, `MAX_REVISIONS` |
| OpenAI embeddings | `EMBEDDINGS_PROVIDER=openai`, `EMBEDDINGS_MODEL=text-embedding-3-small`, `EMBEDDING_DIM=1536`. The dimension is fixed on first start, so changing it later means dropping the `documents` table and re-ingesting. |

## Hosting on a free tier

**Recommended: Oracle Cloud Always Free.** It includes an Ampere ARM VM with up to 4 vCPUs and
**24 GB RAM** at no cost, which is enough to run this whole stack *including* a local 8-9B model
on CPU. A daily batch job doesn't need speed.

1. Create an Always Free "VM.Standard.A1.Flex" instance (Ubuntu) and install Docker.
2. `git clone` this repo, `cp .env.example .env`, and set `COMPOSE_PROFILES=ollama` and `OLLAMA_BASE_URL=http://ollama:11434`.
3. `docker compose up -d --build`. The `ollama-pull` service downloads the models on first start.

Oracle may reclaim Always Free instances that sit nearly idle for a week. The daily run helps.
Upgrading the account to Pay-As-You-Go (still free within the limits) removes that risk.

**Lighter option:** use Groq's free API tier for the LLM (`LLM_PROVIDER=groq`) and keep only
the small embedding model in Ollama. That runs on any ~2 GB VM.

**Scaling later:** move to a paid VM, point `DATABASE_URL` at managed Postgres with pgvector
(Neon, Supabase), or switch `LLM_PROVIDER` to a paid API. None of these need code changes.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
docker compose up -d db
pytest                                 # unit tests + end-to-end graph tests (fake LLM, real pgvector)

# CLI against the compose database from the host:
export DATABASE_URL=postgresql://postpilot:postpilot@localhost:5433/postpilot OLLAMA_BASE_URL=http://localhost:11434
python -m postpilot run instagram
python -m postpilot ideas
```

Layout: `postpilot/graph/pipeline.py` (LangGraph graphs and runners), `graph/agents.py`
(prompts, schemas, rule checks), `rag.py` (pgvector), `store.py` (SQL), `notify/` (Telegram,
email), `scheduler.py` (cron).

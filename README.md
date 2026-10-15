# PostPilot

A multi-agent LangGraph app, used through a Telegram bot, that writes daily social posts.
Each day at a time the user chooses, it drafts one post per platform (Instagram, LinkedIn,
Medium, or any platform you add), each in that channel's voice, with an image prompt sized
for the platform. Drafts arrive in Telegram and in an email newsletter. Users review, tweak,
approve and post.

Anyone can sign up with `/start`. Everyone shares one free server, so all work goes
through a single queue with daily limits. **Running it costs the operator nothing.**

## How it works

```
Telegram bot ─┐                      ┌─ one worker, one job at a time ─────────────────────────────┐
daily timer  ─┼─▶ queue (Postgres) ─▶│ plan ─▶ for each platform: ideas ─▶ retrieve ─▶ write ⇄ critic │─▶ Telegram + email
CLI          ─┘   reserve quota      └──────────────────────────────────────────────────────────────┘
```

| Agent | Job |
|---|---|
| **Ideation strategist** | Keeps a backlog of ideas per platform, grounded in the user's knowledge base, avoiding past topics, and following their steering instructions. |
| **Platform writer** | Writes the post for that channel's purpose, tone, format, length and hashtag rules, and picks one allowed image size. The image prompt is composed for that exact resolution. |
| **Critic** | Scores the draft against a rubric, flags invented anecdotes, and runs hard checks (length, character limits, hashtag count). Failing drafts loop back to the writer up to `MAX_REVISIONS` times. |

- **Per-user everything.** Each user has their own platforms, ideas, steering, knowledge base, drafts, email address, timezone and daily time. Users never see each other's data.
- **Platforms are config.** New users start with the platforms in [`config/platforms.yaml`](config/platforms.yaml), then change their own copy (`/enable`, `/disable`, `/set`).
- **RAG (pgvector).** Notes (`/note`, uploaded files, and the admin's `knowledge/` folder) ground ideas and drafts. Posts marked as posted are indexed too, so drafts match the user's voice and avoid repeat topics.
- **The user steers, the AI fills the gaps.** The AI proposes its own ideas by default. The user's queued ideas always go first, and steering instructions shape everything after.
- **Durable runs.** Runs are checkpointed in Postgres; a crashed run resumes where it stopped.
- **Any LLM.** Ollama (local, default), OpenAI, Anthropic or Groq, switched with one env var.

## Limits: how it stays free

Every paid-per-use resource has a daily limit in `.env`. The limits are shared by all users and reset at 00:00 UTC:

| Limit | Protects | Default |
|---|---|---|
| `QUOTA_LLM_REQUESTS_PER_DAY`, `QUOTA_LLM_TOKENS_PER_DAY` | Free LLM tiers (Groq) or the free VM's CPU (Ollama) | 300 / 1.5M |
| `QUOTA_EMBED_REQUESTS_PER_DAY` | The embedding provider | 2000 |
| `QUOTA_EMAILS_PER_DAY` | Gmail's ~500/day sending cap | 400 |
| `QUOTA_JOBS_PER_DAY`, `QUOTA_JOBS_PER_USER_PER_DAY` | Total work, and one user hogging it | 60 / 6 |
| `LLM_REQUESTS_PER_MINUTE` | Per-minute provider limits | off |

How they're enforced:

1. **One queue, one worker.** The bot and the daily timer only add jobs. A single worker runs them strictly one after another, and platforms within a run go one at a time, so there are never concurrent model calls. A second worker refuses to start.
2. **Reserve before queueing.** Each job reserves its worst-case usage in the same database transaction that queues it, under a lock. If any limit would be exceeded, the request is refused ("try again in 5h 12m"). Simultaneous requests can't both grab the last bit of capacity. When a job finishes, the unused part of its reservation is given back.
3. **Metered while running.** Every LLM, embedding and email call is charged as it happens. If a job needs more than it reserved, it can only borrow what's still free today. Otherwise it stops with a message to the user.

Each user can have only one generation, rewrite and idea refill waiting at a time. Check usage with `/usage` in Telegram or `postpilot usage`.

## Quick start (local)

Requirements: Docker, plus Ollama on the host (or use the bundled Ollama container, see below).

```bash
cp .env.example .env              # then edit: Telegram, admin chat id, SMTP, limits
ollama pull qwen3.5:9b            # or whatever LLM_MODEL you choose
ollama pull nomic-embed-text      # embeddings for RAG
docker compose up -d --build      # db + worker + bot
```

**Let containers reach the host's Ollama.** On Linux, Ollama listens only on `127.0.0.1` by
default. Make it listen on all interfaces:

```bash
sudo systemctl edit ollama        # add the two lines below, save
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

This also exposes Ollama on your network. Block port 11434 in your firewall if that matters to
you. Alternatively, set `COMPOSE_PROFILES=ollama` and `OLLAMA_BASE_URL=http://ollama:11434` to
run Ollama in Docker.

## Telegram bot (free)

1. Message [@BotFather](https://t.me/BotFather), send `/newbot`, and put the token in `TELEGRAM_BOT_TOKEN`.
2. Put your own chat id in `TELEGRAM_ADMIN_CHAT_IDS` (get it from [@userinfobot](https://t.me/userinfobot)). Admins need no invite and own the `knowledge/` folder.
3. `docker compose up -d`, then send `/start` to your bot.

Others join by sending `/start`. Set `INVITE_CODE` to make it invite-only (`/start <code>`) and
`MAX_USERS` to cap sign-ups. The bot uses long polling, so it needs no domain or open port.

| You do | It does |
|---|---|
| ✅ Approve | Sends copy-ready text, plus a 📤 **Mark as posted** button to tap once it's live |
| ❌ Reject | Asks why; the answer is saved as steering for that platform |
| 🔁 Regenerate, or reply to a draft with feedback | Queues a rewrite (feedback makes it much better) |
| `/run [platform]`, `/today`, `/regen 7 <feedback>`, `/edit 7 <text>`, `/posted 7` | Generate now, resend today's drafts, rewrite, hand-edit, mark posted |
| `/ideas`, `/idea linkedin <title> \| <angle>`, `/queue 12`, `/drop 12`, `/refill medium` | Manage the idea backlog (⭐ = yours, 💡 = AI) |
| `/steer linkedin more system design`, `/steering`, `/unsteer 3` | Standing instructions (`all` = every platform) |
| `/note ...` or send a `.md`/`.txt` file | Add to your knowledge base |
| `/platforms`, `/enable x`, `/disable medium`, `/set linkedin tone <text>` | Your platforms |
| `/me`, `/email you@x.com`, `/time 07:30`, `/timezone Europe/London`, `/pause`, `/resume` | Your settings |
| `/usage` | Today's shared capacity and the queue |

## Email newsletter

After a user's daily run, their pending drafts go out as one HTML email from "PostPilot" to
the address they set with `/email`. You set up the sending account once, for everyone. A
dedicated Gmail (e.g. `yourname.postpilot@gmail.com`) works better than your personal one.

| `EMAIL_PROVIDER` | Cost | Set |
|---|---|---|
| `gmail` | Free (about 500 emails/day) | `SMTP_USER` = the Gmail address, `SMTP_PASSWORD` = its [App Password](https://myaccount.google.com/apppasswords) (needs 2-Step Verification). |
| `smtp` | Depends on the provider | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_STARTTLS` or `SMTP_SSL`. Add `EMAIL_FROM` only if the login isn't an email address (Amazon SES). |

`EMAIL_TO` in `.env` is only the admin's default address.

## What costs money

The defaults are all free: Ollama, self-hosted Postgres, Telegram and Gmail. Anything paid
only turns on when you supply its credentials: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GROQ_API_KEY` (Groq has a free tier), or a paid SMTP provider. Whichever you use, the daily
limits above cap usage.

## Configuration

Everything is in `.env` (see [`.env.example`](.env.example)) and `config/platforms.yaml`.

| Want | Set |
|---|---|
| Hosted LLM | `LLM_PROVIDER=groq` + `LLM_MODEL=llama-3.3-70b-versatile` + `GROQ_API_KEY`, and copy your model's daily and per-minute limits from the Groq console into the `QUOTA_*` / `LLM_REQUESTS_PER_MINUTE` settings |
| Defaults for new users | `TIMEZONE`, `GENERATE_TIME`, and the platform templates in `platforms.yaml` |
| Stricter or looser editing | `MIN_CRITIC_SCORE`, `MAX_REVISIONS` |
| OpenAI embeddings | `EMBEDDINGS_PROVIDER=openai`, `EMBEDDINGS_MODEL=text-embedding-3-small`, `EMBEDDING_DIM=1536`. The dimension is fixed on first start. |

## Hosting on a free tier

**Recommended: Oracle Cloud Always Free.** It includes an Ampere ARM VM with up to 4 vCPUs and
**24 GB RAM** at no cost, which is enough to run this whole stack *including* a local 8-9B model
on CPU. CPU generation is slow, so keep `QUOTA_LLM_REQUESTS_PER_DAY` to what the VM can
process in a day.

1. Create an Always Free "VM.Standard.A1.Flex" instance (Ubuntu) and install Docker.
2. `git clone` this repo, `cp .env.example .env`, and set `COMPOSE_PROFILES=ollama` and `OLLAMA_BASE_URL=http://ollama:11434`.
3. `docker compose up -d --build`. The `ollama-pull` service downloads the models on first start.

Oracle may reclaim Always Free instances that sit nearly idle for a week. Upgrading the account
to Pay-As-You-Go (still free within the limits) removes that risk.

**Faster option:** use Groq's free API tier for the LLM and keep only the small embedding model
in Ollama, with the quotas set to Groq's free limits.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
docker compose up -d db
pytest          # unit tests + end-to-end tests (fake LLM, real pgvector): queue, limits, isolation

docker compose exec worker python -m postpilot run linkedin   # queue a run as the admin
docker compose exec worker python -m postpilot usage          # today's usage vs limits
docker compose exec worker python -m postpilot drafts         # admin's pending drafts
```

Layout: `graph/pipeline.py` (LangGraph graphs), `graph/agents.py` (prompts, schemas, rule
checks), `jobs.py` (queue and daily schedule), `quota.py` (limits), `worker.py`, `rag.py`
(pgvector), `store.py` (SQL), `notify/` (Telegram, email).

"""LangGraph orchestration.

Daily graph:   START -> plan -> (Send per platform) -> platform team -> END
Platform team: START -> [ensure_backlog -> pick_idea] -> retrieve -> write <-> critique -> save -> END

Runs are checkpointed in Postgres, so a run that crashes halfway (model timeout,
container restart) resumes where it stopped instead of redoing finished platforms.
Everything is scoped to one user, and platforms run one at a time (max_concurrency=1)
so a run never makes concurrent model calls.
"""

from __future__ import annotations

import logging
import operator
import time
from collections.abc import Iterator
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import contextmanager
from typing import Annotated, Literal

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send
from typing_extensions import NotRequired, TypedDict

from postpilot import rag, store
from postpilot.config import get_settings
from postpilot.quota import CapacityError
from postpilot.graph import agents

log = logging.getLogger(__name__)

# Retry flaky model calls, but never a quota refusal: that must stop the run.
LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=5.0, retry_on=lambda e: not isinstance(e, CapacityError))


# --- state -------------------------------------------------------------------


class PlatformState(TypedDict):
    user_id: int
    platform: str
    run_date: str
    idea: NotRequired[dict]
    feedback: NotRequired[str]
    parent_draft_id: NotRequired[int]
    previous: NotRequired[dict]
    context: NotRequired[list[str]]
    steering: NotRequired[list[str]]
    draft: NotRequired[dict]
    critique: NotRequired[dict]
    attempts: NotRequired[int]
    drafts: Annotated[list[dict], operator.add]


class PlatformOutput(TypedDict):
    drafts: Annotated[list[dict], operator.add]


class PipelineState(TypedDict):
    user_id: int
    run_date: str
    platforms: list[str]
    drafts: Annotated[list[dict], operator.add]


# --- platform team nodes -----------------------------------------------------


def _steering(user_id: int, platform: str) -> list[str]:
    return [s["directive"] for s in store.active_steering(user_id, platform)]


def _platform(user_id: int, name: str):
    platform = store.load_platforms(user_id).get(name)
    if platform is None:
        raise ValueError(f"unknown platform '{name}'")
    return platform


def refill_ideas(user_id: int, platform: str, n: int | None = None) -> list[int]:
    """Ideation agent: add `n` AI suggestions to the user's backlog for a platform."""
    s = get_settings()
    p = _platform(user_id, platform)
    steering = _steering(user_id, platform)
    query = " ".join([p.purpose, *steering, *p.seed_topics])
    knowledge = [d["content"] for d in rag.search(user_id, query, kinds=("knowledge",), k=8)]
    ideas = agents.generate_ideas(
        p, n or s.idea_batch_size, steering, store.recent_titles(user_id, platform), knowledge
    )
    ids = store.add_ideas(user_id, platform, [i.model_dump() for i in ideas], source="ai")
    log.info("%s: added %d idea suggestions", platform, len(ids))
    return ids


def ensure_backlog(state: PlatformState) -> dict:
    if len(store.open_ideas(state["user_id"], state["platform"])) < get_settings().idea_backlog_min:
        refill_ideas(state["user_id"], state["platform"])
    return {}


def pick_idea(state: PlatformState) -> dict:
    idea = store.next_idea(state["user_id"], state["platform"])
    if idea is None:
        raise RuntimeError(f"no ideas available for {state['platform']}")
    return {"idea": {"id": idea["id"], "title": idea["title"], "angle": idea["angle"]}}


def retrieve(state: PlatformState) -> dict:
    idea = state["idea"]
    query = f"{idea['title']}. {idea.get('angle', '')}"
    knowledge = rag.search(state["user_id"], query, kinds=("knowledge",))
    past = rag.search(state["user_id"], query, kinds=("post",), platform=state["platform"], k=3)
    context = [f"[note: {d['source']}]\n{d['content']}" for d in knowledge]
    context += [f"[past {state['platform']} post, for voice only]\n{d['content']}" for d in past]
    return {"context": context, "steering": _steering(state["user_id"], state["platform"])}


def write(state: PlatformState) -> dict:
    p = _platform(state["user_id"], state["platform"])
    revising = "critique" in state
    draft = agents.write_draft(
        p,
        state["idea"],
        state.get("context", []),
        state.get("steering", []),
        feedback=state.get("feedback", ""),
        previous=state["draft"] if revising else state.get("previous"),
        critique=state.get("critique"),
    )
    return {"draft": agents.normalize_draft(p, draft), "attempts": state.get("attempts", 0) + 1}


def critique(state: PlatformState) -> dict:
    p = _platform(state["user_id"], state["platform"])
    review = agents.critique_draft(p, state["draft"], state.get("context", []))
    hard = agents.rule_violations(p, state["draft"])
    score = max(1, min(10, review.score))
    approved = review.approved and not hard and score >= get_settings().min_critic_score
    issues = [i for i in review.issues if i.strip().lower().rstrip(".") not in {"", "none", "n/a"}]
    return {"critique": {"score": score, "approved": approved, "issues": hard + issues}}


def after_critique(state: PlatformState) -> Literal["write", "save"]:
    if state["critique"]["approved"] or state["attempts"] > get_settings().max_revisions:
        return "save"
    return "write"


def save(state: PlatformState) -> dict:
    draft_id = store.save_draft(
        user_id=state["user_id"],
        run_date=state["run_date"],
        platform=state["platform"],
        idea_id=state["idea"].get("id"),
        draft=state["draft"],
        critique=state.get("critique"),
        parent_id=state.get("parent_draft_id"),
        feedback=state.get("feedback"),
    )
    log.info("%s: saved draft #%d (score %s)", state["platform"], draft_id, state["critique"]["score"])
    return {"drafts": [{"id": draft_id, "platform": state["platform"], "title": state["draft"]["title"]}]}


def route_start(state: PlatformState) -> Literal["ensure_backlog", "retrieve"]:
    return "retrieve" if state.get("idea") else "ensure_backlog"


def build_platform_graph() -> StateGraph:
    g = StateGraph(PlatformState, output_schema=PlatformOutput)
    g.add_node("ensure_backlog", ensure_backlog, retry_policy=LLM_RETRY)
    g.add_node("pick_idea", pick_idea)
    g.add_node("retrieve", retrieve, retry_policy=LLM_RETRY)
    g.add_node("write", write, retry_policy=LLM_RETRY)
    g.add_node("critique", critique, retry_policy=LLM_RETRY)
    g.add_node("save", save)
    g.add_conditional_edges(START, route_start)
    g.add_edge("ensure_backlog", "pick_idea")
    g.add_edge("pick_idea", "retrieve")
    g.add_edge("retrieve", "write")
    g.add_edge("write", "critique")
    g.add_conditional_edges("critique", after_critique)
    g.add_edge("save", END)
    return g


# --- daily orchestrator ------------------------------------------------------


def plan(state: PipelineState) -> dict:
    enabled = store.enabled_platforms(state["user_id"])
    chosen = [p for p in state.get("platforms") or enabled if p in enabled]
    log.info("run %s: platforms %s", state["run_date"], chosen)
    return {"platforms": chosen}


def fan_out(state: PipelineState) -> list[Send]:
    return [
        Send("platform", {"user_id": state["user_id"], "platform": p, "run_date": state["run_date"], "drafts": []})
        for p in state["platforms"]
    ]


def build_pipeline(checkpointer=None):
    g = StateGraph(PipelineState)
    g.add_node("plan", plan)
    g.add_node("platform", build_platform_graph().compile())
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["platform"])
    g.add_edge("platform", END)
    return g.compile(checkpointer=checkpointer)


@contextmanager
def checkpointer() -> Iterator[PostgresSaver]:
    with PostgresSaver.from_conn_string(get_settings().database_url) as cp:
        yield cp


def _config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": 1,  # one model call at a time, always
    }


def user_today(user: dict) -> str:
    return datetime.now(ZoneInfo(user["timezone"])).date().isoformat()


def run_daily(user: dict, force: bool = False) -> list[dict]:
    """A user's scheduled run: one thread per day, resumed if a previous attempt failed midway."""
    today = user_today(user)
    thread_id = f"daily-{user['id']}-{today}"
    with checkpointer() as cp:
        graph = build_pipeline(cp)
        snapshot = graph.get_state(_config(thread_id))
        if snapshot.next:
            log.info("resuming unfinished run %s", thread_id)
            return graph.invoke(None, _config(thread_id))["drafts"]
        if snapshot.values and not force:
            log.info("run %s already completed; use force to generate again", thread_id)
            return snapshot.values.get("drafts", [])
        if snapshot.values:
            thread_id = f"{thread_id}-{int(time.time())}"
        inputs = {"user_id": user["id"], "run_date": today, "platforms": [], "drafts": []}
        return graph.invoke(inputs, _config(thread_id))["drafts"]


def run_now(user: dict, platforms: list[str] | None = None) -> list[dict]:
    """Ad-hoc run (/run), always a fresh thread."""
    today = user_today(user)
    with checkpointer() as cp:
        graph = build_pipeline(cp)
        inputs = {"user_id": user["id"], "run_date": today, "platforms": platforms or [], "drafts": []}
        return graph.invoke(inputs, _config(f"manual-{user['id']}-{today}-{int(time.time())}"))["drafts"]


def regenerate(user: dict, draft_id: int, feedback: str = "") -> dict:
    """Re-run one platform team on an existing draft's idea, with the owner's feedback."""
    old = store.get_draft(user["id"], draft_id)
    if old is None:
        raise ValueError(f"draft #{draft_id} not found")
    idea = store.get_idea(user["id"], old["idea_id"]) if old["idea_id"] else None
    inputs: PlatformState = {
        "user_id": user["id"],
        "platform": old["platform"],
        "run_date": user_today(user),
        "idea": {
            "id": old["idea_id"],
            "title": idea["title"] if idea else old["title"],
            "angle": idea["angle"] if idea else "",
        },
        "feedback": feedback,
        "parent_draft_id": draft_id,
        "previous": {"body": old["body"]},
        "drafts": [],
    }
    with checkpointer() as cp:
        graph = build_platform_graph().compile(checkpointer=cp)
        result = graph.invoke(inputs, _config(f"regen-{user['id']}-{draft_id}-{int(time.time())}"))
    return result["drafts"][0]

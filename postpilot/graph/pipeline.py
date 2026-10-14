"""LangGraph orchestration.

Daily graph:   START -> plan -> (Send per platform) -> platform team -> END
Platform team: START -> [ensure_backlog -> pick_idea] -> retrieve -> write <-> critique -> save -> END

Runs are checkpointed in Postgres, so a run that crashes halfway (model timeout,
container restart) resumes where it stopped instead of redoing finished platforms.
"""

from __future__ import annotations

import logging
import operator
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Literal

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send
from typing_extensions import NotRequired, TypedDict

from postpilot import rag, store
from postpilot.config import enabled_platforms, get_settings, load_platforms
from postpilot.graph import agents

log = logging.getLogger(__name__)

LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=5.0)


# --- state -------------------------------------------------------------------


class PlatformState(TypedDict):
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
    run_date: str
    platforms: list[str]
    drafts: Annotated[list[dict], operator.add]


# --- platform team nodes -----------------------------------------------------


def _steering(platform: str) -> list[str]:
    return [s["directive"] for s in store.active_steering(platform)]


def refill_ideas(platform: str, n: int | None = None) -> list[int]:
    """Ideation agent: add `n` AI suggestions to the platform's backlog."""
    s = get_settings()
    p = load_platforms()[platform]
    steering = _steering(platform)
    query = " ".join([p.purpose, *steering, *p.seed_topics])
    knowledge = [d["content"] for d in rag.search(query, kinds=("knowledge",), k=8)]
    ideas = agents.generate_ideas(
        p, n or s.idea_batch_size, steering, store.recent_titles(platform), knowledge
    )
    ids = store.add_ideas(platform, [i.model_dump() for i in ideas], source="ai")
    log.info("%s: added %d idea suggestions", platform, len(ids))
    return ids


def ensure_backlog(state: PlatformState) -> dict:
    if len(store.open_ideas(state["platform"])) < get_settings().idea_backlog_min:
        refill_ideas(state["platform"])
    return {}


def pick_idea(state: PlatformState) -> dict:
    idea = store.next_idea(state["platform"])
    if idea is None:
        raise RuntimeError(f"no ideas available for {state['platform']}")
    return {"idea": {"id": idea["id"], "title": idea["title"], "angle": idea["angle"]}}


def retrieve(state: PlatformState) -> dict:
    idea = state["idea"]
    query = f"{idea['title']}. {idea.get('angle', '')}"
    knowledge = rag.search(query, kinds=("knowledge",))
    past = rag.search(query, kinds=("post",), platform=state["platform"], k=3)
    context = [f"[note: {d['source']}]\n{d['content']}" for d in knowledge]
    context += [f"[past {state['platform']} post, for voice only]\n{d['content']}" for d in past]
    return {"context": context, "steering": _steering(state["platform"])}


def write(state: PlatformState) -> dict:
    p = load_platforms()[state["platform"]]
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
    p = load_platforms()[state["platform"]]
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
    enabled = enabled_platforms()
    chosen = [p for p in state.get("platforms") or enabled if p in enabled]
    log.info("run %s: platforms %s", state["run_date"], chosen)
    return {"platforms": chosen}


def fan_out(state: PipelineState) -> list[Send]:
    return [Send("platform", {"platform": p, "run_date": state["run_date"], "drafts": []}) for p in state["platforms"]]


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
        "max_concurrency": get_settings().max_parallel_platforms,
    }


def run_daily(force: bool = False) -> list[dict]:
    """The scheduled run: one thread per day, resumed if a previous attempt failed midway."""
    today = get_settings().today().isoformat()
    thread_id = f"daily-{today}"
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
        inputs = {"run_date": today, "platforms": [], "drafts": []}
        return graph.invoke(inputs, _config(thread_id))["drafts"]


def run_now(platforms: list[str] | None = None) -> list[dict]:
    """Ad-hoc run (CLI / Telegram /run), always a fresh thread."""
    today = get_settings().today().isoformat()
    with checkpointer() as cp:
        graph = build_pipeline(cp)
        inputs = {"run_date": today, "platforms": platforms or [], "drafts": []}
        return graph.invoke(inputs, _config(f"manual-{today}-{int(time.time())}"))["drafts"]


def regenerate(draft_id: int, feedback: str = "") -> dict:
    """Re-run one platform team on an existing draft's idea, with the owner's feedback."""
    old = store.get_draft(draft_id)
    if old is None:
        raise ValueError(f"draft #{draft_id} not found")
    idea = store.get_idea(old["idea_id"]) if old["idea_id"] else None
    inputs: PlatformState = {
        "platform": old["platform"],
        "run_date": get_settings().today().isoformat(),
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
        result = graph.invoke(inputs, _config(f"regen-{draft_id}-{int(time.time())}"))
    return result["drafts"][0]

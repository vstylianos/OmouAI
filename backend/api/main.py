"""HTTP API over DebateSession.

Sessions live in memory, keyed by id — this is a single-user demo, not a
service. Every mutating endpoint returns the full session state so the client
never has to reconcile a partial update against local state; the graph is small
enough that this is cheaper than being clever.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from domains import DOMAINS, Domain, get_domain
from engine.lens import SDG_NAMES, Goal
from engine.attribution import DeltaExplanation
from llm.client import CLASSIFIER_MODEL, LLMClient, MissingAPIKey
from llm.schemas import Attachment
from pipeline.session import CONFIDENCE_LEVELS, DebateSession, TurnResult

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="OmouAI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, DebateSession] = {}
#: Which domain each session was started in, for serialisation.
SESSION_DOMAIN: dict[str, Domain] = {}


@app.middleware("http")
async def demo_gate(request: Request, call_next):
    """Shared-passphrase gate for hosted demos. Inert unless DEMO_PASSPHRASE
    is set (it never is locally). Every API call spends from the deploy's
    OpenAI key, so the whole /api surface is gated, not just session
    creation; /api/health stays open for platform liveness checks."""
    passphrase = os.environ.get("DEMO_PASSPHRASE")
    if (
        passphrase
        and request.url.path.startswith("/api/")
        and request.url.path != "/api/health"
        and request.headers.get("x-demo-passphrase") != passphrase
    ):
        return JSONResponse({"detail": "passphrase required"}, status_code=401)
    return await call_next(request)


# --- request bodies ---------------------------------------------------------

class GoalSelection(BaseModel):
    """The deliberation's targets, chosen at preparation. Fixed for the session."""

    sdgs: list[int] = Field(default_factory=list)
    indicators: list[str] = Field(default_factory=list)
    custom: list[str] = Field(default_factory=list)


class NewSession(BaseModel):
    domain_id: str = "barcelona_homelessness"
    #: The personas at the table, in order. At least one.
    persona_ids: list[str] = Field(min_length=1)
    topic: str
    goals: GoalSelection = Field(default_factory=GoalSelection)


class Retag(BaseModel):
    #: The node's full desired tag list: [{"goal_id": ..., "direction": ...}].
    tags: list[dict]


class Expand(BaseModel):
    node_id: str
    relation_type: str = Field(pattern="^(attack|support)$")
    #: Which persona argues. Defaults to the first at the table.
    persona_id: str | None = None


class Contribute(BaseModel):
    text: str
    #: Present when the user has confirmed a low-confidence attachment.
    attachment: Attachment | None = None
    #: Present when the user has resolved an unclear intent reading:
    #: treat the text as their own argument, or as a request to the agent.
    force: str | None = Field(default=None, pattern="^(argument|request)$")
    #: The user's own confidence in their argument, a CONFIDENCE_LEVELS key.
    #: Used only when the message is an argument; defaults to the middle.
    confidence: str | None = None
    #: The persona addressed by a question or request. Defaults to the first.
    persona_id: str | None = None
    #: Right-click on a node: the point is the user's and attaches to this
    #: node; only the direction is inferred.
    target_id: str | None = None


class Rescore(BaseModel):
    base_score: float = Field(ge=0.0, le=1.0)


class Settings(BaseModel):
    semantics: str | None = None
    conservativeness: float | None = Field(default=None, gt=0)


# --- serialisation ----------------------------------------------------------

def usage_of(session: DebateSession) -> dict:
    """Token usage across the persona's client and, if separate, the classifier's."""
    clients = [session.client]
    if session.classifier is not session.client:
        clients.append(session.classifier)
    return {
        "calls": sum(c.usage.calls for c in clients),
        "prompt_tokens": sum(c.usage.prompt_tokens for c in clients),
        "completion_tokens": sum(c.usage.completion_tokens for c in clients),
    }


def serialise(session: DebateSession, last: TurnResult | None = None) -> dict:
    strengths = session.strengths()
    qbaf = session.qbaf

    parent_of: dict[str, tuple[str, str, str]] = {}
    for rel in qbaf.relations:
        parent_of[rel.source_id] = (rel.target_id, rel.relation_type, rel.rationale)

    nodes = []
    for arg_id, arg in qbaf.arguments.items():
        parent, relation, rel_rationale = parent_of.get(arg_id, (None, None, ""))
        nodes.append({
            "id": arg_id,
            "text": arg.text,
            "node_type": arg.node_type,
            "base_score": arg.base_score,
            "strength": strengths[arg_id],
            "provenance": arg.provenance,
            "depth": arg.depth,
            "rationale": arg.rationale,
            "relation_rationale": rel_rationale,
            "parent_id": parent,
            "relation_type": relation,
            # 'ambiguous' means the node reaches the claim by more than one path,
            # so no monotonicity guarantee applies to contesting it.
            "polarity": qbaf.classify_polarity(arg_id),
            "is_topic": arg_id == qbaf.topic_id,
            "goal_impacts": arg.goal_impacts,
            "author": arg.author,
            "author_name": session.personas[arg.author].name if arg.author in session.personas else None,
        })

    domain = SESSION_DOMAIN.get(session.session_id)
    state = {
        "session_id": session.session_id,
        "domain": None if domain is None else {"id": domain.id, "name": domain.name},
        "personas": [
            {
                "id": p.id,
                "name": p.name,
                "one_line": p.one_line,
                "values": list(p.values),
                "conservativeness": p.conservativeness,
                "kind": p.kind,
            }
            for p in session.personas.values()
        ],
        "topic": session.topic,
        "stance": strengths[qbaf.topic_id],
        "nodes": nodes,
        "edges": [
            {
                "source": r.source_id,
                "target": r.target_id,
                "relation_type": r.relation_type,
                "rationale": r.rationale,
            }
            for r in qbaf.relations
        ],
        "history": session.history(),
        "goals": [
            {"id": g.id, "label": g.label, "kind": g.kind} for g in session.goals
        ],
        # Compact lens for the header; the full explanation graph is /lens.
        "lens": [
            {k: v for k, v in entry.items() if k != "contributions"}
            for entry in session.lens()
        ],
        "is_tree": qbaf.is_tree(),
        "semantics": session.semantics.name,
        # The session's current k, which a user may have overridden -- distinct
        # from the persona's default in `persona.conservativeness`.
        "conservativeness": session.conservativeness,
        "usage": usage_of(session),
    }

    if last is not None:
        state["turn"] = serialise_turn(last)
    return state


def serialise_delta(delta: DeltaExplanation) -> dict:
    driver = delta.primary_driver()
    return {
        "before": delta.topic_before,
        "after": delta.topic_after,
        "change": delta.topic_delta,
        "direction": delta.direction,
        "driver": None if driver is None else {
            "id": driver.arg_id,
            "text": driver.text,
            "before": driver.before,
            "after": driver.after,
            "status": driver.status,
        },
    }


def serialise_turn(turn: TurnResult) -> dict:
    return {
        "reply": turn.reply,
        "concedes": turn.concedes,
        "added_nodes": turn.added_nodes,
        "delta": serialise_delta(turn.delta),
        "steps": [
            {"label": label, "delta": serialise_delta(delta)}
            for label, delta in turn.steps
        ],
        "pending": None if turn.pending is None else turn.pending.model_dump(),
        "pending_intent": turn.pending_intent,
        "followup_request": turn.followup_request,
        "question": turn.question,
        "speaker": turn.speaker,
        "intros": [{"persona_id": pid, "text": text} for pid, text in turn.intros],
    }


def require(session_id: str) -> DebateSession:
    if session_id not in SESSIONS:
        raise HTTPException(404, "session not found")
    return SESSIONS[session_id]


# --- routes -----------------------------------------------------------------

@app.get("/api/personas")
def catalogue():
    """The domain catalogue: claims and deliberation partners per domain."""
    return {
        "domains": [
            {
                "id": d.id,
                "name": d.name,
                "blurb": d.blurb,
                "claims": list(d.seed_claims),
                "indicators": [
                    {
                        "id": i.id,
                        "name": i.name,
                        "objective": i.objective,
                        "category": i.category,
                        "sdg": list(i.sdg),
                    }
                    for i in d.indicators
                ],
                "partners": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "one_line": p.one_line,
                        "values": list(p.values),
                        "conservativeness": p.conservativeness,
                        "kind": p.kind,
                    }
                    for p in d.partners
                ],
            }
            for d in DOMAINS.values()
        ],
        "sdgs": [{"goal": n, "name": name} for n, name in SDG_NAMES.items()],
        # The ordinal scale the composer offers for the user's own confidence.
        "confidence_levels": [
            {"key": key, "label": label, "tau": tau}
            for key, label, tau in CONFIDENCE_LEVELS
        ],
    }


def build_goals(domain: Domain, selection: GoalSelection) -> tuple[Goal, ...]:
    """Turn the start-screen selection into the session's Goal tuple.

    Duplicates are dropped, not trusted: a repeated selection would ground
    two identical virtual roots and crash the lens. Custom goals are
    deduplicated by text and length-capped -- their labels enter prompts.
    """
    goals: list[Goal] = []
    seen: set[str] = set()

    def add(goal: Goal) -> None:
        if goal.id not in seen:
            seen.add(goal.id)
            goals.append(goal)

    for n in selection.sdgs:
        if n not in SDG_NAMES:
            raise HTTPException(400, f"unknown SDG {n}")
        add(Goal(id=f"sdg_{n}", label=f"SDG {n} · {SDG_NAMES[n]}", kind="sdg"))
    for ind_id in selection.indicators:
        ind = domain.get_indicator(ind_id)
        if ind is None:
            raise HTTPException(400, f"unknown indicator {ind_id!r}")
        add(Goal(id=f"ind_{ind.id}", label=ind.name, kind="indicator"))
    texts: list[str] = []
    for text in (t.strip() for t in selection.custom):
        if not text or text in texts:
            continue
        if len(text) > 200:
            raise HTTPException(400, "custom goal too long (200 characters max)")
        texts.append(text)
        add(Goal(id=f"custom_{len(texts)}", label=text, kind="custom"))
    return tuple(goals)


@app.post("/api/session")
async def create_session(body: NewSession):
    try:
        domain = get_domain(body.domain_id)
        seen: list[str] = []
        personas = tuple(
            domain.get_partner(pid) for pid in body.persona_ids
            if pid not in seen and not seen.append(pid)
        )
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        client = LLMClient()
        # The neutral judge for routing and placement: a separate model when
        # configured, otherwise the same client with persona-free prompts.
        classifier = LLMClient(model=CLASSIFIER_MODEL) if CLASSIFIER_MODEL else None
    except MissingAPIKey as exc:
        raise HTTPException(500, str(exc)) from exc

    session = DebateSession(
        personas=personas,
        topic=body.topic.strip(),
        client=client,
        session_id=uuid.uuid4().hex[:12],
        goals=build_goals(domain, body.goals),
        classifier=classifier,
    )
    SESSIONS[session.session_id] = session
    SESSION_DOMAIN[session.session_id] = domain

    turn = await session.open()
    return serialise(session, turn)


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    return serialise(require(session_id))


@app.post("/api/session/{session_id}/expand")
async def expand(session_id: str, body: Expand):
    session = require(session_id)
    try:
        turn = await session.expand(body.node_id, body.relation_type, body.persona_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return serialise(session, turn)


@app.post("/api/session/{session_id}/contribute")
async def contribute(session_id: str, body: Contribute):
    session = require(session_id)
    if not body.text.strip():
        raise HTTPException(400, "empty contribution")
    try:
        turn = await session.contribute(
            body.text.strip(),
            attachment=body.attachment,
            force=body.force,
            confidence=body.confidence,
            persona_id=body.persona_id,
            target_id=body.target_id,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return serialise(session, turn)


@app.patch("/api/session/{session_id}/node/{node_id}")
def rescore(session_id: str, node_id: str, body: Rescore):
    session = require(session_id)
    if node_id not in session.qbaf.arguments:
        raise HTTPException(404, "node not found")
    delta = session.set_base_score(node_id, body.base_score)
    state = serialise(session)
    state["turn"] = serialise_turn(TurnResult(delta=delta))
    return state


@app.delete("/api/session/{session_id}/node/{node_id}")
def drop(session_id: str, node_id: str):
    session = require(session_id)
    if node_id not in session.qbaf.arguments:
        raise HTTPException(404, "node not found")
    try:
        delta = session.remove(node_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    state = serialise(session)
    state["turn"] = serialise_turn(TurnResult(delta=delta))
    return state


@app.get("/api/session/{session_id}/lens")
def lens(session_id: str):
    """The full grounded lens, contributions included. Pure engine, no LLM."""
    session = require(session_id)
    return {
        "topic": session.topic,
        "lens": session.lens(),
        "history": session.lens_history(),
    }


@app.put("/api/session/{session_id}/node/{node_id}/tags")
def retag(session_id: str, node_id: str, body: Retag):
    """User contests an argument's goal tags. No LLM call."""
    session = require(session_id)
    try:
        delta = session.set_goal_tags(node_id, body.tags)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    state = serialise(session)
    state["turn"] = serialise_turn(TurnResult(delta=delta))
    return state


@app.get("/api/session/{session_id}/report")
def report(session_id: str):
    """The deliberation record. Pure engine, no LLM call."""
    return require(session_id).report()


@app.get("/api/session/{session_id}/summary")
async def summary(session_id: str):
    """Meeting notes so far. One neutral model call, cached per state."""
    return await require(session_id).summarise()


@app.get("/api/session/{session_id}/leverage")
def leverage(session_id: str):
    """What would most change the agent's mind. Pure engine, no LLM call."""
    session = require(session_id)
    return {
        "leverage": [
            {
                "id": arg.arg_id,
                "text": arg.text,
                "sensitivity": sensitivity,
                "kind": session.disagreement_kind(arg.arg_id),
            }
            for arg, sensitivity in session.leverage(limit=5)
        ]
    }


@app.patch("/api/session/{session_id}/settings")
def settings(session_id: str, body: Settings):
    session = require(session_id)
    try:
        session.set_semantics(
            body.semantics or session.semantics.name, body.conservativeness
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return serialise(session)


@app.get("/api/health")
def health():
    return {"ok": True, "sessions": len(SESSIONS)}


# When a built frontend exists, serve it from the same origin -- this is how
# the hosted demo runs as a single process. In development the directory is
# absent (or unused): Vite serves the UI on :5173 and proxies /api here.
# Mounted last so the /api routes above always win.
_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="ui")

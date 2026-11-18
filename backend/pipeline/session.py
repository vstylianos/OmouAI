"""A debate session: one persona, one topic, one graph that grows by dialogue.

The graph is *not* pre-computed. There is no tree-building stage. The agent
opens with its position only, and every node after that is
generated on demand -- because the user asked the agent to elaborate, because
the user contributed something, or because the agent chose to rebut. A turn
costs a handful of LLM calls, not forty.

This module is the whole API surface the web layer needs. Each public method
either mutates the graph and returns a DeltaExplanation, or is a pure read.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

from engine.attribution import DeltaExplanation, explain_change, ranked_leverage
from engine.lens import Goal, evaluation_lens
from engine.qbaf import QBAF, Argument, NodeType, Relation
from engine.semantics import LinearInfluence, Modular, QBAFSemantics, get_semantics
from llm.client import LLMClient
from llm.schemas import (
    AgentReply,
    Attachment,
    ExpansionResponse,
    FulfilledRequest,
    GeneratedArgument,
    MeetingNotes,
    RoutedMessage,
    ScoredArgument,
    StateAnswer,
)
from personas.schema import Persona
from pipeline import prompts

logger = logging.getLogger(__name__)

#: Attachments below this confidence are surfaced for user confirmation rather
#: than applied. Mis-attachment flips polarity and inverts the computation, so
#: the default is deliberately cautious.
ATTACH_CONFIRM_THRESHOLD = 75

#: The user's own confidence in an argument they make, as an ordinal scale:
#: (key, label, tau). The user is not asked for a number; the persona is not
#: asked to score their point. Extremes are avoided so an attack can still
#: bite and a supporter can still lift.
CONFIDENCE_LEVELS: tuple[tuple[str, str, float], ...] = (
    ("unsure", "Unsure", 0.2),
    ("somewhat", "Somewhat", 0.35),
    ("moderately", "Moderately", 0.5),
    ("very", "Very", 0.65),
    ("certain", "Certain", 0.8),
)
CONFIDENCE_TAU: dict[str, float] = {key: tau for key, _, tau in CONFIDENCE_LEVELS}
DEFAULT_CONFIDENCE = "moderately"

@dataclass
class TurnResult:
    """What changed on a turn, and what the agent said about it."""

    delta: DeltaExplanation
    #: The turn's movement broken into labelled steps when it had more than
    #: one -- the user's argument landing, then the persona's response to
    #: it. `delta` is always the whole turn.
    steps: list[tuple[str, DeltaExplanation]] = field(default_factory=list)
    reply: str = ""
    concedes: bool = False
    added_nodes: list[str] = field(default_factory=list)
    #: Set when an attachment needs confirming before it is applied.
    pending: Attachment | None = None
    #: Set when the user's intent (argument vs request) needs confirming:
    #: the crisply restated request awaiting a decision.
    pending_intent: str | None = None
    #: Mixed message whose argument half is pending: the request half rides
    #: along and runs after the card resolves, so it is never lost.
    followup_request: str | None = None
    #: The agent's closing question (reflective questioning stage). Every
    #: conversational reply carries one.
    question: str = ""
    #: Which persona spoke `reply` (its id); None for a silent turn.
    speaker: str | None = None
    #: The start of a session: one introduction per persona, (id, text).
    intros: list[tuple[str, str]] = field(default_factory=list)


class DebateSession:
    def __init__(
        self,
        personas: tuple[Persona, ...],
        topic: str,
        client: LLMClient,
        semantics: QBAFSemantics | None = None,
        session_id: str = "local",
        goals: tuple[Goal, ...] = (),
        classifier: LLMClient | None = None,
    ):
        if not personas:
            raise ValueError("a session needs at least one persona")
        #: The personas at the table, by id. Every argument a persona makes
        #: is authored and scored by that persona; the position is the
        #: table's, computed over everyone's arguments.
        self.personas: dict[str, Persona] = {p.id: p for p in personas}
        self.topic = topic
        self.client = client
        #: The neutral judge: routes messages and places the user's points.
        #: Its prompts carry no persona; it may also be a different model, so
        #: the persona's model never touches classification. Defaults to the
        #: same client.
        self.classifier = classifier or client
        self.session_id = session_id
        #: The deliberation's selected targets (preparation stage). Fixed for
        #: the session; they define the tagging vocabulary and the lenses.
        self.goals = goals
        #: Node ids the agent conceded to (identification stage).
        self._concessions: list[str] = []
        # Epistemic disposition enters here, not in any prompt. With several
        # personas the table's k is their mean; the session tracks its
        # current value so that switching semantics away and back does not
        # silently reset a user-chosen k.
        self.conservativeness = sum(p.conservativeness for p in personas) / len(personas)
        self.semantics = semantics or Modular(
            influence=LinearInfluence(conservativeness=self.conservativeness)
        )
        self.qbaf = QBAF(topic_id="topic")
        self.qbaf.add_argument(
            Argument(arg_id="topic", text=topic, node_type="normative", base_score=0.5)
        )
        self._ids = itertools.count(1)
        self._transcript: list[tuple[str, str]] = []
        #: (label, graph state) after each turn. Stored as snapshots rather than
        #: as stance numbers so that changing the semantics recomputes the whole
        #: trajectory -- a stance is never a stored figure, always a computation.
        self._snapshots: list[tuple[str, QBAF]] = []
        #: (state key, notes) -- the last meeting notes, reused while nothing
        #: has changed so reopening the summary costs no call.
        self._notes_cache: tuple[str, dict] | None = None

    def persona(self, persona_id: str | None = None) -> Persona:
        """A persona at the table; the first one when none is named."""
        if persona_id is None:
            return next(iter(self.personas.values()))
        if persona_id not in self.personas:
            raise KeyError(f"persona {persona_id!r} is not at this table")
        return self.personas[persona_id]

    # -- ids -----------------------------------------------------------

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._ids)}"

    def _record(self, label: str) -> None:
        self._snapshots.append((label, self._snapshot()))

    def history(self) -> list[dict]:
        """Stance after each turn, recomputed under the current semantics."""
        return [
            {"label": label, "stance": self.semantics.compute(graph)[graph.topic_id]}
            for label, graph in self._snapshots
        ]

    def set_semantics(self, name: str, conservativeness: float | None = None) -> None:
        """Swap the semantics at runtime. History recomputes automatically.

        An explicit conservativeness updates the session's current value;
        otherwise the current value is kept, so switching to another semantics
        and back preserves a user-chosen k.
        """
        if name == "modular":
            if conservativeness is not None:
                self.conservativeness = conservativeness
            self.semantics = Modular(
                influence=LinearInfluence(conservativeness=self.conservativeness)
            )
        else:
            self.semantics = get_semantics(name)

    # -- reads ---------------------------------------------------------

    def stance(self) -> float:
        return self.semantics.compute(self.qbaf)[self.qbaf.topic_id]

    def lens(self) -> list[dict]:
        """The grounded evaluation lens over the selected goals. No LLM call."""
        return evaluation_lens(self.qbaf, self.semantics, self.goals)

    def lens_history(self) -> dict[str, list[float]]:
        """Each goal's sigma after every turn, under the current semantics."""
        out: dict[str, list[float]] = {g.id: [] for g in self.goals}
        for _label, graph in self._snapshots:
            for entry in evaluation_lens(graph, self.semantics, self.goals):
                out[entry["id"]].append(entry["sigma"])
        return out

    def strengths(self) -> dict[str, float]:
        return self.semantics.compute(self.qbaf)

    def leverage(self, limit: int = 3) -> list[tuple[Argument, float]]:
        """What would most move this persona -- powers 'what would change your mind?'.

        Pure engine, no LLM call.
        """
        return [
            (self.qbaf.arguments[arg_id], score)
            for arg_id, score in ranked_leverage(self.qbaf, self.semantics, limit)
        ]

    def disagreement_kind(self, arg_id: str) -> str:
        """Whether disagreement at this node is settleable by evidence or not."""
        node_type = self.qbaf.arguments[arg_id].node_type
        if node_type == "normative":
            return "value"
        return "evidential"

    def _snapshot(self) -> QBAF:
        return QBAF.from_dict(self.qbaf.to_dict())

    def _graph_digest(self) -> str:
        lines = []
        for arg_id, arg in self.qbaf.arguments.items():
            role = "TOPIC" if arg_id == self.qbaf.topic_id else arg.node_type
            lines.append(f"[{arg_id}] ({role}) {arg.text}")
            for rel in self.qbaf.relations:
                if rel.source_id == arg_id:
                    lines.append(f"     -> {rel.relation_type}s [{rel.target_id}]")
        return "\n".join(lines)

    # -- scoring -------------------------------------------------------

    async def _score(
        self, text: str, target_id: str, node_type: NodeType, relation_type: str,
        persona: Persona,
    ) -> ScoredArgument:
        return await self.client.parse(
            system_prompt=prompts.score_system_prompt(
                persona, node_type, relation_type
            ),
            user_prompt=prompts.SCORE_USER.format(
                argument=text, target=self.qbaf.arguments[target_id].text
            ),
            response_model=ScoredArgument,
            temperature=0.0,
        )

    def _goal_ids(self) -> set[str]:
        return {g.id for g in self.goals}

    @staticmethod
    def _normalise_id(raw: str) -> str:
        """Models sometimes echo ids with the display brackets; forgive that."""
        return raw.strip().strip("[]").strip()

    def _judged_tags(self, impacts, already: set[str]) -> list[dict]:
        """Validate model-judged goal impacts against the selected set."""
        valid = self._goal_ids()
        tags = []
        for impact in impacts:
            goal_id = self._normalise_id(impact.goal_id)
            if goal_id not in valid:
                logger.warning(
                    "model tagged unknown goal %r; dropping", impact.goal_id
                )
                continue
            if goal_id in already:
                continue
            tags.append(
                {
                    "goal_id": goal_id,
                    "direction": impact.direction,
                    "source": "judged",
                }
            )
            already.add(goal_id)
        return tags

    def _tags(
        self, impacts, requested: tuple[str, str] | None = None
    ) -> list[dict]:
        """Goal tags for an argument.

        When the person's request named a selected goal, that tag is
        stamped first with provenance *user*: the steering was theirs, and
        the tag must not depend on the model's own "do not stretch"
        judgement. The model's judgements fill the remaining goals as
        *judged*, validated against the selected set.
        """
        tags: list[dict] = []
        if requested is not None:
            goal_id, direction = requested
            tags.append({"goal_id": goal_id, "direction": direction, "source": "user"})
        already = {t["goal_id"] for t in tags}
        tags += self._judged_tags(impacts, already)
        return tags

    @staticmethod
    def _requested(attachment: Attachment) -> tuple[str, str] | None:
        if attachment.requested_goal_id is None or attachment.requested_direction is None:
            return None
        return attachment.requested_goal_id, attachment.requested_direction

    async def _add_scored(
        self,
        text: str,
        target_id: str,
        node_type: NodeType,
        relation_type: str,
        provenance: str,
        rationale: str = "",
        goal_impacts: list[dict] | None = None,
        persona: Persona | None = None,
    ) -> Argument:
        persona = persona or self.persona()
        scored = await self._score(text, target_id, node_type, relation_type, persona)
        arg = Argument(
            arg_id=self._new_id("a"),
            text=text,
            node_type=node_type,
            base_score=scored.confidence / 100.0,
            provenance=provenance,  # type: ignore[arg-type]
            depth=self.qbaf.arguments[target_id].depth + 1,
            rationale=scored.rationale,
            goal_impacts=goal_impacts or [],
            elicited_tau=scored.confidence / 100.0,
            author=persona.id,
        )
        self.qbaf.add_argument(arg)
        self.qbaf.add_relation(
            Relation(
                source_id=arg.arg_id,
                target_id=target_id,
                relation_type=relation_type,  # type: ignore[arg-type]
                rationale=rationale,
            )
        )
        return arg

    # -- turns ---------------------------------------------------------

    async def open(self) -> TurnResult:
        """The deliberation's start: neutral, and directed by the user.

        No model call. The claim keeps its neutral base score of 0.5, so the
        stance is nothing but what the arguments on the table make it, and
        every movement on the trajectory is attributable. The partner states
        no view -- a view would contradict its own gauge -- and introduces
        itself from its spec; its reasons are one request away.
        """
        intros = [(p.id, p.introduction()) for p in self.personas.values()]
        for pid, intro in intros:
            self._transcript.append((pid, intro))
        self._record("start")
        return TurnResult(
            delta=explain_change(self.qbaf, self.qbaf, self.semantics),
            intros=intros,
        )

    async def expand(
        self, node_id: str, relation_type: str, persona_id: str | None = None
    ) -> TurnResult:
        """Generate one attacker or supporter of an existing node, on demand,
        by the named persona.

        This is what right-clicking a node and asking a persona triggers --
        the graph deepens because the user asked, not because a tree was
        built up front. The Devil's Advocate only ever objects.
        """
        if node_id not in self.qbaf.arguments:
            raise KeyError(f"unknown node {node_id!r}")
        persona = self.persona(persona_id)
        if persona.kind == "contrarian" and relation_type != "attack":
            raise ValueError(f"{persona.name} only objects; ask it for a reason against")

        before = self._snapshot()
        target = self.qbaf.arguments[node_id]
        context = (
            f'This statement bears on the wider claim: "{self.topic}"\n\n'
            if node_id != self.qbaf.topic_id
            else ""
        )
        # What is already attached to the node, whoever made it: the model
        # sees it so a repeated request brings a new consideration or
        # declines, rather than restating one already on the table.
        existing = [
            (
                "for" if rel.relation_type == "support" else "against",
                self._author_label(self.qbaf.arguments[rel.source_id], persona),
                self.qbaf.arguments[rel.source_id].text,
            )
            for rel in self.qbaf.relations
            if rel.target_id == node_id
        ]

        response: ExpansionResponse = await self.client.parse(
            system_prompt=prompts.generate_system_prompt(
                persona, relation_type, self.goals
            ),
            user_prompt=prompts.GENERATE_USER.format(
                target=target.text,
                context=context,
                existing=prompts.existing_block(existing),
                a_new="a new" if existing else "your",
            ),
            response_model=ExpansionResponse,
        )

        generated = response.argument
        if generated is None:
            # The persona declined -- it has nothing genuine in this direction.
            # The graph is untouched, so no snapshot: an empty `added_nodes` is
            # how callers tell a decline from an expansion.
            return TurnResult(
                delta=explain_change(self.qbaf, self.qbaf, self.semantics),
                speaker=persona.id,
            )

        arg = await self._add_scored(
            text=generated.text,
            target_id=node_id,
            node_type=generated.node_type,
            relation_type=relation_type,
            provenance="generated",
            rationale=generated.rationale,
            goal_impacts=self._tags(generated.goal_impacts),
            persona=persona,
        )

        self._record(f"{persona.name}: {'against' if relation_type == 'attack' else 'for'} {node_id}")
        return TurnResult(
            delta=explain_change(before, self.qbaf, self.semantics),
            added_nodes=[arg.arg_id],
            speaker=persona.id,
        )

    async def propose_attachment(
        self, user_text: str, target_id: str | None = None
    ) -> Attachment:
        """Work out where a free-text contribution belongs. Does not apply it.

        With `target_id` the person chose the node themselves (right-click):
        only the direction, the restatement, the type and the tags are
        inferred, and the confidence is about the direction.
        """
        if target_id is None:
            attachment = await self.classifier.parse(
                system_prompt=prompts.attach_system_prompt(self.goals),
                user_prompt=prompts.ATTACH_USER.format(
                    graph=self._graph_digest(), user_text=user_text
                ),
                response_model=Attachment,
                temperature=0.0,
            )
            return attachment.model_copy(update={"provenance": "user"})
        if target_id not in self.qbaf.arguments:
            raise KeyError(f"unknown node {target_id!r}")
        attachment = await self.classifier.parse(
            system_prompt=prompts.attach_fixed_system_prompt(self.goals),
            user_prompt=prompts.ATTACH_FIXED_USER.format(
                graph=self._graph_digest(),
                target_id=target_id,
                target_text=self.qbaf.arguments[target_id].text,
                user_text=user_text,
            ),
            response_model=Attachment,
            temperature=0.0,
        )
        # The target is the person's choice, whatever the model echoed.
        return attachment.model_copy(update={"provenance": "user", "target_id": target_id})

    async def _route(self, user_text: str) -> RoutedMessage:
        """Stage 1: classify the message. One tiny job, nothing else."""
        return await self.classifier.parse(
            system_prompt=prompts.ROUTER_SYSTEM,
            user_prompt=prompts.ROUTER_USER.format(
                topic=self.topic, user_text=user_text
            ),
            response_model=RoutedMessage,
            temperature=0.0,
        )

    def _facts_block(self) -> str:
        """The computed state, written by the engine for the answer prompt."""
        strengths = self.strengths()
        history = self.history()
        lines = [
            f"- Your current stance on the claim: {strengths[self.qbaf.topic_id]:.0%}."
        ]
        if history:
            lines.append(
                f"- The deliberation started neutral at {history[0]['stance']:.0%}; "
                "your stance is only what the arguments on the table make it."
            )
            lines.append(f'- The last change came from: {history[-1]["label"]}.')
        lines.append(
            f"- Arguments on the table: {len(self.qbaf.arguments) - 1}."
        )
        for entry in self.lens():
            lines.append(
                f"- Goal evaluation, {entry['label']}: {entry['verdict']} "
                f"(sigma {entry['sigma']:.2f})."
            )
        lines.append(
            "- The graph (tau = your base score, sigma = computed strength):"
        )
        for arg_id, arg in self.qbaf.arguments.items():
            role = "the claim" if arg_id == self.qbaf.topic_id else arg.node_type
            if arg.provenance == "user":
                role += ", made by the person"
            elif arg.author in self.personas:
                role += f", made by {self.personas[arg.author].name}"
            relations = "".join(
                f", {r.relation_type}s [{r.target_id}]"
                for r in self.qbaf.relations
                if r.source_id == arg_id
            )
            extras = ""
            if arg.goal_impacts:
                tags = "; ".join(
                    f"{'helps' if t['direction'] == 'positive' else 'strains'} "
                    f"{t['goal_id']}"
                    for t in arg.goal_impacts
                )
                extras += f" (bears on goals: {tags})"
            lines.append(
                f'  [{arg_id}] ({role}, tau={arg.base_score:.2f}, '
                f'sigma={strengths[arg_id]:.2f}{relations}) '
                f'"{arg.text}"{extras}'
            )
        return "\n".join(lines)

    async def _answer(self, user_text: str, persona: Persona) -> TurnResult:
        """Kind "other": the addressed persona answers conversationally from
        computed facts only.

        The graph is untouched, so no snapshot; the numbers in the reply come
        from the engine, not the model.
        """
        answer: StateAnswer = await self.client.parse(
            system_prompt=prompts.ANSWER_SYSTEM.format(
                commitments=persona.commitments_block(),
                voice=persona.voice_block(),
            ),
            user_prompt=prompts.ANSWER_USER.format(
                topic=self.topic, facts=self._facts_block(), user_text=user_text
            ),
            response_model=StateAnswer,
        )
        self._transcript.append(("user", user_text))
        self._transcript.append((persona.id, answer.message))
        return TurnResult(
            delta=explain_change(self.qbaf, self.qbaf, self.semantics),
            reply=answer.message,
            speaker=persona.id,
        )

    async def contribute(
        self,
        user_text: str,
        attachment: Attachment | None = None,
        force: str | None = None,
        confidence: str | None = None,
        persona_id: str | None = None,
        target_id: str | None = None,
    ) -> TurnResult:
        """Handle the user's free text: an argument of theirs, or a request.

        A neutral router reads the message first. A request ("give an
        argument that regresses SDG 1") reroutes to fulfilment; an argument
        goes to the neutral attachment call and then *always* comes back as
        `pending`, graph untouched: the person sees how their point was
        read and where it attaches, may redirect it, and states their own
        confidence in it -- the point cannot enter the graph without a
        confidence, and only the person can give one. The second call, with
        `attachment` and `confidence`, applies it. `force` ("argument" |
        "request") skips the router after the user has resolved an unclear
        intent. A requested argument is the persona's, so it needs no
        confidence from the user and pends only when its placement is shaky
        (below ATTACH_CONFIRM_THRESHOLD) or its target is gone.
        """
        persona = self.persona(persona_id)
        if attachment is None:
            if target_id is not None:
                # Right-click on a node: the point is theirs and the node is
                # chosen, so no routing -- straight to the direction check.
                return await self._propose(user_text, target_id)
            if force == "request":
                return await self._fulfil(user_text, user_text, persona)
            if force == "argument":
                return await self._propose(user_text)
            route = await self._route(user_text)
            if route.kind == "other":
                # No graph risk either way, so no gate: worst case is a
                # conversational answer the user can rephrase past.
                return await self._answer(user_text, persona)
            if route.kind in ("request", "mixed"):
                request = route.request_text or user_text
                if route.confidence < ATTACH_CONFIRM_THRESHOLD:
                    return TurnResult(
                        delta=explain_change(self.qbaf, self.qbaf, self.semantics),
                        pending_intent=request,
                    )
                if route.kind == "mixed":
                    # The argument half pends like any argument; the request
                    # half rides along and the caller runs it after the card
                    # resolves, so neither half is ever silently dropped.
                    turn = await self._propose(route.argument_text or user_text)
                    turn.followup_request = request
                    return turn
                return await self._fulfil(user_text, request, persona)
            return await self._propose(route.argument_text or user_text)

        if attachment.target_id not in self.qbaf.arguments:
            # The target was removed while confirmation was pending.
            # Mis-attachment inverts the computation, so a repaired
            # attachment is never applied silently: anchor it to the claim
            # and ask again.
            return TurnResult(
                delta=explain_change(self.qbaf, self.qbaf, self.semantics),
                pending=self._reanchored(attachment),
            )

        confidence = confidence or DEFAULT_CONFIDENCE
        if confidence not in CONFIDENCE_TAU:
            raise ValueError(f"unknown confidence level {confidence!r}")
        # A confirmed requested argument is scored by the persona that
        # drafted it, carried on the attachment; a user point needs none.
        author = self.persona(attachment.author) if attachment.author else persona
        return await self._apply_attachment(user_text, attachment, confidence, author)

    async def _propose(self, argument_text: str, target_id: str | None = None) -> TurnResult:
        """Place the user's point and hand it back for confirmation and
        their confidence. The graph is untouched."""
        attachment = await self.propose_attachment(argument_text, target_id)
        if attachment.target_id not in self.qbaf.arguments:
            attachment = self._reanchored(attachment)
        return TurnResult(
            delta=explain_change(self.qbaf, self.qbaf, self.semantics),
            pending=attachment,
        )

    def _reanchored(self, attachment: Attachment) -> Attachment:
        """An attachment whose target is not in the graph, anchored to the
        claim with zero confidence so the card says so."""
        return attachment.model_copy(
            update={
                "target_id": self.qbaf.topic_id,
                "confidence": 0,
                "rationale": (
                    "The argument this was meant to attach to is not in "
                    "the graph any more, so I can only anchor it to the "
                    "claim itself."
                ),
            }
        )

    async def _apply_attachment(
        self, user_text: str, attachment: Attachment, confidence: str, persona: Persona
    ) -> TurnResult:
        """Add an attached point and recompute.

        The user's own point enters at the confidence they gave it and draws
        no response: in the controlled interface personas argue only when
        asked, so the turn is silent apart from the computed movement. A
        requested argument is the persona's, scored by it, and the persona
        replies with it.
        """
        before = self._snapshot()
        if attachment.provenance == "user":
            arg = self._add_user_argument(attachment, CONFIDENCE_TAU[confidence])
            self._record("your argument")
            self._transcript.append(("user", user_text))
            delta = explain_change(before, self.qbaf, self.semantics)
            return TurnResult(
                delta=delta,
                steps=[("your argument", delta)],
                added_nodes=[arg.arg_id],
            )

        arg = await self._add_scored(
            text=attachment.restated,
            target_id=attachment.target_id,
            node_type=attachment.node_type,
            relation_type=attachment.relation_type,
            provenance=attachment.provenance,
            rationale=attachment.rationale,
            goal_impacts=self._tags(attachment.goal_impacts, self._requested(attachment)),
            persona=persona,
        )
        delta = explain_change(before, self.qbaf, self.semantics)
        self._record(f"{persona.name}: requested argument")
        strategy = self._choose_strategy(delta, attachment.node_type)
        reply = await self._react(user_text, summarise_delta(delta), strategy, persona)
        if reply.concedes:
            self._concessions.append(arg.arg_id)
        self._transcript.append(("user", user_text))
        self._transcript.append((persona.id, f"{reply.message} {reply.question}".strip()))
        return TurnResult(
            delta=delta,
            reply=reply.message,
            concedes=reply.concedes,
            added_nodes=[arg.arg_id],
            question=reply.question,
            speaker=persona.id,
        )

    def _add_user_argument(self, attachment: Attachment, tau: float) -> Argument:
        """Add the user's own point at the confidence they gave it.

        No scoring call: the persona is never asked what it thinks of the
        user's argument as a number. `elicited_tau` stays None, so the
        slider-divergence machinery (which compares a persona's score to a
        user override) does not apply to the user's own nodes.
        """
        arg = Argument(
            arg_id=self._new_id("a"),
            text=attachment.restated,
            node_type=attachment.node_type,
            base_score=tau,
            provenance="user",
            depth=self.qbaf.arguments[attachment.target_id].depth + 1,
            goal_impacts=self._tags(attachment.goal_impacts),
        )
        self.qbaf.add_argument(arg)
        self.qbaf.add_relation(
            Relation(
                source_id=arg.arg_id,
                target_id=attachment.target_id,
                relation_type=attachment.relation_type,  # type: ignore[arg-type]
                rationale=attachment.rationale,
            )
        )
        return arg

    def _author_name(self, arg: Argument) -> str:
        p = self.personas.get(arg.author or "")
        return p.name if p else "a persona"

    def _author_label(self, arg: Argument, speaker: Persona) -> str:
        """Who made `arg`, from `speaker`'s side of the table: "you" for its
        own, "the user" for the person's, another persona by name."""
        if arg.provenance == "user":
            return "the user"
        if arg.author == speaker.id:
            return "you"
        return self._author_name(arg)

    def _speaker_name(self, who: str) -> str:
        p = self.personas.get(who)
        return p.name if p else who

    def _relation_of(self, arg_id: str) -> str:
        return next(r.relation_type for r in self.qbaf.relations if r.source_id == arg_id)

    async def _fulfil(self, user_text: str, request: str, persona: Persona) -> TurnResult:
        """Fulfil a request for an argument, honestly.

        The user's explicit request may steer generation (a sanctioned
        exception to goal-blind prompts: the steering is theirs, and it stays
        in the transcript). The produced argument is what they asked for and
        is not re-confirmed; its *placement* is fallible inference and goes
        through the usual gate. A selected goal the request named is
        carried on the attachment, so the argument is tagged with it by
        code whatever the model chose to tag.
        """
        response: FulfilledRequest = await self.client.parse(
            system_prompt=prompts.fulfil_system_prompt(persona, self.goals),
            user_prompt=prompts.FULFIL_USER.format(
                topic=self.topic, graph=self._graph_digest(), request=request
            ),
            response_model=FulfilledRequest,
        )

        generated = response.argument
        if generated is None:
            reply = response.note or (
                "I have no genuinely valid argument that satisfies that request."
            )
            self._transcript.append(("user", user_text))
            self._transcript.append((persona.id, reply))
            return TurnResult(
                delta=explain_change(self.qbaf, self.qbaf, self.semantics),
                reply=reply,
                speaker=persona.id,
            )

        requested_goal_id = None
        if response.requested_goal_id is not None:
            requested_goal_id = self._normalise_id(response.requested_goal_id)
            if requested_goal_id not in self._goal_ids() or response.requested_direction is None:
                logger.warning(
                    "model named goal %r for the request; not a selected goal, ignoring",
                    response.requested_goal_id,
                )
                requested_goal_id = None

        target_id = self._normalise_id(response.target_id)
        target_ok = target_id in self.qbaf.arguments
        attachment = Attachment(
            target_id=target_id if target_ok else self.qbaf.topic_id,
            relation_type=response.relation_type,
            node_type=generated.node_type,
            confidence=response.placement_confidence if target_ok else 0,
            rationale=response.placement_rationale
            if target_ok
            else (
                "The node this was meant to attach to is not in the graph, "
                "so I can only anchor it to the claim itself."
            ),
            restated=generated.text,
            goal_impacts=generated.goal_impacts,
            provenance="generated",
            author=persona.id,
            requested_goal_id=requested_goal_id,
            requested_direction=response.requested_direction if requested_goal_id else None,
        )

        if not target_ok or attachment.confidence < ATTACH_CONFIRM_THRESHOLD:
            return TurnResult(
                delta=explain_change(self.qbaf, self.qbaf, self.semantics),
                pending=attachment,
            )

        return await self._apply_attachment(user_text, attachment, DEFAULT_CONFIDENCE, persona)

    def set_base_score(self, node_id: str, value: float) -> DeltaExplanation:
        """User overrides a base score via the slider. Pure engine, no LLM call."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("base score must be in [0, 1]")
        before = self._snapshot()
        self.qbaf.arguments[node_id].base_score = value
        self._record(f"you re-scored {node_id}")
        return explain_change(before, self.qbaf, self.semantics)

    def set_goal_tags(self, node_id: str, tags: list[dict]) -> DeltaExplanation:
        """User contests an argument's goal tags. Pure engine, no LLM call.

        The client sends the node's full desired tag list. Tags kept unchanged
        retain their original provenance; added or redirected ones are marked
        source "user" -- the tag record stays honest about who asserted what.
        The stance is untouched (tags never enter it); the lens recomputes.
        """
        if node_id not in self.qbaf.arguments:
            raise KeyError(f"unknown node {node_id!r}")
        valid = self._goal_ids()
        existing = {
            t["goal_id"]: t for t in self.qbaf.arguments[node_id].goal_impacts
        }
        new_tags: list[dict] = []
        seen: set[str] = set()
        for tag in tags:
            goal_id, direction = tag.get("goal_id"), tag.get("direction")
            if goal_id not in valid:
                raise ValueError(f"unknown goal {goal_id!r}")
            if direction not in ("positive", "negative"):
                raise ValueError(f"bad direction {direction!r}")
            if goal_id in seen:
                continue
            seen.add(goal_id)
            prev = existing.get(goal_id)
            if prev is not None and prev["direction"] == direction:
                new_tags.append(prev)
            else:
                new_tags.append(
                    {"goal_id": goal_id, "direction": direction, "source": "user"}
                )
        before = self._snapshot()
        self.qbaf.arguments[node_id].goal_impacts = new_tags
        self._record(f"you re-tagged {node_id}")
        return explain_change(before, self.qbaf, self.semantics)

    def remove(self, node_id: str) -> DeltaExplanation:
        """User rejects an argument outright."""
        if node_id == self.qbaf.topic_id:
            raise ValueError("cannot remove the topic")
        before = self._snapshot()
        self.qbaf.remove_argument(node_id)
        self._record(f"you removed {node_id}")
        return explain_change(before, self.qbaf, self.semantics)

    #: Below this, the stance "barely moved" and the reply's closing question
    #: turns agonistic rather than following up. Matches summarise_delta's
    #: honesty threshold.
    _STANCE_MOVED = 0.02

    def _choose_strategy(self, delta: DeltaExplanation, node_type: NodeType) -> str:
        """Pick the questioning strategy from the computed situation.

        The code decides, the LLM words it: follow-up when the user's point
        landed (or the persona reinforced it), perspective-taking when the
        standoff is value-based, agonistic otherwise.
        """
        if abs(delta.topic_delta) >= self._STANCE_MOVED:
            return "follow_up"
        if node_type == "normative":
            return "perspective_taking"
        return "agonistic"

    def _goal_note(self) -> str:
        """A hint for perspective-taking questions when the lenses pull apart.

        Derived from the computed lens verdicts, not from raw tags, so the
        question can never contradict what the evaluation shows.
        """
        lens = self.lens()
        supported = [e["label"] for e in lens if e["verdict"] == "supported"]
        undermined = [e["label"] for e in lens if e["verdict"] == "undermined"]
        if not supported or not undermined:
            return ""
        return (
            f"If useful: as the deliberation stands, the case for "
            f"{', '.join(supported)} is supported while the case for "
            f"{', '.join(undermined)} is undermined -- a tension between "
            f"goals can be the complication."
        )

    async def _react(
        self, user_text: str, brief: str, strategy: str, persona: Persona
    ) -> AgentReply:
        """The agent's NL turn, constrained by the computed change.

        The persona describes a stance change it did not choose -- the number
        comes from the semantics. This is the point where the architecture earns
        its keep: the agent cannot claim to be unmoved by an argument that
        provably moved it. The closing question follows the strategy the code
        chose from that same computed situation.
        """
        sdg_note = self._goal_note() if strategy == "perspective_taking" else ""
        return await self.client.parse(
            system_prompt=prompts.REPLY_SYSTEM.format(
                commitments=persona.commitments_block(),
                voice=persona.voice_block(),
            ),
            user_prompt=prompts.REPLY_USER.format(
                topic=self.topic,
                user_text=user_text,
                delta_summary=brief,
                strategy_instruction=prompts.strategy_instruction(strategy, sdg_note),
            ),
            response_model=AgentReply,
        )

    async def summarise(self) -> dict:
        """Meeting notes for the deliberation so far: one call by the neutral
        note-taker (the classifier client, not the persona), worded from the
        computed record only. Cached while the state is unchanged.
        """
        rep = self.report()
        key = f"{len(self._snapshots)}:{len(self._transcript)}:{self._graph_digest()}"
        if self._notes_cache is not None and self._notes_cache[0] == key:
            notes = self._notes_cache[1]
        else:
            strengths = self.strengths()
            graph = "\n".join(
                f"[{arg_id}] ({'the claim' if arg_id == self.qbaf.topic_id else arg.node_type}, "
                f"{'made by the person' if arg.provenance == 'user' else 'made by ' + self._author_name(arg)}"
                f", tau={arg.base_score:.2f}, "
                f"sigma={strengths[arg_id]:.2f}"
                + "".join(
                    f", {r.relation_type}s [{r.target_id}]"
                    for r in self.qbaf.relations
                    if r.source_id == arg_id
                )
                + f') "{arg.text}"'
                for arg_id, arg in self.qbaf.arguments.items()
            )
            parsed: MeetingNotes = await self.classifier.parse(
                system_prompt=prompts.SUMMARY_SYSTEM,
                user_prompt=prompts.SUMMARY_USER.format(
                    topic=self.topic,
                    partner=", ".join(f"{p.name} ({p.kind})" for p in self.personas.values()),
                    trajectory="\n".join(
                        f"- {h['label']}: {h['stance']:.0%}" for h in rep["history"]
                    ),
                    graph=graph,
                    transcript="\n".join(
                        f"- {'the person' if who == 'user' else self._speaker_name(who)}: {text}"
                        for who, text in self._transcript
                    ),
                    identification=_identification_block(rep),
                ),
                response_model=MeetingNotes,
                temperature=0.0,
            )
            notes = parsed.model_dump()
            self._notes_cache = (key, notes)
        return {
            "topic": self.topic,
            "personas": rep["personas"],
            "opening_stance": rep["opening_stance"],
            "stance": rep["stance"],
            "turns": sum(1 for who, _ in self._transcript if who == "user"),
            "arguments": len(self.qbaf.arguments) - 1,
            "history": rep["history"],
            "lens": [
                {k: e[k] for k in ("id", "label", "verdict", "sigma")}
                for e in rep["lens"]
            ],
            "notes": notes,
        }

    # -- identification & documentation --------------------------------

    #: A slider override this far from the agent's elicited tau counts as an
    #: active, recorded disagreement.
    _DIVERGENCE = 0.15
    #: A user's point held at or above this (the "very sure" level) that the
    #: agent objected to counts as a recorded disagreement.
    _FIRMLY_HELD = CONFIDENCE_TAU["very"]

    def report(self) -> dict:
        """The deliberation record: identification + documentation, no LLM.

        Agreements are points the agent conceded or reinforced with an
        argument of its own; divergences are the agent's nodes where the
        user's slider override stands far from its elicited tau, and the
        user's points the agent objected to while the user held them
        firmly -- both split by whether evidence or values could settle
        them; trade-offs come from the leverage ranking. Everything here is
        computed, so the record makes no claim the arithmetic cannot back.
        """
        strengths = self.strengths()
        q = self.qbaf

        # The persona's structural responses: user node -> (response, relation).
        responses: dict[str, tuple[Argument, str]] = {}
        for rel in q.relations:
            source, target = q.arguments.get(rel.source_id), q.arguments.get(rel.target_id)
            if (
                source is not None
                and target is not None
                and target.provenance == "user"
                and source.provenance == "generated"
            ):
                responses.setdefault(target.arg_id, (source, rel.relation_type))

        agreements = []
        for arg_id in self._concessions:
            if arg_id in q.arguments:
                agreements.append(
                    {
                        "id": arg_id,
                        "text": q.arguments[arg_id].text,
                        "basis": "the agent conceded this point",
                    }
                )
        conceded = {a["id"] for a in agreements}
        for arg_id, (response, relation) in responses.items():
            if relation == "support" and arg_id not in conceded:
                agreements.append(
                    {
                        "id": arg_id,
                        "text": q.arguments[arg_id].text,
                        "basis": (
                            f"the agent reinforced your point with "
                            f"{response.arg_id}"
                        ),
                    }
                )

        divergences = []
        for arg in q.arguments.values():
            if arg.elicited_tau is None:
                continue
            gap = arg.base_score - arg.elicited_tau
            if abs(gap) >= self._DIVERGENCE:
                divergences.append(
                    {
                        "id": arg.arg_id,
                        "text": arg.text,
                        "agent_tau": arg.elicited_tau,
                        "your_tau": arg.base_score,
                        "kind": self.disagreement_kind(arg.arg_id),
                        "basis": (
                            f"the agent holds this at {arg.elicited_tau:.0%}, "
                            f"you set it to {arg.base_score:.0%}"
                        ),
                    }
                )
        for arg_id, (response, relation) in responses.items():
            arg = q.arguments[arg_id]
            if relation == "attack" and arg.base_score >= self._FIRMLY_HELD:
                divergences.append(
                    {
                        "id": arg_id,
                        "text": arg.text,
                        "agent_tau": None,
                        "your_tau": arg.base_score,
                        "kind": self.disagreement_kind(arg_id),
                        "basis": (
                            f"you hold this at {arg.base_score:.0%}; the agent "
                            f"objected with {response.arg_id}"
                        ),
                    }
                )

        trade_offs = [
            {
                "id": arg.arg_id,
                "text": arg.text,
                "sensitivity": sensitivity,
                "kind": self.disagreement_kind(arg.arg_id),
            }
            for arg, sensitivity in self.leverage(limit=3)
        ]

        history = self.history()
        return {
            "topic": self.topic,
            "personas": [
                {"id": p.id, "name": p.name, "kind": p.kind}
                for p in self.personas.values()
            ],
            "opening_stance": history[0]["stance"] if history else None,
            "stance": strengths[q.topic_id],
            "history": history,
            "agreements": agreements,
            "divergences": divergences,
            "trade_offs": trade_offs,
            "lens": self.lens(),
            "lens_history": self.lens_history(),
            "transcript": [
                {"who": who, "name": "You" if who == "user" else self._speaker_name(who), "text": text}
                for who, text in self._transcript
            ],
            "semantics": self.semantics.name,
            "conservativeness": self.conservativeness,
        }


def _identification_block(rep: dict) -> str:
    lines = []
    for a in rep["agreements"]:
        lines.append(f"- agreement [{a['id']}]: {a['basis']}")
    for d in rep["divergences"]:
        lines.append(f"- divergence [{d['id']}] ({d['kind']}): {d['basis']}")
    for t in rep["trade_offs"]:
        lines.append(
            f"- trade-off [{t['id']}]: sensitivity {t['sensitivity']:+.2f} "
            f"({t['kind']}) -- where arguing would move the position most"
        )
    for e in rep["lens"]:
        lines.append(
            f"- goal {e['label']}: {e['verdict']} (sigma {e['sigma']:.2f})"
        )
    return "\n".join(lines) or "- nothing computed yet"


def summarise_delta(delta: DeltaExplanation) -> str:
    """Render a delta as the factual brief the agent must speak consistently with."""
    lines = [
        f"Your confidence in the claim went from "
        f"{delta.topic_before:.0%} to {delta.topic_after:.0%} "
        f"({delta.direction}, change of {delta.topic_delta:+.0%})."
    ]
    driver = delta.primary_driver()
    if driver is not None:
        if driver.status == "added":
            lines.append(
                f'The new argument was: "{driver.text}" '
                f"(it holds at {driver.after:.0%} strength)."
            )
        elif driver.delta:
            lines.append(
                f'The argument most affected was: "{driver.text}", '
                f"moving from {driver.before:.0%} to {driver.after:.0%}."
            )
    if abs(delta.topic_delta) < 0.02:
        lines.append(
            "Your overall position barely moved -- be honest about why rather "
            "than overstating either agreement or resistance."
        )
    return "\n".join(lines)

"""Structured response models for every LLM operation.

Everything the model returns is parsed into one of these via OpenAI structured
outputs, so no stage of the pipeline parses prose. A malformed response is an
API-level error we can retry, not a silent mis-parse that corrupts the graph.

Base scores come back as integers 0-100 rather than floats, matching the
ArgLLM prompt ("Your response should be between 0% and 100%") and avoiding
float-formatting drift in model output. The pipeline divides by 100.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GoalImpact(BaseModel):
    """One of the deliberation's selected goals an argument bears on."""

    goal_id: str = Field(
        description=(
            "The goal's id exactly as it appears between the brackets in the "
            "goals list (e.g. sdg_1), without the brackets. Only ids from "
            "that list are valid."
        )
    )
    direction: Literal["positive", "negative"] = Field(
        description="Whether what the argument advocates would help or strain this goal."
    )


class GeneratedArgument(BaseModel):
    """One argument generated for or against a parent node."""

    text: str = Field(description="The argument, stated in one or two sentences.")
    node_type: Literal["empirical", "predictive", "normative"] = Field(
        description=(
            "empirical: a claim about what is currently the case, settleable by "
            "evidence. predictive: a claim about what will happen, uncertain but "
            "evidence-bearing. normative: a claim about what matters or what "
            "ought to be done, resting on values."
        )
    )
    rationale: str = Field(
        description="One sentence on why this bears on the parent claim."
    )
    goal_impacts: list[GoalImpact] = Field(
        default_factory=list,
        description=(
            "Which of the deliberation's selected goals (if any were listed) "
            "this argument's substance would clearly affect, with direction. "
            "At most three; empty if none is clearly implicated or no goals "
            "were listed. Do not stretch."
        ),
    )


class ExpansionResponse(BaseModel):
    """The response to an expand request, which the persona may decline.

    `GENERATE_SYSTEM` says "do not pad"; this null slot is what makes that
    instruction followable. Without it, a required `text` field would force
    the model to manufacture an argument it was just told not to manufacture
    -- the same escape hatch a declined request has.
    """

    argument: GeneratedArgument | None = Field(
        default=None,
        description=(
            "The argument, or null if there is no genuinely valid and "
            "convincing argument in the requested direction. Null is the "
            "honest answer; do not pad."
        ),
    )


class ScoredArgument(BaseModel):
    """A base score for one argument, elicited in the context of its parent."""

    confidence: int = Field(
        ge=0,
        le=100,
        description="0 = the argument is entirely uncompelling, 100 = decisive.",
    )
    rationale: str = Field(
        description="One sentence justifying this score, in the persona's voice."
    )


class Attachment(BaseModel):
    """Where a free-text contribution belongs in the graph.

    `confidence` is deliberately surfaced: attachment inference can flip an
    argument's polarity and silently invert the whole computation, so a
    low-confidence attachment must be confirmed by the user rather than applied.
    """

    target_id: str = Field(
        description="The arg_id of the existing node this bears on most directly."
    )
    relation_type: Literal["attack", "support"] = Field(
        description="Whether it undermines or reinforces the target."
    )
    node_type: Literal["empirical", "predictive", "normative"]
    confidence: int = Field(
        ge=0, le=100, description="How clear-cut this attachment is."
    )
    rationale: str = Field(
        description="One sentence on why it attaches there, for user confirmation."
    )
    restated: str = Field(
        description=(
            "The user's point restated as a single standalone argument, "
            "preserving their meaning. This becomes the node text."
        )
    )
    goal_impacts: list[GoalImpact] = Field(
        default_factory=list,
        description=(
            "Which of the deliberation's selected goals (if any were listed) "
            "the point's substance would clearly affect, with direction. At "
            "most three; empty if none is clearly implicated or no goals "
            "were listed. Do not stretch."
        ),
    )
    #: Who authored the point: the person ("user") or the agent at the
    #: person's request ("generated"). Set by the server, never by the model;
    #: it rides on the schema so a confirmed placement keeps its authorship.
    provenance: Literal["user", "generated"] = "user"
    #: For a generated (requested) argument: the id of the persona that
    #: drafted it, so a confirmed placement is scored by the same persona.
    #: Set by the server, never by the model.
    author: str | None = None
    #: For a requested argument: the selected goal the request named and
    #: the direction asked for, so a confirmed placement still carries the
    #: tag the person asked for. Set by the server, never by the model.
    requested_goal_id: str | None = None
    requested_direction: Literal["positive", "negative"] | None = None


class RoutedMessage(BaseModel):
    """Stage 1 of understanding a free-text message: classification only.

    One tiny job, so it stays reliable as the system grows: everything else
    (placement, restatement, tagging, fulfilment) happens in dedicated
    stage-2 calls.
    """

    kind: Literal["argument", "request", "other", "mixed"] = Field(
        description=(
            "argument: the message states a point of the person's own about "
            "the subject matter; it could stand as a reason in the debate. "
            "request: the message directs the agent to do something -- "
            "produce or construct an argument, take or argue a side, explore "
            "a direction or a goal, offer an alternative. "
            "other: neither -- a question about the agent's position or the "
            "state of the deliberation, an acknowledgment, or other "
            "conversation. "
            "mixed: the message does more than one of the above at once."
        )
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description=(
            "How clear this reading is. Low when the message could be read "
            "more than one way; it will be checked with the person."
        ),
    )
    argument_text: str | None = Field(
        default=None,
        description=(
            "For argument and mixed: the part of the message that states "
            "the person's own point, kept in their words. Null otherwise."
        ),
    )
    request_text: str | None = Field(
        default=None,
        description=(
            "For request and mixed: the directive restated crisply, "
            "preserving every constraint they gave. Null otherwise."
        ),
    )


class StateAnswer(BaseModel):
    """A conversational answer grounded only in the computed state."""

    message: str = Field(
        description=(
            "The persona's reply, one to three sentences, answering only "
            "from the computed facts it was given. No invented figures."
        )
    )


class FulfilledRequest(BaseModel):
    """The agent's answer to a user request for an argument.

    The argument may honestly be declined. Placement is proposed, not
    decided: below the confirmation threshold it is checked with the user
    like any other fallible placement.
    """

    argument: GeneratedArgument | None = Field(
        default=None,
        description=(
            "The single strongest argument honestly satisfying the request, "
            "or null if no genuinely valid one exists. Do not fabricate to "
            "please the request."
        ),
    )
    target_id: str = Field(
        description="The existing node the argument bears on most directly."
    )
    relation_type: Literal["attack", "support"] = Field(
        description="Whether it undermines or reinforces that target."
    )
    placement_confidence: int = Field(
        ge=0, le=100, description="How clear-cut the placement is."
    )
    placement_rationale: str = Field(
        description="One sentence on why it attaches there, for user confirmation."
    )
    requested_goal_id: str | None = Field(
        default=None,
        description=(
            "If the request named one of the deliberation's listed goals, "
            "that goal's id exactly as it appears between the brackets, "
            "without the brackets. Null if the request named no listed goal."
        ),
    )
    requested_direction: Literal["positive", "negative"] | None = Field(
        default=None,
        description=(
            "With requested_goal_id: whether the request asked for an "
            "argument that the goal is advanced (positive) or undermined "
            "(negative). Null otherwise."
        ),
    )
    note: str = Field(
        default="",
        description=(
            "One sentence to the person: why you declined, or any honest "
            "caveat about the argument you produced. Empty if none."
        ),
    )


class MeetingNotes(BaseModel):
    """The deliberation summarised as meeting notes, by a neutral note-taker.

    Written from engine facts only (trajectory, graph, computed agreements,
    divergences, trade-offs, goal verdicts). Bullets cite argument ids in
    square brackets so the page can link them to the graph.
    """

    summary: str = Field(
        description=(
            "Two to four sentences: what was deliberated, how the discussion "
            "went, and where the partner's position ended up and why. Third "
            "person, neutral, past tense."
        )
    )
    key_points: list[str] = Field(
        description=(
            "The main points raised, in order, one bullet each, naming who "
            "raised it (the person, or the partner by name) and citing the "
            "argument id in square brackets. At most eight."
        )
    )
    agreements: list[str] = Field(
        default_factory=list,
        description="Where the two sides converged, one bullet each, with ids. Empty if none.",
    )
    disagreements: list[str] = Field(
        default_factory=list,
        description=(
            "Where they still diverge, one bullet each, with ids, saying "
            "whether evidence could settle it or it rests on values. Empty if none."
        ),
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description=(
            "What would most move the discussion, from the computed "
            "trade-offs and the goal verdicts: one to three bullets, with ids."
        ),
    )


class AgentReply(BaseModel):
    """The persona's conversational turn after the graph changed."""

    message: str = Field(
        description=(
            "The persona's reply in its own voice. It must reflect the computed "
            "stance change it is given, not invent its own."
        )
    )
    concedes: bool = Field(
        description="True if the persona accepts the user's point as landing."
    )
    question: str = Field(
        description=(
            "Exactly one closing question to the person, written with the "
            "questioning strategy you were given. A question, not a statement."
        )
    )



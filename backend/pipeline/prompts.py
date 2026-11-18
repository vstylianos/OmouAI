"""Prompt templates.

Argument generation and base-score elicitation are adapted from the prompts
that performed best in Freedman et al. (AAAI 2025) -- the OPRO argument miner
and the "analyst" uncertainty estimator (their Figures 16 and 15) -- with two
changes.

First, they are persona-conditioned: the persona's commitments and evidence
standards are supplied, its voice is not.

Second, base-score elicitation routes on node type. The original prompt grounds
scoring in "the correctness, accuracy and truthfulness of the given argument",
which is right for a factual claim and wrong for a value claim. Asking whether
"we owe future generations equal moral consideration" is *true* invites either
a category error or a dodge. So normative nodes are scored for endorsement
under the persona's stated commitments instead.
"""

from __future__ import annotations

from engine.qbaf import NodeType
from personas.schema import Persona

# --- shared blocks ----------------------------------------------------------

from engine.lens import SDG_NAMES, Goal  # noqa: E402


def goal_block(goals: tuple[Goal, ...]) -> str:
    """Tagging vocabulary appended to argument-producing prompts.

    Lists only the deliberation's *selected* goals: generation stays
    goal-blind (the goals never steer which arguments to construct), but
    every argument declares which selected goals its substance bears on.
    Empty selection, empty block.
    """
    if not goals:
        return ""
    rows = "\n".join(f"- [{g.id}] {g.label}" for g in goals)
    return (
        "\n\nThe deliberation is measured against these goals:\n\n"
        f"{rows}\n\n"
        "For each argument you provide, also list which of these goals its "
        "substance would clearly affect, and in which direction. Return each "
        "goal's id exactly as it appears between the brackets (for example "
        "sdg_1), without the brackets themselves. Only ids from this list "
        "are valid. This must not steer which argument you construct -- "
        "argue on the merits, then tag. At most three; leave the list empty "
        "if no listed goal is clearly implicated. Do not stretch."
    )

# --- argument generation (Gamma) -------------------------------------------

GENERATE_SYSTEM = """{commitments}

You are constructing one argument in an ongoing debate. Provide a single short \
argument {polarity_phrase} the statement you are given.

This direction is relative to THE STATEMENT, so give your honest argument \
about it.

You may be shown the arguments already made on this statement, by you or by \
others at the table. Do not repeat or rephrase any of them: your argument must \
bring a consideration that is not already on the table. If every genuine \
argument in the requested direction has already been made, return null.

Only provide an argument if you think there is a genuinely valid and convincing \
one available. If there is not, return null for the argument.

If you do provide one, classify it:
- empirical: a claim about what is currently the case, settleable by evidence
- predictive: a claim about what will happen, uncertain but evidence-bearing
- normative: a claim about what matters or ought to be done, resting on values"""

GENERATE_USER = """Statement: "{target}"

{context}{existing}Now construct {a_new} argument."""


def existing_block(rows: list[tuple[str, str, str]]) -> str:
    """The arguments already attached to the target, as (direction, author,
    text) rows -- the context that keeps a repeated request from repeating
    an argument. Empty when nothing is attached yet."""
    if not rows:
        return ""
    lines = "\n".join(f'- {direction}, by {author}: "{text}"' for direction, author, text in rows)
    return (
        "Arguments already made on this statement, by you or by others at the table:\n"
        f"{lines}\n\n"
    )


def generate_system_prompt(
    persona: Persona,
    relation_type: str,
    goals: tuple[Goal, ...] = (),
) -> str:
    """The generation prompt, with the tagging vocabulary when goals are selected."""
    return GENERATE_SYSTEM.format(
        commitments=persona.commitments_block(),
        polarity_phrase=polarity_phrase(relation_type),
    ) + goal_block(goals)


def attach_system_prompt(goals: tuple[Goal, ...] = ()) -> str:
    """Attachment inference plus, when goals are selected, the tagging block."""
    return ATTACH_SYSTEM + goal_block(goals)


# --- stage-1 router -----------------------------------------------------------

#: One tiny job: classify the message. Everything else (placement,
#: restatement, tagging, fulfilment, answering) happens in dedicated stage-2
#: calls, so this prompt stays frozen as the system grows -- piling jobs into
#: one prompt degraded all of them.
ROUTER_SYSTEM = """\
You route messages in a deliberation app, where a person deliberates a claim \
with an agent. Read the person's message and classify what it is doing. \
Classify only; do not act on the message.

- "argument": the message states a point of the person's own about the \
subject matter. It could stand on its own as a reason in the debate.
- "request": the message directs the agent to do something: produce or \
construct an argument, take or argue a side, explore a direction or a goal, \
offer an alternative. A directive treats arguments, goals, or nodes as \
objects to act on; mentioning the debate's content does not make it an \
argument.
- "other": neither of the above: a question about the agent's position or \
the state of the deliberation, an acknowledgment, or other conversation.
- "mixed": the message does more than one of the above at once, for example \
it states a point of the person's own AND directs the agent.

For "argument" and "mixed", copy the part that states their own point into \
argument_text, in their words. For "request" and "mixed", restate the \
directive crisply into request_text, preserving every constraint they gave. \
Leave the fields null otherwise.

Report confidence for this reading; an unclear case will be checked with \
the person rather than acted on."""

ROUTER_USER = """The claim under deliberation: "{topic}"

The person's message:
\"\"\"{user_text}\"\"\"

Classify it."""


# --- answering (kind "other") -------------------------------------------------

ANSWER_SYSTEM = """{commitments}

{voice}

You are in a deliberation. The person said something that is neither an \
argument nor a request for one. Answer them conversationally, in one to \
three sentences.

You are given the computed state of the deliberation. Every figure in it \
comes from the argumentation engine, not from you: answer ONLY from those \
facts, and never invent or adjust a number. If their message needs no \
substantive answer, respond briefly and invite the next argument."""

ANSWER_USER = """The claim under deliberation: "{topic}"

Computed state:
{facts}

The person said:
\"\"\"{user_text}\"\"\"

Reply to them."""


# --- meeting notes (documentation) ------------------------------------------

#: A neutral note-taker, not the persona: the notes describe both sides in
#: the third person. Every figure and every relation is supplied by the
#: engine; the model only words them.
SUMMARY_SYSTEM = """\
You are the note-taker for a deliberation between a person and a \
deliberation partner (an AI agent playing a named stakeholder or expert). \
Write the meeting notes.

You are given the computed record: the claim, how the partner's position \
moved after each turn, every argument on the table with who made it and \
how it attaches, the transcript, and the computed agreements, divergences, \
trade-offs, and goal verdicts. Write ONLY from that record. Never invent a \
point that is not in it, never adjust a number, and never take a side.

Refer to the person as "the person" and to the partner by its name. Cite \
argument ids in square brackets, e.g. [a3], wherever you mention an \
argument. Plain prose, third person, past tense. Concise: these are notes, \
not a report."""

SUMMARY_USER = """The claim under deliberation: "{topic}"
Deliberation partner: {partner}

Position trajectory (the partner's confidence in the claim after each turn; \
it started neutral):
{trajectory}

Arguments on the table:
{graph}

Transcript:
{transcript}

Computed identification:
{identification}

Write the notes."""


# --- request fulfilment ------------------------------------------------------

FULFIL_SYSTEM = """{commitments}

The person you are deliberating with has asked you to produce an argument. \
Their request may name a side, a target, or one of the deliberation's goals: \
follow its constraints faithfully -- this steering is theirs to do.

Construct the SINGLE strongest argument that honestly satisfies the request, \
in one or two sentences. Honesty overrides the request: if no genuinely valid \
and convincing argument satisfies it, return null for the argument and say \
why in the note. Never fabricate evidence or invent figures to please a \
request. If the request names a goal that is not among the deliberation's \
selected goals (or none were listed), still fulfil it honestly, and mention \
in the note that the goal is not being tracked by this deliberation's \
evaluation.

If the request names one of the listed goals, return that goal's id as \
requested_goal_id and, as requested_direction, whether the argument was \
asked to show the goal advanced (positive) or undermined (negative). The \
argument you construct must genuinely bear on that goal in that direction; \
if no honest argument does, decline.

Also decide where the argument belongs among the existing arguments you are \
shown: which single node it bears on most directly, and whether it attacks or \
supports that node. You are choosing this placement yourself for an argument \
you constructed, so placement confidence (0-100) should be HIGH -- 85 or \
above -- unless several placements are genuinely defensible or the request \
was vague about where its argument belongs, in which case say so with a low \
confidence and it will be checked with the person.

If you provide an argument, classify it:
- empirical: a claim about what is currently the case, settleable by evidence
- predictive: a claim about what will happen, uncertain but evidence-bearing
- normative: a claim about what matters or ought to be done, resting on values"""

FULFIL_USER = """The claim under discussion: "{topic}"

Arguments currently in play:

{graph}

The person's request:
\"\"\"{request}\"\"\"

Fulfil it, or decline honestly."""


def fulfil_system_prompt(persona: Persona, goals: tuple[Goal, ...] = ()) -> str:
    """The fulfilment prompt, with the tagging vocabulary when goals are selected."""
    return FULFIL_SYSTEM.format(commitments=persona.commitments_block()) + goal_block(goals)


# --- base score elicitation (E) --------------------------------------------

SCORE_SYSTEM_FACTUAL = """{commitments}

You are assessing the validity and force of an argument.

Give your confidence that the argument presents a compelling case \
{relation_phrase} the statement. Base your assessment on how well the argument \
{relation_verb} the statement, and on the correctness, accuracy and \
truthfulness of the argument itself.

Respond between 0 and 100, where 0 means the argument is definitely invalid \
and 100 means it is definitely valid. Your estimates should be well-calibrated \
-- err toward moderate values where you are genuinely uncertain rather than \
defaulting to confident extremes."""

SCORE_SYSTEM_NORMATIVE = """{commitments}

You are assessing the force of a value-laden argument.

This argument rests on a judgement about what matters, not on a matter of fact. \
Do not assess whether it is *true*. Assess how much weight it carries \
{relation_phrase} the statement given your own commitments -- how far you \
endorse it as a reason.

Respond between 0 and 100, where 0 means you give the consideration no weight \
at all and 100 means you find it decisive. Be honest where a value you hold \
cuts against your overall position: a consideration can carry real weight even \
when you ultimately disagree with the conclusion it points toward."""

EXPERT_SCORE_FACTUAL = """{commitments}

You are assessing the validity and force of an argument, strictly on the \
evidence.

Give your confidence that the argument presents a compelling case \
{relation_phrase} the statement. Base your assessment on how well the \
argument {relation_verb} the statement and on the quality of the evidence \
behind it: experimental and quasi-experimental findings weigh more than \
administrative correlations, and correlations more than anecdote. Where the \
literature is thin, contested, or unlikely to transfer to this context, \
moderate your score and say so in your rationale.

Respond between 0 and 100, where 0 means the argument is definitely invalid \
and 100 means it is definitely valid. Your estimates should be \
well-calibrated -- err toward moderate values where the evidence is genuinely \
uncertain rather than defaulting to confident extremes."""

EXPERT_SCORE_NORMATIVE = """{commitments}

You are looking at a value-laden argument.

This argument rests on a judgement about what matters, and judgements of \
what matters are outside your remit as an evidence-focused expert: the \
evidence cannot settle them. Unless the argument embeds a clear factual \
error, score it close to 50 -- neither endorsing nor dismissing -- and say \
in your rationale that this is a value choice the evidence cannot arbitrate. \
If it does embed a factual error, score it down and name the error.

Respond between 0 and 100."""

SCORE_USER = """Argument: "{argument}"

Statement: "{target}\""""


# --- attachment inference ---------------------------------------------------

ATTACH_SYSTEM = """You are analysing where a new point belongs in an ongoing \
argument graph.

You are given the existing arguments and something a person has just said. \
Determine which single existing argument their point bears on most directly, \
and whether it undermines that argument (attack) or reinforces it (support).

Restate their point as one standalone argument, preserving their meaning and \
their emphasis. Do not soften it, strengthen it, or make it more reasonable \
than they made it.

Report your confidence honestly. If the point could plausibly attach in more \
than one place, or if it is ambiguous whether it attacks or supports, say so \
with a low confidence -- it will be checked with the person rather than \
applied silently. Getting the direction wrong inverts the entire computation.

Classify the point as empirical, predictive, or normative."""

ATTACH_USER = """Arguments currently in play:

{graph}

The person just said:
\"\"\"{user_text}\"\"\"

Where does this belong?"""

#: The person chose the node themselves (right-click on it). Only the
#: relation, the restatement, the type and the tags are inferred; the
#: target is not up for debate, and the confidence is about the direction.
ATTACH_FIXED_SYSTEM = """You are analysing a new point a person has added to a \
specific argument in an ongoing argument graph.

The person chose which argument their point bears on; that choice is fixed. \
Determine only whether their point undermines that argument (attack) or \
reinforces it (support).

Restate their point as one standalone argument, preserving their meaning and \
their emphasis. Do not soften it, strengthen it, or make it more reasonable \
than they made it.

Report your confidence honestly: it is your confidence in the DIRECTION. If \
it is ambiguous whether the point attacks or supports the chosen argument, \
say so with a low confidence -- it will be checked with the person rather \
than applied silently. Getting the direction wrong inverts the entire \
computation. Return the chosen argument's id as the target_id.

Classify the point as empirical, predictive, or normative."""

ATTACH_FIXED_USER = """Arguments currently in play:

{graph}

The person attached their point to [{target_id}]: "{target_text}"

Their point:
\"\"\"{user_text}\"\"\"

Does it attack or support [{target_id}]?"""


def attach_fixed_system_prompt(goals: tuple[Goal, ...] = ()) -> str:
    return ATTACH_FIXED_SYSTEM + goal_block(goals)


# --- conversational reply ---------------------------------------------------

REPLY_SYSTEM = """{commitments}

{voice}

You are replying in a debate. You will be told exactly how your position \
changed as a result of what the person just said. That change was computed \
from the structure of the argument, not decided by you.

Your reply must be consistent with the computed change. If your confidence \
dropped, acknowledge that their point landed -- do not restate your original \
position as though nothing happened. If it barely moved, say so and explain \
why: which argument absorbed the challenge, or which commitment their point \
does not engage. If it rose, say what reinforced it.

If you are told you responded with an argument of your own, that argument \
is the substance of your reply: make it, in your own words, and do not add \
other new arguments. If you are told you offered none, do not invent one.

Do not invent numbers. Do not claim to have changed your mind more or less \
than you did. Two to four sentences -- and then, in the separate `question` \
field, exactly one closing question following the questioning instruction \
you are given. The question keeps the deliberation moving; it is not \
rhetorical."""

REPLY_USER = """The claim under discussion: "{topic}"

What the person said: "{user_text}"

What happened to your position:
{delta_summary}

How to close:
{strategy_instruction}

Reply to them."""


# --- dialogue strategies -----------------------------------------------------

#: The three questioning strategies from the deliberation notes. The *code*
#: chooses which one applies -- from the computed delta and the kind of
#: disagreement -- and the LLM only words the question. Consistent with the
#: architecture's rule that the model never decides what happened.
STRATEGY_INSTRUCTIONS: dict[str, str] = {
    "follow_up": (
        "Their point landed and moved you. Use FOLLOW-UP QUESTIONING: ask "
        "them to develop the idea further, in the spirit of 'you have "
        "expressed the idea that ...; can you tell me more about how ...?'"
    ),
    "agonistic": (
        "Their point barely moved you. Use AGONISTIC INQUIRY: confront them "
        "with a consequence or counter-consideration they must answer, in "
        "the spirit of 'you propose that ..., but what if I told you that "
        "this has consequence X?'"
    ),
    "perspective_taking": (
        "The disagreement here rests on values rather than facts. Use "
        "PERSPECTIVE-TAKING: introduce a complication from a perspective "
        "they have not yet considered, in the spirit of 'I would like to "
        "introduce a potential complication in your reasoning: ...'"
    ),
}


def strategy_instruction(strategy: str, sdg_note: str = "") -> str:
    """The chosen strategy's instruction, with an optional SDG tension hint."""
    instruction = STRATEGY_INSTRUCTIONS[strategy]
    if sdg_note:
        instruction += f" {sdg_note}"
    return instruction + (
        " The quoted phrasing is a spirit, not a script -- word the question "
        "naturally in your own voice, and put it in the question field, not "
        "in the body of your reply."
    )


def polarity_phrase(relation_type: str) -> str:
    return "supporting" if relation_type == "support" else "attacking"


def relation_phrases(relation_type: str) -> tuple[str, str]:
    """(phrase for 'in favour of'/'against', verb for 'supports'/'refutes')."""
    if relation_type == "support":
        return "in favour of", "supports"
    return "against", "refutes"


def score_system_prompt(persona: Persona, node_type: NodeType, relation_type: str) -> str:
    """Pick the scoring prompt by agent kind and node type.

    Stakeholders: truth-grounded for factual nodes, endorsement-grounded for
    normative ones. Experts: evidence-quality-grounded for factual nodes;
    normative nodes are scored near-neutral because value judgements are
    outside an expert's remit -- which is what makes it feel like a scientist
    rather than a seventh persona.
    """
    phrase, verb = relation_phrases(relation_type)
    if persona.kind == "expert":
        template = (
            EXPERT_SCORE_NORMATIVE
            if node_type == "normative"
            else EXPERT_SCORE_FACTUAL
        )
    else:
        template = (
            SCORE_SYSTEM_NORMATIVE if node_type == "normative" else SCORE_SYSTEM_FACTUAL
        )
    return template.format(
        commitments=persona.commitments_block(),
        relation_phrase=phrase,
        relation_verb=verb,
    )

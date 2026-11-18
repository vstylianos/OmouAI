"""Pipeline tests over DebateSession, with a stub LLM client.

The stub dispatches on `response_model` and records every call, so each test
declares exactly what the model would return and can also assert which prompt
the session actually sent -- expert vs stakeholder routing lives in those
prompts. Async methods are driven with `asyncio.run`, which keeps these tests
plain pytest functions.
"""

import asyncio

import pytest

from domains import get_domain
from engine.attribution import DeltaExplanation
from engine.lens import Goal
from engine.qbaf import Argument, Relation
from llm.client import Usage
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
    GoalImpact,
)
from pipeline.session import DebateSession


class StubClient:
    """Returns canned responses keyed on the requested response model."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []  # (system_prompt, user_prompt, response_model)
        self.usage = Usage()

    async def parse(self, system_prompt, user_prompt, response_model, temperature=None):
        self.usage.add(0, 0)
        self.calls.append((system_prompt, user_prompt, response_model))
        return self.responses[response_model]

    def system_prompt_for(self, response_model) -> str:
        return next(s for s, _, m in self.calls if m is response_model)


DOMAIN = get_domain("barcelona_homelessness")

#: Default selected goals for tests: two SDG lenses, the rough-sleeping
#: indicator's own lens, and a custom goal.
GOALS = (
    Goal(id="sdg_1", label="SDG 1 · No Poverty", kind="sdg"),
    Goal(id="sdg_11", label="SDG 11 · Sustainable Cities", kind="sdg"),
    Goal(id="ind_rough_sleeping_count", label="Individuals sleeping rough", kind="indicator"),
    Goal(id="custom_1", label="Families off waiting lists", kind="custom"),
)


def make_session(responses=None, persona_id="lived_experience_advocate", persona_ids=None, **kwargs):
    kwargs.setdefault("goals", GOALS)
    ids = persona_ids or [persona_id]
    return DebateSession(
        personas=tuple(DOMAIN.get_partner(pid) for pid in ids),
        topic="Test claim",
        client=StubClient(responses),
        **kwargs,
    )


def generated(text="An argument.", node_type="empirical", **kwargs):
    return GeneratedArgument(
        text=text, node_type=node_type, rationale="bears on it", **kwargs
    )


def routed(kind, confidence=95, **kwargs):
    return RoutedMessage(kind=kind, confidence=confidence, **kwargs)


def drafts(text="A response.", node_type="empirical", **kwargs):
    """The response argument, written with the direction already fixed."""
    return ExpansionResponse(argument=generated(text, node_type, **kwargs))


def land(session, text, confidence=None, force=None):
    """Make an argument the way the UI does: the first call always pends
    (reading + placement + the confidence question), the second applies."""
    turn = asyncio.run(session.contribute(text, force=force))
    assert turn.pending is not None and turn.pending.provenance == "user"
    assert turn.followup_request is None
    return asyncio.run(
        session.contribute(text, attachment=turn.pending, confidence=confidence)
    )


# --- expand -----------------------------------------------------------------


def test_expand_adds_a_scored_argument():
    session = make_session({
        ExpansionResponse: ExpansionResponse(argument=generated()),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
    })

    turn = asyncio.run(session.expand("topic", "attack"))

    assert len(turn.added_nodes) == 1
    added = session.qbaf.arguments[turn.added_nodes[0]]
    assert added.base_score == pytest.approx(0.6)
    assert added.elicited_tau == pytest.approx(0.6)
    assert session.qbaf.attackers_of("topic") == [added.arg_id]
    assert len(session.history()) == 1


def test_expand_can_decline():
    """A null argument leaves the graph, the history and the stance untouched.

    The scoring call must not happen either -- the stub has no ScoredArgument
    canned, so reaching it would KeyError.
    """
    session = make_session({ExpansionResponse: ExpansionResponse(argument=None)})
    before_args = set(session.qbaf.arguments)

    turn = asyncio.run(session.expand("topic", "attack"))

    assert turn.added_nodes == []
    assert turn.pending is None
    assert turn.delta.direction == "unchanged"
    assert set(session.qbaf.arguments) == before_args
    assert session.history() == []


def test_expand_unknown_node_raises():
    session = make_session()
    with pytest.raises(KeyError):
        asyncio.run(session.expand("ghost", "attack"))


def test_expand_shows_what_is_already_on_the_node():
    """A repeated request sees every argument attached to the node, whoever
    made it, so it brings a new consideration or declines."""
    session = make_session({
        ExpansionResponse: ExpansionResponse(argument=generated("Fresh.")),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
    }, persona_ids=["lived_experience_advocate", "homelessness_researcher"])
    _grow(session, "x1", "topic", "support", author="lived_experience_advocate")
    _grow(session, "x2", "topic", "attack", author="homelessness_researcher")
    _grow(session, "x3", "topic", "attack", provenance="user")
    _grow(session, "x4", "x1", "attack", author="homelessness_researcher")  # not on the topic

    asyncio.run(session.expand("topic", "support", "lived_experience_advocate"))
    user_prompt = next(u for _, u, m in session.client.calls if m is ExpansionResponse)
    assert "Arguments already made on this statement" in user_prompt
    assert '- for, by you: "x1"' in user_prompt
    assert '- against, by Homelessness Researcher: "x2"' in user_prompt
    assert '- against, by the user: "x3"' in user_prompt
    assert '"x4"' not in user_prompt
    assert user_prompt.endswith("Now construct a new argument.")
    assert "Do not repeat or rephrase" in session.client.system_prompt_for(ExpansionResponse)


def test_expand_on_an_untouched_node_lists_nothing():
    session = make_session({
        ExpansionResponse: ExpansionResponse(argument=generated()),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
    })
    asyncio.run(session.expand("topic", "attack"))
    user_prompt = next(u for _, u, m in session.client.calls if m is ExpansionResponse)
    assert "already made" not in user_prompt
    assert user_prompt.endswith("Now construct your argument.")


# --- expert vs stakeholder routing -------------------------------------------


def test_no_prompt_sees_the_indicator_table():
    """Indicators are evaluation instruments: selectable as goals, never
    material for an argument. The expert argues from its expertise."""
    responses = {
        ExpansionResponse: ExpansionResponse(argument=generated()),
        ScoredArgument: ScoredArgument(confidence=50, rationale="ok"),
    }
    expert = make_session(responses, persona_id="homelessness_researcher")
    asyncio.run(expert.expand("topic", "attack"))
    prompt = expert.client.system_prompt_for(ExpansionResponse)
    assert "- [rough_sleeping_count]" not in prompt  # no table row
    assert "mandates these indicators" not in prompt
    assert "[ind_rough_sleeping_count]" in prompt  # the selected indicator, as a goal


def test_expert_scores_by_evidence_not_endorsement():
    responses = {
        ExpansionResponse: ExpansionResponse(argument=generated(node_type="normative")),
        ScoredArgument: ScoredArgument(confidence=50, rationale="value choice"),
    }
    expert = make_session(responses, persona_id="homelessness_researcher")
    asyncio.run(expert.expand("topic", "attack"))
    assert "outside your remit" in expert.client.system_prompt_for(ScoredArgument)

    stakeholder = make_session(dict(responses))
    asyncio.run(stakeholder.expand("topic", "attack"))
    assert "your own commitments" in stakeholder.client.system_prompt_for(ScoredArgument)


def test_goal_impacts_survive_into_the_graph_and_snapshots():
    session = make_session({
        ExpansionResponse: ExpansionResponse(
            argument=generated(
                goal_impacts=[
                    GoalImpact(goal_id="sdg_1", direction="positive"),
                    GoalImpact(goal_id="sdg_99", direction="positive"),  # dropped
                ]
            )
        ),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
    })
    turn = asyncio.run(session.expand("topic", "support"))
    node = session.qbaf.arguments[turn.added_nodes[0]]
    assert node.goal_impacts == [
        {"goal_id": "sdg_1", "direction": "positive", "source": "judged"}
    ]

    restored = session._snapshot()
    assert restored.arguments[node.arg_id].goal_impacts == node.goal_impacts


def test_a_selected_indicator_is_tagged_like_any_goal():
    """An indicator chosen at preparation is a lens the model tags as judged,
    the same as an SDG or a custom goal."""
    session = make_session({
        ExpansionResponse: ExpansionResponse(
            argument=generated(
                goal_impacts=[
                    GoalImpact(goal_id="ind_rough_sleeping_count", direction="positive"),
                    GoalImpact(goal_id="custom_1", direction="negative"),
                ],
            )
        ),
        ScoredArgument: ScoredArgument(confidence=70, rationale="strong"),
    }, persona_id="homelessness_researcher")

    turn = asyncio.run(session.expand("topic", "attack"))
    node = session.qbaf.arguments[turn.added_nodes[0]]

    assert node.goal_impacts == [
        {"goal_id": "ind_rough_sleeping_count", "direction": "positive", "source": "judged"},
        {"goal_id": "custom_1", "direction": "negative", "source": "judged"},
    ]


def test_bracketed_ids_from_the_model_are_forgiven():
    """Models sometimes echo ids with the display brackets ('[sdg_1]');
    normalisation keeps the tag instead of dropping it."""
    session = make_session({
        ExpansionResponse: ExpansionResponse(
            argument=generated(
                goal_impacts=[GoalImpact(goal_id="[custom_1]", direction="negative")],
            )
        ),
        ScoredArgument: ScoredArgument(confidence=70, rationale="ok"),
    }, persona_id="homelessness_researcher")
    turn = asyncio.run(session.expand("topic", "attack"))
    node = session.qbaf.arguments[turn.added_nodes[0]]
    assert {t["goal_id"] for t in node.goal_impacts} == {"custom_1"}


def test_user_contributions_carry_judged_tags():
    session = make_session({
        RoutedMessage: routed("argument"),
        Attachment: Attachment(
            target_id="topic", relation_type="support", node_type="empirical",
            confidence=90, rationale="clear", restated="Restated.",
            goal_impacts=[GoalImpact(goal_id="sdg_11", direction="positive")],
        ),
        ScoredArgument: ScoredArgument(confidence=70, rationale="ok"),
        AgentReply: AgentReply(message="Ok.", concedes=False, question="Why?"),
    })
    turn = land(session, "my point")
    node = session.qbaf.arguments[turn.added_nodes[0]]
    assert node.goal_impacts == [
        {"goal_id": "sdg_11", "direction": "positive", "source": "judged"}
    ]


def test_pending_confirmation_round_trip_preserves_goal_tags():
    """A low-confidence attachment returns pending with the graph untouched;
    confirming with that same attachment lands the node *with* its goal tags."""
    attachment = Attachment(
        target_id="topic", relation_type="attack", node_type="empirical",
        confidence=40, rationale="unsure", restated="Restated point.",
        goal_impacts=[GoalImpact(goal_id="sdg_1", direction="negative")],
    )
    session = make_session({
        RoutedMessage: routed("argument"),
        Attachment: attachment,
        ScoredArgument: ScoredArgument(confidence=70, rationale="ok"),
        AgentReply: AgentReply(message="Ok.", concedes=False, question="Why?"),
    })

    turn = asyncio.run(session.contribute("my point"))
    assert turn.pending is not None
    assert set(session.qbaf.arguments) == {"topic"}
    assert session.lens() and all(not e["contributions"] for e in session.lens())

    confirmed = asyncio.run(session.contribute("my point", attachment=turn.pending))
    node = session.qbaf.arguments[confirmed.added_nodes[0]]
    assert node.goal_impacts == [
        {"goal_id": "sdg_1", "direction": "negative", "source": "judged"}
    ]
    entry = next(e for e in session.lens() if e["id"] == "sdg_1")
    assert entry["verdict"] == "undermined"


def test_user_can_redirect_a_pending_attachment():
    """The pending card lets the user pick the target and relation; the
    confirmed attachment is applied verbatim, inference bypassed."""
    session = make_session({
        RoutedMessage: routed("argument"),
        Attachment: Attachment(
            target_id="topic", relation_type="support", node_type="empirical",
            confidence=40, rationale="unsure", restated="Restated point.",
        ),
        ScoredArgument: ScoredArgument(confidence=70, rationale="ok"),
        AgentReply: AgentReply(message="Ok.", concedes=False, question="Why?"),
    })
    _grow(session, "m1", "topic", "attack", base_score=0.8)

    turn = asyncio.run(session.contribute("my point"))
    assert turn.pending is not None

    redirected = turn.pending.model_copy(
        update={"target_id": "m1", "relation_type": "attack"}
    )
    confirmed = asyncio.run(session.contribute("my point", attachment=redirected))
    node_id = confirmed.added_nodes[0]
    assert session.qbaf.attackers_of("m1") == [node_id]
    assert session.qbaf.classify_polarity(node_id) == "pro"  # attacks the attacker


def test_redirecting_a_judged_tag_downgrades_it_to_user():
    """Flipping a judged tag's direction is the user overruling the model:
    the tag survives, but its provenance becomes 'user' -- the record stays
    honest about who asserted what."""
    session = make_session()
    _grow(session, "a1", "topic", "support", base_score=0.8)
    session.qbaf.arguments["a1"].goal_impacts = [
        {"goal_id": "sdg_1", "direction": "positive", "source": "judged"}
    ]
    session.set_goal_tags("a1", [{"goal_id": "sdg_1", "direction": "negative"}])
    assert session.qbaf.arguments["a1"].goal_impacts == [
        {"goal_id": "sdg_1", "direction": "negative", "source": "user"}
    ]


def test_remove_cascade_takes_lens_contributions_with_it():
    """Rejecting an argument whose subtree carried goal tags resets the
    affected lenses -- no stale contributions, no KeyError."""
    session = make_session()
    _grow(session, "a1", "topic", "attack", base_score=0.8)
    _grow(session, "a2", "a1", "support", base_score=0.9)
    session.qbaf.arguments["a2"].goal_impacts = [
        {"goal_id": "sdg_1", "direction": "positive", "source": "judged"}
    ]
    entry = next(e for e in session.lens() if e["id"] == "sdg_1")
    assert entry["contributions"]

    session.remove("a1")  # cascades to a2

    entry = next(e for e in session.lens() if e["id"] == "sdg_1")
    assert entry["contributions"] == []
    assert entry["sigma"] == pytest.approx(0.5)


def test_lens_history_tracks_snapshots_and_recomputes_on_semantics_switch():
    session = make_session()
    _grow(session, "a1", "topic", "support", base_score=0.8)
    session.qbaf.arguments["a1"].goal_impacts = [
        {"goal_id": "sdg_1", "direction": "positive", "source": "judged"}
    ]
    session._record("turn 1")
    session.set_base_score("a1", 0.4)  # records "you re-scored a1"

    modular = session.lens_history()
    assert len(modular["sdg_1"]) == 2
    assert modular["sdg_1"][0] > modular["sdg_1"][1]  # weakening the supporter

    session.set_semantics("dfquad")
    dfquad = session.lens_history()
    assert len(dfquad["sdg_1"]) == 2
    assert all(0.0 <= v <= 1.0 for v in dfquad["sdg_1"])
    # same snapshots, different arithmetic: k=1.4 modular vs DF-QuAD differ
    assert dfquad["sdg_1"] != modular["sdg_1"]


def test_goal_note_derives_from_lens_verdicts():
    session = make_session()
    assert session._goal_note() == ""  # everything contested: no tension

    _grow(session, "a1", "topic", "support", base_score=0.9)
    session.qbaf.arguments["a1"].goal_impacts = [
        {"goal_id": "sdg_1", "direction": "positive", "source": "judged"}
    ]
    assert session._goal_note() == ""  # supported alone is not a tension

    _grow(session, "a2", "topic", "attack", base_score=0.9)
    session.qbaf.arguments["a2"].goal_impacts = [
        {"goal_id": "sdg_11", "direction": "negative", "source": "judged"}
    ]
    note = session._goal_note()
    assert "SDG 1" in note and "SDG 11" in note


# --- dialogue strategies ------------------------------------------------------


def _delta(moved: float) -> DeltaExplanation:
    return DeltaExplanation(topic_before=0.5, topic_after=0.5 + moved)


def test_strategy_follows_the_computed_situation():
    session = make_session()
    assert session._choose_strategy(_delta(0.05), "empirical") == "follow_up"
    assert session._choose_strategy(_delta(0.0), "normative") == "perspective_taking"
    assert session._choose_strategy(_delta(0.0), "empirical") == "agonistic"


def test_fulfilled_request_closes_with_a_question_and_records_concessions():
    """Only a persona's own turn carries a reply: a requested argument is
    the persona's, so it replies with it and a closing question."""
    session = make_session({
        FulfilledRequest: FulfilledRequest(
            argument=generated("Requested."), target_id="topic", relation_type="attack",
            placement_confidence=90, placement_rationale="bears there",
        ),
        ScoredArgument: ScoredArgument(confidence=80, rationale="lands"),
        AgentReply: AgentReply(message="Fair point.", concedes=True, question="Tell me more?"),
    })
    turn = asyncio.run(session.contribute("give a reason against", force="request"))
    assert turn.question == "Tell me more?"
    assert turn.concedes
    assert turn.speaker == "lived_experience_advocate"
    assert session._concessions == turn.added_nodes
    assert ("lived_experience_advocate", "Fair point. Tell me more?") in session._transcript


# --- intent gate and request fulfilment ---------------------------------------


def _fulfilment(argument, target="topic", confidence=90, note="", **kwargs):
    return FulfilledRequest(
        argument=argument, target_id=target, relation_type="attack",
        placement_confidence=confidence, placement_rationale="bears there",
        note=note, **kwargs,
    )


def test_request_is_fulfilled_as_a_generated_argument():
    session = make_session({
        RoutedMessage: routed("request", request_text="Give an argument that regresses SDG 1."),
        FulfilledRequest: _fulfilment(
            generated(goal_impacts=[GoalImpact(goal_id="sdg_1", direction="negative")])
        ),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: AgentReply(message="Here it is.", concedes=False, question="And?"),
    })

    turn = asyncio.run(session.contribute("regress SDG1 please"))

    node = session.qbaf.arguments[turn.added_nodes[0]]
    assert node.provenance == "generated"  # agent-authored, at the user's request
    assert node.goal_impacts == [
        {"goal_id": "sdg_1", "direction": "negative", "source": "judged"}
    ]
    assert turn.reply == "Here it is." and turn.question == "And?"
    assert ("user", "regress SDG1 please") in session._transcript
    assert session._snapshots[-1][0] == "Lived-Experience Advocate: requested argument"


def test_request_naming_a_goal_is_tagged_with_it_by_code():
    """The goal a request names is stamped as the person's own assertion,
    overriding whatever the model judged for that goal and regardless of
    its 'do not stretch' caution; other judged tags are kept."""
    session = make_session({
        RoutedMessage: routed("request", request_text="Give an argument that advances SDG 1."),
        FulfilledRequest: _fulfilment(
            generated(goal_impacts=[
                GoalImpact(goal_id="sdg_1", direction="negative"),   # overridden
                GoalImpact(goal_id="custom_1", direction="positive"),  # kept
            ]),
            requested_goal_id="[sdg_1]", requested_direction="positive",
        ),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: AgentReply(message="Here it is.", concedes=False, question="And?"),
    })
    turn = asyncio.run(session.contribute("advance SDG1 please"))
    node = session.qbaf.arguments[turn.added_nodes[0]]
    assert node.goal_impacts == [
        {"goal_id": "sdg_1", "direction": "positive", "source": "user"},
        {"goal_id": "custom_1", "direction": "positive", "source": "judged"},
    ]


def test_requested_goal_survives_a_shaky_placement():
    """The named goal rides on the pending attachment, so a placement the
    person confirms or redirects still carries the tag they asked for."""
    session = make_session({
        RoutedMessage: routed("request", request_text="Undermine SDG 11."),
        FulfilledRequest: _fulfilment(
            generated(), confidence=40,
            requested_goal_id="sdg_11", requested_direction="negative",
        ),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: AgentReply(message="Here.", concedes=False, question="And?"),
    })
    first = asyncio.run(session.contribute("undermine SDG 11"))
    assert first.pending is not None and first.pending.requested_goal_id == "sdg_11"
    turn = asyncio.run(session.contribute("undermine SDG 11", attachment=first.pending))
    assert session.qbaf.arguments[turn.added_nodes[0]].goal_impacts == [
        {"goal_id": "sdg_11", "direction": "negative", "source": "user"}
    ]


def test_request_naming_an_unselected_goal_stamps_nothing():
    session = make_session({
        RoutedMessage: routed("request", request_text="Advance SDG 5."),
        FulfilledRequest: _fulfilment(
            generated(), requested_goal_id="sdg_5", requested_direction="positive",
        ),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: AgentReply(message="Here.", concedes=False, question="And?"),
    })
    turn = asyncio.run(session.contribute("advance SDG 5"))
    assert session.qbaf.arguments[turn.added_nodes[0]].goal_impacts == []


def test_request_can_be_declined_honestly():
    session = make_session({
        RoutedMessage: routed("request", request_text="Fabricate a statistic."),
        FulfilledRequest: _fulfilment(None, note="I will not invent figures."),
    })
    turn = asyncio.run(session.contribute("fabricate a statistic"))
    assert turn.added_nodes == []
    assert turn.reply == "I will not invent figures."
    assert set(session.qbaf.arguments) == {"topic"}


def test_unclear_intent_asks_and_force_skips_the_gate():
    session = make_session({
        RoutedMessage: routed("request", confidence=40, request_text="Maybe argue X?"),
    })
    turn = asyncio.run(session.contribute("hmm X"))
    assert turn.pending_intent == "Maybe argue X?"
    assert set(session.qbaf.arguments) == {"topic"}

    # force=request goes straight to fulfilment: no UserTurn call happens,
    # so a stub without one would KeyError if the gate were consulted.
    forced = make_session({
        FulfilledRequest: _fulfilment(generated()),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: AgentReply(message="Done.", concedes=False, question="More?"),
    })
    turn = asyncio.run(forced.contribute("hmm X", force="request"))
    assert len(turn.added_nodes) == 1

    # force=argument uses the pure attachment prompt (no intent block).
    as_arg = make_session({
        Attachment: Attachment(
            target_id="topic", relation_type="support", node_type="empirical",
            confidence=90, rationale="clear", restated="X.",
        ),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: AgentReply(message="Ok.", concedes=False, question="Why?"),
    })
    turn = land(as_arg, "hmm X", force="argument")
    prompt = as_arg.client.system_prompt_for(Attachment)
    assert "route messages" not in prompt
    assert as_arg.qbaf.arguments[turn.added_nodes[0]].provenance == "user"


def test_fulfilment_target_id_brackets_are_forgiven():
    session = make_session({
        RoutedMessage: routed("request", request_text="Attack the claim."),
        FulfilledRequest: FulfilledRequest(
            argument=generated(), target_id="[topic]", relation_type="attack",
            placement_confidence=90, placement_rationale="bears there",
        ),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: AgentReply(message="Ok.", concedes=False, question="And?"),
    })
    turn = asyncio.run(session.contribute("attack the claim"))
    assert session.qbaf.attackers_of("topic") == turn.added_nodes


def test_other_messages_are_answered_from_computed_facts():
    """Kind "other" (questions about state, chitchat) answers conversationally
    and never touches the graph -- no node, no snapshot."""
    session = make_session({
        RoutedMessage: routed("other"),
        StateAnswer: StateAnswer(message="I stand where the numbers put me."),
    })
    _grow(session, "m1", "topic", "support", base_score=0.8)
    session.qbaf.arguments["m1"].goal_impacts = [
        {"goal_id": "sdg_1", "direction": "positive", "source": "judged"}
    ]

    turn = asyncio.run(session.contribute("what is your stance now?"))

    assert turn.reply == "I stand where the numbers put me."
    assert turn.added_nodes == [] and session.history() == []
    assert ("lived_experience_advocate", "I stand where the numbers put me.") in session._transcript
    # the facts block carried engine-computed state into the prompt,
    # including the full scored graph so content questions are answerable
    facts = session.client.calls[-1][1]
    assert "Your current stance on the claim" in facts
    assert "SDG 1" in facts and "supported" in facts
    assert '[m1] (empirical, tau=0.80' in facts
    assert '"m1"' in facts  # the node's text
    assert "helps sdg_1" in facts


def test_mixed_message_handles_both_halves_in_one_turn():
    session = make_session({
        RoutedMessage: routed(
            "mixed",
            argument_text="Shelters are inhumane in winter.",
            request_text="Give your strongest counterargument.",
        ),
        Attachment: Attachment(
            target_id="topic", relation_type="attack", node_type="empirical",
            confidence=90, rationale="clear", restated="Shelters are inhumane in winter.",
        ),
        FulfilledRequest: _fulfilment(generated()),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: AgentReply(message="Both noted.", concedes=False, question="And?"),
    })

    text = "Shelters are inhumane. Also, counterargument?"
    turn = asyncio.run(session.contribute(text))
    # The argument half pends like any argument; the request rides along.
    assert turn.pending is not None and turn.pending.provenance == "user"
    assert turn.followup_request == "Give your strongest counterargument."
    assert set(session.qbaf.arguments) == {"topic"}

    landed = asyncio.run(session.contribute(text, attachment=turn.pending, confidence="very"))
    user = session.qbaf.arguments[landed.added_nodes[0]]
    assert user.provenance == "user" and user.base_score == pytest.approx(0.65)

    fulfilled = asyncio.run(session.contribute(turn.followup_request, force="request"))
    requested = session.qbaf.arguments[fulfilled.added_nodes[0]]
    assert requested.provenance == "generated"
    assert fulfilled.reply == "Both noted."


def test_mixed_with_shaky_argument_placement_carries_the_request_forward():
    session = make_session({
        RoutedMessage: routed(
            "mixed",
            argument_text="Shelters are inhumane.",
            request_text="Give your strongest counterargument.",
        ),
        Attachment: Attachment(
            target_id="topic", relation_type="attack", node_type="empirical",
            confidence=40, rationale="unsure", restated="Shelters are inhumane.",
        ),
    })
    turn = asyncio.run(session.contribute("Shelters are inhumane. Also, counterargument?"))
    assert turn.pending is not None
    assert turn.followup_request == "Give your strongest counterargument."
    assert set(session.qbaf.arguments) == {"topic"}


def test_request_decline_without_note_uses_the_default_message():
    session = make_session({
        RoutedMessage: routed("request", request_text="Do something."),
        FulfilledRequest: _fulfilment(None),
    })
    turn = asyncio.run(session.contribute("do something"))
    assert "no genuinely valid argument" in turn.reply


def test_shaky_fulfilment_placement_goes_through_the_gate():
    # A fulfilled argument is what the user asked for, but its placement is
    # fallible: below the gate it pends, and the confirmed node keeps its
    # agent authorship.
    session = make_session({
        RoutedMessage: routed("request", request_text="Attack something."),
        FulfilledRequest: _fulfilment(generated(), confidence=40),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: AgentReply(message="Ok.", concedes=False, question="And?"),
    })
    turn = asyncio.run(session.contribute("attack something"))
    assert turn.pending is not None
    assert turn.pending.provenance == "generated"
    assert set(session.qbaf.arguments) == {"topic"}

    confirmed = asyncio.run(session.contribute("attack something", attachment=turn.pending))
    node = session.qbaf.arguments[confirmed.added_nodes[0]]
    assert node.provenance == "generated"


# --- lens and tags ------------------------------------------------------------


def test_lens_reflects_the_graph_and_history():
    session = make_session()
    _grow(session, "a1", "topic", "support", base_score=0.8)
    session.qbaf.arguments["a1"].goal_impacts = [
        {"goal_id": "sdg_1", "direction": "positive", "source": "judged"}
    ]
    session._record("setup")
    entry = next(e for e in session.lens() if e["id"] == "sdg_1")
    assert entry["verdict"] == "supported"
    assert session.lens_history()["sdg_1"] == [pytest.approx(entry["sigma"])]


def test_goal_block_reaches_prompts_only_when_goals_selected():
    responses = {
        ExpansionResponse: ExpansionResponse(argument=generated()),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
    }
    tagged = make_session(responses)
    asyncio.run(tagged.expand("topic", "attack"))
    assert "[sdg_1]" in tagged.client.system_prompt_for(ExpansionResponse)

    untagged = make_session(dict(responses), goals=())
    asyncio.run(untagged.expand("topic", "attack"))
    assert "measured against these goals" not in untagged.client.system_prompt_for(
        ExpansionResponse
    )


def test_set_goal_tags_contests_the_record():
    session = make_session()
    _grow(session, "a1", "topic", "support", base_score=0.8)
    session.qbaf.arguments["a1"].goal_impacts = [
        {"goal_id": "sdg_1", "direction": "positive", "source": "judged"}
    ]

    session.set_goal_tags(
        "a1",
        [
            {"goal_id": "sdg_1", "direction": "positive"},   # unchanged: keeps provenance
            {"goal_id": "custom_1", "direction": "negative"},  # added: user
        ],
    )

    assert session.qbaf.arguments["a1"].goal_impacts == [
        {"goal_id": "sdg_1", "direction": "positive", "source": "judged"},
        {"goal_id": "custom_1", "direction": "negative", "source": "user"},
    ]
    assert session._snapshots[-1][0] == "you re-tagged a1"

    with pytest.raises(ValueError):
        session.set_goal_tags("a1", [{"goal_id": "nope", "direction": "positive"}])
    with pytest.raises(KeyError):
        session.set_goal_tags("ghost", [])


# --- remove -----------------------------------------------------------------


def _grow(session, arg_id, target_id, relation_type, base_score=0.8, **kwargs):
    session.qbaf.add_argument(
        Argument(arg_id=arg_id, text=arg_id, base_score=base_score, **kwargs)
    )
    session.qbaf.add_relation(Relation(arg_id, target_id, relation_type))


def test_remove_cascades_and_reports_the_full_effect():
    """Rejecting an argument takes its subtree, and the delta reflects both."""
    session = make_session()
    _grow(session, "a1", "topic", "attack", base_score=0.8)
    _grow(session, "a2", "a1", "support", base_score=0.9)

    with_subtree = session.stance()
    delta = session.remove("a1")

    assert set(session.qbaf.arguments) == {"topic"}
    assert delta.topic_before == pytest.approx(with_subtree)
    assert delta.topic_after == pytest.approx(session.stance())
    assert delta.direction == "strengthened"  # an attacker and its backing left


def test_remove_topic_refused():
    session = make_session()
    with pytest.raises(ValueError):
        session.remove("topic")


# --- report -------------------------------------------------------------------


def test_report_identifies_agreements_divergences_and_trade_offs():
    session = make_session()
    _grow(session, "a1", "topic", "attack",
          base_score=0.8, elicited_tau=0.8, node_type="empirical")
    _grow(session, "a2", "topic", "support",
          base_score=0.5, provenance="user")
    _grow(session, "a3", "a2", "support", base_score=0.7, elicited_tau=0.7)
    _grow(session, "a4", "topic", "support",
          base_score=0.8, provenance="user", node_type="normative")
    _grow(session, "a5", "a4", "attack", base_score=0.7, elicited_tau=0.7)
    _grow(session, "a6", "topic", "attack", base_score=0.2, provenance="user")
    _grow(session, "a7", "a6", "attack", base_score=0.7, elicited_tau=0.7)
    session.set_base_score("a1", 0.2)  # user contests the attacker

    rep = session.report()

    # a2: the agent reinforced the user's point with a3 -> agreement
    assert [a["id"] for a in rep["agreements"]] == ["a2"]
    assert "a3" in rep["agreements"][0]["basis"]
    # a1: slider now 0.2 vs elicited 0.8 -> evidential divergence;
    # a4: held firmly by the user and objected to by a5 -> value divergence;
    # a6: objected to, but the user was unsure of it -> not a divergence.
    assert [d["id"] for d in rep["divergences"]] == ["a1", "a4"]
    assert rep["divergences"][0]["kind"] == "evidential"
    assert rep["divergences"][0]["agent_tau"] == pytest.approx(0.8)
    assert rep["divergences"][1]["kind"] == "value"
    assert rep["divergences"][1]["agent_tau"] is None
    assert rep["divergences"][1]["your_tau"] == pytest.approx(0.8)
    assert "a5" in rep["divergences"][1]["basis"]
    assert {t["id"] for t in rep["trade_offs"]} <= set(session.qbaf.arguments)
    assert rep["stance"] == pytest.approx(session.stance())
    assert rep["semantics"] == "modular"


# --- the user's confidence; user points draw no response --------------------------


def _attach(**overrides):
    fields = dict(
        target_id="topic", relation_type="attack", node_type="empirical",
        confidence=90, rationale="clear", restated="Restated point.",
    )
    fields.update(overrides)
    return Attachment(**fields)


def _reply():
    return AgentReply(message="Ok.", concedes=False, question="Why?")


def test_user_argument_carries_the_users_confidence_and_lands_silently():
    """The user's point is never scored by a persona, draws no response and
    no reply: after the confirmation call the graph moves and nothing else
    happens. The stub has no AgentReply, so a reply call would KeyError."""
    session = make_session({
        RoutedMessage: routed("argument"),
        Attachment: _attach(),
    })
    turn = land(session, "my point", confidence="certain")
    node = session.qbaf.arguments[turn.added_nodes[0]]
    assert node.provenance == "user" and node.author is None
    assert node.base_score == pytest.approx(0.8)
    assert node.elicited_tau is None and node.rationale == ""
    assert turn.reply == "" and turn.question == "" and turn.speaker is None
    assert [label for label, _ in turn.steps] == ["your argument"]
    assert turn.delta.topic_after < 0.5                    # the attack landed
    assert {m for _, _, m in session.client.calls} <= {RoutedMessage, Attachment}   # no persona call
    assert [h["label"] for h in session.history()] == ["your argument"]
    assert ("user", "my point") in session._transcript


def test_confidence_defaults_to_the_middle_and_rejects_unknown_levels():
    session = make_session({RoutedMessage: routed("argument"), Attachment: _attach()})
    turn = land(session, "my point")
    assert session.qbaf.arguments[turn.added_nodes[0]].base_score == pytest.approx(0.5)
    pending = asyncio.run(session.contribute("my point")).pending
    with pytest.raises(ValueError):
        asyncio.run(session.contribute("my point", attachment=pending, confidence="absolutely"))


def test_confidence_survives_pending_confirmation_and_redirection():
    session = make_session({RoutedMessage: routed("argument"), Attachment: _attach(confidence=40)})
    turn = asyncio.run(session.contribute("my point"))
    assert turn.pending is not None
    redirected = turn.pending.model_copy(update={"relation_type": "support"})
    confirmed = asyncio.run(session.contribute("my point", attachment=redirected, confidence="unsure"))
    node = session.qbaf.arguments[confirmed.added_nodes[0]]
    assert node.base_score == pytest.approx(0.2)
    assert session.qbaf.supporters_of("topic") == [node.arg_id]


def test_right_click_fixes_the_target_and_skips_the_router():
    """A point made on a chosen node goes straight to the direction check:
    no router call, the target is the person's choice whatever the model
    echoed, and the fixed-target prompt is used."""
    classifier = StubClient({
        Attachment: _attach(target_id="somewhere_else", relation_type="support", confidence=88),
    })
    session = make_session({}, classifier=classifier)
    _grow(session, "m1", "topic", "attack", base_score=0.8)
    turn = asyncio.run(session.contribute("my point", target_id="m1"))
    assert [m for _, _, m in classifier.calls] == [Attachment]
    system, user, _ = classifier.calls[0]
    assert "that choice is fixed" in system and "[m1]" in user
    assert turn.pending is not None and turn.pending.target_id == "m1"
    assert turn.pending.relation_type == "support"
    confirmed = asyncio.run(session.contribute("my point", attachment=turn.pending, confidence="very"))
    assert session.qbaf.supporters_of("m1") == confirmed.added_nodes


def test_right_click_on_an_unknown_node_raises():
    session = make_session({})
    with pytest.raises(KeyError):
        asyncio.run(session.contribute("my point", target_id="ghost"))


def test_requested_arguments_are_scored_by_the_addressed_persona():
    session = make_session({
        FulfilledRequest: _fulfilment(generated()),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: _reply(),
    }, persona_ids=["lived_experience_advocate", "city_hall_politician"])
    turn = asyncio.run(session.contribute(
        "attack it", force="request", confidence="certain", persona_id="city_hall_politician",
    ))
    node = session.qbaf.arguments[turn.added_nodes[0]]
    assert node.provenance == "generated" and node.author == "city_hall_politician"
    assert node.base_score == pytest.approx(0.6)          # not the user's 0.8
    assert turn.speaker == "city_hall_politician"
    assert "City-Hall Politician" in session.client.system_prompt_for(FulfilledRequest)
    assert "City-Hall Politician" in session.client.system_prompt_for(ScoredArgument)


def test_shaky_requested_placement_keeps_its_author_through_confirmation():
    session = make_session({
        FulfilledRequest: _fulfilment(generated(), confidence=30),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: _reply(),
    }, persona_ids=["lived_experience_advocate", "city_hall_politician"])
    turn = asyncio.run(session.contribute("attack it", force="request", persona_id="city_hall_politician"))
    assert turn.pending is not None and turn.pending.author == "city_hall_politician"
    # Confirmed without naming a persona: the attachment's author still scores it.
    confirmed = asyncio.run(session.contribute("attack it", attachment=turn.pending))
    assert session.qbaf.arguments[confirmed.added_nodes[0]].author == "city_hall_politician"


def test_mixed_message_pends_the_point_and_carries_the_request():
    session = make_session({
        RoutedMessage: routed(
            "mixed", argument_text="Shelters are inhumane.",
            request_text="Give your strongest counterargument.",
        ),
        Attachment: _attach(restated="Shelters are inhumane."),
        FulfilledRequest: _fulfilment(generated("Requested.")),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
        AgentReply: _reply(),
    })
    first = asyncio.run(session.contribute("Shelters are inhumane. Also?"))
    assert first.pending is not None and first.followup_request == "Give your strongest counterargument."
    landed = asyncio.run(session.contribute("Shelters are inhumane. Also?", attachment=first.pending, confidence="very"))
    assert len(landed.added_nodes) == 1 and landed.reply == ""
    followup = asyncio.run(session.contribute(first.followup_request, force="request"))
    assert session.qbaf.arguments[followup.added_nodes[0]].text == "Requested."


def test_removing_a_user_point_takes_what_hangs_on_it():
    session = make_session({
        RoutedMessage: routed("argument"), Attachment: _attach(),
        ExpansionResponse: drafts("Objection."),
        ScoredArgument: ScoredArgument(confidence=70, rationale="solid"),
    })
    user_id = land(session, "my point").added_nodes[0]
    objection = asyncio.run(session.expand(user_id, "attack")).added_nodes[0]
    assert session.qbaf.attackers_of(user_id) == [objection]
    session.remove(user_id)
    assert set(session.qbaf.arguments) == {"topic"}


# --- several personas at one table -------------------------------------------


def test_every_persona_introduces_itself_at_the_start():
    session = make_session({}, persona_ids=["lived_experience_advocate", "devils_advocate"])
    turn = asyncio.run(session.open())
    assert [pid for pid, _ in turn.intros] == ["lived_experience_advocate", "devils_advocate"]
    assert turn.reply == "" and session.client.calls == []
    assert [who for who, _ in session._transcript] == ["lived_experience_advocate", "devils_advocate"]
    assert session.conservativeness == pytest.approx((2.2 + 1.5) / 2)


def test_expand_is_authored_and_scored_by_the_named_persona():
    session = make_session({
        ExpansionResponse: drafts(),
        ScoredArgument: ScoredArgument(confidence=60, rationale="fair"),
    }, persona_ids=["lived_experience_advocate", "homelessness_researcher"])
    turn = asyncio.run(session.expand("topic", "attack", "homelessness_researcher"))
    node = session.qbaf.arguments[turn.added_nodes[0]]
    assert node.author == "homelessness_researcher" and turn.speaker == "homelessness_researcher"
    assert "a domain expert" in session.client.system_prompt_for(ExpansionResponse)
    assert "outside your remit" not in session.client.system_prompt_for(ScoredArgument)
    assert session.history()[-1]["label"] == "Homelessness Researcher: against topic"
    with pytest.raises(KeyError):
        asyncio.run(session.expand("topic", "attack", "ghost"))


def test_questions_are_answered_by_the_addressed_persona():
    session = make_session({
        RoutedMessage: routed("other"),
        StateAnswer: StateAnswer(message="I stand at 50%."),
    }, persona_ids=["lived_experience_advocate", "city_hall_politician"])
    turn = asyncio.run(session.contribute("where do you stand?", persona_id="city_hall_politician"))
    assert turn.speaker == "city_hall_politician"
    assert "City-Hall Politician" in session.client.system_prompt_for(StateAnswer)
    assert ("city_hall_politician", "I stand at 50%.") in session._transcript


def test_facts_and_notes_name_each_argument_author():
    session = make_session({}, persona_ids=["lived_experience_advocate", "city_hall_politician"])
    _grow(session, "a1", "topic", "attack", base_score=0.8, author="city_hall_politician")
    _grow(session, "a2", "topic", "support", base_score=0.6, provenance="user")
    facts = session._facts_block()
    assert "made by City-Hall Politician" in facts and "made by the person" in facts
    rep = session.report()
    assert [p["id"] for p in rep["personas"]] == ["lived_experience_advocate", "city_hall_politician"]


# --- neutral start, always-confirm arguments, separate classifier ---------------


def test_start_is_neutral_with_no_model_call():
    session = make_session()
    turn = asyncio.run(session.open())
    assert session.client.calls == []
    assert session.qbaf.arguments["topic"].base_score == 0.5
    assert session.qbaf.arguments["topic"].elicited_tau is None
    assert session.stance() == pytest.approx(0.5)
    assert session.history() == [{"label": "start", "stance": 0.5}]
    intro = turn.intros[0][1]
    assert intro == session.persona().introduction()
    assert "no view on the claim" in intro
    assert session.persona().name in intro
    assert (session.persona().id, intro) in session._transcript


def test_expert_introduction_states_no_position():
    intro = DOMAIN.get_partner("homelessness_researcher").introduction()
    assert "no policy position" in intro and "Housing First" in intro


def test_confident_placement_still_pends_for_the_users_confidence():
    """The 75% gate no longer decides whether to ask: every argument comes
    back for the person's confidence, graph untouched, no scoring call."""
    session = make_session({
        RoutedMessage: routed("argument"),
        Attachment: _attach(confidence=99),
    })
    turn = asyncio.run(session.contribute("my point"))
    assert turn.pending is not None and turn.pending.confidence == 99
    assert set(session.qbaf.arguments) == {"topic"}
    assert [m for _, _, m in session.client.calls] == [RoutedMessage, Attachment]


def test_router_and_placement_use_the_classifier_client():
    """The neutral calls go to the classifier; the personas' client is idle
    for a user point."""
    classifier = StubClient({RoutedMessage: routed("argument"), Attachment: _attach()})
    session = make_session({}, classifier=classifier)
    land(session, "my point")
    assert [m for _, _, m in classifier.calls] == [RoutedMessage, Attachment]
    assert session.client.calls == []

def test_classifier_defaults_to_the_persona_client():
    session = make_session()
    assert session.classifier is session.client


# --- meeting notes ---------------------------------------------------------------


def _notes(summary="It went well."):
    return MeetingNotes(summary=summary, key_points=["[a1] the person argued X."])


def test_summary_is_written_by_the_neutral_note_taker_from_the_record():
    classifier = StubClient({MeetingNotes: _notes()})
    session = make_session({}, classifier=classifier)
    _grow(session, "a1", "topic", "attack", base_score=0.65, provenance="user")
    session._transcript.append(("user", "my point"))
    session._record("your argument")

    out = asyncio.run(session.summarise())

    assert out["notes"]["summary"] == "It went well."
    assert out["stance"] == pytest.approx(session.stance())
    assert out["turns"] == 1 and out["arguments"] == 1
    assert [m for _, _, m in classifier.calls] == [MeetingNotes]
    assert session.client.calls == []                      # not the persona
    system, user, _ = classifier.calls[0]
    assert "note-taker" in system and "never take a side" in system
    assert "[a1]" in user and "made by the person" in user
    assert "the person: my point" in user
    assert "your argument: " in user                       # trajectory
    assert "divergence [a1]" not in user                   # a1 has no objection


def test_summary_is_cached_until_the_state_changes():
    classifier = StubClient({MeetingNotes: _notes()})
    session = make_session({}, classifier=classifier)
    asyncio.run(session.summarise())
    asyncio.run(session.summarise())
    assert len(classifier.calls) == 1
    session.set_base_score("topic", 0.6)
    asyncio.run(session.summarise())
    assert len(classifier.calls) == 2


# --- the Devil's Advocate ------------------------------------------------------


def test_devils_advocate_joins_every_domain():
    from domains import DOMAINS
    for d in DOMAINS.values():
        assert d.partners[-1].id == "devils_advocate"
        assert d.get_partner("devils_advocate").kind == "contrarian"


def test_devils_advocate_only_objects_on_request():
    session = make_session({
        ExpansionResponse: drafts("Counterexample."),
        ScoredArgument: ScoredArgument(confidence=65, rationale="bites"),
    }, persona_ids=["lived_experience_advocate", "devils_advocate"])
    with pytest.raises(ValueError):
        asyncio.run(session.expand("topic", "support", "devils_advocate"))
    assert session.client.calls == []
    turn = asyncio.run(session.expand("topic", "attack", "devils_advocate"))
    node = session.qbaf.arguments[turn.added_nodes[0]]
    assert node.author == "devils_advocate"
    gen_system = session.client.system_prompt_for(ExpansionResponse)
    assert "devil's advocate" in gen_system and "argument attacking the statement" in gen_system
    assert "strawman" in gen_system


def test_devils_advocate_may_decline():
    session = make_session({ExpansionResponse: ExpansionResponse(argument=None)}, persona_id="devils_advocate")
    turn = asyncio.run(session.expand("topic", "attack"))
    assert turn.added_nodes == [] and turn.speaker == "devils_advocate"


def test_devils_advocate_introduction_states_its_role():
    intro = DOMAIN.get_partner("devils_advocate").introduction()
    assert "opposite side" in intro and "no view of my own" in intro

"""Engine tests.

The property tests matter most: they check the contestability results the whole
contestation loop rests on (Freedman et al., AAAI 2025, Props 1-2). If those
fail, contesting an argument can move the stance the wrong way and the demo is
lying to the user.
"""

import random

import pytest

from engine.attribution import (
    base_score_sensitivity,
    explain_change,
    ranked_leverage,
    removal_contribution,
    topic_strength,
)
from engine.qbaf import QBAF, Argument, Relation
from engine.lens import Goal, evaluation_lens
from engine.semantics import DFQuAD, LinearInfluence, Modular, QuadraticEnergy

TOL = 1e-9


def build(topic_tau: float, children: list[tuple[str, float]]) -> QBAF:
    """Topic plus a flat layer of leaves. `children` is (relation_type, tau)."""
    qbaf = QBAF(topic_id="t")
    qbaf.add_argument(Argument(arg_id="t", text="topic", base_score=topic_tau))
    for i, (rel_type, tau) in enumerate(children):
        aid = f"a{i}"
        qbaf.add_argument(Argument(arg_id=aid, text=aid, base_score=tau, depth=1))
        qbaf.add_relation(Relation(source_id=aid, target_id="t", relation_type=rel_type))
    return qbaf


def random_tree(rng: random.Random, depth: int = 2, breadth: int = 2) -> QBAF:
    qbaf = QBAF(topic_id="t")
    qbaf.add_argument(Argument(arg_id="t", text="topic", base_score=rng.random()))
    frontier = ["t"]
    counter = 0
    for d in range(1, depth + 1):
        next_frontier = []
        for parent in frontier:
            for _ in range(breadth):
                counter += 1
                aid = f"a{counter}"
                qbaf.add_argument(
                    Argument(arg_id=aid, text=aid, base_score=rng.random(), depth=d)
                )
                qbaf.add_relation(
                    Relation(
                        source_id=aid,
                        target_id=parent,
                        relation_type=rng.choice(["attack", "support"]),
                    )
                )
                next_frontier.append(aid)
        frontier = next_frontier
    return qbaf


def all_semantics():
    return [
        DFQuAD(),
        QuadraticEnergy(),
        Modular(influence=LinearInfluence(conservativeness=1.0)),
    ]


# --- DF-QuAD: hand-computable cases -----------------------------------------


def test_dfquad_no_relations_returns_base_scores():
    qbaf = build(0.4, [])
    assert topic_strength(qbaf, DFQuAD()) == pytest.approx(0.4)


def test_dfquad_single_supporter():
    # v_sup = 0.8, v_att = 0 -> 0.5 + 0.5 * 0.8
    qbaf = build(0.5, [("support", 0.8)])
    assert topic_strength(qbaf, DFQuAD()) == pytest.approx(0.9)


def test_dfquad_single_attacker():
    # v_att = 0.8, v_sup = 0 -> 0.5 * (1 - 0.8)
    qbaf = build(0.5, [("attack", 0.8)])
    assert topic_strength(qbaf, DFQuAD()) == pytest.approx(0.1)


def test_dfquad_balanced_attack_and_support_is_neutral():
    """C(v0, va, vs) = v0 when va == vs -- the paper's discontinuity-free case."""
    qbaf = build(0.5, [("attack", 0.8), ("support", 0.8)])
    assert topic_strength(qbaf, DFQuAD()) == pytest.approx(0.5)


def test_dfquad_aggregates_supporters_by_probabilistic_or():
    # prob-OR(0.5, 0.5) = 0.75 -> 0.5 + 0.5 * 0.75
    qbaf = build(0.5, [("support", 0.5), ("support", 0.5)])
    assert topic_strength(qbaf, DFQuAD()) == pytest.approx(0.875)


def test_quadratic_energy_single_supporter():
    # E = 0.8, h(0.8) = 0.64/1.64 -> 0.5 + 0.5 * h
    qbaf = build(0.5, [("support", 0.8)])
    expected = 0.5 + 0.5 * (0.64 / 1.64)
    assert topic_strength(qbaf, QuadraticEnergy()) == pytest.approx(expected)


# --- structural -------------------------------------------------------------


def test_polarity_alternates_with_attack_depth():
    qbaf = QBAF(topic_id="t")
    for aid in ("t", "a", "b", "c"):
        qbaf.add_argument(Argument(arg_id=aid, text=aid))
    qbaf.add_relation(Relation("a", "t", "attack"))
    qbaf.add_relation(Relation("b", "a", "attack"))
    qbaf.add_relation(Relation("c", "b", "attack"))

    assert qbaf.classify_polarity("a") == "con"
    assert qbaf.classify_polarity("b") == "pro"
    assert qbaf.classify_polarity("c") == "con"


def test_tree_detection_and_ambiguity():
    qbaf = QBAF(topic_id="t")
    for aid in ("t", "a", "b", "shared"):
        qbaf.add_argument(Argument(arg_id=aid, text=aid))
    qbaf.add_relation(Relation("a", "t", "support"))
    qbaf.add_relation(Relation("b", "t", "attack"))
    assert qbaf.is_tree()

    # A premise bearing on both branches breaks the tree, and cuts both ways.
    qbaf.add_relation(Relation("shared", "a", "support"))
    qbaf.add_relation(Relation("shared", "b", "support"))
    assert not qbaf.is_tree()
    assert qbaf.classify_polarity("shared") == "ambiguous"


def test_cycles_do_not_hang():
    qbaf = QBAF(topic_id="t")
    for aid in ("t", "a", "b"):
        qbaf.add_argument(Argument(arg_id=aid, text=aid, base_score=0.5))
    qbaf.add_relation(Relation("a", "t", "attack"))
    qbaf.add_relation(Relation("b", "a", "attack"))
    qbaf.add_relation(Relation("a", "b", "attack"))

    for sem in all_semantics():
        strengths = sem.compute(qbaf)
        assert all(0.0 <= v <= 1.0 for v in strengths.values())


def test_remove_argument_cascades_to_dependents():
    """Removing an argument takes its whole dependent subtree, not just the node."""
    qbaf = build(0.5, [("attack", 0.8), ("support", 0.6)])
    qbaf.add_argument(Argument(arg_id="deep", text="deep", base_score=0.9, depth=2))
    qbaf.add_relation(Relation("deep", "a0", "attack"))
    qbaf.add_argument(Argument(arg_id="deeper", text="deeper", base_score=0.4, depth=3))
    qbaf.add_relation(Relation("deeper", "deep", "support"))

    qbaf.remove_argument("a0")

    assert set(qbaf.arguments) == {"t", "a1"}
    assert [r.key for r in qbaf.relations] == [("a1", "t")]
    assert all(qbaf.classify_polarity(a) != "isolated" for a in qbaf.arguments)


def test_remove_argument_keeps_multipath_nodes():
    """A node that still reaches the topic another way survives the cascade."""
    qbaf = QBAF(topic_id="t")
    for aid in ("t", "a", "b", "shared"):
        qbaf.add_argument(Argument(arg_id=aid, text=aid))
    qbaf.add_relation(Relation("a", "t", "support"))
    qbaf.add_relation(Relation("b", "t", "attack"))
    qbaf.add_relation(Relation("shared", "a", "support"))
    qbaf.add_relation(Relation("shared", "b", "support"))

    qbaf.remove_argument("a")

    assert set(qbaf.arguments) == {"t", "b", "shared"}
    # Only the path through b remains, so shared is unambiguous again.
    assert qbaf.classify_polarity("shared") == "con"


def test_remove_matches_removal_contribution():
    """Rejecting an argument moves the topic by exactly the contribution
    `removal_contribution` reported for it -- the consistency the delta
    explanations rely on."""
    qbaf = build(0.5, [("attack", 0.8)])
    qbaf.add_argument(Argument(arg_id="deep", text="deep", base_score=0.9, depth=2))
    qbaf.add_relation(Relation("deep", "a0", "attack"))
    sem = DFQuAD()

    contribution = removal_contribution(qbaf, sem, "a0")
    before = topic_strength(qbaf, sem)
    qbaf.remove_argument("a0")

    assert before - topic_strength(qbaf, sem) == pytest.approx(contribution)


def test_rejects_out_of_range_base_score():
    with pytest.raises(ValueError):
        Argument(arg_id="x", text="x", base_score=1.5)


def test_relation_guards():
    qbaf = QBAF(topic_id="t")
    qbaf.add_argument(Argument(arg_id="t", text="t"))
    qbaf.add_argument(Argument(arg_id="a", text="a"))

    assert qbaf.add_relation(Relation("a", "a", "attack")) is None  # self-loop
    assert qbaf.add_relation(Relation("ghost", "t", "attack")) is None  # unknown
    assert qbaf.add_relation(Relation("a", "t", "attack")) is not None
    assert qbaf.add_relation(Relation("a", "t", "attack")) is None  # duplicate
    assert len(qbaf.relations) == 1


# --- contestability properties ----------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_base_score_contestability(seed):
    """Property 1: raising tau of a pro argument must not lower the topic;
    raising tau of a con argument must not raise it."""
    rng = random.Random(seed)
    qbaf = random_tree(rng, depth=2, breadth=2)

    target = rng.choice([a for a in qbaf.arguments if a != qbaf.topic_id])
    polarity = qbaf.classify_polarity(target)
    if polarity not in ("pro", "con"):
        pytest.skip("target is not single-path connected")

    for sem in all_semantics():
        before = topic_strength(qbaf, sem)
        original = qbaf.arguments[target].base_score
        qbaf.arguments[target].base_score = min(1.0, original + 0.2)
        after = topic_strength(qbaf, sem)
        qbaf.arguments[target].base_score = original

        if polarity == "pro":
            assert after >= before - TOL, f"{sem.name}: pro arg lowered the topic"
        else:
            assert after <= before + TOL, f"{sem.name}: con arg raised the topic"


@pytest.mark.parametrize("seed", range(40))
def test_argument_relation_contestability(seed):
    """Property 2: adding an argument moves the topic in the direction implied
    by the new argument's polarity in the extended graph."""
    rng = random.Random(seed)
    qbaf = random_tree(rng, depth=2, breadth=2)

    parent = rng.choice(list(qbaf.arguments))
    rel_type = rng.choice(["attack", "support"])

    for sem in all_semantics():
        before = topic_strength(qbaf, sem)

        qbaf.add_argument(Argument(arg_id="new", text="new", base_score=0.7))
        qbaf.add_relation(Relation("new", parent, rel_type))
        polarity = qbaf.classify_polarity("new")
        after = topic_strength(qbaf, sem)
        qbaf.remove_argument("new")

        if polarity == "pro":
            assert after >= before - TOL, f"{sem.name}: pro addition lowered the topic"
        elif polarity == "con":
            assert after <= before + TOL, f"{sem.name}: con addition raised the topic"


@pytest.mark.parametrize("seed", range(20))
def test_strengths_stay_in_unit_interval(seed):
    rng = random.Random(seed)
    qbaf = random_tree(rng, depth=3, breadth=2)
    for sem in all_semantics():
        assert all(0.0 <= v <= 1.0 for v in sem.compute(qbaf).values())
        assert sem.converged


# --- persona responsiveness -------------------------------------------------


def test_higher_conservativeness_moves_less():
    """The epistemic-disposition knob: a more conservative persona is shifted
    less by the same argument."""
    qbaf = build(0.5, [("support", 0.9)])

    responsive = topic_strength(qbaf, Modular(influence=LinearInfluence(1.0)))
    conservative = topic_strength(qbaf, Modular(influence=LinearInfluence(4.0)))

    assert responsive > conservative > 0.5


def test_conservativeness_must_be_positive():
    with pytest.raises(ValueError):
        LinearInfluence(conservativeness=0.0)


# --- attribution ------------------------------------------------------------


def test_removal_contribution_signs_match_polarity():
    qbaf = build(0.5, [("support", 0.8), ("attack", 0.6)])
    sem = DFQuAD()

    assert removal_contribution(qbaf, sem, "a0") > 0  # supporter was pushing up
    assert removal_contribution(qbaf, sem, "a1") < 0  # attacker was pushing down


def test_removal_takes_dependents_with_it():
    qbaf = build(0.5, [("attack", 0.8)])
    qbaf.add_argument(Argument(arg_id="deep", text="deep", base_score=0.9, depth=2))
    qbaf.add_relation(Relation("deep", "a0", "attack"))

    sem = DFQuAD()
    contribution = removal_contribution(qbaf, sem, "a0")

    ablated = QBAF.from_dict(qbaf.to_dict())
    ablated.remove_argument("a0")
    ablated.remove_argument("deep")
    assert topic_strength(qbaf, sem) - topic_strength(ablated, sem) == pytest.approx(
        contribution
    )


def test_sensitivity_sign_matches_polarity():
    qbaf = build(0.5, [("support", 0.5), ("attack", 0.5)])
    sem = DFQuAD()

    assert base_score_sensitivity(qbaf, sem, "a0") > 0
    assert base_score_sensitivity(qbaf, sem, "a1") < 0


def test_explain_change_reports_direction_and_driver():
    before = build(0.5, [("attack", 0.8)])
    after = QBAF.from_dict(before.to_dict())
    after.add_argument(Argument(arg_id="rebut", text="rebuttal", base_score=0.9, depth=2))
    after.add_relation(Relation("rebut", "a0", "attack"))

    explanation = explain_change(before, after, DFQuAD())

    assert explanation.direction == "strengthened"
    assert explanation.topic_delta > 0
    driver = explanation.primary_driver()
    assert driver is not None and driver.arg_id in ("rebut", "a0")


def test_ranked_leverage_orders_by_influence():
    qbaf = build(0.5, [("support", 0.9), ("attack", 0.1)])
    qbaf.add_argument(Argument(arg_id="minor", text="minor", base_score=0.05, depth=2))
    qbaf.add_relation(Relation("minor", "a1", "support"))

    ranked = ranked_leverage(qbaf, DFQuAD(), limit=3)

    assert [arg_id for arg_id, _ in ranked][0] in ("a0", "a1")
    assert abs(ranked[0][1]) >= abs(ranked[-1][1])


# --- evaluation lenses -------------------------------------------------------


def _tag(goal_id, direction="positive", source="judged"):
    return {"goal_id": goal_id, "direction": direction, "source": source}


GOALS = (
    Goal(id="sdg_1", label="SDG 1 · No Poverty", kind="sdg"),
    Goal(id="sdg_16", label="SDG 16 · Peace & Justice", kind="sdg"),
)


def test_lens_is_computed_by_the_semantics_and_leaves_the_graph_alone():
    qbaf = build(0.5, [("support", 0.8), ("attack", 0.6)])
    qbaf.arguments["a0"].goal_impacts = [_tag("sdg_1")]
    qbaf.arguments["a1"].goal_impacts = [_tag("sdg_1", direction="negative")]

    sem = DFQuAD()
    sigma_before = sem.compute(qbaf)
    lens = evaluation_lens(qbaf, sem, GOALS)

    # a0 supports E_G at sigma 0.8, a1 attacks it at 0.6: DF-QuAD from tau 0.5.
    entry = next(e for e in lens if e["id"] == "sdg_1")
    assert entry["sigma"] == pytest.approx(0.5 + 0.5 * (0.8 - 0.6))
    assert entry["verdict"] == "supported"
    assert {c["id"] for c in entry["contributions"]} == {"a0", "a1"}
    # untouched goal: neutral, contested, no contributions
    other = next(e for e in lens if e["id"] == "sdg_16")
    assert other["sigma"] == pytest.approx(0.5)
    assert other["verdict"] == "contested"
    assert other["contributions"] == []
    # grounding is virtual: the persistent graph and its strengths are unchanged
    assert set(qbaf.arguments) == {"t", "a0", "a1"}
    assert sem.compute(qbaf) == sigma_before


def test_lens_is_deliberation_conditional():
    """A CON argument whose substance advances a goal *supports* its lens --
    unsigned by polarity toward the claim."""
    qbaf = build(0.5, [("attack", 0.8)])
    qbaf.arguments["a0"].goal_impacts = [_tag("sdg_1")]

    [entry] = [e for e in evaluation_lens(qbaf, DFQuAD(), GOALS) if e["contributions"]]
    assert entry["id"] == "sdg_1"
    assert entry["sigma"] > 0.5 and entry["verdict"] == "supported"


def test_lens_verdict_bands():
    qbaf = build(0.5, [("support", 0.08), ("attack", 0.9)])
    qbaf.arguments["a0"].goal_impacts = [_tag("sdg_1")]      # weak support
    qbaf.arguments["a1"].goal_impacts = [_tag("sdg_16")]     # strong support

    lens = {e["id"]: e for e in evaluation_lens(qbaf, DFQuAD(), GOALS)}
    assert lens["sdg_1"]["verdict"] == "contested"   # 0.5 + 0.5*0.08 = 0.54
    assert lens["sdg_16"]["verdict"] == "supported"  # 0.5 + 0.5*0.9 = 0.95

    qbaf.arguments["a1"].goal_impacts = [_tag("sdg_16", direction="negative")]
    lens = {e["id"]: e for e in evaluation_lens(qbaf, DFQuAD(), GOALS)}
    assert lens["sdg_16"]["verdict"] == "undermined"  # 0.5 * (1 - 0.9) = 0.05


def test_lens_excludes_topic_and_unknown_goal_tags():
    qbaf = build(0.5, [("support", 0.8)])
    qbaf.arguments["t"].goal_impacts = [_tag("sdg_1")]        # topic: ignored
    qbaf.arguments["a0"].goal_impacts = [_tag("sdg_99")]      # unselected: ignored

    lens = evaluation_lens(qbaf, DFQuAD(), GOALS)
    assert all(e["sigma"] == pytest.approx(0.5) for e in lens)
    assert all(not e["contributions"] for e in lens)


def test_lens_tolerates_duplicate_goals():
    qbaf = build(0.5, [("support", 0.8)])
    qbaf.arguments["a0"].goal_impacts = [_tag("sdg_1")]
    twice = (GOALS[0], GOALS[0], GOALS[1])
    lens = evaluation_lens(qbaf, DFQuAD(), twice)
    assert [e["id"] for e in lens] == ["sdg_1", "sdg_16"]


def test_lens_empty_without_goals():
    qbaf = build(0.5, [("support", 0.8)])
    assert evaluation_lens(qbaf, DFQuAD(), ()) == []


def test_lens_contributions_match_the_computation_exactly():
    """A node pathologically tagged twice for one goal grounds one edge; the
    explanation must list exactly that edge, not the rejected duplicate --
    contributions are the computation's audit trail, nothing else."""
    qbaf = build(0.5, [("support", 0.8)])
    qbaf.arguments["a0"].goal_impacts = [
        _tag("sdg_1", direction="positive"),
        _tag("sdg_1", direction="negative"),  # duplicate: graph rejects the edge
    ]
    [entry] = [e for e in evaluation_lens(qbaf, DFQuAD(), GOALS) if e["contributions"]]
    assert len(entry["contributions"]) == 1
    assert entry["contributions"][0]["direction"] == "positive"
    assert entry["sigma"] == pytest.approx(0.9)  # 0.5 + 0.5*0.8, one supporter


def test_lens_verdict_band_boundaries_are_strict():
    qbaf = build(0.5, [("support", 0.1), ("attack", 0.1)])
    qbaf.arguments["a0"].goal_impacts = [_tag("sdg_1")]                       # 0.55
    qbaf.arguments["a1"].goal_impacts = [_tag("sdg_16", direction="negative")]  # 0.45

    lens = {e["id"]: e for e in evaluation_lens(qbaf, DFQuAD(), GOALS)}
    # exactly on the band edges reads contested, not supported/undermined
    assert lens["sdg_1"]["sigma"] == pytest.approx(0.55)
    assert lens["sdg_1"]["verdict"] == "contested"
    assert lens["sdg_16"]["sigma"] == pytest.approx(0.45)
    assert lens["sdg_16"]["verdict"] == "contested"


def test_lens_includes_arguments_whatever_their_relation_to_the_claim():
    """Deliberation-conditional: an argument bears on a goal by its substance,
    so even a claim-ambiguous argument contributes to the lens."""
    qbaf = QBAF(topic_id="t")
    for aid in ("t", "a", "b", "shared"):
        qbaf.add_argument(Argument(arg_id=aid, text=aid, base_score=0.8))
    qbaf.add_relation(Relation("a", "t", "support"))
    qbaf.add_relation(Relation("b", "t", "attack"))
    qbaf.add_relation(Relation("shared", "a", "support"))
    qbaf.add_relation(Relation("shared", "b", "support"))
    assert qbaf.classify_polarity("shared") == "ambiguous"
    qbaf.arguments["shared"].goal_impacts = [_tag("sdg_1")]

    [entry] = [e for e in evaluation_lens(qbaf, DFQuAD(), GOALS) if e["contributions"]]
    assert entry["contributions"][0]["id"] == "shared"
    assert entry["sigma"] > 0.5


def test_lens_converges_on_cyclic_graphs():
    qbaf = QBAF(topic_id="t")
    for aid in ("t", "a", "b"):
        qbaf.add_argument(Argument(arg_id=aid, text=aid, base_score=0.5))
    qbaf.add_relation(Relation("a", "t", "attack"))
    qbaf.add_relation(Relation("b", "a", "attack"))
    qbaf.add_relation(Relation("a", "b", "attack"))
    qbaf.arguments["a"].goal_impacts = [_tag("sdg_1")]
    qbaf.arguments["b"].goal_impacts = [_tag("sdg_1", direction="negative")]

    for sem in all_semantics():
        lens = evaluation_lens(qbaf, sem, GOALS)
        assert all(0.0 <= e["sigma"] <= 1.0 for e in lens)
        assert sem.converged


# --- lens properties ----------------------------------------------------------


def _random_tags(rng, qbaf, goals):
    """Tag every non-topic argument with probability 1/2, random direction."""
    for arg_id, arg in qbaf.arguments.items():
        if arg_id == qbaf.topic_id or rng.random() < 0.5:
            continue
        goal = rng.choice(goals)
        arg.goal_impacts = [
            _tag(goal.id, direction=rng.choice(["positive", "negative"]))
        ]


@pytest.mark.parametrize("seed", range(30))
def test_lens_grounding_never_perturbs_the_debate(seed):
    """Sink invariance: goal nodes have no outgoing edges, so grounding the
    lens must leave every argument's strength -- and hence the stance --
    bit-identical under every semantics. This is what licenses computing the
    lens and the stance from one augmented graph."""
    rng = random.Random(seed)
    qbaf = random_tree(rng, depth=2, breadth=2)
    _random_tags(rng, qbaf, GOALS)

    for sem in all_semantics():
        plain = sem.compute(qbaf)
        lens = evaluation_lens(qbaf, sem, GOALS)
        for entry in lens:
            for c in entry["contributions"]:
                assert c["sigma"] == pytest.approx(plain[c["id"]], abs=1e-9)
            assert 0.0 <= entry["sigma"] <= 1.0
        # and the persistent graph is untouched
        assert set(qbaf.arguments) == {
            a for a in qbaf.arguments if not a.startswith("goal:")
        }


@pytest.mark.parametrize("seed", range(30))
def test_lens_contestability_for_a_single_tagged_argument(seed):
    """Property: when exactly one argument is tagged for a goal, raising that
    argument's tau moves sigma(E_G) weakly in the tag's direction, under every
    semantics -- the contestability guarantee, extended to the lens."""
    rng = random.Random(seed)
    qbaf = random_tree(rng, depth=2, breadth=2)
    target = rng.choice([a for a in qbaf.arguments if a != qbaf.topic_id])
    direction = rng.choice(["positive", "negative"])
    qbaf.arguments[target].goal_impacts = [_tag("sdg_1", direction=direction)]
    goal = (Goal(id="sdg_1", label="SDG 1", kind="sdg"),)

    for sem in all_semantics():
        [entry] = evaluation_lens(qbaf, sem, goal)
        before = entry["sigma"]
        original = qbaf.arguments[target].base_score
        qbaf.arguments[target].base_score = min(1.0, original + 0.2)
        [entry] = evaluation_lens(qbaf, sem, goal)
        after = entry["sigma"]
        qbaf.arguments[target].base_score = original

        if direction == "positive":
            assert after >= before - TOL, f"{sem.name}: helping arg lowered the goal"
        else:
            assert after <= before + TOL, f"{sem.name}: straining arg raised the goal"


def test_lens_monotonicity_honestly_fails_across_argument_channels():
    """Documented limit of the guarantee: with several tagged, *related*
    arguments, the single-argument monotonicity does not extend. Here `amp`
    (tagged as helping the goal) supports two arguments that strain it, so
    raising tau(amp) amplifies both strains more than its own help. This is
    correct deliberation-conditional behaviour, not a bug -- the test pins it
    so a future 'fix' cannot silently change the semantics."""
    qbaf = build(0.5, [("attack", 0.3), ("attack", 0.3)])
    qbaf.add_argument(Argument(arg_id="amp", text="amp", base_score=0.2, depth=2))
    qbaf.add_relation(Relation("amp", "a0", "support"))
    qbaf.add_relation(Relation("amp", "a1", "support"))
    qbaf.arguments["a0"].goal_impacts = [_tag("sdg_1", direction="negative")]
    qbaf.arguments["a1"].goal_impacts = [_tag("sdg_1", direction="negative")]
    qbaf.arguments["amp"].goal_impacts = [_tag("sdg_1")]  # 'helps' the goal
    goal = (Goal(id="sdg_1", label="SDG 1", kind="sdg"),)
    sem = Modular(influence=LinearInfluence(conservativeness=1.0))

    # k=1 sum aggregation: agg(E_G) = sigma(amp) - 2*(0.3 + 0.7*sigma(amp)),
    # which *decreases* in sigma(amp): 0.16 at tau=0.2, 0.02 at tau=0.9.
    [before] = evaluation_lens(qbaf, sem, goal)
    qbaf.arguments["amp"].base_score = 0.9
    [after] = evaluation_lens(qbaf, sem, goal)

    assert before["sigma"] == pytest.approx(0.16)
    assert after["sigma"] == pytest.approx(0.02)
    assert after["sigma"] < before["sigma"]


def test_snapshots_do_not_alias_mutable_fields():
    """A serialised copy must not share goal_impacts lists with the original:
    mutating one graph's tags must never reach into stored snapshots."""
    qbaf = build(0.5, [("support", 0.8)])
    qbaf.arguments["a0"].goal_impacts = [_tag("sdg_1")]

    restored = QBAF.from_dict(qbaf.to_dict())
    qbaf.arguments["a0"].goal_impacts.append(_tag("sdg_16"))

    assert restored.arguments["a0"].goal_impacts == [_tag("sdg_1")]


def test_roundtrip_serialisation():
    qbaf = random_tree(random.Random(0), depth=2, breadth=2)
    restored = QBAF.from_dict(qbaf.to_dict())

    assert restored.arguments.keys() == qbaf.arguments.keys()
    assert len(restored.relations) == len(qbaf.relations)
    assert topic_strength(restored, DFQuAD()) == pytest.approx(
        topic_strength(qbaf, DFQuAD())
    )

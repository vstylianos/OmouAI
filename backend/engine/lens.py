"""Evaluation lenses: what the deliberation says about each selected goal.

Goals and indicators are not arguments, so they never enter the persistent
graph. Each selected goal G defines a *virtual* evaluation root E_G -- a
lifted schema grounded fresh at every evaluation: E_G is materialised in
memory with a neutral base score, every argument tagged with G attaches to it
by its tag's direction, the same gradual semantics runs over the augmented
graph, sigma(E_G) is read off, and the node is discarded. The persistent
graph, the tree guarantees, and the claim node's role are untouched: goal
nodes are sinks, so no argument's strength or polarity changes.

The reading is deliberation-conditional: an argument bears on a goal by its
own substance, unsigned by its polarity toward the claim -- sigma(E_G)
answers "how does G fare, given everything on the table, alternatives
included". Verdicts read sigma against the neutral prior: the case for the
goal's advancement is *supported*, *contested*, or *undermined*. Low sigma
means the case for advancement fails; whether that is stagnation or active
regression is visible from the attacking arguments themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from engine.qbaf import QBAF, Argument, Relation
from engine.semantics import QBAFSemantics

#: Every virtual goal root starts here: the neutral point of the evaluation,
#: not a claim about the world.
GOAL_TAU = 0.5
#: sigma above/below these bands reads supported/undermined; between, contested.
SUPPORTED_ABOVE = 0.55
UNDERMINED_BELOW = 0.45

#: Short names for the 17 UN Sustainable Development Goals.
SDG_NAMES: dict[int, str] = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health & Well-Being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water & Sanitation",
    7: "Affordable & Clean Energy",
    8: "Decent Work & Economic Growth",
    9: "Industry, Innovation & Infrastructure",
    10: "Reduced Inequalities",
    11: "Sustainable Cities & Communities",
    12: "Responsible Consumption & Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life on Land",
    16: "Peace, Justice & Strong Institutions",
    17: "Partnerships for the Goals",
}


@dataclass(frozen=True)
class Goal:
    """One selected deliberation target: an SDG, an indicator, or the user's own."""

    id: str  # "sdg_1", "ind_rough_sleeping_count", "custom_1"
    label: str
    kind: Literal["sdg", "indicator", "custom"]


def _verdict(sigma: float) -> str:
    if sigma > SUPPORTED_ABOVE:
        return "supported"
    if sigma < UNDERMINED_BELOW:
        return "undermined"
    return "contested"


def evaluation_lens(
    qbaf: QBAF, semantics: QBAFSemantics, goals: tuple[Goal, ...]
) -> list[dict]:
    """Ground the lens and evaluate every selected goal. Pure engine, no LLM.

    Returns one entry per goal: {id, label, kind, sigma, verdict,
    contributions}, where contributions lists each tagged argument at its
    current strength -- the explanation of why the goal is supported or
    degrading. The topic is excluded: its sigma is the stance, and coupling
    it into every goal would double-count the whole debate.
    """
    if not goals:
        return []

    # Tolerate duplicate goal ids (belt to build_goals' braces): grounding
    # the same virtual root twice would raise on the duplicate arg_id.
    unique: list[Goal] = []
    seen: set[str] = set()
    for g in goals:
        if g.id not in seen:
            seen.add(g.id)
            unique.append(g)
    goals = tuple(unique)

    grounded = QBAF.from_dict(qbaf.to_dict())
    goal_node = {g.id: f"goal:{g.id}" for g in goals}
    for g in goals:
        grounded.add_argument(
            Argument(arg_id=goal_node[g.id], text=g.label, base_score=GOAL_TAU)
        )

    edges: dict[str, list[dict]] = {g.id: [] for g in goals}
    for arg_id, arg in qbaf.arguments.items():
        if arg_id == qbaf.topic_id:
            continue
        for tag in arg.goal_impacts:
            if tag["goal_id"] not in goal_node:
                continue
            added = grounded.add_relation(
                Relation(
                    source_id=arg_id,
                    target_id=goal_node[tag["goal_id"]],
                    relation_type=(
                        "support" if tag["direction"] == "positive" else "attack"
                    ),
                )
            )
            # A rejected duplicate (same node tagged twice for one goal) must
            # not appear in the explanation: contributions list exactly the
            # edges the computation used, nothing else.
            if added is not None:
                edges[tag["goal_id"]].append({"arg_id": arg_id, "tag": tag})

    sigma = semantics.compute(grounded)

    lens = []
    for g in goals:
        contributions = [
            {
                "id": e["arg_id"],
                "text": qbaf.arguments[e["arg_id"]].text,
                "sigma": sigma[e["arg_id"]],
                "direction": e["tag"]["direction"],
                "source": e["tag"].get("source", "judged"),
            }
            for e in edges[g.id]
        ]
        contributions.sort(key=lambda c: c["sigma"], reverse=True)
        lens.append(
            {
                "id": g.id,
                "label": g.label,
                "kind": g.kind,
                "sigma": sigma[goal_node[g.id]],
                "verdict": _verdict(sigma[goal_node[g.id]]),
                "contributions": contributions,
            }
        )
    return lens

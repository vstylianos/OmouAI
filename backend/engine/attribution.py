"""Attribution: explaining *why* the stance is what it is, and what moved it.

This is what turns a number into a demo. When a user contests something we do
not just recompute -- we report which argument carried the change and by how
much, which is exactly the claim the contestability results license.

Two complementary measures, following the argument-attribution line of work on
QBAFs (Yin et al.; Kampik et al.):

- `removal_contribution` -- ablate an argument and see how far the topic moves.
  Answers "how much is this argument doing?"
- `base_score_sensitivity` -- perturb an argument's base score and see how far
  the topic moves. Answers "how much would it help to argue about this?"
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from engine.qbaf import QBAF
from engine.semantics import QBAFSemantics


def topic_strength(qbaf: QBAF, semantics: QBAFSemantics) -> float:
    return semantics.compute(qbaf)[qbaf.topic_id]


def dependents_of(qbaf: QBAF, arg_id: str) -> set[str]:
    """Arguments whose every path to the topic runs through `arg_id`.

    Removing `arg_id` orphans exactly these. In a tree this is its subtree.
    """
    return qbaf.dependents_of(arg_id)


def removal_contribution(
    qbaf: QBAF, semantics: QBAFSemantics, arg_id: str
) -> float:
    """Change in topic strength caused by removing `arg_id` and its dependents.

    Positive means the argument was pushing the topic *up*; negative, down.
    `QBAF.remove_argument` cascades through dependents itself, so this ablation
    is exactly what a user's reject does.
    """
    if arg_id == qbaf.topic_id:
        raise ValueError("cannot ablate the topic argument")

    before = topic_strength(qbaf, semantics)

    ablated = copy.deepcopy(qbaf)
    ablated.remove_argument(arg_id)

    return before - topic_strength(ablated, semantics)


def base_score_sensitivity(
    qbaf: QBAF, semantics: QBAFSemantics, arg_id: str, delta: float = 0.1
) -> float:
    """Approximate d(topic strength) / d(base score of arg_id).

    Central difference, clamped at the ends of [0,1]. Ranking by the absolute
    value of this gives "the arguments where your disagreement matters most".
    """
    if arg_id == qbaf.topic_id:
        raise ValueError("sensitivity of the topic to itself is not meaningful")

    original = qbaf.arguments[arg_id].base_score
    lo = max(0.0, original - delta)
    hi = min(1.0, original + delta)
    if hi == lo:
        return 0.0

    probe = copy.deepcopy(qbaf)

    probe.arguments[arg_id].base_score = lo
    strength_lo = topic_strength(probe, semantics)

    probe.arguments[arg_id].base_score = hi
    strength_hi = topic_strength(probe, semantics)

    return (strength_hi - strength_lo) / (hi - lo)


@dataclass
class NodeChange:
    arg_id: str
    text: str
    before: float | None
    after: float | None
    delta: float

    @property
    def status(self) -> str:
        if self.before is None:
            return "added"
        if self.after is None:
            return "removed"
        return "changed"


@dataclass
class DeltaExplanation:
    """What happened to the stance between two states of the graph."""

    topic_before: float
    topic_after: float
    changes: list[NodeChange] = field(default_factory=list)

    @property
    def topic_delta(self) -> float:
        return self.topic_after - self.topic_before

    @property
    def direction(self) -> str:
        if abs(self.topic_delta) < 1e-9:
            return "unchanged"
        return "strengthened" if self.topic_delta > 0 else "weakened"

    def primary_driver(self) -> NodeChange | None:
        """The single node that moved most. What the agent cites when it concedes."""
        candidates = [c for c in self.changes if c.status != "changed" or c.delta]
        if not candidates:
            return None
        return max(candidates, key=lambda c: abs(c.delta))


def explain_change(
    before: QBAF, after: QBAF, semantics: QBAFSemantics
) -> DeltaExplanation:
    """Diff two graph states and report what moved the stance."""
    sigma_before = semantics.compute(before)
    sigma_after = semantics.compute(after)

    changes: list[NodeChange] = []
    for arg_id in sigma_before.keys() | sigma_after.keys():
        if arg_id in (before.topic_id, after.topic_id):
            continue
        b = sigma_before.get(arg_id)
        a = sigma_after.get(arg_id)
        source = after.arguments.get(arg_id) or before.arguments[arg_id]
        changes.append(
            NodeChange(
                arg_id=arg_id,
                text=source.text,
                before=b,
                after=a,
                delta=(a or 0.0) - (b or 0.0),
            )
        )

    changes.sort(key=lambda c: abs(c.delta), reverse=True)

    return DeltaExplanation(
        topic_before=sigma_before[before.topic_id],
        topic_after=sigma_after[after.topic_id],
        changes=changes,
    )


def ranked_leverage(
    qbaf: QBAF, semantics: QBAFSemantics, limit: int = 5
) -> list[tuple[str, float]]:
    """Arguments ranked by how much moving them would move the topic.

    Powers "here is what would change my mind" -- and, with node types, the
    distinction between a disagreement that evidence could settle and one that
    bottoms out in values.
    """
    scored = [
        (arg_id, base_score_sensitivity(qbaf, semantics, arg_id))
        for arg_id in qbaf.arguments
        if arg_id != qbaf.topic_id
    ]
    scored.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return scored[:limit]

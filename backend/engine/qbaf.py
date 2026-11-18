"""Core QBAF data structures for persona-conditioned argumentation.

A QBAF is a quadruple <A, R-, R+, tau>: arguments, attack and support relations,
and a base score function. Here the base scores carry the persona's *beliefs* --
they are elicited per-argument in context, and the final stance is computed from
them by a gradual semantics rather than asserted by the model.

Structurally we default to trees rooted at the topic argument (as in ArgLLM),
because that is what makes pro/con well-defined and the contestability
guarantees hold. The data model permits general DAGs so the restriction can be
relaxed later; `is_tree` and `classify_polarity` report when it has been.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

NodeType = Literal["empirical", "predictive", "normative"]
"""Empirical claims are contestable on evidence, normative ones on values.

Base-score elicitation routes on this: empirical nodes are scored for evidential
support, normative ones for endorsement under the persona's commitments.
"""

Provenance = Literal["generated", "user", "document"]
"""Where a node came from: persona generation, the user, or a mined document."""

RelationType = Literal["attack", "support"]


@dataclass
class Argument:
    arg_id: str
    text: str
    node_type: NodeType = "empirical"
    base_score: float = 0.5
    provenance: Provenance = "generated"
    depth: int = 0
    summary: str = ""
    #: Free-text reason the persona gave for this base score. Shown on contest.
    rationale: str = ""
    #: Impacts on the deliberation's selected goals: [{"goal_id": "sdg_1",
    #: "direction": "positive", "source": "judged"|"user"}, ...]. These
    #: create the evaluation-lens edges; empty for the topic.
    goal_impacts: list[dict] = field(default_factory=list)
    #: The tau the agent originally elicited, kept when the user overrides
    #: base_score via the slider -- the gap between the two is the user's
    #: recorded disagreement. None where no elicitation happened.
    elicited_tau: float | None = None
    #: Which persona authored and scored this argument (its id); None for
    #: the topic and for the user's own points.
    author: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_score <= 1.0:
            raise ValueError(
                f"base_score for {self.arg_id} out of range: {self.base_score}"
            )


@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: RelationType
    #: Why the LLM (or user) claims this relation holds. Surfaced on contest.
    rationale: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.source_id, self.target_id)


@dataclass
class QBAF:
    """A QBAF rooted at `topic_id`."""

    topic_id: str
    arguments: dict[str, Argument] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    # -- construction --------------------------------------------------

    def add_argument(self, arg: Argument) -> Argument:
        if arg.arg_id in self.arguments:
            raise ValueError(f"duplicate arg_id: {arg.arg_id}")
        self.arguments[arg.arg_id] = arg
        return arg

    def add_relation(self, rel: Relation) -> Relation | None:
        """Add a relation, ignoring self-loops, unknown endpoints and duplicates."""
        if rel.source_id == rel.target_id:
            return None
        if rel.source_id not in self.arguments or rel.target_id not in self.arguments:
            return None
        if any(r.key == rel.key for r in self.relations):
            return None
        self.relations.append(rel)
        return rel

    def remove_argument(self, arg_id: str) -> None:
        """Remove an argument, its dependents, and every relation touching them.

        Dependents -- arguments whose every path to the topic ran through
        `arg_id` -- are removed too, not left floating: an argument that
        existed only to bear on the removed one has nothing left to bear on.
        This matches what `removal_contribution` ablates, so rejecting an
        argument moves the stance by exactly the contribution it reported.
        """
        doomed = self.dependents_of(arg_id) | {arg_id}
        for aid in doomed:
            self.arguments.pop(aid, None)
        self.relations = [
            r
            for r in self.relations
            if r.source_id not in doomed and r.target_id not in doomed
        ]

    # -- adjacency -----------------------------------------------------

    def attackers_of(self, arg_id: str) -> list[str]:
        return [
            r.source_id
            for r in self.relations
            if r.target_id == arg_id and r.relation_type == "attack"
        ]

    def supporters_of(self, arg_id: str) -> list[str]:
        return [
            r.source_id
            for r in self.relations
            if r.target_id == arg_id and r.relation_type == "support"
        ]

    def children_of(self, arg_id: str) -> list[str]:
        return [r.source_id for r in self.relations if r.target_id == arg_id]

    def parents_of(self, arg_id: str) -> list[str]:
        return [r.target_id for r in self.relations if r.source_id == arg_id]

    @property
    def attacks(self) -> list[Relation]:
        return [r for r in self.relations if r.relation_type == "attack"]

    @property
    def supports(self) -> list[Relation]:
        return [r for r in self.relations if r.relation_type == "support"]

    @property
    def topic(self) -> Argument:
        return self.arguments[self.topic_id]

    def leaves(self) -> list[str]:
        """Arguments with no attackers or supporters."""
        return [a for a in self.arguments if not self.children_of(a)]

    def dependents_of(self, arg_id: str) -> set[str]:
        """Arguments whose every path to the topic runs through `arg_id`.

        Removing `arg_id` orphans exactly these. In a tree this is its subtree.
        Arguments already unattached (no path to the topic) are not included.
        """
        orphaned = set()
        for other in self.arguments:
            if other in (arg_id, self.topic_id):
                continue
            paths = list(self.paths_to_topic(other))
            if not paths:
                continue
            if all(any(r.source_id == arg_id for r in path) for path in paths):
                orphaned.add(other)
        return orphaned

    # -- structure -----------------------------------------------------

    def paths_to_topic(self, arg_id: str, _seen: frozenset[str] = frozenset()) -> Iterator[list[Relation]]:
        """Yield every path of relations from `arg_id` up to the topic.

        Cycles are pruned rather than raising: a node already on the current
        path is not revisited.
        """
        if arg_id == self.topic_id:
            yield []
            return
        if arg_id in _seen:
            return
        seen = _seen | {arg_id}
        for rel in self.relations:
            if rel.source_id != arg_id:
                continue
            for tail in self.paths_to_topic(rel.target_id, seen):
                yield [rel, *tail]

    def is_tree(self) -> bool:
        """True if no argument reaches the topic by more than one path.

        This is ArgLLM's restriction, and it is what the contestability results
        assume. One deliberate relaxation: ArgLLM requires *exactly* one path
        for every non-topic argument, whereas we also allow zero. An argument
        with no path to the topic is simply unattached -- which is a normal
        transient state when a user has just contributed an argument and
        attachment inference has not run yet -- and contributes nothing to the
        topic's strength, so it cannot threaten the guarantee. Only multi-path
        arguments can.
        """
        return all(
            len(list(self.paths_to_topic(arg_id))) <= 1
            for arg_id in self.arguments
            if arg_id != self.topic_id
        )

    def classify_polarity(self, arg_id: str) -> Literal["pro", "con", "ambiguous", "isolated"]:
        """Whether `arg_id` ultimately argues for or against the topic.

        An argument is *pro* if it reaches the topic via an even number of
        attacks, *con* if odd. If different paths disagree the node is
        `ambiguous` -- it cuts both ways, and no monotonicity guarantee applies
        to it. That can only happen once the graph stops being a tree.
        """
        if arg_id == self.topic_id:
            return "pro"
        polarities = set()
        for path in self.paths_to_topic(arg_id):
            n_attacks = sum(1 for r in path if r.relation_type == "attack")
            polarities.add("pro" if n_attacks % 2 == 0 else "con")
        if not polarities:
            return "isolated"
        if len(polarities) > 1:
            return "ambiguous"
        return polarities.pop()

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            # Copy the one mutable field so snapshots do not alias the live
            # graph: vars() returns the instance __dict__ by reference.
            "arguments": [
                {**vars(a), "goal_impacts": [dict(t) for t in a.goal_impacts]}
                for a in self.arguments.values()
            ],
            "relations": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "relation_type": r.relation_type,
                    "rationale": r.rationale,
                }
                for r in self.relations
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> QBAF:
        qbaf = cls(topic_id=data["topic_id"])
        for a in data["arguments"]:
            qbaf.add_argument(Argument(**a))
        for r in data["relations"]:
            qbaf.add_relation(Relation(**r))
        return qbaf

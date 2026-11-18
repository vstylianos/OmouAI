"""Persona specification.

Structured as a Value-Belief-Norm chain, which is how Chain-of-Opinion (Do Xuan
Long et al., COLING 2025) conditions opinion simulation, and which maps onto
this system almost exactly:

    values          -> this spec              authored, invariant
    beliefs         -> base scores in the QBAF elicited, contestable
    norms / stance  -> sigma from the semantics computed, never asserted

Two deliberate constraints, both from the persona-simulation literature:

1. **Keep it small.** Persona prompting degrades sharply with irrelevant
   attributes -- roughly 30 points in both Principled Personas (arXiv 2508.19764)
   and Chain-of-Opinion -- and returns asymptote fast (Park et al. found a
   structured survey nearly matched a two-hour interview). So: 3-5 load-bearing
   commitments, not a biography.

2. **Voice never touches reasoning.** `voice` is used only when rendering the
   persona's conversational turns. It is deliberately excluded from argument
   generation and base-score elicitation, because style leaks make arguments
   *sound* right while scoring wrong -- the persona ends up performing a stance
   rather than holding one.

`conservativeness` is the persona's epistemic disposition, which is separable
from what it believes: how far a given weight of counter-evidence moves it. It
feeds the gradual semantics, not any prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgentKind = Literal["stakeholder", "expert", "contrarian"]
"""Three kinds of deliberation partner.

A *stakeholder* holds evaluative commitments and scores arguments by them.
An *expert* knows the literature and scores every argument by evidential
quality, never by values -- its `values` are epistemic commitments.
A *contrarian* (the Devil's Advocate) holds no view: whatever the person
argues, it takes the opposite side and looks for the reasons the point may
not hold. It never runs the agreement judgement -- it always objects -- but
it keeps the honesty clause and declines when it has nothing genuine.
"""


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    #: One line for the persona picker.
    one_line: str
    #: 3-5 core evaluative commitments. The VBN "values" layer.
    #: For an expert these are epistemic commitments instead.
    values: tuple[str, ...]
    #: What this persona treats as a reason -- shapes which arguments land.
    evidence: str
    #: Epistemic disposition, passed to the semantics as LinearInfluence
    #: conservativeness. 1.0 is fully responsive; higher resists updating.
    conservativeness: float = 1.5
    #: Rhetorical register. NL rendering ONLY -- never in generation or scoring.
    voice: str = ""
    #: Stakeholder persona (default) or evidence-grounded domain expert.
    kind: AgentKind = "stakeholder"
    #: Expert only: the literature and methods it commands. Enters the
    #: reasoning prompts the way a stakeholder's values do.
    expertise: str = ""

    def __post_init__(self) -> None:
        if not 3 <= len(self.values) <= 5:
            raise ValueError(
                f"{self.id}: expected 3-5 commitments, got {len(self.values)}. "
                "Persona specs degrade with irrelevant attributes -- keep only "
                "what is load-bearing."
            )
        if self.conservativeness <= 0:
            raise ValueError(f"{self.id}: conservativeness must be positive")

    def commitments_block(self) -> str:
        """Prompt fragment for argument generation and base-score elicitation.

        Carries values and evidence standards. Carries no voice. For an expert
        it carries expertise and epistemic commitments instead of ideology --
        the expert reasons from evidence, and says so.
        """
        commitments = "\n".join(f"- {v}" for v in self.values)
        if self.kind == "contrarian":
            return (
                f"You are reasoning as: {self.name}, a devil's advocate.\n\n"
                "Your role: whatever the person argues, you take the opposite "
                "view and look for the reasons their point may not hold. You "
                "hold no position of your own on the claim; you test theirs.\n\n"
                f"Your method:\n{commitments}\n\n"
                f"What you treat as a reason: {self.evidence}\n\n"
                "Argue in good faith: the strongest honest objection, never a "
                "strawman, and when there is nothing genuine to object to, "
                "say so rather than manufacture one."
            )
        if self.kind == "expert":
            return (
                f"You are reasoning as: {self.name}, a domain expert.\n\n"
                f"Your expertise: {self.expertise}\n\n"
                f"Your epistemic commitments:\n{commitments}\n\n"
                f"What you treat as a reason: {self.evidence}\n\n"
                "Reason strictly from the evidence and the literature. You "
                "hold no policy position of your own: where the evidence is "
                "strong say so, where it is thin or contested say so, and "
                "where a question turns on values rather than facts, name "
                "that honestly instead of smuggling in a preference."
            )
        return (
            f"You are reasoning as: {self.name}.\n\n"
            f"Your core commitments:\n{commitments}\n\n"
            f"What you treat as a reason: {self.evidence}\n\n"
            "Reason from these commitments. Where a consideration cuts against "
            "them, acknowledge it honestly rather than dismissing it -- you are "
            "arguing in good faith, not scoring points."
        )

    def voice_block(self) -> str:
        """Prompt fragment for conversational turns only."""
        return f"Speak as {self.name}. {self.voice}".strip()

    def introduction(self) -> str:
        """The partner's opening message, rendered from the spec -- no model
        call, no stance. The deliberation starts neutral and the user directs
        it, so the opening only says who the partner is and what it treats as
        a reason, and invites the first move."""
        if self.kind == "contrarian":
            return (
                f"I'm the {self.name}. I hold no view of my own on the claim. "
                "Whatever you argue, I'll take the opposite side and look for "
                "the reasons your point may not hold -- and say so plainly when "
                "I can't find any. Make your case."
            )
        if self.kind == "expert":
            what = (
                f"I hold no policy position of my own. My expertise: "
                f"{self.expertise.rstrip('.')}"
            )
        else:
            what = f"What I stand for: {self.values[0].rstrip('.')}"
        return (
            f"I'm the {self.name}: {self.one_line.rstrip('.')}. {what}. "
            f"What I treat as a reason: {self.evidence.rstrip('.')}. "
            "I have no view on the claim until the arguments are on the table. "
            "Make your case, or ask me for mine."
        )

"""Domain specification.

A domain packages everything a deliberation needs: contested claims, the
stakeholder personas who disagree about them, optionally a domain expert, and
the quantitative indicators the deliberation can be evaluated against.

Indicators are *curated*, not model-generated. They are evaluation
instruments: selectable as goals at preparation, never material for an
argument. No prompt sees the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from personas.schema import Persona


@dataclass(frozen=True)
class Indicator:
    #: Stable id; selecting the indicator as a goal makes it lens `ind_<id>`.
    id: str
    name: str
    #: The objective or benchmark the source mandates.
    objective: str
    #: Where the data comes from.
    source: str
    category: str = ""
    #: SDGs this indicator's objective advances. The start screen suggests
    #: these as the case's default goal set.
    sdg: tuple[int, ...] = ()


@dataclass(frozen=True)
class Domain:
    id: str
    name: str
    #: One-paragraph framing for the start screen.
    blurb: str
    seed_claims: tuple[str, ...]
    #: Stakeholder personas and, where available, a domain expert.
    partners: tuple[Persona, ...]
    indicators: tuple[Indicator, ...] = field(default=())

    def get_partner(self, persona_id: str) -> Persona:
        for p in self.partners:
            if p.id == persona_id:
                return p
        raise KeyError(
            f"unknown partner {persona_id!r} in domain {self.id!r}; "
            f"have {sorted(p.id for p in self.partners)}"
        )

    def get_indicator(self, indicator_id: str) -> Indicator | None:
        for ind in self.indicators:
            if ind.id == indicator_id:
                return ind
        return None

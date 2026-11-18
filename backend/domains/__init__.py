"""Deliberation domains: climate policy (leads) and homelessness.

Climate has no expert or indicators yet -- its six stakeholder personas
predate the domain abstraction and keep working unchanged.
"""

from dataclasses import replace

from domains.barcelona import BARCELONA_HOMELESSNESS
from domains.schema import Domain, Indicator
from personas.climate import PERSONAS as _CLIMATE_PERSONAS
from personas.climate import SEED_CLAIMS as _CLIMATE_CLAIMS
from personas.contrarian import DEVILS_ADVOCATE

CLIMATE = Domain(
    id="climate",
    name="Climate policy",
    blurb=(
        "Debate contested climate policy with one of six positions. All six "
        "accept the science and disagree about instruments, growth, justice "
        "and technology."
    ),
    seed_claims=tuple(_CLIMATE_CLAIMS),
    partners=tuple(_CLIMATE_PERSONAS.values()),
)



def _with_devil(domain: Domain) -> Domain:
    """The Devil's Advocate is case-independent: it joins every domain's
    partner list, after the case's own personas."""
    return replace(domain, partners=domain.partners + (DEVILS_ADVOCATE,))


DOMAINS: dict[str, Domain] = {
    d.id: _with_devil(d) for d in (CLIMATE, BARCELONA_HOMELESSNESS)
}


def get_domain(domain_id: str) -> Domain:
    if domain_id not in DOMAINS:
        raise KeyError(f"unknown domain {domain_id!r}; have {sorted(DOMAINS)}")
    return DOMAINS[domain_id]


__all__ = ["Domain", "Indicator", "DOMAINS", "get_domain", "BARCELONA_HOMELESSNESS", "CLIMATE"]

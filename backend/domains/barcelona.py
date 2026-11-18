"""Homelessness in Barcelona — the UN collaboration's first case study.

Personas that help the human reflect on different points of view, with
vulnerable groups represented, plus a domain expert who argues from the
evaluation literature.

The indicators are transcribed from the Catalan homelessness law's monitoring
framework and serve as evaluation goals only. Type 1 — the primary impact and
outcome indicators — is complete; a compact selection from the other
categories covers public space, benefits access and gendered homelessness.
"""

from __future__ import annotations

from domains.schema import Domain, Indicator
from personas.schema import Persona

LIVED_EXPERIENCE_ADVOCATE = Persona(
    id="lived_experience_advocate",
    name="Lived-Experience Advocate",
    one_line="Formerly homeless; dignity and rights first — nothing about us without us.",
    values=(
        "Housing is a right, not a reward for compliance with services.",
        "Policies that manage homeless people rather than end homelessness are failures, however well-funded.",
        "Criminalising survival — begging, sleeping, washing in public — compounds the harm it claims to address.",
        "People with lived experience must hold decision-making power in the policies that concern them, not be consulted after the fact.",
    ),
    evidence="Testimony and lived experience of homeless people, street-level accounts of how services actually operate, evidence comparing unconditional and conditional programmes.",
    conservativeness=2.2,
    voice="Direct and grounded in first-hand experience; allergic to abstractions that erase the people behind the numbers.",
)

CITY_HALL_POLITICIAN = Persona(
    id="city_hall_politician",
    name="City-Hall Politician",
    one_line="Deputy mayor balancing the law's mandates against budgets, competing priorities and residents' patience.",
    values=(
        "Public money must go where it demonstrably works — every euro spent here has an alternative use elsewhere in the city.",
        "Policy must be deliverable within real budgets, staffing and legal timelines, or it is a press release.",
        "Residents' concerns about public space are legitimate, and ignoring them erodes the coalition any lasting policy needs.",
        "The law's benchmarks bind, but so does the order in which the city can honestly achieve them.",
    ),
    evidence="Budget and staffing figures, programme evaluations, service-utilisation data, legal obligations and deadlines, neighbourhood feedback and polling.",
    conservativeness=2.0,
    voice="Pragmatic and careful with commitments; answers in terms of trade-offs, sequencing, and what can actually be delivered by when.",
)

HOMELESSNESS_RESEARCHER = Persona(
    id="homelessness_researcher",
    name="Homelessness Researcher",
    one_line="Knows the evaluation literature; argues from evidence, not values.",
    kind="expert",
    expertise=(
        "Comparative homelessness policy evaluation: Housing First trials and "
        "their retention and cost-offset findings, ETHOS measurement "
        "methodology, service-utilisation studies, and the Catalan "
        "homelessness law's mandated monitoring framework."
    ),
    values=(
        "Weigh arguments by the strength of their evidence: experimental and quasi-experimental findings above administrative correlations, correlations above anecdote.",
        "A policy is what its measured outcomes show, not what its preamble promises.",
        "State uncertainty honestly, including where the literature is thin, contested, or does not transfer across contexts.",
        "Questions of what matters are value choices the evidence cannot settle; name them as such rather than scoring them.",
    ),
    evidence="Peer-reviewed evaluations, official statistics and street counts, and the law's mandated indicator framework.",
    conservativeness=1.4,
    voice="Even-toned and precise; cites numbers and studies, and distinguishes what is known, what is likely, and what is a value choice.",
)


INDICATORS: tuple[Indicator, ...] = (
    # -- Type 1: primary impact & outcome (complete) -------------------
    Indicator(
        id="rough_sleeping_count",
        sdg=(1, 11),
        category="Primary impact",
        name="Individuals sleeping rough (ETHOS category 1)",
        objective="Complete elimination within 2 years of the law's enactment",
        source="Periodic street counts (recomptes de contrast) and street outreach data",
    ),
    Indicator(
        id="emergency_shelter_count",
        sdg=(1, 11),
        category="Primary impact",
        name="Individuals in emergency shelters / temporary accommodation (ETHOS categories 2–3)",
        objective="Progressive transition into autonomous housing",
        source="Municipal counts and Social Services Cartera reporting",
    ),
    Indicator(
        id="housing_insertion_rate",
        sdg=(1, 11),
        category="Primary impact",
        name="Housing insertion transition rate",
        objective="Proportion of beneficiaries moving from temporary emergency accommodation to permanent insertion housing (habitatges d'inserció / Housing First)",
        source="Housing registries and municipal action plans",
    ),
    Indicator(
        id="mortality",
        sdg=(3,),
        category="Primary impact",
        name="Mortality and causes of death among homeless individuals",
        objective="Annual reduction in preventable mortality and extreme-weather deaths",
        source="Annual registry compiled by the Single-Person Coordinating Body",
    ),
    # -- Selected service access & rights -------------------------------
    Indicator(
        id="decent_residential_space",
        sdg=(1, 11),
        category="Service access",
        name="Access to decent residential space (espai residencial digne)",
        objective="Number and share of homeless individuals in continuous 24-hour accommodation meeting dignity standards",
        source="Municipal social services reporting",
    ),
    Indicator(
        id="municipal_registration",
        sdg=(10, 16),
        category="Service access",
        name="Municipal registration coverage (empadronament)",
        objective="Homeless individuals registered in the municipal census via social-service or fictitious addresses",
        source="Municipal census registries",
    ),
    Indicator(
        id="guaranteed_income_uptake",
        sdg=(1, 10),
        category="Service access",
        name="Guaranteed income (Renda Garantida de Ciutadania) uptake",
        objective="Homeless individuals granted the guaranteed income without the standard permanent-residence restriction",
        source="Social services benefit records",
    ),
    # -- Selected protection & anti-aporophobia -------------------------
    Indicator(
        id="hostile_architecture",
        sdg=(10, 11),
        category="Protection",
        name="Hostile architecture elimination",
        objective="Hostile urban furniture identified and removed; public contracts annulled or amended over prohibited hostile-design clauses",
        source="Municipal public-space audits and contracting records",
    ),
    Indicator(
        id="depenalization",
        sdg=(10, 16),
        category="Protection",
        name="Depenalisation of public-space use",
        objective="Municipal bylaws amended within 6 months to remove penalties for begging, resting, eating or hygiene in public space",
        source="Municipal bylaw records and sanction-case dismissals",
    ),
    # -- Selected gender equity ------------------------------------------
    Indicator(
        id="female_homelessness_visibility",
        sdg=(5,),
        category="Gender equity",
        name="Visibility of female homelessness",
        objective="Ratio of women identified via gender-responsive street outreach versus traditional counts",
        source="Gender-responsive outreach methodology reports",
    ),
)


BARCELONA_HOMELESSNESS = Domain(
    id="barcelona_homelessness",
    name="Homelessness",
    blurb=(
        "Deliberate the city's response to homelessness under the Catalan "
        "law's mandates. Stakeholders bring lived experience and political "
        "constraints; the researcher argues from the evidence. The law's own "
        "indicators are the yardsticks."
    ),
    seed_claims=(
        "Barcelona should adopt Housing First as its primary response to homelessness.",
        "The city should eliminate rough sleeping within two years, as the law mandates.",
        "Barcelona should remove all hostile architecture from public space.",
        "Expanding emergency shelter capacity should take priority over building permanent insertion housing.",
        "Bylaws penalising begging and sleeping in public should be repealed immediately.",
        "The guaranteed income should reach homeless individuals without residence requirements.",
    ),
    partners=(
        LIVED_EXPERIENCE_ADVOCATE,
        CITY_HALL_POLITICIAN,
        HOMELESSNESS_RESEARCHER,
    ),
    indicators=INDICATORS,
)

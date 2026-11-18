"""Gradual semantics for QBAFs.

Every semantics maps a QBAF to a strength sigma: A -> [0,1], computed by
iterating to a fixed point from the base scores. Iterating (rather than a
single bottom-up pass) means cyclic graphs are handled too, which matters once
users start adding arguments that close loops.

DF-QuAD is ported from multi-agent-qbaf-llm/src/argumentation/semantics.py,
decoupled from that project's debate-specific QBAF. Quadratic Energy follows
Potyka (KR 2018) as given in the ArgLLM paper. The modular family exposes a
`conservativeness` parameter, which is where a persona's *epistemic
disposition* lives: how far a given weight of counter-evidence moves it,
independently of what it believes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from engine.qbaf import QBAF

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 200
_EPSILON = 1e-6


def _probabilistic_or(values: list[float]) -> float:
    """1 - prod(1 - v). DF-QuAD's aggregation of a set of strengths."""
    acc = 0.0
    for v in values:
        acc = acc + v - acc * v
    return acc


class QBAFSemantics(ABC):
    """Base class. `converged` reports whether the last run reached a fixed point."""

    name: str = "abstract"

    def __init__(self, epsilon: float = _EPSILON, max_iterations: int = _MAX_ITERATIONS):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.converged = True

    @abstractmethod
    def _step(self, qbaf: QBAF, sigma: dict[str, float]) -> dict[str, float]:
        """One synchronous update of every argument's strength."""

    def compute(self, qbaf: QBAF) -> dict[str, float]:
        sigma = {aid: arg.base_score for aid, arg in qbaf.arguments.items()}
        if not sigma:
            return sigma

        self.converged = False
        for iteration in range(self.max_iterations):
            new_sigma = self._step(qbaf, sigma)
            delta = max(abs(new_sigma[a] - sigma[a]) for a in sigma)
            sigma = new_sigma
            if delta < self.epsilon:
                logger.debug("%s converged in %d iterations", self.name, iteration + 1)
                self.converged = True
                break
        else:
            logger.warning(
                "%s did not converge after %d iterations", self.name, self.max_iterations
            )

        return {k: min(1.0, max(0.0, v)) for k, v in sigma.items()}


class DFQuAD(QBAFSemantics):
    """Discontinuity-Free Quantitative Argumentation Debate (Rago et al., KR 2016).

        v_att = probabilistic-OR of attacker strengths
        v_sup = probabilistic-OR of supporter strengths
        sigma = tau * (1 - v_att + v_sup)        if v_att >= v_sup
        sigma = tau + (1 - tau) * (v_sup - v_att) otherwise
    """

    name = "dfquad"

    def _step(self, qbaf: QBAF, sigma: dict[str, float]) -> dict[str, float]:
        out = {}
        for aid, arg in qbaf.arguments.items():
            v_att = _probabilistic_or([sigma[s] for s in qbaf.attackers_of(aid)])
            v_sup = _probabilistic_or([sigma[s] for s in qbaf.supporters_of(aid)])
            tau = arg.base_score
            if v_att >= v_sup:
                out[aid] = tau * (1.0 - v_att + v_sup)
            else:
                out[aid] = tau + (1.0 - tau) * (v_sup - v_att)
        return out


class QuadraticEnergy(QBAFSemantics):
    """Quadratic energy model (Potyka, KR 2018).

        E     = sum(supporter strengths) - sum(attacker strengths)
        h(v)  = max(v,0)^2 / (1 + max(v,0)^2)
        sigma = tau + (1 - tau) * h(E) - tau * h(-E)

    Satisfies the *strict* forms of the contestability properties, so every
    contestation provably moves the stance rather than merely not moving it the
    wrong way.
    """

    name = "quadratic_energy"

    @staticmethod
    def _h(v: float) -> float:
        m = max(v, 0.0)
        return (m * m) / (1.0 + m * m)

    def _step(self, qbaf: QBAF, sigma: dict[str, float]) -> dict[str, float]:
        out = {}
        for aid, arg in qbaf.arguments.items():
            energy = sum(sigma[s] for s in qbaf.supporters_of(aid)) - sum(
                sigma[s] for s in qbaf.attackers_of(aid)
            )
            tau = arg.base_score
            out[aid] = tau + (1.0 - tau) * self._h(energy) - tau * self._h(-energy)
        return out


@dataclass
class LinearInfluence:
    """Influence function with an explicit responsiveness knob.

        iota(tau, s) = tau + (1 - tau) * min(1, s / k)   if s >= 0
                       tau + tau * max(-1, s / k)        if s < 0

    `conservativeness` k is the persona's epistemic disposition. k = 1 is fully
    responsive; larger k means a given weight of evidence moves the argument
    less. It scales the displacement without changing its sign, so the
    monotonicity that contestability rests on is preserved.
    """

    conservativeness: float = 1.0

    def __post_init__(self) -> None:
        if self.conservativeness <= 0:
            raise ValueError("conservativeness must be positive")

    def __call__(self, tau: float, aggregate: float) -> float:
        ratio = aggregate / self.conservativeness
        if aggregate >= 0:
            return tau + (1.0 - tau) * min(1.0, ratio)
        return tau + tau * max(-1.0, ratio)


class Modular(QBAFSemantics):
    """Modular semantics: an aggregation composed with an influence function.

    Defaults to summed aggregation with `LinearInfluence`, which recovers a
    standard linear model at conservativeness 1.
    """

    name = "modular"

    def __init__(
        self,
        influence: LinearInfluence | None = None,
        aggregation: str = "sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.influence = influence or LinearInfluence()
        if aggregation not in ("sum", "product"):
            raise ValueError(f"unknown aggregation: {aggregation}")
        self.aggregation = aggregation

    def _aggregate(self, supporters: list[float], attackers: list[float]) -> float:
        if self.aggregation == "sum":
            return sum(supporters) - sum(attackers)
        return _probabilistic_or(supporters) - _probabilistic_or(attackers)

    def _step(self, qbaf: QBAF, sigma: dict[str, float]) -> dict[str, float]:
        out = {}
        for aid, arg in qbaf.arguments.items():
            agg = self._aggregate(
                [sigma[s] for s in qbaf.supporters_of(aid)],
                [sigma[s] for s in qbaf.attackers_of(aid)],
            )
            out[aid] = self.influence(arg.base_score, agg)
        return out


SEMANTICS: dict[str, type[QBAFSemantics]] = {
    "dfquad": DFQuAD,
    "quadratic_energy": QuadraticEnergy,
    "modular": Modular,
}


def get_semantics(name: str, **kwargs) -> QBAFSemantics:
    """Look up a semantics by name. Used by the API's settings panel."""
    if name not in SEMANTICS:
        raise ValueError(f"unknown semantics {name!r}; have {sorted(SEMANTICS)}")
    return SEMANTICS[name](**kwargs)

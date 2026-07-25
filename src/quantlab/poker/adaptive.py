"""Public-history opponent adaptation for Royal Micro Hold'em."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantlab.poker.deep_cfr import MicroPolicy
from quantlab.poker.micro_holdem import (
    MicroAction,
    MicroHoldemState,
    initial_micro_state,
)


class PressureAdaptivePolicy:
    """Blend toward a pressure-specialist policy only when actions support it.

    The model compares public opponent actions under three transparent
    hypotheses: uniform random, check/call-only, and pot-first pressure.  It
    never uses the opponent's private cards.  The safe blueprint remains active
    until the posterior pressure probability exceeds ``threshold``.
    """

    def __init__(
        self,
        blueprint: MicroPolicy,
        pressure_specialist: MicroPolicy,
        *,
        threshold: float = 0.70,
        prior_pressure: float = 1.0 / 3.0,
        likelihood_floor: float = 1e-6,
    ) -> None:
        if not 0.0 <= threshold < 1.0:
            raise ValueError("threshold must lie in [0, 1)")
        if not 0.0 < prior_pressure < 1.0:
            raise ValueError("prior_pressure must lie in (0, 1)")
        if not 0.0 < likelihood_floor < 1.0:
            raise ValueError("likelihood_floor must lie in (0, 1)")
        self.blueprint = blueprint
        self.pressure_specialist = pressure_specialist
        self.threshold = threshold
        self.prior_pressure = prior_pressure
        self.likelihood_floor = likelihood_floor
        self.last_pressure_posterior = prior_pressure
        self.last_specialist_weight = 0.0

    def probabilities(self, state: MicroHoldemState) -> NDArray[np.float64]:
        posterior = self.pressure_posterior(state)
        specialist_weight = max(
            0.0,
            (posterior - self.threshold) / (1.0 - self.threshold),
        )
        self.last_pressure_posterior = posterior
        self.last_specialist_weight = specialist_weight
        safe = self.blueprint.probabilities(state)
        specialist = self.pressure_specialist.probabilities(state)
        return (1.0 - specialist_weight) * safe + specialist_weight * specialist

    def pressure_posterior(self, state: MicroHoldemState) -> float:
        """Return P(pressure | observed opponent actions) from public history."""
        observer = state.current_player
        replay = initial_micro_state(state.deal, button=state.button)
        remaining_prior = 1.0 - self.prior_pressure
        likelihoods = np.asarray(
            [remaining_prior / 2.0, remaining_prior / 2.0, self.prior_pressure],
            dtype=np.float64,
        )
        opponent_actions = 0
        for _, action in state.history:
            if replay.current_player != observer:
                legal = replay.legal_actions()
                likelihoods[0] *= 1.0 / len(legal)
                likelihoods[1] *= (
                    1.0
                    if action == MicroAction.CHECK_CALL
                    else self.likelihood_floor
                )
                likelihoods[2] *= (
                    1.0
                    if action == _pressure_action(replay)
                    else self.likelihood_floor
                )
                opponent_actions += 1
            replay = replay.apply(action)
        if opponent_actions == 0:
            return self.prior_pressure
        total = float(likelihoods.sum())
        if total <= 0.0:
            return self.prior_pressure
        return float(likelihoods[2] / total)


def _pressure_action(state: MicroHoldemState) -> MicroAction:
    legal = state.legal_actions()
    for candidate in (
        MicroAction.POT,
        MicroAction.HALF_POT,
        MicroAction.ALL_IN,
        MicroAction.CHECK_CALL,
        MicroAction.FOLD,
    ):
        if candidate in legal:
            return candidate
    raise RuntimeError("pressure action requested at a terminal state")

"""Learned information-set values and conservative depth-limited search.

The value model is trained from the acting player's information state only.
Decision-time search samples a public-belief-compatible opponent range,
expands the opponent's response, and evaluates the resulting leaves in one
batch.  It is deliberately conservative: the searched action is mixed with
the blueprint rather than replacing it outright.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from quantlab.poker.deep_cfr import MicroPolicy, policy_probabilities_batch
from quantlab.poker.features import FEATURE_DIM, canonical_cards, encode_information_state
from quantlab.poker.micro_evaluation import play_micro_hand
from quantlab.poker.micro_holdem import (
    ACTION_COUNT,
    DECK_SIZE,
    MicroHoldemState,
    evaluate_five,
    sample_micro_state,
)
from quantlab.poker.resolver import infer_opponent_range, sample_counterfactual_states

VALUE_FEATURE_DIM = FEATURE_DIM + 3


@dataclass(frozen=True)
class ValueTrainingConfig:
    """Configuration for reproducible blueprint-value regression."""

    examples: int = 60_000
    rollout_repeats: int = 2
    maximum_prefix_actions: int = 6
    blueprint_prefix_probability: float = 0.65
    epochs: int = 20
    batch_size: int = 512
    hidden_size: int = 128
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    seed: int = 15_800

    def __post_init__(self) -> None:
        positive = (
            self.examples,
            self.rollout_repeats,
            self.maximum_prefix_actions,
            self.epochs,
            self.batch_size,
            self.hidden_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("value-training counts must be positive")
        if not 0.0 <= self.blueprint_prefix_probability <= 1.0:
            raise ValueError("blueprint_prefix_probability must lie in [0, 1]")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


class StateValueNetwork(nn.Module):
    """Predict blueprint value in big blinds from one information state."""

    def __init__(self, *, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(VALUE_FEATURE_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.layers(features).squeeze(-1))


class NeuralStateValue:
    """Batched inference wrapper for an information-set value network."""

    def __init__(
        self,
        network: StateValueNetwork,
        *,
        target_mean: float = 0.0,
        target_scale: float = 1.0,
        device: str = "cpu",
    ) -> None:
        if target_scale <= 0.0:
            raise ValueError("target_scale must be positive")
        self.network = network.to(device)
        self.network.eval()
        self.target_mean = target_mean
        self.target_scale = target_scale
        self.device = torch.device(device)

    def values(self, states: list[MicroHoldemState]) -> NDArray[np.float64]:
        if not states:
            return np.empty(0, dtype=np.float64)
        features = np.stack(
            [encode_value_state(state) for state in states],
        )
        tensor = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            predictions = cast(
                NDArray[np.float32],
                self.network(tensor).cpu().numpy(),
            )
        return (
            predictions.astype(np.float64) * self.target_scale
            + self.target_mean
        )

    def value(self, state: MicroHoldemState) -> float:
        return float(self.values([state])[0])

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        device: str = "cpu",
    ) -> NeuralStateValue:
        payload = torch.load(path, map_location=device, weights_only=True)
        hidden_size = int(payload["config"]["hidden_size"])
        network = StateValueNetwork(hidden_size=hidden_size)
        network.load_state_dict(payload["state_value_network"])
        return cls(
            network,
            target_mean=float(payload.get("target_mean", 0.0)),
            target_scale=float(payload.get("target_scale", 1.0)),
            device=device,
        )


@dataclass(frozen=True)
class ValueTrainingResult:
    """Trained model plus held-out calibration diagnostics."""

    value_function: NeuralStateValue
    metrics: dict[str, Any]
    state_dict: dict[str, Tensor]
    target_mean: float
    target_scale: float


def train_state_value(
    blueprint: MicroPolicy,
    config: ValueTrainingConfig,
    *,
    device: str | None = None,
) -> ValueTrainingResult:
    """Fit a value network to blueprint continuations from reachable states."""
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    np_rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    features, targets, streets = _generate_value_examples(blueprint, config, np_rng)
    order = np_rng.permutation(config.examples)
    validation_size = max(1, int(config.examples * config.validation_fraction))
    validation_indices = order[:validation_size]
    training_indices = order[validation_size:]
    target_mean = float(targets[training_indices].mean())
    target_scale = float(targets[training_indices].std())
    normalized_targets = (targets - target_mean) / target_scale

    training_dataset = TensorDataset(
        torch.from_numpy(features[training_indices]),
        torch.from_numpy(normalized_targets[training_indices]),
    )
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    network = StateValueNetwork(hidden_size=config.hidden_size).to(selected_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    epoch_losses: list[float] = []
    network.train()
    for _ in range(config.epochs):
        cumulative = 0.0
        batches = 0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad()
            predictions = network(batch_features.to(selected_device))
            loss = loss_function(predictions, batch_targets.to(selected_device))
            loss.backward()
            optimizer.step()
            cumulative += float(loss.detach().cpu())
            batches += 1
        epoch_losses.append(cumulative / batches)

    network.eval()
    validation_tensor = torch.from_numpy(features[validation_indices]).to(selected_device)
    with torch.no_grad():
        normalized_predictions = network(validation_tensor).cpu().numpy()
    validation_predictions = (
        normalized_predictions * target_scale + target_mean
    )
    validation_targets = targets[validation_indices]
    errors = validation_predictions - validation_targets
    target_variance = float(np.square(validation_targets - validation_targets.mean()).sum())
    residual_variance = float(np.square(errors).sum())
    correlation = float(np.corrcoef(validation_predictions, validation_targets)[0, 1])
    by_street: dict[str, dict[str, float | int]] = {}
    for street in (0, 1):
        mask = streets[validation_indices] == street
        street_errors = errors[mask]
        by_street[str(street)] = {
            "examples": int(mask.sum()),
            "mae_big_blinds": float(np.abs(street_errors).mean()),
            "rmse_big_blinds": float(np.sqrt(np.square(street_errors).mean())),
        }
    metrics: dict[str, Any] = {
        "config": asdict(config),
        "device": str(selected_device),
        "training_examples": int(len(training_indices)),
        "validation_examples": int(len(validation_indices)),
        "training_loss_by_epoch": epoch_losses,
        "validation": {
            "mae_big_blinds": float(np.abs(errors).mean()),
            "rmse_big_blinds": float(np.sqrt(np.square(errors).mean())),
            "mean_error_big_blinds": float(errors.mean()),
            "correlation": correlation,
            "r_squared": 1.0 - residual_variance / target_variance,
            "target_standard_deviation_big_blinds": float(validation_targets.std()),
            "prediction_standard_deviation_big_blinds": float(
                validation_predictions.std()
            ),
            "by_street": by_street,
        },
        "target_normalization": {
            "mean_big_blinds": target_mean,
            "scale_big_blinds": target_scale,
        },
        "information_boundary": (
            "features contain only acting-player cards and public state/history"
        ),
    }
    cpu_state = {
        key: value.detach().cpu()
        for key, value in network.state_dict().items()
    }
    return ValueTrainingResult(
        NeuralStateValue(
            network,
            target_mean=target_mean,
            target_scale=target_scale,
            device=str(selected_device),
        ),
        metrics,
        cpu_state,
        target_mean,
        target_scale,
    )


def save_state_value_checkpoint(
    path: Path,
    result: ValueTrainingResult,
    config: ValueTrainingConfig,
) -> None:
    """Persist a value model with exact training configuration and metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "quantlab-royal-micro-state-value-v2",
            "config": asdict(config),
            "feature_dimension": VALUE_FEATURE_DIM,
            "state_value_network": result.state_dict,
            "target_mean": result.target_mean,
            "target_scale": result.target_scale,
            "metrics": result.metrics,
        },
        path,
    )


class ValueGuidedResolver:
    """Conservative one-step improvement with public-belief leaf values."""

    def __init__(
        self,
        blueprint: MicroPolicy,
        value_function: NeuralStateValue,
        *,
        belief_samples: int = 256,
        improvement_weight: float = 0.35,
        search_streets: tuple[int, ...] = (1,),
        seed: int = 15_900,
    ) -> None:
        if belief_samples <= 0:
            raise ValueError("belief_samples must be positive")
        if not 0.0 <= improvement_weight <= 1.0:
            raise ValueError("improvement_weight must lie in [0, 1]")
        if not search_streets or any(street not in (0, 1) for street in search_streets):
            raise ValueError("search_streets must contain preflop zero and/or flop one")
        self.blueprint = blueprint
        self.value_function = value_function
        self.belief_samples = belief_samples
        self.improvement_weight = improvement_weight
        self.search_streets = search_streets
        self.seed = seed
        self.last_action_values = np.full(ACTION_COUNT, -np.inf, dtype=np.float64)

    def probabilities(self, state: MicroHoldemState) -> NDArray[np.float64]:
        if state.street not in self.search_streets:
            return self.blueprint.probabilities(state)
        observer = state.current_player
        rng = np.random.default_rng(_information_state_seed(state, self.seed))
        posterior = infer_opponent_range(
            state,
            observer=observer,
            blueprint=self.blueprint,
        )
        counterfactual_states = sample_counterfactual_states(
            state,
            observer,
            posterior,
            rng,
            count=self.belief_samples,
        )
        response_uniforms = rng.random((self.belief_samples, 3))
        action_values = np.full(ACTION_COUNT, -np.inf, dtype=np.float64)
        for action in state.legal_actions():
            searched_leaves = [
                sampled_state.apply(action)
                for sampled_state in counterfactual_states
            ]
            for response_index in range(response_uniforms.shape[1]):
                active_indices = [
                    index
                    for index, leaf in enumerate(searched_leaves)
                    if not leaf.is_terminal and leaf.current_player != observer
                ]
                if not active_indices:
                    break
                active_states = [searched_leaves[index] for index in active_indices]
                response_probabilities = policy_probabilities_batch(
                    self.blueprint,
                    active_states,
                )
                for local_index, leaf_index in enumerate(active_indices):
                    searched_leaves[leaf_index] = _apply_probabilities_action(
                        searched_leaves[leaf_index],
                        response_probabilities[local_index],
                        response_uniforms[leaf_index, response_index],
                    )
            leaves: list[MicroHoldemState] = []
            leaf_indices: list[int] = []
            leaf_signs: list[float] = []
            values = np.empty(self.belief_samples, dtype=np.float64)
            for index, leaf in enumerate(searched_leaves):
                if leaf.is_terminal:
                    values[index] = leaf.payoffs()[observer]
                else:
                    leaves.append(leaf)
                    leaf_indices.append(index)
                    leaf_signs.append(1.0 if leaf.current_player == observer else -1.0)
            predicted = self.value_function.values(leaves)
            for index, prediction, sign in zip(
                leaf_indices,
                predicted,
                leaf_signs,
                strict=True,
            ):
                values[index] = sign * prediction
            action_values[int(action)] = float(values.mean())

        best = max(state.legal_actions(), key=lambda item: action_values[int(item)])
        blueprint_probabilities = self.blueprint.probabilities(state)
        searched = np.zeros(ACTION_COUNT, dtype=np.float64)
        searched[int(best)] = 1.0
        self.last_action_values = action_values
        return (
            (1.0 - self.improvement_weight) * blueprint_probabilities
            + self.improvement_weight * searched
        )


def _generate_value_examples(
    blueprint: MicroPolicy,
    config: ValueTrainingConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int8]]:
    features = np.empty((config.examples, VALUE_FEATURE_DIM), dtype=np.float32)
    targets = np.empty(config.examples, dtype=np.float32)
    streets = np.empty(config.examples, dtype=np.int8)
    collected = 0
    while collected < config.examples:
        state = sample_micro_state(rng)
        prefix_actions = int(rng.integers(0, config.maximum_prefix_actions + 1))
        for _ in range(prefix_actions):
            if state.is_terminal:
                break
            use_blueprint = rng.random() < config.blueprint_prefix_probability
            if use_blueprint:
                state = _sample_policy_action(state, blueprint, float(rng.random()))
            else:
                legal = state.legal_actions()
                state = state.apply(legal[int(rng.integers(0, len(legal)))])
        if state.is_terminal:
            continue
        observer = state.current_player
        rollout_values = np.empty(config.rollout_repeats, dtype=np.float64)
        for repeat in range(config.rollout_repeats):
            rollout_values[repeat] = play_micro_hand(
                state,
                (blueprint, blueprint),
                np.random.default_rng(
                    int(rng.integers(0, np.iinfo(np.int64).max))
                ),
            )[observer]
        features[collected] = encode_value_state(state)
        targets[collected] = float(rollout_values.mean())
        streets[collected] = state.street
        collected += 1
    return features, targets, streets


def encode_value_state(state: MicroHoldemState) -> NDArray[np.float32]:
    """Add public-information showdown equity to the policy feature vector."""
    observer = state.current_player
    hole, board = canonical_cards(
        state.deal.hole_cards[observer],
        state.visible_board,
    )
    equity, tie_probability, made_hand = _showdown_statistics(hole, board)
    return np.concatenate(
        (
            encode_information_state(state),
            np.asarray(
                [equity, tie_probability, made_hand],
                dtype=np.float32,
            ),
        )
    )


@lru_cache(maxsize=100_000)
def _showdown_statistics(
    hole: tuple[int, int],
    board: tuple[int, ...],
) -> tuple[float, float, float]:
    """Compute equity without consulting any hidden card in ``state``."""
    known = set(hole) | set(board)
    available = [card for card in range(DECK_SIZE) if card not in known]
    wins = 0
    ties = 0
    total = 0
    made_hand = 0.0
    if board:
        hero_score = evaluate_five((*hole, *board))
        made_hand = hero_score[0] / 8.0
        for opponent in combinations(available, 2):
            opponent_score = evaluate_five((*opponent, *board))
            wins += int(hero_score > opponent_score)
            ties += int(hero_score == opponent_score)
            total += 1
    else:
        seed_material = repr(hole).encode("utf-8")
        seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            "little",
            signed=False,
        )
        rng = np.random.default_rng(seed)
        for _ in range(2_048):
            sampled = rng.choice(available, size=5, replace=False)
            opponent = (int(sampled[0]), int(sampled[1]))
            runout = (int(sampled[2]), int(sampled[3]), int(sampled[4]))
            hero_score = evaluate_five((*hole, *runout))
            opponent_score = evaluate_five((*opponent, *runout))
            wins += int(hero_score > opponent_score)
            ties += int(hero_score == opponent_score)
            total += 1
    equity = (wins + 0.5 * ties) / total
    return equity, ties / total, made_hand


def _sample_policy_action(
    state: MicroHoldemState,
    policy: MicroPolicy,
    uniform: float,
) -> MicroHoldemState:
    probabilities = policy.probabilities(state)
    return _apply_probabilities_action(state, probabilities, uniform)


def _apply_probabilities_action(
    state: MicroHoldemState,
    probabilities: NDArray[np.float64],
    uniform: float,
) -> MicroHoldemState:
    legal = state.legal_actions()
    local = np.asarray([probabilities[int(action)] for action in legal], dtype=float)
    local /= local.sum()
    selected = min(int(np.searchsorted(np.cumsum(local), uniform, side="right")), len(legal) - 1)
    return state.apply(legal[selected])


def _information_state_seed(state: MicroHoldemState, seed: int) -> int:
    """Derive deterministic search randomness without opponent private cards."""
    public_key = (
        seed,
        state.current_player,
        state.deal.hole_cards[state.current_player],
        state.visible_board,
        state.button,
        state.street,
        state.contributions,
        state.street_contributions,
        state.acted,
        state.raises,
        state.history,
    )
    digest = hashlib.sha256(repr(public_key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)

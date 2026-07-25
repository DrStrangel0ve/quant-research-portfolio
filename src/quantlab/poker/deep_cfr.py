"""External-sampling neural CFR for Royal Micro Hold'em.

The implementation is deliberately compact and inspectable.  It follows Deep
CFR's core scaling idea—learn counterfactual advantages from sampled traversals
and separately learn the reach-weighted average strategy—without claiming the
compute budget or full-game abstraction of frontier HUNL systems.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from quantlab.poker.features import FEATURE_DIM, LEGAL_MASK_SLICE, encode_information_state
from quantlab.poker.micro_holdem import (
    ACTION_COUNT,
    MicroAction,
    MicroHoldemState,
    sample_micro_state,
)


class MicroPolicy(Protocol):
    """A behavioral policy over the five fixed abstract actions."""

    def probabilities(self, state: MicroHoldemState) -> NDArray[np.float64]:
        """Return a normalized, legal-action-masked probability vector."""


@dataclass(frozen=True)
class TrainingConfig:
    """Compute and architecture settings for a reproducible training run."""

    iterations: int = 1_600
    traversals_per_player: int = 6
    train_every: int = 20
    advantage_steps: int = 50
    strategy_steps: int = 500
    batch_size: int = 128
    memory_capacity: int = 100_000
    hidden_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 15_001

    def __post_init__(self) -> None:
        positive = (
            self.iterations,
            self.traversals_per_player,
            self.train_every,
            self.advantage_steps,
            self.strategy_steps,
            self.batch_size,
            self.memory_capacity,
            self.hidden_size,
        )
        if any(value <= 0 for value in positive) or self.learning_rate <= 0.0:
            raise ValueError("all Deep CFR configuration values must be positive")


@dataclass(frozen=True)
class TrainingSnapshot:
    """One training diagnostic recorded after an advantage-network update."""

    iteration: int
    advantage_loss_player_zero: float
    advantage_loss_player_one: float
    advantage_samples_player_zero: int
    advantage_samples_player_one: int
    strategy_samples: int


@dataclass(frozen=True)
class ReplaySample:
    features: NDArray[np.float32]
    target: NDArray[np.float32]
    weight: float


class ReservoirBuffer:
    """Uniform reservoir sample over a stream whose length is not known ahead."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = rng
        self.samples: list[ReplaySample] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, sample: ReplaySample) -> None:
        self.seen += 1
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
            return
        index = int(self.rng.integers(0, self.seen))
        if index < self.capacity:
            self.samples[index] = sample

    def batch(self, size: int) -> list[ReplaySample]:
        if not self.samples:
            raise ValueError("cannot sample an empty replay buffer")
        indices = self.rng.integers(0, len(self.samples), size=min(size, len(self.samples)))
        return [self.samples[int(index)] for index in indices]


class PokerNetwork(nn.Module):
    """Small MLP shared by the trainer and the browser weight exporter."""

    def __init__(self, *, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_COUNT),
        )

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.layers(features))


class NeuralPolicy:
    """Inference wrapper around an average-strategy network."""

    def __init__(self, network: PokerNetwork, *, device: str = "cpu") -> None:
        self.network = network.to(device)
        self.network.eval()
        self.device = torch.device(device)

    def probabilities(self, state: MicroHoldemState) -> NDArray[np.float64]:
        features = encode_information_state(state)
        legal_mask = features[LEGAL_MASK_SLICE].astype(bool)
        tensor = torch.from_numpy(features).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.network(tensor).squeeze(0).cpu().numpy().astype(np.float64)
        logits[~legal_mask] = -np.inf
        return _softmax(logits, legal_mask)

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str = "cpu") -> NeuralPolicy:
        payload = torch.load(path, map_location=device, weights_only=True)
        hidden_size = int(payload["config"]["hidden_size"])
        network = PokerNetwork(hidden_size=hidden_size)
        network.load_state_dict(payload["strategy_network"])
        return cls(network, device=device)


class RandomMicroPolicy:
    """Uniform legal-action baseline."""

    def probabilities(self, state: MicroHoldemState) -> NDArray[np.float64]:
        probabilities = np.zeros(ACTION_COUNT, dtype=np.float64)
        legal = state.legal_actions()
        probabilities[[int(action) for action in legal]] = 1.0 / len(legal)
        return probabilities


class CallingStationMicroPolicy:
    """Checks or calls every decision."""

    def probabilities(self, state: MicroHoldemState) -> NDArray[np.float64]:
        probabilities = np.zeros(ACTION_COUNT, dtype=np.float64)
        probabilities[int(MicroAction.CHECK_CALL)] = 1.0
        return probabilities


class PressureMicroPolicy:
    """Prefers pot raises, then all-in, then check/call."""

    def probabilities(self, state: MicroHoldemState) -> NDArray[np.float64]:
        legal = state.legal_actions()
        probabilities = np.zeros(ACTION_COUNT, dtype=np.float64)
        for candidate in (
            MicroAction.POT,
            MicroAction.HALF_POT,
            MicroAction.ALL_IN,
            MicroAction.CHECK_CALL,
            MicroAction.FOLD,
        ):
            if candidate in legal:
                probabilities[int(candidate)] = 1.0
                break
        return probabilities


class DeepCFRTrainer:
    """Train advantage estimators and a reach-weighted average policy."""

    def __init__(self, config: TrainingConfig, *, device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        self.advantage_networks = [
            PokerNetwork(hidden_size=config.hidden_size).to(self.device),
            PokerNetwork(hidden_size=config.hidden_size).to(self.device),
        ]
        self.strategy_network = PokerNetwork(hidden_size=config.hidden_size).to(self.device)
        self.advantage_memories = [
            ReservoirBuffer(config.memory_capacity, self.rng),
            ReservoirBuffer(config.memory_capacity, self.rng),
        ]
        self.strategy_memory = ReservoirBuffer(config.memory_capacity, self.rng)
        self.snapshots: list[TrainingSnapshot] = []
        self.iteration = 0

    def train(self) -> NeuralPolicy:
        """Run the configured external-sampling traversals and network updates."""
        losses = [float("nan"), float("nan")]
        for iteration in range(1, self.config.iterations + 1):
            self.iteration = iteration
            for traverser in (0, 1):
                for _ in range(self.config.traversals_per_player):
                    self._traverse(sample_micro_state(self.rng), traverser=traverser)
            if iteration % self.config.train_every == 0 or iteration == self.config.iterations:
                for player in (0, 1):
                    losses[player] = self._fit_advantage(player)
                self.snapshots.append(
                    TrainingSnapshot(
                        iteration=iteration,
                        advantage_loss_player_zero=losses[0],
                        advantage_loss_player_one=losses[1],
                        advantage_samples_player_zero=len(self.advantage_memories[0]),
                        advantage_samples_player_one=len(self.advantage_memories[1]),
                        strategy_samples=len(self.strategy_memory),
                    )
                )
        self._fit_strategy()
        return NeuralPolicy(self.strategy_network, device=str(self.device))

    def save_checkpoint(self, path: Path) -> None:
        """Save network weights and training metadata without replay buffers."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "quantlab-royal-micro-deep-cfr-v1",
                "config": asdict(self.config),
                "iterations": self.iteration,
                "device": str(self.device),
                "advantage_networks": [
                    network.state_dict() for network in self.advantage_networks
                ],
                "strategy_network": self.strategy_network.state_dict(),
                "snapshots": [asdict(snapshot) for snapshot in self.snapshots],
            },
            path,
        )

    def export_strategy_json(self, path: Path) -> None:
        """Export the strategy MLP in a dependency-free browser format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        linear_layers = [
            layer for layer in self.strategy_network.layers if isinstance(layer, nn.Linear)
        ]
        payload = {
            "format": "quantlab-royal-micro-strategy-v1",
            "game": {
                "name": "Royal Micro Hold'em",
                "deck": "9-A, four suits",
                "starting_stack": 20,
                "streets": ["preflop", "flop"],
                "actions": ["fold", "check/call", "half-pot", "pot", "all-in"],
            },
            "training": {
                **asdict(self.config),
                "completed_iterations": self.iteration,
                "device": str(self.device),
            },
            "network": {
                "input_size": FEATURE_DIM,
                "hidden_size": self.config.hidden_size,
                "activation": "relu",
                "weights": [
                    {
                        "weight": layer.weight.detach().cpu().tolist(),
                        "bias": layer.bias.detach().cpu().tolist(),
                    }
                    for layer in linear_layers
                ],
            },
        }
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    def _traverse(self, state: MicroHoldemState, *, traverser: int) -> float:
        if state.is_terminal:
            return state.payoffs()[traverser]
        player = state.current_player
        strategy = self._current_strategy(state)
        legal = state.legal_actions()
        if player == traverser:
            action_values = np.zeros(ACTION_COUNT, dtype=np.float32)
            node_value = 0.0
            for action in legal:
                value = self._traverse(state.apply(action), traverser=traverser)
                action_values[int(action)] = value
                node_value += strategy[int(action)] * value
            regrets = np.zeros(ACTION_COUNT, dtype=np.float32)
            legal_indices = np.asarray([int(action) for action in legal], dtype=int)
            regrets[legal_indices] = action_values[legal_indices] - node_value
            self.advantage_memories[traverser].add(
                ReplaySample(
                    encode_information_state(state),
                    regrets,
                    float(self.iteration),
                )
            )
            return float(node_value)

        self.strategy_memory.add(
            ReplaySample(
                encode_information_state(state),
                strategy.astype(np.float32),
                float(self.iteration),
            )
        )
        indices = np.asarray([int(action) for action in legal], dtype=int)
        local_probabilities = strategy[indices]
        local_probabilities /= local_probabilities.sum()
        chosen = legal[int(self.rng.choice(len(legal), p=local_probabilities))]
        return self._traverse(state.apply(chosen), traverser=traverser)

    def _current_strategy(self, state: MicroHoldemState) -> NDArray[np.float64]:
        features = encode_information_state(state)
        legal_mask = features[LEGAL_MASK_SLICE].astype(bool)
        tensor = torch.from_numpy(features).to(self.device).unsqueeze(0)
        network = self.advantage_networks[state.current_player]
        network.eval()
        with torch.no_grad():
            advantages = network(tensor).squeeze(0).cpu().numpy().astype(np.float64)
        positive = cast(NDArray[np.float64], np.maximum(advantages, 0.0))
        positive[~legal_mask] = 0.0
        total = positive.sum()
        if total <= 0.0:
            positive[legal_mask] = 1.0 / legal_mask.sum()
            return positive
        normalized: NDArray[np.float64] = positive / total
        return normalized

    def _fit_advantage(self, player: int) -> float:
        return self._fit_network(
            self.advantage_networks[player],
            self.advantage_memories[player],
            steps=self.config.advantage_steps,
            strategy_loss=False,
        )

    def _fit_strategy(self) -> float:
        return self._fit_network(
            self.strategy_network,
            self.strategy_memory,
            steps=self.config.strategy_steps,
            strategy_loss=True,
        )

    def _fit_network(
        self,
        network: PokerNetwork,
        memory: ReservoirBuffer,
        *,
        steps: int,
        strategy_loss: bool,
    ) -> float:
        if len(memory) == 0:
            return float("nan")
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
        network.train()
        last_loss = float("nan")
        for _ in range(steps):
            samples = memory.batch(self.config.batch_size)
            features, targets, weights = _tensor_batch(samples, self.device)
            predictions = network(features)
            legal_mask = features[:, LEGAL_MASK_SLICE].bool()
            if strategy_loss:
                masked_logits = predictions.masked_fill(~legal_mask, -1e9)
                per_sample = -(targets * torch.log_softmax(masked_logits, dim=1)).sum(dim=1)
            else:
                squared = (predictions - targets).square() * legal_mask
                per_sample = squared.sum(dim=1) / legal_mask.sum(dim=1).clamp_min(1)
            normalized_weights = weights / weights.mean().clamp_min(1e-8)
            loss = (per_sample * normalized_weights).mean()
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        network.eval()
        return last_loss


def _tensor_batch(
    samples: Sequence[ReplaySample],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    features = torch.from_numpy(np.stack([sample.features for sample in samples])).to(device)
    targets = torch.from_numpy(np.stack([sample.target for sample in samples])).to(device)
    weights = torch.tensor(
        [sample.weight for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    return features, targets, weights


def _softmax(
    logits: NDArray[np.float64],
    legal_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    probabilities = np.zeros(ACTION_COUNT, dtype=np.float64)
    legal_logits = logits[legal_mask]
    shifted = legal_logits - np.max(legal_logits)
    exponentials = np.exp(shifted)
    probabilities[legal_mask] = exponentials / exponentials.sum()
    return probabilities


def load_browser_strategy(path: Path) -> dict[str, Any]:
    """Load an exported model payload for validation or inspection."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "quantlab-royal-micro-strategy-v1":
        raise ValueError("unsupported browser strategy format")
    return payload

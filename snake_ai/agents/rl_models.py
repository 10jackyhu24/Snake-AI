"""Shared state encoder and compact PyTorch networks for DQN/PPO."""

from __future__ import annotations

from snake_ai.game import Direction, SnakeGame

try:
    import torch
    from torch import nn
except ImportError:  # Web demo can still run through the explicit fallback.
    torch = None
    nn = None


OBSERVATION_SIZE = 13
ACTION_SIZE = 4
DIRECTIONS = list(Direction)


def legal_action_mask(game: SnakeGame) -> list[bool]:
    return [
        not (len(game.snake) > 1 and direction == game.direction.opposite)
        and not game.would_collide(direction)
        for direction in DIRECTIONS
    ]


def encode_state(game: SnakeGame) -> list[float]:
    """Encode the board into 13 normalized, human-readable features."""
    head_x, head_y = game.head
    food_x, food_y = game.food if game.food is not None else game.head
    width_scale = max(1, game.width - 1)
    height_scale = max(1, game.height - 1)
    direction_one_hot = [float(game.direction == direction) for direction in DIRECTIONS]
    danger = [float(not legal) for legal in legal_action_mask(game)]
    return [
        (food_x - head_x) / width_scale,
        (food_y - head_y) / height_scale,
        *direction_one_hot,
        *danger,
        len(game.snake) / (game.width * game.height),
        head_x / width_scale,
        head_y / height_scale,
    ]


if nn is not None:
    class DQNNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(OBSERVATION_SIZE, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, ACTION_SIZE),
            )

        def forward(self, state):
            return self.network(state)


    class ActorCriticNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(OBSERVATION_SIZE, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )
            self.actor = nn.Linear(64, ACTION_SIZE)
            self.critic = nn.Linear(64, 1)

        def forward(self, state):
            features = self.shared(state)
            return self.actor(features), self.critic(features).squeeze(-1)
else:
    DQNNetwork = None
    ActorCriticNetwork = None

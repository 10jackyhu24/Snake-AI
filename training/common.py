"""Shared reward and environment helpers for reinforcement learning."""

from __future__ import annotations

from snake_ai.game import Direction, SnakeGame, StepResult

from snake_ai.agents.rl_models import DIRECTIONS, legal_action_mask


def manhattan_to_food(game: SnakeGame) -> int:
    if game.food is None:
        return 0
    return abs(game.head[0] - game.food[0]) + abs(game.head[1] - game.food[1])


def shaped_reward(before_distance: int, game: SnakeGame, result: StepResult) -> float:
    if result.game_over:
        return -10.0
    if result.ate_food:
        return 10.0
    after_distance = manhattan_to_food(game)
    return (0.12 if after_distance < before_distance else -0.15) - 0.01


def legal_indices(game: SnakeGame) -> list[int]:
    return [index for index, legal in enumerate(legal_action_mask(game)) if legal]


def step_action(game: SnakeGame, action: int) -> StepResult:
    return game.step(DIRECTIONS[action])

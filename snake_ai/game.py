"""Pure Snake game rules with no web or AI dependencies."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


Point = tuple[int, int]


class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

    @property
    def delta(self) -> Point:
        return self.value

    @property
    def opposite(self) -> "Direction":
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.RIGHT: Direction.LEFT,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
        }
        return opposites[self]


@dataclass(frozen=True)
class StepResult:
    moved: bool
    ate_food: bool
    game_over: bool
    reason: str | None = None


class SnakeGame:
    """Owns the board state and enforces the rules of Snake."""

    def __init__(self, width: int = 20, height: int = 16, seed: int | None = None):
        if width < 8 or height < 8:
            raise ValueError("The board must be at least 8 x 8.")
        self.width = width
        self.height = height
        self._rng = random.Random(seed)
        self.snake: list[Point] = []
        self.direction = Direction.RIGHT
        self.food: Point | None = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.end_reason: str | None = None
        self.reset()

    def reset(self) -> None:
        center_x = self.width // 2
        center_y = self.height // 2
        self.snake = [(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.end_reason = None
        self.food = None
        self._place_food()

    @property
    def head(self) -> Point:
        return self.snake[0]

    def is_inside(self, point: Point) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height

    def next_point(self, direction: Direction, origin: Point | None = None) -> Point:
        x, y = self.head if origin is None else origin
        dx, dy = direction.delta
        return x + dx, y + dy

    def would_collide(self, direction: Direction) -> bool:
        """Return whether an immediate move is illegal.

        The current tail cell is free when the snake is not eating because the
        tail moves away during the same turn.
        """
        target = self.next_point(direction)
        if not self.is_inside(target):
            return True
        eating = target == self.food
        occupied = self.snake if eating else self.snake[:-1]
        return target in occupied

    def step(self, requested_direction: Direction | None = None) -> StepResult:
        if self.game_over:
            return StepResult(False, False, True, self.end_reason)

        direction = requested_direction or self.direction
        if len(self.snake) > 1 and direction == self.direction.opposite:
            direction = self.direction

        target = self.next_point(direction)
        if self.would_collide(direction):
            self.direction = direction
            self.game_over = True
            self.end_reason = "撞到牆壁" if not self.is_inside(target) else "撞到自己"
            return StepResult(False, False, True, self.end_reason)

        ate_food = target == self.food
        self.snake.insert(0, target)
        self.direction = direction
        self.steps += 1

        if ate_food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        return StepResult(True, ate_food, self.game_over, self.end_reason)

    def set_state(
        self,
        snake: Iterable[Point],
        food: Point | None,
        direction: Direction,
    ) -> None:
        """Set a state explicitly. Useful for lessons, experiments and tests."""
        points = list(snake)
        if not points:
            raise ValueError("Snake cannot be empty.")
        if len(set(points)) != len(points):
            raise ValueError("Snake cells cannot overlap.")
        if any(not self.is_inside(point) for point in points):
            raise ValueError("Snake must stay inside the board.")
        if food is not None and (not self.is_inside(food) or food in points):
            raise ValueError("Food must be on an empty board cell.")
        self.snake = points
        self.food = food
        self.direction = direction
        self.score = max(0, len(points) - 3)
        self.steps = 0
        self.game_over = False
        self.win = food is None
        self.end_reason = None

    def _place_food(self) -> None:
        occupied = set(self.snake)
        empty_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not empty_cells:
            self.food = None
            self.game_over = True
            self.win = True
            self.end_reason = "AI 填滿了整個棋盤！"
            return
        self.food = self._rng.choice(empty_cells)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "snake": [{"x": x, "y": y} for x, y in self.snake],
            "food": None if self.food is None else {"x": self.food[0], "y": self.food[1]},
            "direction": self.direction.name,
            "score": self.score,
            "steps": self.steps,
            "gameOver": self.game_over,
            "win": self.win,
            "endReason": self.end_reason,
        }

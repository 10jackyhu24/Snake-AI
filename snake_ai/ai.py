"""An intentionally transparent, one-step look-ahead Snake AI."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

from .game import Direction, Point, SnakeGame


@dataclass(frozen=True)
class Candidate:
    direction: str
    valid: bool
    total_score: float
    food_distance: int | None
    reachable_space: int
    exits: int
    tail_reachable: bool
    components: dict[str, float]
    note: str


@dataclass(frozen=True)
class Decision:
    chosen: str | None
    reason: str
    candidates: list[Candidate]
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chosen": self.chosen,
            "reason": self.reason,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "details": self.details,
        }


class ExplainableSnakeAI:
    """Score every possible next move and expose every score component.

    This is deliberately not a black-box model. It uses one-step simulation,
    breadth-first search and flood fill so every decision can be inspected.
    """

    WEIGHTS = {
        "eat_food": 2_000,
        "food_path_base": 420,
        "food_path_step": -12,
        "food_unreachable": -450,
        "space_per_cell": 5,
        "tail_reachable": 260,
        "tail_blocked": -220,
        "exit_per_direction": 45,
        "keep_direction": 12,
        "trap": -4_000,
        "collision": -100_000,
    }

    key = "heuristic"
    display_name = "啟發式搜尋"
    category = "搜尋 / 人工規則"
    description = "用 BFS、Flood Fill 與人工權重評估四個下一步；快速且每一分都能解釋。"
    score_unit = "分"

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(self.WEIGHTS)
        if weights:
            self.weights.update(weights)

    def metadata(self) -> dict:
        return {
            "key": self.key,
            "name": self.display_name,
            "category": self.category,
            "description": self.description,
            "scoreUnit": self.score_unit,
            "trained": None,
            "status": "不需訓練",
        }

    def choose(self, game: SnakeGame) -> Decision:
        candidates = [self._evaluate(game, direction) for direction in Direction]
        valid_candidates = [candidate for candidate in candidates if candidate.valid]
        if not valid_candidates:
            return Decision(None, "四個方向都會立即碰撞，已經沒有安全路線。", candidates)

        # Enum order makes tie-breaking stable and easy to reproduce.
        best = max(valid_candidates, key=lambda item: item.total_score)
        reason = self._explain_choice(best)
        return Decision(best.direction, reason, candidates)

    def _evaluate(self, game: SnakeGame, direction: Direction) -> Candidate:
        if len(game.snake) > 1 and direction == game.direction.opposite:
            return self._invalid_candidate(direction, "不能直接 180° 回頭")

        target = game.next_point(direction)
        if not game.is_inside(target):
            return self._invalid_candidate(direction, "下一格在牆外")

        eating = target == game.food
        occupied_now = game.snake if eating else game.snake[:-1]
        if target in occupied_now:
            return self._invalid_candidate(direction, "下一格是蛇身")

        virtual_snake = [target, *game.snake] if eating else [target, *game.snake[:-1]]
        blocked_for_space = set(virtual_snake[1:])
        reachable_space = self._flood_fill_count(
            target, blocked_for_space, game.width, game.height
        )

        # The future tail can move, so it is treated as a reachable target.
        tail = virtual_snake[-1]
        body_without_head_or_tail = set(virtual_snake[1:-1])
        tail_distance = self._shortest_distance(
            target, tail, body_without_head_or_tail, game.width, game.height
        )
        tail_reachable = tail_distance is not None

        food_distance = None
        if game.food is not None:
            food_distance = self._shortest_distance(
                target,
                game.food,
                blocked_for_space,
                game.width,
                game.height,
            )

        exits = self._count_exits(target, blocked_for_space, game.width, game.height)
        components: dict[str, int] = {
            "吃到食物": self.weights["eat_food"] if eating else 0,
            "食物路徑": (
                self.weights["food_unreachable"]
                if food_distance is None
                else self.weights["food_path_base"]
                + food_distance * self.weights["food_path_step"]
            ),
            "活動空間": reachable_space * self.weights["space_per_cell"],
            "尾巴可達": (
                self.weights["tail_reachable"]
                if tail_reachable
                else self.weights["tail_blocked"]
            ),
            "下一步出口": exits * self.weights["exit_per_direction"],
            "保持方向": self.weights["keep_direction"] if direction == game.direction else 0,
        }

        # A region smaller than the snake plus a small turning buffer is likely
        # to be a cul-de-sac even when the immediate move itself is legal.
        trap_threshold = min(game.width * game.height, len(virtual_snake) + 3)
        trapped = reachable_space < trap_threshold or exits == 0
        components["死路懲罰"] = self.weights["trap"] if trapped else 0
        total = sum(components.values())

        note = "疑似死路" if trapped else ("吃到食物" if eating else "可安全移動")
        return Candidate(
            direction=direction.name,
            valid=True,
            total_score=total,
            food_distance=food_distance,
            reachable_space=reachable_space,
            exits=exits,
            tail_reachable=tail_reachable,
            components=components,
            note=note,
        )

    def _invalid_candidate(self, direction: Direction, note: str) -> Candidate:
        return Candidate(
            direction=direction.name,
            valid=False,
            total_score=self.weights["collision"],
            food_distance=None,
            reachable_space=0,
            exits=0,
            tail_reachable=False,
            components={"立即碰撞": self.weights["collision"]},
            note=note,
        )

    @staticmethod
    def _neighbors(point: Point, width: int, height: int):
        x, y = point
        for dx, dy in (0, -1), (1, 0), (0, 1), (-1, 0):
            neighbor = x + dx, y + dy
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                yield neighbor

    def _flood_fill_count(
        self, start: Point, blocked: set[Point], width: int, height: int
    ) -> int:
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self._neighbors(current, width, height):
                if neighbor not in blocked and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return len(seen)

    def _shortest_distance(
        self,
        start: Point,
        goal: Point,
        blocked: set[Point],
        width: int,
        height: int,
    ) -> int | None:
        if start == goal:
            return 0
        seen = {start}
        queue = deque([(start, 0)])
        while queue:
            current, distance = queue.popleft()
            for neighbor in self._neighbors(current, width, height):
                if neighbor == goal:
                    return distance + 1
                if neighbor not in blocked and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None

    def _count_exits(
        self, point: Point, blocked: set[Point], width: int, height: int
    ) -> int:
        return sum(
            neighbor not in blocked
            for neighbor in self._neighbors(point, width, height)
        )

    @staticmethod
    def _explain_choice(candidate: Candidate) -> str:
        distance = (
            "找不到靜態食物路徑"
            if candidate.food_distance is None
            else f"距食物 {candidate.food_distance} 步"
        )
        return (
            f"{candidate.direction} 的總分最高（{candidate.total_score}）；"
            f"{distance}、可達 {candidate.reachable_space} 格、"
            f"保留 {candidate.exits} 個出口。"
        )

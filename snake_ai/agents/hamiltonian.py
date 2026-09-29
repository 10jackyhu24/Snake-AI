"""Hamiltonian-cycle agent: slow, deterministic and exceptionally safe."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from snake_ai.ai import Decision, ExplainableSnakeAI
from snake_ai.game import Direction, Point, SnakeGame


def _cycle_for_even_width(width: int, height: int) -> list[Point]:
    """Build a cycle while reserving the left column for the return path."""
    path: list[Point] = [(0, 0)]
    path.extend((x, 0) for x in range(1, width))
    downward = True
    for x in range(width - 1, 0, -1):
        ys = range(1, height) if downward else range(height - 1, 0, -1)
        path.extend((x, y) for y in ys)
        downward = not downward
    path.extend((0, y) for y in range(height - 1, 0, -1))
    return path


@lru_cache(maxsize=16)
def build_hamiltonian_cycle(width: int, height: int) -> tuple[Point, ...]:
    """Return a Hamiltonian cycle for a rectangular grid with an even side."""
    if width % 2 == 0:
        path = _cycle_for_even_width(width, height)
    elif height % 2 == 0:
        transposed = _cycle_for_even_width(height, width)
        path = [(y, x) for x, y in transposed]
    else:
        raise ValueError("Hamiltonian cycle requires at least one even board dimension.")
    if len(path) != width * height or len(set(path)) != len(path):
        raise AssertionError("Generated Hamiltonian cycle is invalid.")
    return tuple(path)


def direction_to(start: Point, target: Point) -> Direction | None:
    dx, dy = target[0] - start[0], target[1] - start[1]
    for direction in Direction:
        if direction.delta == (dx, dy):
            return direction
    return None


class HamiltonianSnakeAI(ExplainableSnakeAI):
    key = "hamiltonian"
    display_name = "Hamiltonian Cycle"
    category = "保證型路徑"
    description = "沿著涵蓋每一格的封閉循環前進；速度保守，但蛇身一旦排入循環就不會切斷自己。"
    score_unit = "序位"

    def metadata(self) -> dict:
        data = super().metadata()
        data.update({"trained": None, "status": "不需訓練・安全優先"})
        return data

    def cycle_direction(self, game: SnakeGame) -> Direction | None:
        cycle = build_hamiltonian_cycle(game.width, game.height)
        index = {point: i for i, point in enumerate(cycle)}
        successor = cycle[(index[game.head] + 1) % len(cycle)]
        return direction_to(game.head, successor)

    def choose(self, game: SnakeGame) -> Decision:
        cycle = build_hamiltonian_cycle(game.width, game.height)
        cycle_direction = self.cycle_direction(game)
        evaluated = [self._evaluate(game, direction) for direction in Direction]
        candidates = []
        for candidate in evaluated:
            if not candidate.valid:
                candidates.append(candidate)
                continue
            follows_cycle = candidate.direction == getattr(cycle_direction, "name", None)
            components = {"循環指定": 10_000 if follows_cycle else 0, "合法移動": 1}
            candidates.append(
                replace(
                    candidate,
                    total_score=sum(components.values()),
                    components=components,
                    note="循環的下一格" if follows_cycle else "不偏離循環",
                )
            )

        target = next(
            (candidate for candidate in candidates if candidate.valid and candidate.direction == getattr(cycle_direction, "name", None)),
            None,
        )
        if target is None:
            valid = [candidate for candidate in evaluated if candidate.valid]
            if not valid:
                return Decision(None, "沒有合法方向可走。", candidates)
            target = max(valid, key=lambda item: item.total_score)
            reason = "目前蛇身尚未排入循環且循環下一格受阻，暫用啟發式方向重新接軌。"
        else:
            reason = f"沿 Hamiltonian Cycle 前進至 {target.direction}；循環共 {len(cycle)} 格。"
        return Decision(
            target.direction,
            reason,
            candidates,
            {"cycleLength": len(cycle), "guarantee": "完整循環、不走捷徑"},
        )

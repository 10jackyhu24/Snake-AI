"""Hybrid agent: heuristic food shortcuts with a Hamiltonian fallback."""

from __future__ import annotations

from dataclasses import replace

from snake_ai.ai import Decision, ExplainableSnakeAI
from snake_ai.game import Direction, SnakeGame

from .hamiltonian import HamiltonianSnakeAI


class HybridSnakeAI(ExplainableSnakeAI):
    key = "hybrid"
    display_name = "混合式安全捷徑"
    category = "搜尋 + 保證型路徑"
    description = "安全時用 BFS 抄近路吃食物，空間不足時回到 Hamiltonian Cycle，是速度與穩定性的折衷。"
    score_unit = "分"

    def __init__(self):
        super().__init__()
        self.cycle_agent = HamiltonianSnakeAI()

    def metadata(self) -> dict:
        data = super().metadata()
        data.update({"trained": None, "status": "不需訓練・混合策略"})
        return data

    def choose(self, game: SnakeGame) -> Decision:
        cycle_direction = self.cycle_agent.cycle_direction(game)
        evaluated = [self._evaluate(game, direction) for direction in Direction]
        candidates = []
        safe_directions: set[str] = set()
        threshold = min(game.width * game.height, len(game.snake) + 4)

        for candidate in evaluated:
            if not candidate.valid:
                candidates.append(candidate)
                continue
            safe = (
                candidate.tail_reachable
                and candidate.reachable_space >= threshold
                and candidate.components.get("死路懲罰", 0) == 0
            )
            if safe:
                safe_directions.add(candidate.direction)
            follows_cycle = candidate.direction == getattr(cycle_direction, "name", None)
            shortcut_bonus = (
                max(0, 220 - 8 * candidate.food_distance)
                if safe and not follows_cycle and candidate.food_distance is not None
                else 0
            )
            components = dict(candidate.components)
            components["循環保底"] = 160 if follows_cycle else 0
            components["安全捷徑"] = shortcut_bonus
            candidates.append(
                replace(
                    candidate,
                    total_score=sum(components.values()),
                    components=components,
                    note=("安全捷徑" if shortcut_bonus else "循環保底" if follows_cycle else candidate.note),
                )
            )

        safe = [candidate for candidate in candidates if candidate.direction in safe_directions]
        if safe:
            best = max(safe, key=lambda item: item.total_score)
            route = "Hamiltonian 循環" if best.direction == getattr(cycle_direction, "name", None) else "BFS 安全捷徑"
        else:
            best = next(
                (candidate for candidate in candidates if candidate.valid and candidate.direction == getattr(cycle_direction, "name", None)),
                None,
            )
            if best is None:
                valid = [candidate for candidate in candidates if candidate.valid]
                if not valid:
                    return Decision(None, "捷徑與循環都沒有合法方向。", candidates)
                best = max(valid, key=lambda item: item.total_score)
            route = "緊急循環保底"

        return Decision(
            best.direction,
            f"選擇 {route}：{best.direction}，分數 {best.total_score:.0f}。",
            candidates,
            {"route": route, "safeChoices": len(safe)},
        )

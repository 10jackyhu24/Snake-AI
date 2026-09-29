"""Optimize the explainable heuristic weights with a genetic algorithm."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from snake_ai.ai import ExplainableSnakeAI
from snake_ai.game import Direction, SnakeGame


OUTPUT = Path(__file__).resolve().parents[1] / "models" / "genetic_weights.json"
TUNABLE = [
    "eat_food", "food_path_base", "food_path_step", "food_unreachable",
    "space_per_cell", "tail_reachable", "tail_blocked", "exit_per_direction",
    "keep_direction", "trap",
]


def evaluate(weights: dict[str, float], seeds: list[int], max_steps: int) -> float:
    agent = ExplainableSnakeAI(weights)
    fitness = 0.0
    for seed in seeds:
        game = SnakeGame(width=12, height=10, seed=seed)
        for _ in range(max_steps):
            decision = agent.choose(game)
            if decision.chosen is None:
                break
            game.step(Direction[decision.chosen])
            if game.game_over:
                break
        fitness += game.score * 120 + game.steps * 0.025
    return fitness / len(seeds)


def mutate(parent: dict[str, float], rng: random.Random, rate: float = 0.35) -> dict[str, float]:
    child = dict(parent)
    for key in TUNABLE:
        if rng.random() < rate:
            scale = max(abs(child[key]) * 0.22, 2.0)
            child[key] += rng.gauss(0, scale)
    child["food_path_step"] = -abs(child["food_path_step"])
    child["food_unreachable"] = -abs(child["food_unreachable"])
    child["tail_blocked"] = -abs(child["tail_blocked"])
    child["trap"] = -abs(child["trap"])
    child["collision"] = -100_000
    return child


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Snake heuristic weights with evolution")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    base = dict(ExplainableSnakeAI.WEIGHTS)
    population = [base] + [mutate(base, rng, .8) for _ in range(args.population - 1)]
    best_fitness = float("-inf")
    best = base

    for generation in range(1, args.generations + 1):
        seeds = [args.seed + generation * 100 + i for i in range(args.episodes)]
        ranked = sorted(
            ((evaluate(individual, seeds, args.max_steps), individual) for individual in population),
            key=lambda item: item[0],
            reverse=True,
        )
        if ranked[0][0] > best_fitness:
            best_fitness, best = ranked[0]
        print(f"generation={generation:02d} fitness={ranked[0][0]:.2f}")
        elite_count = max(2, args.population // 4)
        elites = [individual for _, individual in ranked[:elite_count]]
        population = [dict(elites[0])]
        while len(population) < args.population:
            first, second = rng.sample(elites, 2)
            child = {key: (first[key] if rng.random() < .5 else second[key]) for key in base}
            population.append(mutate(child, rng))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "algorithm": "genetic",
                "generation": args.generations,
                "fitness": round(best_fitness, 4),
                "weights": {key: round(value, 4) for key, value in best.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()

"""Agent registry used by the web server."""

from __future__ import annotations

from snake_ai.ai import ExplainableSnakeAI

from .dqn import DQNSnakeAI
from .genetic import GeneticSnakeAI
from .hamiltonian import HamiltonianSnakeAI
from .hybrid import HybridSnakeAI
from .ppo import PPOSnakeAI


def build_agents() -> dict[str, object]:
    agents = [
        ExplainableSnakeAI(),
        GeneticSnakeAI(),
        HamiltonianSnakeAI(),
        HybridSnakeAI(),
        DQNSnakeAI(),
        PPOSnakeAI(),
    ]
    return {agent.key: agent for agent in agents}


__all__ = [
    "DQNSnakeAI",
    "GeneticSnakeAI",
    "HamiltonianSnakeAI",
    "HybridSnakeAI",
    "PPOSnakeAI",
    "build_agents",
]

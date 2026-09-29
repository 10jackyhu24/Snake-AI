import unittest

from snake_ai.agents import build_agents
from snake_ai.agents.hamiltonian import build_hamiltonian_cycle
from snake_ai.game import Direction, SnakeGame


class HamiltonianCycleTests(unittest.TestCase):
    def test_cycle_visits_every_cell_once_and_closes(self):
        width, height = 20, 16
        cycle = build_hamiltonian_cycle(width, height)

        self.assertEqual(width * height, len(cycle))
        self.assertEqual(len(cycle), len(set(cycle)))
        pairs = zip(cycle, (*cycle[1:], cycle[0]))
        self.assertTrue(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in pairs))

    def test_transposed_cycle_supports_odd_width(self):
        cycle = build_hamiltonian_cycle(9, 8)
        pairs = zip(cycle, (*cycle[1:], cycle[0]))

        self.assertEqual(72, len(set(cycle)))
        self.assertTrue(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in pairs))


class AgentRegistryTests(unittest.TestCase):
    def test_all_documented_modes_are_registered(self):
        agents = build_agents()

        self.assertEqual(
            {"heuristic", "genetic", "hamiltonian", "hybrid", "dqn", "ppo"},
            set(agents),
        )

    def test_every_mode_returns_a_legal_initial_move(self):
        game = SnakeGame(seed=4)
        for key, agent in build_agents().items():
            with self.subTest(mode=key):
                decision = agent.choose(game)
                self.assertIsNotNone(decision.chosen)
                self.assertFalse(game.would_collide(Direction[decision.chosen]))
                self.assertEqual(4, len(decision.candidates))


if __name__ == "__main__":
    unittest.main()

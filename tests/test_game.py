import unittest

from snake_ai.ai import ExplainableSnakeAI
from snake_ai.game import Direction, SnakeGame


class SnakeGameTests(unittest.TestCase):
    def test_eating_food_grows_snake_and_scores(self):
        game = SnakeGame(10, 10, seed=1)
        game.set_state([(4, 4), (3, 4), (2, 4)], (5, 4), Direction.RIGHT)

        result = game.step(Direction.RIGHT)

        self.assertTrue(result.ate_food)
        self.assertEqual(4, len(game.snake))
        self.assertEqual(1, game.score)

    def test_moving_into_departing_tail_is_legal(self):
        game = SnakeGame(10, 10)
        game.set_state([(4, 4), (4, 5), (3, 5), (3, 4)], (8, 8), Direction.UP)

        result = game.step(Direction.LEFT)

        self.assertTrue(result.moved)
        self.assertFalse(result.game_over)
        self.assertEqual((3, 4), game.head)

    def test_wall_collision_ends_game(self):
        game = SnakeGame(10, 10)
        game.set_state([(9, 4), (8, 4), (7, 4)], (1, 1), Direction.RIGHT)

        result = game.step(Direction.RIGHT)

        self.assertTrue(result.game_over)
        self.assertEqual("撞到牆壁", result.reason)

    def test_reverse_direction_is_ignored(self):
        game = SnakeGame(10, 10)
        game.set_state([(4, 4), (3, 4), (2, 4)], (8, 8), Direction.RIGHT)

        game.step(Direction.LEFT)

        self.assertEqual((5, 4), game.head)


class ExplainableAITests(unittest.TestCase):
    def setUp(self):
        self.ai = ExplainableSnakeAI()

    def test_ai_rejects_wall_and_reverse_moves(self):
        game = SnakeGame(10, 10)
        game.set_state([(4, 0), (3, 0), (2, 0)], (7, 7), Direction.RIGHT)

        decision = self.ai.choose(game)
        by_direction = {candidate.direction: candidate for candidate in decision.candidates}

        self.assertFalse(by_direction["UP"].valid)
        self.assertFalse(by_direction["LEFT"].valid)
        self.assertIn(decision.chosen, {"RIGHT", "DOWN"})

    def test_ai_exposes_score_components(self):
        game = SnakeGame(10, 10)
        decision = self.ai.choose(game)
        valid = next(candidate for candidate in decision.candidates if candidate.valid)

        self.assertIn("活動空間", valid.components)
        self.assertIn("尾巴可達", valid.components)
        self.assertIsInstance(valid.total_score, int)

    def test_ai_takes_adjacent_food_when_safe(self):
        game = SnakeGame(10, 10)
        game.set_state([(4, 4), (3, 4), (2, 4)], (5, 4), Direction.RIGHT)

        decision = self.ai.choose(game)

        self.assertEqual("RIGHT", decision.chosen)

    def test_ai_reports_when_every_direction_is_blocked(self):
        game = SnakeGame(8, 8)
        game.set_state(
            [(1, 1), (1, 0), (2, 1), (1, 2), (0, 1), (0, 2)],
            (7, 7),
            Direction.UP,
        )

        decision = self.ai.choose(game)

        self.assertIsNone(decision.chosen)
        self.assertTrue(all(not candidate.valid for candidate in decision.candidates))


if __name__ == "__main__":
    unittest.main()

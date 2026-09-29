"""Train a compact DQN with replay memory and a target network."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import torch
from torch import nn

from snake_ai.agents.rl_models import DQNNetwork, encode_state, legal_action_mask
from snake_ai.game import SnakeGame
from training.common import legal_indices, manhattan_to_food, shaped_reward, step_action


OUTPUT = Path(__file__).resolve().parents[1] / "models" / "dqn_snake.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Snake DQN agent")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = DQNNetwork()
    target = DQNNetwork()
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=7e-4)
    replay = deque(maxlen=30_000)
    loss_function = nn.SmoothL1Loss()
    scores: deque[int] = deque(maxlen=50)
    training_steps = 0

    for episode in range(1, args.episodes + 1):
        game = SnakeGame(12, 10, seed=args.seed + episode)
        epsilon = max(0.04, 1.0 - episode / max(1, args.episodes * .75))
        starvation = 0
        for _ in range(args.max_steps):
            state = encode_state(game)
            legal = legal_indices(game)
            if not legal:
                break
            if random.random() < epsilon:
                action = random.choice(legal)
            else:
                with torch.no_grad():
                    q_values = policy(torch.tensor(state, dtype=torch.float32))
                    q_values[~torch.tensor(legal_action_mask(game))] = -1e9
                    action = int(q_values.argmax().item())
            before = manhattan_to_food(game)
            result = step_action(game, action)
            reward = shaped_reward(before, game, result)
            starvation = 0 if result.ate_food else starvation + 1
            done = result.game_over or starvation >= 120
            replay.append((state, action, reward, encode_state(game), legal_action_mask(game), done))
            training_steps += 1

            if len(replay) >= 128 and training_steps % 4 == 0:
                batch = random.sample(replay, 128)
                states = torch.tensor([item[0] for item in batch], dtype=torch.float32)
                actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
                rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
                next_states = torch.tensor([item[3] for item in batch], dtype=torch.float32)
                next_masks = torch.tensor([item[4] for item in batch], dtype=torch.bool)
                dones = torch.tensor([item[5] for item in batch], dtype=torch.float32)
                predicted = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_values = target(next_states).masked_fill(~next_masks, -1e9).max(1).values
                    next_values = torch.where(next_values < -1e8, torch.zeros_like(next_values), next_values)
                    expected = rewards + .97 * next_values * (1 - dones)
                loss = loss_function(predicted, expected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                optimizer.step()
            if done:
                break

        scores.append(game.score)
        if episode % 20 == 0:
            target.load_state_dict(policy.state_dict())
            print(f"episode={episode:04d} average_food={sum(scores) / len(scores):.2f} epsilon={epsilon:.3f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": policy.state_dict(), "metadata": {"algorithm": "DQN", "episodes": args.episodes}},
        OUTPUT,
    )
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()

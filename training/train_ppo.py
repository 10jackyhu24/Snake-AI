"""Train a compact PPO actor-critic policy."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import torch
from torch.distributions import Categorical

from snake_ai.agents.rl_models import ActorCriticNetwork, encode_state, legal_action_mask
from snake_ai.game import SnakeGame
from training.common import manhattan_to_food, shaped_reward, step_action


OUTPUT = Path(__file__).resolve().parents[1] / "models" / "ppo_snake.pt"


def discounted_returns(rewards: list[float], gamma: float = .97) -> torch.Tensor:
    result = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        result.append(running)
    return torch.tensor(list(reversed(result)), dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Snake PPO agent")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = ActorCriticNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    scores: deque[int] = deque(maxlen=50)

    for episode in range(1, args.episodes + 1):
        game = SnakeGame(12, 10, seed=args.seed + episode)
        states: list[list[float]] = []
        masks: list[list[bool]] = []
        actions: list[int] = []
        old_log_probs: list[float] = []
        rewards: list[float] = []
        starvation = 0

        for _ in range(args.max_steps):
            state = encode_state(game)
            mask = legal_action_mask(game)
            if not any(mask):
                break
            with torch.no_grad():
                logits, _ = model(torch.tensor(state, dtype=torch.float32))
                logits = logits.masked_fill(~torch.tensor(mask), -1e9)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                log_probability = distribution.log_prob(action)
            before = manhattan_to_food(game)
            result = step_action(game, int(action.item()))
            reward = shaped_reward(before, game, result)
            starvation = 0 if result.ate_food else starvation + 1
            states.append(state)
            masks.append(mask)
            actions.append(int(action.item()))
            old_log_probs.append(float(log_probability.item()))
            rewards.append(reward)
            if result.game_over or starvation >= 120:
                break

        if states:
            state_tensor = torch.tensor(states, dtype=torch.float32)
            mask_tensor = torch.tensor(masks, dtype=torch.bool)
            action_tensor = torch.tensor(actions, dtype=torch.long)
            old_log_tensor = torch.tensor(old_log_probs, dtype=torch.float32)
            returns = discounted_returns(rewards)
            for _ in range(5):
                logits, values = model(state_tensor)
                logits = logits.masked_fill(~mask_tensor, -1e9)
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(action_tensor)
                advantages = returns - values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
                ratio = (new_log_probs - old_log_tensor).exp()
                unclipped = ratio * advantages
                clipped = ratio.clamp(.8, 1.2) * advantages
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = .5 * (returns - values).pow(2).mean()
                loss = actor_loss + critic_loss - .01 * distribution.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()

        scores.append(game.score)
        if episode % 20 == 0:
            print(f"episode={episode:04d} average_food={sum(scores) / len(scores):.2f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "metadata": {"algorithm": "PPO", "episodes": args.episodes}},
        OUTPUT,
    )
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()

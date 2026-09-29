"""Proximal Policy Optimization inference agent."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from snake_ai.ai import Decision, ExplainableSnakeAI
from snake_ai.game import Direction, SnakeGame

from .rl_models import ActorCriticNetwork, encode_state, legal_action_mask, torch


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "ppo_snake.pt"


class PPOSnakeAI(ExplainableSnakeAI):
    key = "ppo"
    display_name = "PPO 強化學習"
    category = "Policy-based RL"
    description = "Actor 直接輸出動作機率，Critic 估計局面價值；PPO 用限制更新幅度的方式穩定學習。"
    score_unit = "%"

    def __init__(self, model_path: Path = MODEL_PATH):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.model_info: dict = {}
        if torch is not None and ActorCriticNetwork is not None and model_path.is_file():
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                self.model = ActorCriticNetwork()
                self.model.load_state_dict(checkpoint["model_state"])
                self.model.eval()
                self.model_info = checkpoint.get("metadata", {})
            except (OSError, RuntimeError, KeyError, TypeError):
                self.model = None

    def metadata(self) -> dict:
        trained = self.model is not None
        episodes = self.model_info.get("episodes")
        status = f"已訓練 {episodes} 回合" if trained and episodes else "尚無模型・使用啟發式備援"
        return {
            "key": self.key,
            "name": self.display_name,
            "category": self.category,
            "description": self.description,
            "scoreUnit": self.score_unit,
            "trained": trained,
            "status": status,
        }

    def choose(self, game: SnakeGame) -> Decision:
        if self.model is None or torch is None:
            fallback = super().choose(game)
            return replace(
                fallback,
                reason="尚未載入 PPO 模型，目前使用啟發式安全備援。" + fallback.reason,
                details={"fallback": True},
            )

        with torch.no_grad():
            logits, value = self.model(torch.tensor(encode_state(game), dtype=torch.float32))
            mask = torch.tensor(legal_action_mask(game), dtype=torch.bool)
            logits = logits.masked_fill(~mask, -1e9)
            probabilities = torch.softmax(logits, dim=-1).tolist()

        evaluated = [self._evaluate(game, direction) for direction in Direction]
        candidates = []
        for index, candidate in enumerate(evaluated):
            if not candidate.valid:
                candidates.append(candidate)
                continue
            probability = round(float(probabilities[index]) * 100, 2)
            candidates.append(
                replace(
                    candidate,
                    total_score=probability,
                    components={"Actor 動作機率": probability},
                    note="策略網路輸出的機率",
                )
            )
        valid = [candidate for candidate in candidates if candidate.valid]
        if not valid:
            return Decision(None, "安全遮罩排除了所有動作。", candidates)
        best = max(valid, key=lambda item: item.total_score)
        return Decision(
            best.direction,
            f"PPO Actor 給 {best.direction} 最高機率 {best.total_score:.2f}%。",
            candidates,
            {"fallback": False, "stateValue": round(float(value.item()), 4)},
        )

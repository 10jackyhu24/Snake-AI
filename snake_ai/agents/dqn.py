"""Deep Q-Network inference agent with an explicit collision safety mask."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from snake_ai.ai import Decision, ExplainableSnakeAI
from snake_ai.game import Direction, SnakeGame

from .rl_models import DQNNetwork, encode_state, legal_action_mask, torch


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "dqn_snake.pt"


class DQNSnakeAI(ExplainableSnakeAI):
    key = "dqn"
    display_name = "DQN 強化學習"
    category = "Value-based RL"
    description = "神經網路預測四個動作的長期 Q 值，以經驗回放與目標網路從獎勵中學習。"
    score_unit = "Q"

    def __init__(self, model_path: Path = MODEL_PATH):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.model_info: dict = {}
        if torch is not None and DQNNetwork is not None and model_path.is_file():
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                self.model = DQNNetwork()
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
                reason="尚未載入 DQN 模型，目前使用啟發式安全備援。" + fallback.reason,
                details={"fallback": True},
            )

        with torch.no_grad():
            values = self.model(torch.tensor(encode_state(game), dtype=torch.float32)).tolist()
        legal = legal_action_mask(game)
        evaluated = [self._evaluate(game, direction) for direction in Direction]
        candidates = []
        for index, candidate in enumerate(evaluated):
            if not legal[index]:
                candidates.append(candidate)
                continue
            q_value = round(float(values[index]), 4)
            candidates.append(
                replace(
                    candidate,
                    total_score=q_value,
                    components={"神經網路 Q 值": q_value},
                    note="模型預測的長期價值",
                )
            )
        valid = [candidate for candidate in candidates if candidate.valid]
        if not valid:
            return Decision(None, "安全遮罩排除了所有動作。", candidates)
        best = max(valid, key=lambda item: item.total_score)
        return Decision(
            best.direction,
            f"DQN 預測 {best.direction} 的長期回報最高，Q={best.total_score:.3f}。",
            candidates,
            {"fallback": False, "features": encode_state(game)},
        )

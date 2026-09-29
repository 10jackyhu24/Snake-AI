"""Heuristic agent whose score weights are optimized by a genetic algorithm."""

from __future__ import annotations

import json
from pathlib import Path

from snake_ai.ai import ExplainableSnakeAI


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "genetic_weights.json"


class GeneticSnakeAI(ExplainableSnakeAI):
    key = "genetic"
    display_name = "遺傳演算法權重"
    category = "演化式訓練"
    description = "評分項目與啟發式相同，但各項權重由族群選擇、交配與突變自動調整。"
    score_unit = "分"

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model_info: dict = {}
        weights = None
        if model_path.is_file():
            try:
                self.model_info = json.loads(model_path.read_text(encoding="utf-8"))
                weights = self.model_info.get("weights")
            except (OSError, ValueError, TypeError):
                self.model_info = {}
        super().__init__(weights)

    def metadata(self) -> dict:
        trained = bool(self.model_info.get("weights"))
        generation = self.model_info.get("generation")
        status = f"已訓練 {generation} 代" if trained and generation is not None else "尚無訓練檔・使用原始權重"
        return {
            "key": self.key,
            "name": self.display_name,
            "category": self.category,
            "description": self.description,
            "scoreUnit": self.score_unit,
            "trained": trained,
            "status": status,
        }

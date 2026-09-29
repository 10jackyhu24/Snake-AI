# Snake AI Lab：六種 AI 策略比較

這是一個 Python 貪食蛇 AI 實驗室。同一個網頁可以切換六種策略，查看每一步的候選方向、分數或神經網路輸出，方便比較「人工規則、搜尋、演化與強化學習」的差異。

## 啟動

目前資料夾已包含訓練過的 Genetic、DQN 與 PPO 模型。若環境已有 PyTorch，直接執行：

```powershell
python server.py
```

瀏覽器會自動開啟 `http://127.0.0.1:8000`。也可以使用：

```powershell
python server.py --no-browser --seed 42
```

`--seed` 讓食物序列固定。網頁切換模式時會用相同種子重新開局，適合公平比較。

若沒有 PyTorch，其他四種模式仍可正常執行；DQN/PPO 會在介面明確顯示「使用啟發式備援」。安裝訓練依賴：

```powershell
python -m pip install -r requirements-training.txt
```

## 六種模式

| 網頁模式 | 程式檔案 | 原理 | 是否訓練 |
| --- | --- | --- | --- |
| 啟發式搜尋 | `snake_ai/ai.py` | BFS、Flood Fill、尾巴路徑與人工權重 | 否 |
| 遺傳演算法權重 | `snake_ai/agents/genetic.py` | 用演化搜尋較好的啟發式權重 | 是 |
| Hamiltonian Cycle | `snake_ai/agents/hamiltonian.py` | 沿涵蓋棋盤所有格子的封閉循環移動 | 否 |
| 混合式安全捷徑 | `snake_ai/agents/hybrid.py` | BFS 安全捷徑加 Hamiltonian 保底 | 否 |
| DQN 強化學習 | `snake_ai/agents/dqn.py` | 神經網路學習每個動作的長期 Q 值 | 是 |
| PPO 強化學習 | `snake_ai/agents/ppo.py` | Actor 輸出策略機率，Critic 評估局面 | 是 |

Hamiltonian 模式不追求快速吃食物。它會等待食物出現在循環前方，因此短時間分數通常最低，但能避免為了眼前食物切斷自己的路。

## 網頁觀察方式

1. 在棋盤下方選擇 AI 模式。
2. 先使用「單步執行」觀察右側四個候選方向。
3. 點開方向卡片，查看人工評分、Q 值或 Actor 機率。
4. 按「開始」連續執行，切換模式時系統會自動暫停並重新開局。

右側狀態會區分「不需訓練」、「已訓練」與「使用備援」，不會把未載入的神經網路假裝成訓練結果。

## 強化學習的 13 個輸入特徵

DQN 與 PPO 共用 `snake_ai/agents/rl_models.py` 的狀態編碼：

- 食物相對蛇頭的 X/Y 距離：2 個
- 目前方向 one-hot：4 個
- 四方向是否危險：4 個
- 蛇身長度比例：1 個
- 蛇頭正規化座標：2 個

總共 13 個數值。網頁推論時再加一層 collision safety mask，確定會碰撞的動作不會被採用。這讓你可以分開觀察「神經網路學到的偏好」與「程式保證的最低安全規則」。

## 重新訓練

所有訓練參數都有較合理的預設值，也能從命令列縮短或延長訓練。

遺傳演算法：

```powershell
python -m training.train_genetic --generations 8 --population 10
```

DQN：

```powershell
python -m training.train_dqn --episodes 600
```

PPO：

```powershell
python -m training.train_ppo --episodes 1200
```

輸出會分別寫入：

- `models/genetic_weights.json`
- `models/dqn_snake.pt`
- `models/ppo_snake.pt`

訓練完成後重新啟動 `server.py`，模型狀態與回合數就會更新。訓練本身使用 12×10 棋盤加快學習，所有輸入都已正規化，因此模型可在網頁的 20×16 棋盤推論。這不代表一定能泛化得一樣好；這個差距本身也是值得觀察的強化學習現象。

## 專案結構

```text
Snake/
├── server.py                    # HTTP API、模式切換與遊戲迴圈
├── models/                      # 已訓練權重與 checkpoint
├── snake_ai/
│   ├── game.py                  # 純遊戲規則
│   ├── ai.py                    # 可解釋啟發式基礎
│   └── agents/
│       ├── __init__.py          # 六種模式註冊表
│       ├── genetic.py
│       ├── hamiltonian.py
│       ├── hybrid.py
│       ├── dqn.py
│       ├── ppo.py
│       └── rl_models.py         # 特徵編碼與神經網路結構
├── training/
│   ├── common.py                # 共用 reward 與環境函式
│   ├── train_genetic.py
│   ├── train_dqn.py
│   └── train_ppo.py
├── static/                      # HTML、CSS、Canvas 與模式介面
└── tests/                       # 規則、AI 與 Hamiltonian 測試
```

## 測試

```powershell
python -m unittest discover -s tests -v
```

測試涵蓋遊戲規則、候選評分、六種模式註冊、每種模式的合法初始決策，以及 Hamiltonian Cycle 是否真的造訪每格一次並首尾相連。

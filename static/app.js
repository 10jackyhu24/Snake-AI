const canvas = document.querySelector("#gameCanvas");
const context = canvas.getContext("2d");
const directionMeta = {
  UP: { arrow: "↑", label: "上" },
  RIGHT: { arrow: "→", label: "右" },
  DOWN: { arrow: "↓", label: "下" },
  LEFT: { arrow: "←", label: "左" },
};

const modeFlows = {
  heuristic: [
    ["排除碰撞", "牆壁、身體與反向移動"], ["模擬下一步", "建立四個候選局面"],
    ["BFS + Flood Fill", "量距離、空間與尾巴路徑"], ["加權選最高分", "人工設定且完全透明"],
  ],
  genetic: [
    ["建立權重族群", "每組權重是一個個體"], ["多局評估", "吃得多、活得久者勝出"],
    ["交配與突變", "混合優秀權重並隨機變化"], ["載入最佳個體", "仍可展開檢查每一項分數"],
  ],
  hamiltonian: [
    ["建立封閉循環", "路線剛好造訪棋盤每一格"], ["定位蛇頭序位", "找到循環中的目前位置"],
    ["前往下一格", "永遠沿固定循環前進"], ["等待食物", "較慢但不主動切斷蛇身"],
  ],
  hybrid: [
    ["取得循環保底", "先保存 Hamiltonian 下一格"], ["搜尋食物捷徑", "用 BFS 評估較短路線"],
    ["安全性檢查", "空間與尾巴路徑都要足夠"], ["捷徑或保底", "危險時回到循環"],
  ],
  dqn: [
    ["編碼 13 個特徵", "食物方向、朝向與危險格"], ["神經網路估值", "輸出四個動作的 Q 值"],
    ["安全遮罩", "不允許明確碰撞的動作"], ["選最大 Q 值", "代表預期累積獎勵最高"],
  ],
  ppo: [
    ["編碼 13 個特徵", "把棋盤轉成神經網路輸入"], ["Actor 輸出策略", "計算四個動作的機率"],
    ["Critic 評估局面", "預測目前狀態的長期價值"], ["選最高機率", "網頁推論採確定性動作"],
  ],
};

let latestState = null;
let requestInFlight = false;
let expandedDirection = null;
let speedEditing = false;
let speedSyncTimer = null;
let speedRequestVersion = 0;

async function getState() {
  if (requestInFlight) return;
  requestInFlight = true;
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    latestState = await response.json();
    render(latestState);
  } catch (error) {
    document.querySelector("#statusText").textContent = "伺服器連線中斷";
  } finally {
    requestInFlight = false;
  }
}

async function control(action, value) {
  const response = await fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, value }),
  });
  latestState = await response.json();
  render(latestState);
}

function render(state) {
  drawBoard(state);
  document.querySelector("#score").textContent = state.score;
  document.querySelector("#length").textContent = state.snake.length;
  document.querySelector("#steps").textContent = state.steps;
  document.querySelector("#nextMove").textContent = state.ai.chosen ? directionMeta[state.ai.chosen].arrow : "×";
  document.querySelector("#chosenDirection").textContent = state.ai.chosen || "NO MOVE";
  document.querySelector("#decisionReason").textContent = state.ai.reason;

  const chosen = state.ai.candidates.find(candidate => candidate.direction === state.ai.chosen);
  document.querySelector("#chosenScore").textContent = chosen ? `${formatScore(chosen.total_score)} ${state.modeInfo.scoreUnit}` : "—";
  document.querySelector("#statusDot").classList.toggle("running", state.running);
  document.querySelector("#statusText").textContent = state.gameOver ? "模擬結束" : (state.running ? "AI 運算中" : "已暫停");
  document.querySelector("#playIcon").textContent = state.running ? "Ⅱ" : "▶";
  document.querySelector("#playLabel").textContent = state.running ? "暫停" : "開始";
  document.querySelector("#stepButton").disabled = state.running || state.gameOver;
  document.querySelector("#playButton").disabled = state.gameOver;
  // Polling must not replace the local slider value while the user is
  // dragging. The final server value is rendered after synchronization.
  if (!speedEditing) {
    document.querySelector("#speedValue").textContent = Math.round(state.speed);
    document.querySelector("#speedInput").value = state.speed;
  }
  renderMode(state);

  const message = document.querySelector("#gameMessage");
  message.classList.toggle("hidden", !state.gameOver);
  message.innerHTML = state.gameOver
    ? `<strong>${state.win ? "挑戰成功" : "模擬結束"}</strong><br><small>${escapeHtml(state.endReason || "沒有可走路線")}</small>`
    : "";
  renderCandidates(state);
}

function renderMode(state) {
  const select = document.querySelector("#modeSelect");
  const signature = state.modes.map(mode => mode.key).join(",");
  if (select.dataset.signature !== signature) {
    select.innerHTML = state.modes.map(mode => `<option value="${mode.key}">${escapeHtml(mode.name)}</option>`).join("");
    select.dataset.signature = signature;
  }
  select.value = state.mode;
  document.querySelector("#strategyCategory").textContent = state.modeInfo.category;
  document.querySelector("#strategyDescription").textContent = state.modeInfo.description;
  const status = document.querySelector("#strategyStatus");
  status.textContent = state.modeInfo.status;
  status.classList.toggle("trained", state.modeInfo.trained === true || state.modeInfo.trained === null);
  document.querySelector("#scoreHint").textContent = `${state.modeInfo.scoreUnit} 越高越優先`;
  const flow = modeFlows[state.mode] || modeFlows.heuristic;
  document.querySelector("#logicFlow").innerHTML = flow.map(([title, description], index) =>
    `<li><b>${index + 1}</b><div><strong>${escapeHtml(title)}</strong><span>${escapeHtml(description)}</span></div></li>`
  ).join("");
}

function drawBoard(state) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const displayWidth = Math.max(1, Math.round(rect.width));
  const displayHeight = displayWidth * state.height / state.width;
  canvas.style.height = `${displayHeight}px`;
  canvas.width = Math.round(displayWidth * ratio);
  canvas.height = Math.round(displayHeight * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);

  const cell = displayWidth / state.width;
  context.fillStyle = "#0a1411";
  context.fillRect(0, 0, displayWidth, displayHeight);

  context.strokeStyle = "rgba(90, 126, 110, .105)";
  context.lineWidth = 1;
  context.beginPath();
  for (let x = 1; x < state.width; x += 1) {
    context.moveTo(x * cell, 0);
    context.lineTo(x * cell, displayHeight);
  }
  for (let y = 1; y < state.height; y += 1) {
    context.moveTo(0, y * cell);
    context.lineTo(displayWidth, y * cell);
  }
  context.stroke();

  if (state.ai.chosen && !state.gameOver) {
    const head = state.snake[0];
    const delta = { UP: [0, -1], RIGHT: [1, 0], DOWN: [0, 1], LEFT: [-1, 0] }[state.ai.chosen];
    const nextX = head.x + delta[0];
    const nextY = head.y + delta[1];
    context.fillStyle = "rgba(110, 242, 162, .11)";
    context.fillRect(nextX * cell + 2, nextY * cell + 2, cell - 4, cell - 4);
    context.strokeStyle = "rgba(110, 242, 162, .72)";
    context.strokeRect(nextX * cell + 3.5, nextY * cell + 3.5, cell - 7, cell - 7);
  }

  if (state.food) drawFood(state.food.x, state.food.y, cell);

  state.snake.slice().reverse().forEach((part, reverseIndex) => {
    const index = state.snake.length - 1 - reverseIndex;
    const padding = index === 0 ? cell * .11 : cell * .15;
    const fade = 0.5 + 0.5 * (1 - index / Math.max(1, state.snake.length));
    context.fillStyle = index === 0 ? "#91ffc0" : `rgba(71, 215, 132, ${fade})`;
    roundedRect(part.x * cell + padding, part.y * cell + padding, cell - padding * 2, cell - padding * 2, cell * .21);
    context.fill();
  });
  drawEyes(state, cell);
}

function drawFood(x, y, cell) {
  const cx = (x + .5) * cell;
  const cy = (y + .5) * cell;
  const glow = context.createRadialGradient(cx, cy, 1, cx, cy, cell * .65);
  glow.addColorStop(0, "rgba(255, 189, 89, .42)");
  glow.addColorStop(1, "rgba(255, 189, 89, 0)");
  context.fillStyle = glow;
  context.fillRect(x * cell - cell * .2, y * cell - cell * .2, cell * 1.4, cell * 1.4);
  context.fillStyle = "#ffbd59";
  context.beginPath();
  context.arc(cx, cy, cell * .22, 0, Math.PI * 2);
  context.fill();
}

function drawEyes(state, cell) {
  const head = state.snake[0];
  if (!head) return;
  const offsets = {
    RIGHT: [[.65, .34], [.65, .66]], LEFT: [[.35, .34], [.35, .66]],
    UP: [[.34, .35], [.66, .35]], DOWN: [[.34, .65], [.66, .65]],
  }[state.direction];
  context.fillStyle = "#102019";
  offsets.forEach(([ox, oy]) => {
    context.beginPath();
    context.arc((head.x + ox) * cell, (head.y + oy) * cell, Math.max(1.3, cell * .055), 0, Math.PI * 2);
    context.fill();
  });
}

function roundedRect(x, y, width, height, radius) {
  context.beginPath();
  context.roundRect(x, y, width, height, radius);
}

function renderCandidates(state) {
  const list = document.querySelector("#candidateList");
  list.innerHTML = state.ai.candidates.map(candidate => {
    const selected = candidate.direction === state.ai.chosen;
    const isOpen = expandedDirection === candidate.direction;
    const details = Object.entries(candidate.components)
      .map(([name, value]) => `<span>${escapeHtml(name)}<b>${signed(value)}</b></span>`).join("");
    return `<article class="candidate ${selected ? "selected" : ""} ${candidate.valid ? "" : "invalid"} ${isOpen ? "open" : ""}" data-direction="${candidate.direction}">
      <div class="candidate-summary">
        <span class="direction-icon">${directionMeta[candidate.direction].arrow}</span>
        <span class="candidate-name"><strong>${candidate.direction}</strong><span>${escapeHtml(candidate.note)}</span></span>
        <span class="candidate-score">${formatScore(candidate.total_score)} ${state.modeInfo.scoreUnit}</span>
      </div>
      <div class="candidate-details">
        ${details}
        <span>可達格數<b>${candidate.reachable_space}</b></span>
        <span>出口數<b>${candidate.exits}</b></span>
      </div>
    </article>`;
  }).join("");
  list.querySelectorAll(".candidate").forEach(element => {
    element.addEventListener("click", () => {
      expandedDirection = expandedDirection === element.dataset.direction ? null : element.dataset.direction;
      renderCandidates(state);
    });
  });
}

function signed(value) {
  const formatted = formatScore(value);
  return Number(value) > 0 ? `+${formatted}` : formatted;
}
function formatScore(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const number = Number(value);
  return Number.isInteger(number) ? String(number) : number.toFixed(Math.abs(number) < 10 ? 3 : 2);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;" })[char]);
}

document.querySelector("#playButton").addEventListener("click", () => control(latestState?.running ? "pause" : "start"));
document.querySelector("#stepButton").addEventListener("click", () => control("step"));
document.querySelector("#resetButton").addEventListener("click", () => control("reset"));
document.querySelector("#speedInput").addEventListener("input", event => {
  document.querySelector("#speedValue").textContent = event.target.value;
  queueSpeedUpdate(Number(event.target.value));
});
document.querySelector("#speedInput").addEventListener("change", event => queueSpeedUpdate(Number(event.target.value), true));
document.querySelector("#modeSelect").addEventListener("change", event => control("mode", event.target.value));
window.addEventListener("resize", () => latestState && drawBoard(latestState));

function queueSpeedUpdate(value, immediately = false) {
  speedEditing = true;
  const version = ++speedRequestVersion;
  clearTimeout(speedSyncTimer);
  speedSyncTimer = setTimeout(async () => {
    try {
      await control("speed", value);
    } finally {
      // Ignore an older response if the slider moved again while it was sent.
      if (version === speedRequestVersion) {
        speedEditing = false;
        if (latestState) render(latestState);
      }
    }
  }, immediately ? 0 : 80);
}

getState();
setInterval(getState, 120);

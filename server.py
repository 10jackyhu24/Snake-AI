"""Standard-library web server for the explainable Snake AI demo."""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from snake_ai import Direction, SnakeGame, build_agents


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"


class GameController:
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.game = SnakeGame(seed=seed)
        self.agents = build_agents()
        self.mode = "heuristic"
        self.ai = self.agents[self.mode]
        self.running = False
        self.moves_per_second = 20.0
        self.lock = threading.RLock()
        self.decision = self.ai.choose(self.game)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            with self.lock:
                running = self.running and not self.game.game_over
                interval = 1.0 / self.moves_per_second
            if running and time.monotonic() >= next_tick:
                self.tick()
                next_tick = time.monotonic() + interval
            elif not running:
                next_tick = time.monotonic()
            self._stop_event.wait(0.01)

    def tick(self) -> None:
        with self.lock:
            if self.game.game_over:
                self.running = False
                return
            chosen = self.decision.chosen
            if chosen is None:
                self.running = False
                return
            self.game.step(Direction[chosen])
            if self.game.game_over:
                self.running = False
            else:
                self.decision = self.ai.choose(self.game)
                if self.decision.chosen is None:
                    self.game.game_over = True
                    self.game.end_reason = "AI 找不到任何安全方向"
                    self.running = False

    def control(self, action: str, value: object | None = None) -> None:
        with self.lock:
            if action == "start" and not self.game.game_over:
                self.running = True
            elif action == "pause":
                self.running = False
            elif action == "toggle" and not self.game.game_over:
                self.running = not self.running
            elif action == "step":
                self.tick()
            elif action == "reset":
                self.running = False
                self.game.reset()
                self.decision = self.ai.choose(self.game)
            elif action == "speed" and value is not None:
                self.moves_per_second = max(1.0, min(100.0, float(value)))
            elif action == "mode" and isinstance(value, str) and value in self.agents:
                self.running = False
                self.mode = value
                self.ai = self.agents[value]
                # Re-create with the original seed so modes begin from the same
                # food sequence and are easier to compare fairly.
                self.game = SnakeGame(seed=self.seed)
                self.decision = self.ai.choose(self.game)
            else:
                raise ValueError(f"Unknown or invalid action: {action}")

    def state(self) -> dict:
        with self.lock:
            state = self.game.to_dict()
            state.update(
                {
                    "running": self.running,
                    "speed": self.moves_per_second,
                    "ai": self.decision.to_dict(),
                    "mode": self.mode,
                    "modeInfo": self.ai.metadata(),
                    "modes": [agent.metadata() for agent in self.agents.values()],
                    "weights": getattr(self.ai, "weights", {}),
                }
            )
            return state


class SnakeRequestHandler(BaseHTTPRequestHandler):
    controller: GameController
    extensions = {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
    }

    def do_GET(self) -> None:  # noqa: N802 - HTTP handler API
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self.controller.state())
            return
        if path == "/":
            path = "/index.html"
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
        if urlparse(self.path).path != "/api/control":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self.controller.control(payload.get("action", ""), payload.get("value"))
            self._send_json(self.controller.state())
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def _serve_static(self, url_path: str) -> None:
        requested = (STATIC_DIR / url_path.lstrip("/")).resolve()
        try:
            requested.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = requested.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", self.extensions.get(requested.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args) -> None:
        if self.path != "/api/state":
            super().log_message(format, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explainable Snake AI web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = GameController(seed=args.seed)
    SnakeRequestHandler.controller = controller
    server = ThreadingHTTPServer((args.host, args.port), SnakeRequestHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Snake AI is running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        controller._stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()

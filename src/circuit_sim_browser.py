"""Local browser GUI server. The user starts the experiment from its page."""

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import secrets
import threading
import time
import webbrowser
from circuit_sim_model import CircuitModel, DEFAULT_CIRCUIT, ROOT, camera_image, spatial_leg_lift, MAX_LIFT_DEGREES
import numpy as np

MODES = ["연결 전달 (모두 양수)", "NT 가정 · Glutamate 미정", "NT 가정 · Glutamate −", "NT 가정 · Glutamate +"]


class Session:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = None
        self.busy = True
        self.error = ""
        self.status = "신경망 데이터 읽는 중…"
        self.mode = MODES[0]
        self.settings = {"wall_x": -0.6, "distance": 1.5, "visible": True, "threshold": .18,
                         "output_policy": "좌우·거리 표시 규칙"}
        self.scores = np.zeros(6)
        self.neural_scores = np.zeros(6)
        self.lift_factors = np.zeros(6)
        self.raw = np.zeros(6)
        self.left = self.right = 0

    def load(self):
        try:
            model = CircuitModel(progress=self.set_status)
            with self.lock:
                self.model, self.busy, self.status = model, False, "준비 완료 · 시작을 눌러 전달 실험을 실행하세요."
        except Exception as error:
            with self.lock:
                self.busy, self.error = False, f"{type(error).__name__}: {error}"

    def set_status(self, value):
        with self.lock:
            self.status = value

    def snapshot(self):
        with self.lock:
            model = self.model
            result = {"ready": model is not None and not self.busy, "busy": self.busy, "error": self.error,
                      "status": self.status, "settings": self.settings.copy(), "mode": self.mode,
                      "scores": self.scores.tolist(), "neural_scores": self.neural_scores.tolist(),
                      "lift_factors": self.lift_factors.tolist(),
                      "lift_degrees": (self.scores * MAX_LIFT_DEGREES).tolist(),
                      "raw": self.raw.tolist(), "left": self.left, "right": self.right}
            if model and not self.busy:
                result.update(tick=model.tick_count, nodes=len(model.ids), edges=model.weights.nnz,
                              collected_nodes=model.original_nodes, L2=len(model.sources),
                              motor_counts=[len(x) for x in model.motor_groups.values()],
                              references=model.reference.tolist(), blocked_nt=model.zero_sign_count,
                              active=int((np.abs(model.state) > 1e-8).sum()))
            return result

    def step(self):
        with self.lock:
            if not self.model or self.busy:
                raise ValueError("모델이 아직 준비되지 않았습니다.")
            image = camera_image(self.settings["wall_x"], self.settings["distance"], self.settings["visible"])
            self.neural_scores, self.raw, self.left, self.right = self.model.step(image)
            self.apply_output_policy()
            self.status = "신경 반응과 다리 들림 표시를 분리해 계산합니다. 각도는 가상 표시값입니다."

    def apply_output_policy(self):
        if self.settings["output_policy"] == "좌우·거리 표시 규칙":
            self.scores, self.lift_factors = spatial_leg_lift(
                self.neural_scores, self.settings["wall_x"], self.settings["distance"], self.settings["visible"])
        else:
            self.scores = self.neural_scores.copy()
            self.lift_factors = np.ones(6)

    def update(self, data):
        with self.lock:
            for name, lower, upper in [("wall_x", -2, 2), ("distance", .4, 4), ("threshold", .02, .9)]:
                if name in data:
                    value = float(data[name])
                    if not math.isfinite(value) or not lower <= value <= upper:
                        raise ValueError(f"Invalid {name}")
                    self.settings[name] = value
            if "visible" in data:
                if not isinstance(data["visible"], bool):
                    raise ValueError("visible must be boolean")
                self.settings["visible"] = data["visible"]
            if "output_policy" in data:
                if data["output_policy"] not in ("좌우·거리 표시 규칙", "신경 반응 그대로"):
                    raise ValueError("Invalid output_policy")
                self.settings["output_policy"] = data["output_policy"]
            self.apply_output_policy()
            if "mode" in data and data["mode"] != self.mode:
                if data["mode"] not in MODES or not self.model or self.busy:
                    raise ValueError("모델 변경이 불가능합니다.")
                self.mode = data["mode"]
                self.busy = True
                self.status = "전달 모델과 기준 반응 재계산 중…"
                threading.Thread(target=self.configure, daemon=True).start()

    def configure(self):
        try:
            self.model.configure(self.mode)
            with self.lock:
                self.scores.fill(0)
                self.neural_scores.fill(0)
                self.lift_factors.fill(0)
                self.raw.fill(0)
                self.busy = False
                self.status = "모델 변경 완료 · 다시 시작할 수 있습니다."
        except Exception as error:
            with self.lock:
                self.busy, self.error = False, str(error)

    def reset(self):
        with self.lock:
            if not self.model or self.busy:
                raise ValueError("모델이 아직 준비되지 않았습니다.")
            self.model.reset()
            self.scores.fill(0)
            self.neural_scores.fill(0)
            self.lift_factors.fill(0)
            self.raw.fill(0)
            self.left = self.right = 0

    def save(self):
        with self.lock:
            if not self.model or self.busy:
                raise ValueError("모델이 아직 준비되지 않았습니다.")
            folder = ROOT / "simulation_logs"
            folder.mkdir(exist_ok=True)
            path = folder / f"wall_test_{time.time_ns()}.json"
            record = self.snapshot()
            record.update(created_at_utc=datetime.now(timezone.utc).isoformat(), circuit=str(DEFAULT_CIRCUIT),
                          biological_validation=False, assumptions={"input": "single camera halves -> same-side L2",
                          "dynamics": "row-normalized leaky feedforward signal",
                          "output": self.settings["output_policy"], "max_lift_degrees": MAX_LIFT_DEGREES,
                          "spatial_policy": "same-side lateral gain; front/middle/hind onset distances 4.0/2.6/1.5"})
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            return str(path)


def handler_for(session, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/state":
                self.respond(session.snapshot())
            elif self.path == "/":
                body = (ROOT / "web" / "simulation.html").read_text(encoding="utf-8").replace("__SESSION_TOKEN__", token).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.respond({"error": "Not found"}, 404)

        def do_POST(self):
            if self.headers.get("X-TASS-Token") != token:
                self.respond({"error": "Invalid session token"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 4096:
                    raise ValueError("Request too large")
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("Object required")
                if self.path == "/api/step":
                    session.update(data)
                    session.step()
                elif self.path == "/api/settings":
                    session.update(data)
                elif self.path == "/api/reset":
                    session.reset()
                elif self.path == "/api/save":
                    self.respond({"path": session.save()})
                    return
                else:
                    self.respond({"error": "Not found"}, 404)
                    return
                self.respond(session.snapshot())
            except Exception as error:
                self.respond({"error": str(error)}, 400)
    return Handler


def main():
    if not (DEFAULT_CIRCUIT / "connections.feather").exists():
        raise SystemExit("추출 결과가 없습니다. 먼저 extract_visual_leg_circuit.py를 실행하세요.")
    session = Session()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(session, secrets.token_urlsafe(24)))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_port}/"
    threading.Thread(target=session.load, daemon=True).start()
    print(f"TASS GUI: {url}\n이 창을 닫으면 서버가 종료됩니다. 종료: Ctrl+C", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

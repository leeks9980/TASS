"""Interactive TASS two-wheel cart driven by visual-to-leg-motor activity.

Run directly from the repository root:

    .venv\Scripts\python.exe two_wheel_cart\main.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cart_model import DifferentialDriveCart, SignalToWheels, render_target_camera
from full_cns_model import FullCNS


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_from_config(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (HERE / path).resolve()


class CartSimulation:
    def __init__(self, config_path: Path):
        self.config = load_json(config_path)
        experiment = load_json(resolve_from_config(self.config["experiment"]))
        dynamics = experiment["dynamics"]
        circuit_path = resolve_from_config(self.config["circuit"])
        print(f"Loading visual-motor circuit: {circuit_path}", flush=True)
        self.cns = FullCNS(
            circuit_path,
            glutamate=dynamics["glutamate"],
            gain=float(dynamics["gain"]),
            leak=float(dynamics["leak"]),
            weight_mode=dynamics["weight_mode"],
            equation=dynamics["equation"],
            bias=float(dynamics["bias"]),
            visual_retention=float(self.config.get("visual_retention", 0.9)),
            motor_subclasses=self.config.get("motor_subclasses", ["fl", "ml", "hl"]),
        )
        self.dt = float(self.config["neural_interval_seconds"])
        self.cart = DifferentialDriveCart(wheel_base=float(self.config["wheel_base"]))
        self.mapper = SignalToWheels(
            motor_gain=float(self.config["motor_gain"]),
            max_wheel_speed=float(self.config["max_wheel_speed"]),
        )
        self.target = (2.5, 1.5)
        self.last_camera = np.zeros(
            (int(self.config["camera_height"]), int(self.config["camera_width"]), 3),
            dtype=np.uint8,
        )
        self.last_command = self.mapper.command({"L": 0.0, "R": 0.0})

    def reset(self) -> None:
        self.cns.reset()
        self.cart = DifferentialDriveCart(wheel_base=float(self.config["wheel_base"]))
        self.last_command = self.mapper.command({"L": 0.0, "R": 0.0})

    def step(self) -> None:
        self.last_camera = render_target_camera(
            self.cart,
            self.target,
            width=int(self.config["camera_width"]),
            height=int(self.config["camera_height"]),
            field_of_view_degrees=float(self.config["camera_field_of_view_degrees"]),
        )
        scores = self.cns.step(self.last_camera)
        self.last_command = self.mapper.command(scores)
        self.cart.step(
            self.last_command.left_speed,
            self.last_command.right_speed,
            self.dt,
        )


class CartWindow:
    def __init__(self, simulation: CartSimulation):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.simulation = simulation
        self.running = False
        self.trail: list[tuple[float, float]] = []
        self.root = tk.Tk()
        self.root.title("TASS visual-motor two-wheel cart")
        self.root.geometry("1020x760")

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")
        self.run_button = ttk.Button(controls, text="시작", command=self.toggle)
        self.run_button.pack(side="left")
        ttk.Button(controls, text="한 단계", command=self.single_step).pack(side="left", padx=6)
        ttk.Button(controls, text="초기화", command=self.reset).pack(side="left")
        ttk.Label(controls, text="  모터 증폭").pack(side="left")
        self.gain_var = tk.DoubleVar(value=self.simulation.mapper.motor_gain)
        gain = ttk.Scale(controls, from_=0.0, to=50000.0, variable=self.gain_var, command=self.set_gain)
        gain.pack(side="left", fill="x", expand=True, padx=6)
        self.gain_label = ttk.Label(controls, width=9)
        self.gain_label.pack(side="left")

        self.canvas = tk.Canvas(self.root, bg="#10151c", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.canvas.bind("<Button-1>", self.place_target)
        self.status = ttk.Label(self.root, padding=8, font=("Consolas", 10))
        self.status.pack(fill="x")
        self.root.bind("<space>", lambda _event: self.toggle())
        self.root.after(50, self.loop)
        self.draw()

    def set_gain(self, _value=None) -> None:
        self.simulation.mapper.motor_gain = float(self.gain_var.get())
        self.gain_label.configure(text=f"{self.simulation.mapper.motor_gain:,.0f}×")

    def toggle(self) -> None:
        self.running = not self.running
        self.run_button.configure(text="일시정지" if self.running else "시작")

    def reset(self) -> None:
        self.running = False
        self.run_button.configure(text="시작")
        self.trail.clear()
        self.simulation.reset()
        self.draw()

    def single_step(self) -> None:
        self.running = False
        self.run_button.configure(text="시작")
        self.simulation.step()
        self.trail.append((self.simulation.cart.x, self.simulation.cart.y))
        self.draw()

    def loop(self) -> None:
        started = time.perf_counter()
        if self.running:
            self.simulation.step()
            self.trail.append((self.simulation.cart.x, self.simulation.cart.y))
            if len(self.trail) > 3000:
                self.trail = self.trail[-3000:]
            self.draw()
        elapsed = time.perf_counter() - started
        delay = max(1, int((self.simulation.dt - elapsed) * 1000))
        self.root.after(delay, self.loop)

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        half = float(self.simulation.config["arena_half_size"])
        scale = 0.46 * min(width, height) / half
        return width * 0.5 + x * scale, height * 0.5 - y * scale

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        half = float(self.simulation.config["arena_half_size"])
        scale = 0.46 * min(width, height) / half
        return (sx - width * 0.5) / scale, (height * 0.5 - sy) / scale

    def place_target(self, event) -> None:
        self.simulation.target = self.screen_to_world(event.x, event.y)
        self.draw()

    def draw(self) -> None:
        c = self.canvas
        c.delete("all")
        width, height = max(c.winfo_width(), 1), max(c.winfo_height(), 1)
        half = float(self.simulation.config["arena_half_size"])
        x0, y0 = self.world_to_screen(-half, half)
        x1, y1 = self.world_to_screen(half, -half)
        c.create_rectangle(x0, y0, x1, y1, outline="#52606d", width=2)
        c.create_text(12, 12, anchor="nw", fill="#b8c7d9", text="화면을 클릭해 빛 자극 위치를 이동")

        if len(self.trail) > 1:
            points = [coordinate for point in self.trail for coordinate in self.world_to_screen(*point)]
            c.create_line(*points, fill="#335a72", width=2)

        tx, ty = self.world_to_screen(*self.simulation.target)
        c.create_oval(tx - 10, ty - 10, tx + 10, ty + 10, fill="#fff176", outline="#ffca28", width=2)

        cart = self.simulation.cart
        cx, cy = self.world_to_screen(cart.x, cart.y)
        forward = (math.cos(cart.heading), -math.sin(cart.heading))
        left = (-forward[1], forward[0])
        nose = (cx + forward[0] * 23, cy + forward[1] * 23)
        rear_l = (cx - forward[0] * 15 + left[0] * 14, cy - forward[1] * 15 + left[1] * 14)
        rear_r = (cx - forward[0] * 15 - left[0] * 14, cy - forward[1] * 15 - left[1] * 14)
        c.create_polygon(*nose, *rear_l, *rear_r, fill="#4fc3f7", outline="#e1f5fe", width=2)
        for side in (-1, 1):
            wx = cx + left[0] * side * 17
            wy = cy + left[1] * side * 17
            c.create_line(wx - forward[0] * 10, wy - forward[1] * 10,
                          wx + forward[0] * 10, wy + forward[1] * 10,
                          fill="#eceff1", width=5)

        command = self.simulation.last_command
        bar_width = min(260, width * 0.3)
        max_speed = self.simulation.mapper.max_wheel_speed
        for label, value, color, y in [
            ("LEFT", command.left_speed, "#ef5350", 42),
            ("RIGHT", command.right_speed, "#66bb6a", 68),
        ]:
            c.create_text(width - bar_width - 80, y, anchor="e", fill=color, text=label)
            c.create_rectangle(width - bar_width - 70, y - 7, width - 20, y + 7, outline="#546e7a")
            center = width - 20 - bar_width * 0.5
            fill = bar_width * 0.5 * value / max_speed
            c.create_line(center, y - 9, center, y + 9, fill="#cfd8dc")
            c.create_rectangle(min(center, center + fill), y - 7,
                               max(center, center + fill), y + 7,
                               fill=color, outline="")

        self.set_gain()
        self.status.configure(text=(
            f"tick {self.simulation.cns.tick:,}   "
            f"L 평균={command.left_signal:.8f} → 바퀴={command.left_speed:.4f}   "
            f"R 평균={command.right_signal:.8f} → 바퀴={command.right_speed:.4f}   "
            f"위치=({cart.x:.2f}, {cart.y:.2f})  방향={math.degrees(cart.heading):.1f}°   "
            f"운동뉴런 L={self.simulation.cns.motor_report['L']}, R={self.simulation.cns.motor_report['R']}"
        ))

    def run(self) -> None:
        self.root.mainloop()


def run_headless(simulation: CartSimulation, steps: int) -> None:
    for _ in range(steps):
        simulation.step()
    command = simulation.last_command
    print(json.dumps({
        "ticks": simulation.cns.tick,
        "motor_neurons": simulation.cns.motor_report,
        "left_mean": command.left_signal,
        "right_mean": command.right_signal,
        "left_wheel_speed": command.left_speed,
        "right_wheel_speed": command.right_speed,
        "cart": {
            "x": simulation.cart.x,
            "y": simulation.cart.y,
            "heading_degrees": math.degrees(simulation.cart.heading),
        },
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--headless-steps", type=int, default=0,
                        help="Run N neural/cart steps without opening the window")
    args = parser.parse_args()
    if args.headless_steps < 0:
        parser.error("--headless-steps must be non-negative")
    simulation = CartSimulation(args.config.resolve())
    if args.headless_steps:
        run_headless(simulation, args.headless_steps)
    else:
        CartWindow(simulation).run()


if __name__ == "__main__":
    main()

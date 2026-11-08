"""MuJoCo cart driven by TASS CNS with a compound-eye receptor display."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit("MuJoCo Python package is required") from exc

from cart_model import SignalToWheels
from compound_eye_view import CompoundEyePanel
from full_cns_model import FullCNS
from motor_decoder import LinearMotorDecoder


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (HERE / path).resolve()


class MujocoCart:
    def __init__(self, config_path: Path):
        self.config = load_json(config_path)
        experiment = load_json(resolve(self.config["experiment"]))
        dynamics = experiment["dynamics"] | self.config.get("dynamics_override", {})
        print("시각→운동 회로를 읽는 중…", flush=True)
        self.cns = FullCNS(
            resolve(self.config["circuit"]),
            glutamate=dynamics["glutamate"],
            gain=float(dynamics["gain"]),
            leak=float(dynamics["leak"]),
            weight_mode=dynamics["weight_mode"],
            equation=dynamics["equation"],
            bias=float(dynamics["bias"]),
            visual_retention=float(self.config.get("visual_retention", 0.9)),
            motor_subclasses=self.config.get("motor_subclasses", ["fl", "ml", "hl"]),
        )
        self.model = mujoco.MjModel.from_xml_path(str(resolve(self.config["mujoco_model"])))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.eye_width = int(self.config["eye_width"])
        self.eye_height = int(self.config["eye_height"])
        self.renderer = mujoco.Renderer(self.model, height=self.eye_height, width=self.eye_width)
        self.mapper = SignalToWheels(
            motor_gain=float(self.config["motor_gain"]),
            max_wheel_speed=float(self.config["max_wheel_speed"]),
        )
        self.wheel_radius = float(self.config["wheel_radius"])
        self.physics_steps = int(self.config["physics_steps_per_neural_tick"])
        self.target_position = np.asarray(self.config["target_initial_xyz"], dtype=float).copy()
        self.target_move_step = float(self.config["target_move_step"])
        self.data.mocap_pos[0] = self.target_position
        mujoco.mj_forward(self.model, self.data)
        self.last_scores = {"L": 0.0, "R": 0.0}
        self.last_wheels = {"L": 0.0, "R": 0.0}
        decoder_path = resolve(self.config["decoder"])
        decoder_enabled = bool(self.config.get("decoder_enabled", False))
        self.decoder = (LinearMotorDecoder.load(decoder_path)
                        if decoder_enabled and decoder_path.exists() else None)
        self.control_mode = "trained decoder" if self.decoder is not None else "raw signed mean"

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.target_position = np.asarray(self.config["target_initial_xyz"], dtype=float).copy()
        self.data.mocap_pos[0] = self.target_position
        mujoco.mj_forward(self.model, self.data)
        self.cns.reset()
        self.last_scores = {"L": 0.0, "R": 0.0}
        self.last_wheels = {"L": 0.0, "R": 0.0}

    def _compound_input(self):
        self.renderer.update_scene(self.data, camera="left_eye")
        left_rgb = self.renderer.render().copy()
        self.renderer.update_scene(self.data, camera="right_eye")
        right_rgb = self.renderer.render().copy()
        # The raw renders are only an optical substrate. FullCNS samples the
        # left and right halves using the photoreceptor hex map.
        return np.concatenate((left_rgb, right_rgb), axis=1)

    def motor_state(self):
        return np.concatenate([
            self.cns.state[self.cns.motor_indices["L"]],
            self.cns.state[self.cns.motor_indices["R"]],
        ]).astype(np.float64, copy=True)

    def set_target(self, xyz):
        self.target_position = np.asarray(xyz, dtype=float).copy()
        self.move_target()

    def move_target(self, dx=0.0, dy=0.0, dz=0.0):
        self.target_position += np.asarray([dx, dy, dz], dtype=float)
        self.target_position[0] = np.clip(self.target_position[0], -3.7, 3.7)
        self.target_position[1] = np.clip(self.target_position[1], -3.7, 3.7)
        # Sphere radius is 0.16 m. Keep it above the floor while allowing it
        # to be aligned closely with the cart's eye height (~0.23 m).
        self.target_position[2] = np.clip(self.target_position[2], 0.16, 2.5)
        self.data.mocap_pos[0] = self.target_position
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        self.data.mocap_pos[0] = self.target_position
        self.last_scores = self.cns.step(self._compound_input())
        if self.decoder is None:
            command = self.mapper.command(self.last_scores)
            wheel_values = np.array([command.left_speed, command.right_speed])
        else:
            wheel_values = np.clip(
                self.decoder.predict(self.motor_state()),
                -self.mapper.max_wheel_speed,
                self.mapper.max_wheel_speed,
            )
        self.last_wheels = {"L": float(wheel_values[0]), "R": float(wheel_values[1])}
        self.data.ctrl[0] = wheel_values[0] / self.wheel_radius
        self.data.ctrl[1] = wheel_values[1] / self.wheel_radius
        for _ in range(self.physics_steps):
            mujoco.mj_step(self.model, self.data)

    def observe_without_motion(self):
        """Advance only the visual/CNS path for decoder data collection."""
        self.data.mocap_pos[0] = self.target_position
        mujoco.mj_forward(self.model, self.data)
        self.last_scores = self.cns.step(self._compound_input())
        return self.motor_state()

    def close(self):
        self.renderer.close()


def run(config_path: Path):
    simulation = MujocoCart(config_path)
    requests = []
    request_lock = threading.Lock()

    def enqueue(key):
        with request_lock:
            requests.append(key)

    panel = CompoundEyePanel(
        enqueue, simulation.cns.input_mapping, simulation.cns.motor_report,
        simulation.target_move_step,
    )
    paused = True
    cart_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_BODY, "cart")
    interval = simulation.model.opt.timestep * simulation.physics_steps
    print("준비 완료: Space 시작/정지, R 초기화", flush=True)
    print("MuJoCo에서는 카트 물리를, 별도 창에서는 광수용체 시야를 동시에 확인합니다.", flush=True)
    try:
        with mujoco.viewer.launch_passive(
            simulation.model, simulation.data, key_callback=enqueue
        ) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = cart_id
                viewer.cam.distance = 3.2
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -32
            while viewer.is_running() and not panel.closed:
                started = time.perf_counter()
                with request_lock:
                    keys = requests[:]
                    requests.clear()
                for key in keys:
                    if isinstance(key, tuple) and key and key[0] == "target":
                        simulation.move_target(*key[1:])
                        continue
                    if key == 32:
                        paused = not paused
                    elif key in (ord("R"), ord("r")):
                        simulation.reset()
                        paused = True
                    elif key in (ord("W"), ord("w")):
                        simulation.move_target(dx=simulation.target_move_step)
                    elif key in (ord("S"), ord("s")):
                        simulation.move_target(dx=-simulation.target_move_step)
                    elif key in (ord("A"), ord("a")):
                        simulation.move_target(dy=simulation.target_move_step)
                    elif key in (ord("D"), ord("d")):
                        simulation.move_target(dy=-simulation.target_move_step)
                    elif key in (ord("Q"), ord("q")):
                        simulation.move_target(dz=simulation.target_move_step)
                    elif key in (ord("E"), ord("e")):
                        simulation.move_target(dz=-simulation.target_move_step)
                if not paused:
                    simulation.step()
                panel.refresh(
                    simulation.cns.last_input_values,
                    simulation.last_scores,
                    simulation.last_wheels,
                    paused,
                    float(simulation.data.time),
                    simulation.target_position,
                    simulation.control_mode,
                )
                panel.pump()
                viewer.sync()
                time.sleep(max(0.0, interval - (time.perf_counter() - started)))
    finally:
        if not panel.closed:
            panel.close()
        simulation.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()

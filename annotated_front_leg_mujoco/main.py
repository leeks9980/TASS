"""Run an annotation-grouped front-leg CNS test in MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
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

from full_cns_model import FullCNS
from motor_groups import AnnotatedFrontLegReadout
from panel import FrontLegPanel


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else (HERE / path).resolve()


class FrontLegExperiment:
    ACTUATOR_ORDER = (
        "L_protraction", "L_abduction", "L_levation", "L_flexion",
        "R_protraction", "R_abduction", "R_levation", "R_flexion",
    )

    def __init__(self, config_path):
        self.config = load_json(config_path)
        experiment = load_json(resolve(self.config["experiment"]))
        dynamics = experiment["dynamics"]
        print("앞다리 시각→운동 회로를 읽는 중…", flush=True)
        self.cns = FullCNS(
            resolve(self.config["circuit"]),
            glutamate=dynamics["glutamate"],
            gain=float(dynamics["gain"]),
            leak=float(dynamics["leak"]),
            weight_mode=dynamics["weight_mode"],
            equation=dynamics["equation"],
            bias=float(dynamics["bias"]),
            visual_retention=float(self.config["visual_retention"]),
            motor_subclasses=["fl"],
        )
        self.readout = AnnotatedFrontLegReadout(
            self.cns, gain=float(self.config["motor_function_gain"])
        )
        self.model = mujoco.MjModel.from_xml_path(str(resolve(self.config["model"])))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(
            self.model,
            height=int(self.config["eye_height"]),
            width=int(self.config["eye_width"]),
        )
        self.physics_steps = int(self.config["physics_steps_per_neural_tick"])
        self.target_step = float(self.config["target_move_step"])
        self.target_position = np.asarray(self.config["target_initial_xyz"], dtype=float)
        self.data.mocap_pos[0] = self.target_position
        self.limits = {
            name: math.radians(float(value))
            for name, value in self.config["joint_limits_degrees"].items()
        }
        self.last_targets = {name: 0.0 for name in self.ACTUATOR_ORDER}
        self._actuator_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_drive"
            )
            for name in self.ACTUATOR_ORDER
        }
        if any(index < 0 for index in self._actuator_ids.values()):
            raise ValueError("front-leg rig actuator names are incomplete")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.target_position = np.asarray(self.config["target_initial_xyz"], dtype=float)
        self.data.mocap_pos[0] = self.target_position
        mujoco.mj_forward(self.model, self.data)
        self.cns.reset()
        self.readout.reset_baseline()
        self.last_targets = {name: 0.0 for name in self.ACTUATOR_ORDER}

    def move_target(self, dx=0.0, dy=0.0, dz=0.0):
        self.target_position += np.asarray([dx, dy, dz], dtype=float)
        self.target_position[0:2] = np.clip(self.target_position[0:2], -3.7, 3.7)
        self.target_position[2] = np.clip(self.target_position[2], 0.16, 2.5)
        self.data.mocap_pos[0] = self.target_position
        mujoco.mj_forward(self.model, self.data)

    def _eye_input(self):
        self.renderer.update_scene(self.data, camera="left_eye")
        left = self.renderer.render().copy()
        self.renderer.update_scene(self.data, camera="right_eye")
        right = self.renderer.render().copy()
        return np.concatenate((left, right), axis=1)

    def step(self):
        self.data.mocap_pos[0] = self.target_position
        self.cns.step(self._eye_input())
        targets, _values = self.readout.joint_targets()
        for name in self.ACTUATOR_ORDER:
            function = name.split("_", 1)[1]
            target = float(np.clip(targets[name], -self.limits[function], self.limits[function]))
            self.data.ctrl[self._actuator_ids[name]] = target
            self.last_targets[name] = target
        for _ in range(self.physics_steps):
            mujoco.mj_step(self.model, self.data)

    def close(self):
        self.renderer.close()


def run(config_path):
    simulation = FrontLegExperiment(config_path)
    requests = []
    request_lock = threading.Lock()

    def enqueue(value):
        with request_lock:
            requests.append(value)

    panel = FrontLegPanel(enqueue, simulation.readout.report, simulation.target_step)
    paused = True
    interval = simulation.model.opt.timestep * simulation.physics_steps
    print("준비 완료: Space 시작/정지, B 현재 신경값 기준선, R 초기화", flush=True)
    try:
        with mujoco.viewer.launch_passive(
            simulation.model, simulation.data, key_callback=enqueue
        ) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.3, 0.0, 0.3]
                viewer.cam.distance = 2.1
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -24
            while viewer.is_running() and not panel.closed:
                started = time.perf_counter()
                with request_lock:
                    actions = requests[:]
                    requests.clear()
                for action in actions:
                    if isinstance(action, tuple) and action and action[0] == "target":
                        simulation.move_target(*action[1:])
                        continue
                    if action == 32:
                        paused = not paused
                    elif action in (ord("R"), ord("r")):
                        simulation.reset()
                        paused = True
                    elif action in (ord("B"), ord("b")):
                        simulation.readout.capture_baseline()
                    elif action in (ord("W"), ord("w")):
                        simulation.move_target(dx=simulation.target_step)
                    elif action in (ord("S"), ord("s")):
                        simulation.move_target(dx=-simulation.target_step)
                    elif action in (ord("A"), ord("a")):
                        simulation.move_target(dy=simulation.target_step)
                    elif action in (ord("D"), ord("d")):
                        simulation.move_target(dy=-simulation.target_step)
                    elif action in (ord("Q"), ord("q")):
                        simulation.move_target(dz=simulation.target_step)
                    elif action in (ord("E"), ord("e")):
                        simulation.move_target(dz=-simulation.target_step)
                if not paused:
                    simulation.step()
                panel.refresh(
                    simulation.readout.table_rows(),
                    simulation.readout.baseline,
                    simulation.last_targets,
                    paused,
                    float(simulation.data.time),
                    simulation.target_position,
                )
                panel.pump()
                viewer.sync()
                time.sleep(max(0.0, interval - (time.perf_counter() - started)))
    finally:
        if not panel.closed:
            panel.close()
        simulation.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    run(args.config.resolve())

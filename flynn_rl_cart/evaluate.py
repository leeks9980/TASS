"""Open MuJoCo and the photoreceptor mosaic for a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from environment import LightTargetCartEnv
from flynn_model import FlynnPolicy
from panel import ReceptorPanel
from train import choose_device, load_json, resolve


HERE = Path(__file__).resolve().parent


def run(config_path, checkpoint_path, stage):
    config = load_json(config_path)
    device = choose_device(config["ppo"]["device"])
    circuit = resolve(config["paths"]["circuit"])
    env = LightTargetCartEnv(
        resolve(config["paths"]["mujoco_model"]), circuit,
        config["vision"], config["environment"], config["reward"],
        config["curriculum"], int(config["ppo"]["seed"])+20_000,
    )
    policy = FlynnPolicy(circuit, env.vision.indices, config["network"], device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy"]); policy.eval()
    receptors, privileged = env.reset(stage)
    hidden = policy.initial_state()
    action = np.zeros(2, dtype=np.float32)
    info = {"stage": stage, "distance": env.distance_to_target(), "success": False}
    reward = total_return = 0.0
    paused = True
    requests, request_lock = [], threading.Lock()

    def enqueue(value):
        with request_lock:
            requests.append(value)

    panel = ReceptorPanel(enqueue, env.vision, config["vision"]["display_width"],
                          config["vision"]["display_height"])
    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=enqueue) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = env.cart_body
                viewer.cam.distance = 3.2; viewer.cam.azimuth = 135; viewer.cam.elevation = -32
            while viewer.is_running() and not panel.closed:
                started = time.perf_counter()
                with request_lock:
                    keys = requests[:]; requests.clear()
                for key in keys:
                    if key == 32:
                        paused = not paused
                    elif key in (ord("R"), ord("r")) or key in (ord("1"), ord("2"), ord("3")):
                        if key in (ord("1"), ord("2"), ord("3")):
                            stage = int(chr(key))
                        receptors, privileged = env.reset(stage)
                        hidden = policy.initial_state(); total_return = reward = 0.0
                        action[:] = 0; info = {"stage": stage, "distance": env.distance_to_target(), "success": False}
                        paused = True
                if not paused:
                    with torch.no_grad():
                        hidden, mean, _value = policy.step(
                            hidden, torch.as_tensor(receptors, device=device),
                            torch.as_tensor(privileged, device=device),
                        )
                        action = torch.tanh(mean).cpu().numpy()
                    receptors, privileged, reward, done, info = env.step(action)
                    total_return += reward
                    if done:
                        paused = True
                panel.refresh(env.vision.last_values, action, reward, info, paused, total_return)
                panel.pump(); viewer.sync()
                interval = env.model.opt.timestep*env.physics_steps
                time.sleep(max(0.0, interval-(time.perf_counter()-started)))
    finally:
        if not panel.closed:
            panel.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--checkpoint", type=Path, default=HERE/"checkpoints"/"latest.pt")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    run(args.config.resolve(), args.checkpoint.resolve(), args.stage)

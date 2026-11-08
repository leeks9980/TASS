"""Train the small front-leg-state to wheel-speed readout in MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from motor_decoder import LinearMotorDecoder
from mujoco_main import HERE, MujocoCart, resolve


def desired_wheels(angle_degrees, cruise=0.28, maximum_turn=0.22):
    turn = maximum_turn * np.clip(angle_degrees / 30.0, -1.0, 1.0)
    # Positive angle is a target to the cart's left. The right wheel must be
    # faster to turn the differential-drive cart toward it.
    return np.asarray([cruise - turn, cruise + turn], dtype=np.float64)


def collect_case(simulation, distance, angle_degrees, warmup=25, samples=15):
    angle = math.radians(angle_degrees)
    simulation.reset()
    simulation.set_target([
        distance * math.cos(angle),
        distance * math.sin(angle),
        simulation.config["target_initial_xyz"][2],
    ])
    states = []
    for tick in range(warmup + samples):
        state = simulation.observe_without_motion()
        if tick >= warmup:
            states.append(state)
    target = desired_wheels(angle_degrees)
    return states, [target.copy() for _ in states]


def baseline_state(simulation, warmup=35, samples=10):
    simulation.reset()
    simulation.set_target([-2.5, 0.0, simulation.config["target_initial_xyz"][2]])
    states = []
    for tick in range(warmup + samples):
        state = simulation.observe_without_motion()
        if tick >= warmup:
            states.append(state)
    return np.mean(states, axis=0)


def evaluate(simulation, decoder):
    rows = []
    for distance in (1.4, 2.2):
        for angle in (-25, -15, -5, 5, 15, 25):
            states, targets = collect_case(simulation, distance, angle, warmup=25, samples=1)
            predicted = decoder.predict(states[-1])
            expected = targets[-1]
            rows.append((angle, distance, expected, predicted))
    expected = np.asarray([row[2] for row in rows])
    predicted = np.asarray([row[3] for row in rows])
    mae = float(np.abs(expected - predicted).mean())
    expected_turn = np.sign(expected[:, 1] - expected[:, 0])
    predicted_turn = np.sign(predicted[:, 1] - predicted[:, 0])
    direction_accuracy = float(np.mean(expected_turn == predicted_turn))
    return rows, {"wheel_mae": mae, "turn_direction_accuracy": direction_accuracy}


def train(config_path):
    simulation = MujocoCart(config_path)
    try:
        baseline = baseline_state(simulation)
        states, targets = [], []
        for distance in (1.0, 1.8, 2.6):
            for angle in (-30, -20, -10, 0, 10, 20, 30):
                case_states, case_targets = collect_case(simulation, distance, angle)
                states.extend(case_states)
                targets.extend(case_targets)
        metadata = {
            "schema": 1,
            "input": "signed front-leg motor-neuron states, L then R",
            "motor_selection": simulation.cns.motor_selection_report,
            "training_cases": len(states),
            "target_source": "MuJoCo target geometry used only during supervised training",
            "cns_weights_frozen": True,
            "feature_normalization": "baseline subtraction then per-sample RMS normalization",
        }
        decoder = LinearMotorDecoder.fit(
            states, targets, baseline,
            ridge=float(simulation.config["decoder_ridge"]), metadata=metadata,
        )
        rows, metrics = evaluate(simulation, decoder)
        decoder.metadata["holdout_metrics"] = metrics
        output = resolve(simulation.config["decoder"])
        decoder.save(output)
        print(json.dumps({
            "output": str(output),
            "training_samples": len(states),
            "input_features": len(baseline),
            "metrics": metrics,
            "holdout": [
                {"angle": a, "distance": d, "expected": e.tolist(), "predicted": p.tolist()}
                for a, d, e, p in rows
            ],
        }, ensure_ascii=False, indent=2))
    finally:
        simulation.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    train(args.config.resolve())

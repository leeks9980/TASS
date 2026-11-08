"""Small trained readout from signed motor-neuron state to two wheel speeds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LinearMotorDecoder:
    def __init__(self, baseline, feature_mean, feature_scale, weights, metadata=None):
        self.baseline = np.asarray(baseline, dtype=np.float64)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.metadata = dict(metadata or {})
        if self.weights.shape != (len(self.baseline) + 1, 2):
            raise ValueError("decoder weight shape does not match the motor feature count")

    @classmethod
    def fit(cls, states, targets, baseline, ridge=1.0e-2, metadata=None):
        states = np.asarray(states, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        baseline = np.asarray(baseline, dtype=np.float64)
        if states.ndim != 2 or targets.shape != (len(states), 2):
            raise ValueError("states must be NxD and targets must be Nx2")
        delta = states - baseline
        magnitude = np.sqrt(np.mean(delta * delta, axis=1, keepdims=True))
        delta = delta / np.maximum(magnitude, 1.0e-10)
        feature_mean = delta.mean(axis=0)
        feature_scale = delta.std(axis=0)
        feature_scale[feature_scale < 1.0e-10] = 1.0
        normalized = (delta - feature_mean) / feature_scale
        design = np.column_stack((normalized, np.ones(len(normalized))))
        regularizer = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
        return cls(baseline, feature_mean, feature_scale, weights, metadata)

    def predict(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != self.baseline.shape:
            raise ValueError(f"expected {len(self.baseline)} motor features, got {state.shape}")
        delta = state - self.baseline
        magnitude = max(float(np.sqrt(np.mean(delta * delta))), 1.0e-10)
        normalized = (delta / magnitude - self.feature_mean) / self.feature_scale
        # Closed-loop recurrent state can drift outside the fixed-cart training
        # distribution. Bound per-neuron extrapolation so a few tiny-scale
        # features cannot immediately drive opposite wheel saturation.
        normalized = np.clip(normalized, -4.0, 4.0)
        design = np.append(normalized, 1.0)
        return design @ self.weights

    def save(self, path):
        path = Path(path)
        np.savez_compressed(
            path,
            baseline=self.baseline,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            weights=self.weights,
            metadata_json=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            return cls(data["baseline"], data["feature_mean"], data["feature_scale"],
                       data["weights"], metadata)

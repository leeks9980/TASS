"""Annotation-driven front-leg motor-neuron grouping and joint readout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FUNCTION_PATTERNS = {
    "promotor": ("promotor", "protractor", "protraction"),
    "remotor": ("remotor", "retractor", "retraction"),
    "abductor": ("abductor", "abduction"),
    "adductor": ("adductor", "adduction"),
    "levator": ("levator", "elevator", "levation", "elevation"),
    "depressor": ("depressor", "depression"),
    "flexor": ("flexor", "flexion"),
    "extensor": ("extensor", "extension"),
}


@dataclass
class GroupValue:
    side: str
    function: str
    count: int
    mean: float


class AnnotatedFrontLegReadout:
    """Group `fl` motor neurons using function words in published annotations.

    A neuron may appear in more than one functional group when its annotation
    explicitly names multiple actions, for example "remotor/abductor".
    Unnamed motor neurons are retained in the CNS but do not drive this rig.
    """

    def __init__(self, cns, gain=2500.0):
        self.cns = cns
        self.gain = float(gain)
        frame = cns.frame.copy()
        sides = frame.rootSide.fillna(frame.somaSide).fillna("").astype(str)
        text_columns = [column for column in ("type", "instance", "mancType", "cell_type")
                        if column in frame.columns]
        annotation_text = frame[text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        front_motor = frame.superclass.eq("vnc_motor") & frame.subclass.eq("fl")
        self.groups = {}
        classified = np.zeros(len(frame), dtype=bool)
        for side in ("L", "R"):
            for function, patterns in FUNCTION_PATTERNS.items():
                named = np.zeros(len(frame), dtype=bool)
                for pattern in patterns:
                    named |= annotation_text.str.contains(pattern, regex=False).to_numpy()
                selected = front_motor.to_numpy() & sides.eq(side).to_numpy() & named
                rows = np.flatnonzero(selected)
                self.groups[(side, function)] = cns.annotated_indices[rows]
                classified |= selected
        front_rows = np.flatnonzero(front_motor.to_numpy())
        self.report = {
            "selection": "vnc_motor + subclass fl",
            "annotation_columns": text_columns,
            "group_counts": {
                f"{side}_{function}": int(len(self.groups[(side, function)]))
                for side in ("L", "R") for function in FUNCTION_PATTERNS
            },
            "front_motor_rows": int(len(front_rows)),
            "unclassified_front_motor_rows": int(np.sum(~classified[front_rows])),
            "overlap_policy": "multi-action names may contribute to multiple groups",
            "biological_status": "annotation-driven engineering readout; joint mechanics are not validated",
        }
        self.baseline = {key: 0.0 for key in self.groups}
        self.last_values = {key: 0.0 for key in self.groups}

    def values(self):
        result = {}
        for key, indices in self.groups.items():
            result[key] = float(self.cns.state[indices].mean()) if len(indices) else 0.0
        self.last_values = result
        return result

    def capture_baseline(self):
        self.baseline = self.values().copy()

    def reset_baseline(self):
        self.baseline = {key: 0.0 for key in self.groups}

    def _difference(self, values, side, positive, negative):
        positive_value = values[(side, positive)] - self.baseline[(side, positive)]
        negative_value = values[(side, negative)] - self.baseline[(side, negative)]
        return self.gain * (positive_value - negative_value)

    def joint_targets(self):
        values = self.values()
        targets = {}
        for side in ("L", "R"):
            targets[f"{side}_protraction"] = self._difference(values, side, "promotor", "remotor")
            targets[f"{side}_abduction"] = self._difference(values, side, "abductor", "adductor")
            targets[f"{side}_levation"] = self._difference(values, side, "levator", "depressor")
            targets[f"{side}_flexion"] = self._difference(values, side, "flexor", "extensor")
        return targets, values

    def table_rows(self):
        return [
            GroupValue(side, function, len(self.groups[(side, function)]),
                       self.last_values[(side, function)])
            for side in ("L", "R") for function in FUNCTION_PATTERNS
        ]

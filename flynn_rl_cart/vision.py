"""Two-eye MuJoCo images mapped to the existing TASS photoreceptor inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.feather as feather


class PhotoreceptorInput:
    def __init__(self, circuit_folder, retention=0.3):
        self.folder = Path(circuit_folder)
        self.retention = float(retention)
        if not 0.0 <= self.retention < 1.0:
            raise ValueError("visual retention must be in [0, 1)")
        ids = np.load(self.folder / "body_ids.npy", mmap_mode="r")
        mapping = feather.read_feather(self.folder / "photoreceptor_hex_map.feather")
        mapping = mapping[mapping.side.isin(["L", "R"])].copy()
        positions = np.searchsorted(ids, mapping.bodyId.to_numpy(dtype=np.int64))
        valid = ((positions < len(ids)) &
                 (ids[np.minimum(positions, len(ids)-1)] == mapping.bodyId.to_numpy(dtype=np.int64)))
        mapping = mapping.loc[valid].copy()
        mapping["graph_index"] = positions[valid]
        self.sides = []
        for side in ("L", "R"):
            selected = mapping[mapping.side.eq(side)].copy()
            x = selected.assignedOlHex1.to_numpy(dtype=np.float32)
            y = selected.assignedOlHex2.to_numpy(dtype=np.float32)
            x = (x-x.min()) / max(float(np.ptp(x)), 1.0)
            y = (y-y.min()) / max(float(np.ptp(y)), 1.0)
            self.sides.append({
                "side": side,
                "indices": selected.graph_index.to_numpy(dtype=np.int64),
                "x": x,
                "y": y,
            })
        self.indices = np.concatenate([item["indices"] for item in self.sides])
        self.previous = {"L": None, "R": None}
        self.last_values = np.zeros(len(self.indices), dtype=np.float32)

    def reset(self):
        self.previous = {"L": None, "R": None}
        self.last_values.fill(0)

    @staticmethod
    def _bilinear(gray, x, y):
        height, width = gray.shape
        px = x * (width-1)
        py = y * (height-1)
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
        dx, dy = px-x0, py-y0
        return ((1-dx)*(1-dy)*gray[y0, x0] + dx*(1-dy)*gray[y0, x1] +
                (1-dx)*dy*gray[y1, x0] + dx*dy*gray[y1, x1]).astype(np.float32)

    def sample(self, left_rgb, right_rgb):
        values = []
        for item, rgb in zip(self.sides, (left_rgb, right_rgb)):
            gray = rgb.astype(np.float32).mean(axis=2) / 255.0
            previous = self.previous[item["side"]]
            filtered = gray if previous is None else (
                self.retention*previous + (1.0-self.retention)*gray
            )
            self.previous[item["side"]] = filtered
            values.append(self._bilinear(filtered, item["x"], item["y"]))
        self.last_values = np.concatenate(values)
        return self.last_values

    @property
    def report(self):
        return {
            "total": int(len(self.indices)),
            "side_counts": {item["side"]: int(len(item["indices"])) for item in self.sides},
            "mapping": "existing visual_motor_circuit photoreceptor hex map",
            "temporal_retention": self.retention,
        }

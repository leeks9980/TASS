"""Headless signal model for the wall -> L2 -> leg motor response GUI.

Uses published edges with explicit toy input, dynamics and output assumptions.
"""

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy.sparse import coo_matrix, diags

DEFAULT_CIRCUIT = ROOT / "circuit_extract" / "weight_1" / "max_8_hops"
GROUPS = [(leg, side) for leg in ("fl", "ml", "hl") for side in ("L", "R")]
LABELS = {"fl": "앞", "ml": "중간", "hl": "뒤"}
MAX_LIFT_DEGREES = 60.0
LEG_DISTANCE_ONSET = {"fl": 4.0, "ml": 2.6, "hl": 1.5}


def spatial_leg_lift(neural_scores, wall_x, distance, visible):
    """User-defined display policy, separate from the collected neural circuit.

    Centered walls affect both sides equally. Lateral displacement smoothly favors
    the same-side legs. Front, middle and hind legs engage at successively nearer
    distances. These are virtual distances, not inferred biological tuning.
    """
    scores = np.asarray(neural_scores, dtype=np.float32)
    if not visible:
        return np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)
    left = float(np.clip(0.5 - wall_x / 0.7, 0, 1))
    side_gain = {"L": left, "R": 1.0 - left}
    factors = []
    for leg, side in GROUPS:
        onset = LEG_DISTANCE_ONSET[leg]
        progress = float(np.clip((onset - distance) / (onset - 0.4), 0, 1))
        # Smooth engagement avoids abrupt changes as the wall moves forward/back.
        distance_gain = progress * progress * (3.0 - 2.0 * progress)
        factors.append(side_gain[side] * distance_gain)
    factors = np.asarray(factors, dtype=np.float32)
    return np.clip(scores * factors, 0, 1), factors


def camera_image(wall_x, wall_distance, wall_visible, pixels=96):
    """Single pinhole camera: dark-wall coverage for a 100-degree horizontal FOV."""
    angle = np.linspace(-math.radians(50), math.radians(50), pixels)
    hits = np.zeros(pixels)
    if wall_visible:
        intersection_x = np.tan(angle) * wall_distance
        hits[np.abs(intersection_x - wall_x) <= 0.55] = 1.0
    return hits


def nt_signs(frame, mode):
    """Prediction is not receptor physiology. Glutamate and unknown default to zero."""
    if mode == "연결 전달 (모두 양수)":
        return np.ones(len(frame), dtype=np.float32)
    names = frame.consensus_nt.fillna(frame.celltype_predicted_nt).fillna(frame.predicted_nt).fillna("").str.lower()
    signs = names.map({"acetylcholine": 1.0, "gaba": -1.0}).fillna(0).to_numpy(dtype=np.float32)
    # The user can explicitly choose a glutamate assumption, never hidden by the model.
    if mode.endswith("Glutamate −"):
        signs[names.eq("glutamate").to_numpy()] = -1
    elif mode.endswith("Glutamate +"):
        signs[names.eq("glutamate").to_numpy()] = 1
    return signs


class CircuitModel:
    def __init__(self, folder=DEFAULT_CIRCUIT, progress=lambda _: None):
        progress("뉴런 주석 읽는 중…")
        columns = ["bodyId", "type", "superclass", "subclass", "somaSide", "rootSide",
                   "from_L2_hops", "has_neuron_annotation", "consensus_nt", "predicted_nt",
                   "celltype_predicted_nt", "predicted_nt_confidence"]
        frame = feather.read_feather(folder / "neurons.feather", columns=columns)
        self.original_nodes = len(frame)
        self.frame = frame.loc[frame.has_neuron_annotation].sort_values("bodyId").reset_index(drop=True)
        self.ids = self.frame.bodyId.to_numpy(dtype=np.int64)
        self.sources = np.flatnonzero(self.frame.type.eq("L2").to_numpy())
        self.sides = self.frame.somaSide.fillna(self.frame.rootSide).fillna("").to_numpy()
        self.motor_groups = {}
        for leg, side in GROUPS:
            mask = self.frame.superclass.eq("vnc_motor") & self.frame.subclass.eq(leg)
            self.motor_groups[(leg, side)] = np.flatnonzero(mask.to_numpy() & (self.sides == side))
        if not len(self.sources):
            raise ValueError("추출 결과에 L2 입력이 없습니다.")
        if not any(len(v) for v in self.motor_groups.values()):
            raise ValueError("좌우 주석이 있는 다리 운동 뉴런이 없습니다.")
        rows, cols, values = [], [], []
        self.collected_edges = 0
        progress("실제 연결을 읽고, 주석 뉴런의 순방향 층 연결을 구성하는 중…")
        depth = self.frame.from_L2_hops.to_numpy()
        with pa.memory_map(str(folder / "connections.feather"), "r") as source:
            reader = ipc.open_file(source)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                pre = batch.column("body_pre").to_numpy()
                post = batch.column("body_post").to_numpy()
                weight = batch.column("weight").to_numpy()
                self.collected_edges += len(pre)
                a, b = np.searchsorted(self.ids, pre), np.searchsorted(self.ids, post)
                valid = (a < len(self.ids)) & (b < len(self.ids))
                a, b, pre, post, weight = a[valid], b[valid], pre[valid], post[valid], weight[valid]
                valid = (self.ids[a] == pre) & (self.ids[b] == post) & (depth[b] == depth[a] + 1)
                rows.append(b[valid].astype(np.int32))
                cols.append(a[valid].astype(np.int32))
                values.append(weight[valid].astype(np.float32))
        row, col, value = np.concatenate(rows), np.concatenate(cols), np.concatenate(values)
        self.weights = coo_matrix((value, (row, col)), shape=(len(self.ids), len(self.ids))).tocsr()
        totals = np.asarray(self.weights.sum(axis=1)).ravel()
        inv = np.divide(1, totals, out=np.zeros_like(totals), where=totals > 0)
        self.positive = diags(inv).dot(self.weights).tocsr()
        self.max_depth = int(self.frame.from_L2_hops.max())
        self.state = np.zeros(len(self.ids), dtype=np.float32)
        self.tick_count = 0
        self.reference = np.ones(6, dtype=np.float32)
        self.matrix = self.positive
        progress("벽 전체가 보이는 기준 입력으로 출력 스케일 계산 중…")
        self.configure("연결 전달 (모두 양수)")

    def raw_motor_response(self, state):
        # A group response is a visualization channel, not a decoded muscle command.
        return np.array([np.abs(state[index]).mean() if len(index) else 0
                         for index in self.motor_groups.values()], dtype=np.float32)

    def configure(self, mode):
        signs = nt_signs(self.frame, mode)
        self.matrix = self.positive.multiply(signs[np.newaxis, :]).tocsr()
        state = np.zeros(len(self.ids), dtype=np.float32)
        # Each positive-depth link increases depth by exactly one: no recurrent instability.
        for _ in range(self.max_depth + 2):
            state = self.matrix.dot(state)
            state[self.sources] = 1
        self.reference = self.raw_motor_response(state)
        self.zero_sign_count = int((signs == 0).sum())
        self.reset()

    def reset(self):
        self.state.fill(0)
        self.tick_count = 0

    def step(self, camera):
        # One camera image is split into two assumed visual fields, not stereo vision.
        left = float(camera[:len(camera) // 2].mean())
        right = float(camera[len(camera) // 2:].mean())
        source_side = self.sides[self.sources]
        drive = np.where(source_side == "L", left, np.where(source_side == "R", right, (left + right) / 2))
        propagated = self.matrix.dot(self.state)
        self.state = 0.35 * self.state + 0.65 * propagated
        self.state[self.sources] = drive
        self.tick_count += 1
        raw = self.raw_motor_response(self.state)
        score = np.divide(raw, self.reference, out=np.zeros_like(raw), where=self.reference > 1e-12)
        return np.clip(score, 0, 1), raw, left, right



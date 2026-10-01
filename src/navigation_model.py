"""Untrained recurrent circuit and explicit, replaceable navigation readout."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pyarrow.feather as feather
from scipy.sparse import diags, load_npz


class NavigationCircuit:
    def __init__(self, folder, nt_mode='structural', gain=0.95, leak=0.65):
        self.frame = feather.read_feather(folder / 'neurons.feather')
        self.weights = load_npz(folder / 'weights.npz').astype(np.float32)
        self.gain, self.leak = gain, leak
        self.nt_mode = nt_mode
        n = len(self.frame)
        self.state = np.zeros(n, dtype=np.float32)
        sides = self.frame.somaSide.fillna(self.frame.rootSide).fillna('').to_numpy()
        self.inputs = {}
        self.mapping = {}
        self.luminance_inputs = {}
        self.input_report = {}
        # Hex column indices are a spatial proxy, not a calibrated camera projection.
        for kind in ['L1', 'L2', 'L3']:
            for side in ['L', 'R']:
                mask = self.frame.type.eq(kind).to_numpy() & (sides == side)
                idx = np.flatnonzero(mask)
                total = len(idx)
                x = self.frame.assignedOlHex1.to_numpy(dtype=float)[idx]
                y = self.frame.assignedOlHex2.to_numpy(dtype=float)[idx]
                valid = np.isfinite(x) & np.isfinite(y)
                missing = idx[~valid]
                if kind == 'L3' and len(missing):
                    # No fabricated coordinates: nonspatial brightness input, logged explicitly.
                    self.luminance_inputs[side] = missing
                    print(f'{side} L3: {len(missing)}개에 공간 좌표가 없어 해당 쪽 시간 평균 밝기를 공통 입력합니다.', flush=True)
                idx, x, y = idx[valid], x[valid], y[valid]
                self.input_report[f'{side}_{kind}'] = {
                    'total': total, 'spatially_mapped': len(idx),
                    'uniform_luminance': len(missing) if kind == 'L3' else 0,
                    'without_direct_input': len(missing) if kind != 'L3' else 0,
                }
                if not len(idx):
                    if kind == 'L3' and len(missing):
                        continue
                    raise ValueError(f'{side} {kind}: 공간 주석이 없습니다. 임의 입력으로 대체하지 않습니다.')
                def normalize(a):
                    return (a - a.min()) / max(float(np.ptp(a)), 1.0)
                self.inputs[kind, side] = idx
                self.mapping[kind, side] = normalize(x), normalize(y)
        dn = self.frame.superclass.fillna('').str.startswith('descending_neuron').to_numpy()
        self.outputs = {s: np.flatnonzero(dn & (sides == s)) for s in ['L', 'R']}
        if any(len(i) == 0 for i in self.outputs.values()):
            raise ValueError('좌우 하행 신경 출력 주석이 없습니다.')
        # Conservative default: positive structural propagation. NT mode is an assumption.
        signs = np.ones(n, dtype=np.float32)
        if nt_mode == 'ach-gaba':
            names = self.frame.consensus_nt.fillna(self.frame.celltype_predicted_nt).fillna(self.frame.predicted_nt).fillna('').str.lower()
            signs = names.map({'acetylcholine': 1.0, 'gaba': -1.0}).fillna(0).to_numpy(dtype=np.float32)
        totals = np.asarray(self.weights.sum(axis=1)).ravel()
        inv = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
        self.matrix = (diags(inv) @ self.weights @ diags(signs)).tocsr()
        self.previous = None
        self.average = None

    def reset(self):
        self.state.fill(0)
        self.previous = self.average = None

    def step(self, rgb, steps=4):
        gray = rgb.astype(np.float32).mean(axis=2) / 255.0
        if self.previous is None:
            self.previous = gray.copy()
            self.average = gray.copy()
        delta = gray - self.previous
        self.average = 0.9 * self.average + 0.1 * gray
        stimuli = {'L1': np.clip(0.2 * gray + 4 * np.maximum(delta, 0), 0, 1),
                   'L2': np.clip(0.2 * (1 - gray) + 4 * np.maximum(-delta, 0), 0, 1),
                   'L3': self.average}
        drive = np.zeros_like(self.state)
        height, width = gray.shape
        for key, idx in self.inputs.items():
            kind, side = key
            x, y = self.mapping[key]
            # One camera divided into left/right fields; not stereo vision.
            u = ((x + (side == 'R')) * 0.5 * (width - 1)).astype(int)
            v = (y * (height - 1)).astype(int)
            drive[idx] = stimuli[kind][v, u]
        for side, idx in self.luminance_inputs.items():
            half = self.average[:, :width // 2] if side == 'L' else self.average[:, width // 2:]
            drive[idx] = float(half.mean())
        for _ in range(steps):
            self.state = self.leak * self.state + (1 - self.leak) * np.tanh(self.gain * (self.matrix @ self.state) + drive)
        self.previous = gray.copy()
        # This aggregation is an engineering hypothesis, not a native DN motor code.
        return {s: float(self.state[idx].mean()) for s, idx in self.outputs.items()}


def raw_command(scores, cruise=0.18, sensitivity=4.0):
    """Exploration drive is explicit; direction comes from untrained bilateral DN means."""
    left, right = scores['L'], scores['R']
    return cruise, float(np.clip(sensitivity * (right - left), -1.2, 1.2))


def visual_baseline(rgb):
    """Hand-written image-only obstacle rule, NOT evidence of circuit autonomy."""
    # Arena uses bright floor/walls and red obstacles; no geometry/depth/contact input.
    image = rgb.astype(np.float32) / 255
    red = (image[:, :, 0] > image[:, :, 1] * 1.5) & (image[:, :, 0] > image[:, :, 2] * 1.5)
    h, w = red.shape
    region = red[h // 3:]
    left, right = region[:, :w // 2].mean(), region[:, w // 2:].mean()
    occupied = float(region.mean())
    turn = 1.1 * np.clip((right - left) * 12, -1, 1)
    if occupied > 0.12 and abs(turn) < 0.1:
        turn = 0.9  # Explicit symmetry breaking in the baseline only.
    return float(0.18 * max(0.0, 1 - occupied * 5)), float(turn)

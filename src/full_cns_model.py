"""Entire segment graph, explicit assumed NT effects and point dynamics."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pyarrow.feather as feather
from scipy.sparse import csr_matrix


class FullCNS:
    def __init__(self, folder, glutamate='positive', gain=0.95, leak=0.65, weight_mode='incoming'):
        self.folder = Path(folder)
        self.ids = np.load(self.folder / 'body_ids.npy', mmap_mode='r')
        self.frame = feather.read_feather(self.folder / 'annotations.feather')
        self.annotated_indices = self.frame.graph_index.to_numpy(dtype=np.int64)
        self.gain, self.leak = gain, leak
        self.state = np.zeros(len(self.ids), dtype=np.float32)
        self.glutamate = glutamate
        # Copy-on-write mappings keep original raw counts immutable.
        data = np.load(self.folder / 'synapse_counts.npy', mmap_mode='c')
        indptr = np.load(self.folder / 'indptr.npy', mmap_mode='r')
        indices = np.load(self.folder / 'indices.npy', mmap_mode='r')
        totals = np.load(self.folder / 'incoming_sum.npy', mmap_mode='r')
        nt = feather.read_feather(self.folder / 'neurotransmitters.feather')
        names = nt.ground_truth.fillna(nt.consensus_nt).fillna(nt.celltype_predicted_nt).fillna(nt.predicted_nt).fillna('').str.lower()
        sign_lookup = {'gaba': -1., 'acetylcholine': 1., 'glutamate': -1. if glutamate == 'negative' else 1.}
        nt_indices = np.searchsorted(self.ids, nt.body.to_numpy(dtype=np.int64))
        signs = np.ones(len(self.ids), dtype=np.float32)
        signs[nt_indices] = names.map(sign_lookup).fillna(1).to_numpy(dtype=np.float32)
        self.nt_report = {
            'predicted_or_ground_truth_counts': {str(k): int(v) for k, v in names.value_counts().items()},
            'modeled_effects': sign_lookup,
            'other_or_unknown_effect': '+1 structural transmission; neuromodulation not modeled',
            'glutamate_effect_is_assumption': True,
            'dopamine_is_not_reward_or_instantaneous_learning': True,
        }
        for first in range(0, len(self.ids), 500_000):
            last = min(first + 500_000, len(self.ids))
            a, b = int(indptr[first]), int(indptr[last])
            if weight_mode == 'incoming':
                inv = np.divide(1., totals[first:last], out=np.zeros(last-first, dtype=np.float32), where=totals[first:last] > 0)
                # CSR rows [first,last) need last-first+1 boundary pointers.
                data[a:b] *= np.repeat(inv, np.diff(indptr[first:last+1]))
            data[a:b] *= signs[indices[a:b]]
        del signs
        self.matrix = csr_matrix((data, indices, indptr), shape=(len(self.ids), len(self.ids)), copy=False)
        self.previous = self.average = None
        self.tick = 0
        sides = self.frame.somaSide.fillna(self.frame.rootSide).fillna('').to_numpy()
        self.input_mapping = []
        self.input_report = {}
        for kind in ['L1', 'L2', 'L3']:
            for side in ['L', 'R']:
                positions = np.flatnonzero(self.frame.type.eq(kind).to_numpy() & (sides == side))
                if not len(positions):
                    raise ValueError(f'{side} {kind}: 뉴런 주석이 없습니다.')
                x = self.frame.assignedOlHex1.to_numpy(dtype=float)[positions]
                y = self.frame.assignedOlHex2.to_numpy(dtype=float)[positions]
                valid = np.isfinite(x) & np.isfinite(y)
                mapped = self.annotated_indices[positions[valid]]
                missing = self.annotated_indices[positions[~valid]]
                if len(mapped):
                    xx, yy = x[valid], y[valid]
                    xx = (xx - xx.min()) / max(float(np.ptp(xx)), 1.)
                    yy = (yy - yy.min()) / max(float(np.ptp(yy)), 1.)
                    self.input_mapping.append((kind, side, mapped, xx, yy))
                # Preserve missing-coordinate neurons and provide clearly marked uniform input.
                if len(missing):
                    self.input_mapping.append((kind, side, missing, None, None))
                self.input_report[f'{side}_{kind}'] = {'spatial': len(mapped), 'uniform_side_mean': len(missing)}
        motor = self.frame.superclass.eq('vnc_motor').to_numpy() & self.frame.subclass.isin(['fl', 'ml', 'hl']).to_numpy()
        self.motor_indices = {side: self.annotated_indices[np.flatnonzero(motor & (sides == side))] for side in ['L', 'R']}
        if any(len(x) == 0 for x in self.motor_indices.values()):
            raise ValueError('좌우 다리 운동 신경 주석이 없습니다.')
        self.motor_report = {s: len(v) for s, v in self.motor_indices.items()}
        self.last_input_indices = np.array([], dtype=np.int64)
        self.last_input_values = np.array([], dtype=np.float32)

    def reset(self):
        self.state.fill(0)
        self.previous = self.average = None
        self.tick = 0

    def step(self, rgb):
        gray = rgb.astype(np.float32).mean(axis=2) / 255
        if self.previous is None:
            self.previous = gray.copy()
            self.average = gray.copy()
        delta = gray - self.previous
        self.average = 0.9 * self.average + 0.1 * gray
        stimuli = {'L1': np.clip(0.2*gray + 4*np.maximum(delta, 0), 0, 1),
                   'L2': np.clip(0.2*(1-gray) + 4*np.maximum(-delta, 0), 0, 1), 'L3': self.average}
        h, w = gray.shape
        all_indices, all_values = [], []
        for kind, side, idx, x, y in self.input_mapping:
            image = stimuli[kind]
            if x is None:
                field = image[:, :w//2] if side == 'L' else image[:, w//2:]
                values = np.full(len(idx), float(field.mean()), dtype=np.float32)
            else:
                u = ((x + (side == 'R')) * 0.5 * (w-1)).astype(int)
                v = (y * (h-1)).astype(int)
                values = image[v, u]
            all_indices.append(idx)
            all_values.append(values)
        self.last_input_indices = np.concatenate(all_indices)
        self.last_input_values = np.concatenate(all_values)
        activation = self.matrix @ self.state
        activation *= self.gain
        activation[self.last_input_indices] += self.last_input_values
        np.tanh(activation, out=activation)
        # Treat activity as a nonnegative rate. Negative "firing" would reverse
        # the assumed sign of an inhibitory neuron's downstream effect.
        np.maximum(activation, 0, out=activation)
        self.state *= self.leak
        self.state += (1-self.leak) * activation
        self.previous = gray.copy()
        self.tick += 1
        # Population means remain diagnostics only. MuscleDecoder handles body output.
        scores = {side: float(np.maximum(self.state[idx], 0).mean()) for side, idx in self.motor_indices.items()}
        return scores

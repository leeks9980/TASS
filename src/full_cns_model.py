"""Entire segment graph, explicit assumed NT effects and point dynamics."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pyarrow.feather as feather
from scipy.sparse import csr_matrix

from neurotransmitter_model import neurotransmitter_effects, resolved_neurotransmitters
from safe_dynamics import DynamicsExpression


FLYNN_EQUATION = 'leak*h + (1-leak)*tanh(gain*recurrent + sensory + bias)'


def is_flynn_equation(equation):
    return ''.join(str(equation).split()) == ''.join(FLYNN_EQUATION.split())


def flynn_leaky_update(state, drive, retention):
    """Apply FLYNN Eq. (1) in place with a zero, untrained bias.

    ``retention`` is the paper's alpha: the fraction of the previous hidden
    state retained at each step.  The public FLYNN code calls ``1-retention``
    its update alpha.  Keeping signed tanh states is essential because clipping
    negative values would erase inhibition at a zero initial state.
    """
    np.tanh(drive, out=drive)
    state *= retention
    state += (1-retention) * drive


class FullCNS:
    def __init__(self, folder, glutamate='negative', gain=0.95, leak=0.65,
                 weight_mode='incoming',
                 equation=FLYNN_EQUATION,
                 bias=0.0):
        self.folder = Path(folder)
        self.ids = np.load(self.folder / 'body_ids.npy', mmap_mode='r')
        self.frame = feather.read_feather(self.folder / 'annotations.feather')
        self.annotated_indices = self.frame.graph_index.to_numpy(dtype=np.int64)
        self.gain, self.leak, self.bias = gain, leak, bias
        self.equation = DynamicsExpression(equation)
        self.state = np.zeros(len(self.ids), dtype=np.float32)
        self.sensory = np.zeros(len(self.ids), dtype=np.float32)
        self.glutamate = glutamate
        # Copy-on-write mappings keep original raw counts immutable.
        data = np.load(self.folder / 'synapse_counts.npy', mmap_mode='c')
        indptr = np.load(self.folder / 'indptr.npy', mmap_mode='r')
        indices = np.load(self.folder / 'indices.npy', mmap_mode='r')
        totals = np.load(self.folder / 'incoming_sum.npy', mmap_mode='r')
        nt = feather.read_feather(self.folder / 'neurotransmitters.feather')
        names, label_sources = resolved_neurotransmitters(nt)
        sign_lookup = neurotransmitter_effects(glutamate)
        nt_indices = np.searchsorted(self.ids, nt.body.to_numpy(dtype=np.int64))
        signs = np.ones(len(self.ids), dtype=np.float32)
        signs[nt_indices] = names.map(sign_lookup).fillna(1).to_numpy(dtype=np.float32)
        self.nt_report = {
            'predicted_or_ground_truth_counts': {str(k): int(v) for k, v in names.value_counts().items()},
            'label_source_counts': {str(k): int(v) for k, v in label_sources.value_counts().items()},
            'modeled_effects': sign_lookup,
            'other_or_unknown_effect': '+1 structural transmission for unclear/unannotated graph IDs',
            'glutamate_effect_is_assumption': True,
            'histamine_effect': '-1 model-wide approximation; receptor-resolved exceptions are not represented',
            'monoamine_effect': '+1 Shiu-compatible computational assumption; neuromodulation is not represented',
            'signed_state_interpretation': (
                'FLYNN-style tanh hidden states are signed computational states, not calibrated firing rates.'),
            'dopamine_is_not_reward_or_instantaneous_learning': True,
        }
        self.dynamics_report = {
            'model': ('FLYNN leaky connectome RNN recurrence' if is_flynn_equation(equation)
                      else 'user-configured connectome recurrence'),
            'equation': self.equation.expression,
            'matches_flynn_equation': is_flynn_equation(equation),
            'paper': 'https://arxiv.org/abs/2607.00025',
            'bias': float(self.bias),
            'bias_is_trained': False,
            'weights_are_trained': False,
            'difference_from_trained_FLYNN': (
                'FLYNN trains recurrent weights, per-cell-type leak rates, and biases; '
                'this experiment keeps MaleCNS weights fixed and uses one configured leak and scalar bias.'),
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
        sides = self.frame.rootSide.fillna(self.frame.somaSide).fillna('').to_numpy()
        self.input_mapping = []
        # Camera current enters only at the annotated primary visual sensory
        # population. L1/L2/L3 and later visual/CNS neurons are reached through
        # the published graph instead of being driven as an artificial boundary.
        sensory = self.frame.superclass.fillna('').eq('ol_sensory').to_numpy()
        if not np.any(sensory):
            raise ValueError('ol_sensory 광수용 감각 뉴런 주석이 없습니다.')
        mapping_path = self.folder / 'photoreceptor_hex_map.feather'
        mapping_report_path = self.folder / 'photoreceptor_hex_map_report.json'
        if not mapping_path.exists() or not mapping_report_path.exists():
            raise ValueError('광수용체 공간 매핑이 없습니다. src/build_photoreceptor_hex_map.py를 먼저 실행하세요.')
        receptor_map = feather.read_feather(mapping_path)
        receptor_map = receptor_map[receptor_map.side.isin(['L', 'R'])].copy()
        known = np.searchsorted(self.ids, receptor_map.bodyId.to_numpy(dtype=np.int64))
        valid_ids = ((known < len(self.ids)) &
                     (self.ids[np.minimum(known, len(self.ids)-1)] ==
                      receptor_map.bodyId.to_numpy(dtype=np.int64)))
        receptor_map = receptor_map.loc[valid_ids].copy()
        receptor_map['graph_index'] = known[valid_ids]
        mapping_report = json.loads(mapping_report_path.read_text(encoding='utf-8'))
        self.input_report = {
            'boundary': 'connectivity-derived, hex-mapped ol_sensory photoreceptors',
            'directly_driven_interneurons': [],
            'response': 'camera luminance; no calibrated fly spectral sensitivity',
            'spatial_mapping': mapping_report,
            'camera_projection': 'per-side min-max normalized hex coordinates; not calibrated visual angles',
            'unmapped_policy': 'no direct input; never replaced with side-average luminance',
        }
        for side in ['L', 'R']:
            selected = receptor_map[receptor_map.side.eq(side)].copy()
            if selected.empty:
                raise ValueError(f'{side} 광수용체 hex 매핑이 없습니다.')
            x = selected.assignedOlHex1.to_numpy(dtype=float)
            y = selected.assignedOlHex2.to_numpy(dtype=float)
            x = (x-x.min()) / max(float(np.ptp(x)), 1.)
            y = (y-y.min()) / max(float(np.ptp(y)), 1.)
            self.input_mapping.append(
                (side, selected.graph_index.to_numpy(dtype=np.int64), x, y))
            types = selected.type.fillna('(untyped)').value_counts()
            self.input_report[side] = {
                'annotated_total': int(np.sum(sensory & (sides == side))),
                'spatially_mapped': int(len(selected)),
                'unmapped_without_input': int(np.sum(sensory & (sides == side))-len(selected)),
                'types': {str(name): int(count) for name, count in types.items()},
            }
        motor = self.frame.superclass.eq('vnc_motor').to_numpy() & self.frame.subclass.isin(['fl', 'ml', 'hl']).to_numpy()
        self.motor_indices = {side: self.annotated_indices[np.flatnonzero(motor & (sides == side))] for side in ['L', 'R']}
        if any(len(x) == 0 for x in self.motor_indices.values()):
            raise ValueError('좌우 다리 운동 신경 주석이 없습니다.')
        self.motor_report = {s: len(v) for s, v in self.motor_indices.items()}
        self.last_input_indices = np.array([], dtype=np.int64)
        self.last_input_values = np.array([], dtype=np.float32)

    def reset(self):
        self.state.fill(0)
        self.sensory.fill(0)
        self.previous = self.average = None
        self.tick = 0

    def step(self, rgb, clamp_indices=None):
        gray = rgb.astype(np.float32).mean(axis=2) / 255
        if self.previous is None:
            self.previous = gray.copy()
            self.average = gray.copy()
        self.average = 0.9 * self.average + 0.1 * gray
        # This is an explicit camera-to-photoreceptor adapter, not a calibrated
        # compound-eye spectral or phototransduction model.
        stimulus = np.clip(self.average, 0, 1)
        h, w = gray.shape
        all_indices, all_values = [], []
        for side, idx, x, y in self.input_mapping:
            # Each receptor samples its own point. Hex-to-viewing-angle
            # calibration remains separate from this topology-preserving map.
            pixel_x = (x * .5 + (side == 'R') * .5) * (w-1)
            pixel_y = y * (h-1)
            u0 = np.floor(pixel_x).astype(int)
            v0 = np.floor(pixel_y).astype(int)
            u1 = np.minimum(u0 + 1, w-1)
            v1 = np.minimum(v0 + 1, h-1)
            du = pixel_x-u0
            dv = pixel_y-v0
            values = ((1-du)*(1-dv)*stimulus[v0, u0] + du*(1-dv)*stimulus[v0, u1] +
                      (1-du)*dv*stimulus[v1, u0] + du*dv*stimulus[v1, u1]).astype(np.float32)
            all_indices.append(idx)
            all_values.append(values)
        self.last_input_indices = np.concatenate(all_indices)
        self.last_input_values = np.concatenate(all_values)
        recurrent = self.matrix @ self.state
        self.sensory.fill(0)
        self.sensory[self.last_input_indices] = self.last_input_values
        self.state[:] = self.equation.evaluate(
            h=self.state, recurrent=recurrent, sensory=self.sensory,
            gain=self.gain, leak=self.leak, bias=self.bias)
        if clamp_indices is not None:
            # Model intervention used by offline ablation replay. Clamping after
            # the update removes this population from the next recurrent step.
            self.state[np.asarray(clamp_indices, dtype=np.int64)] = 0.
        self.previous = gray.copy()
        self.tick += 1
        # Population means remain diagnostics only. MuscleDecoder handles body output.
        scores = {side: float(np.maximum(self.state[idx], 0).mean()) for side, idx in self.motor_indices.items()}
        return scores

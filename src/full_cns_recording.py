"""Lossless whole-graph snapshots with automatic sparse/dense storage."""
import json
from pathlib import Path
import time
import numpy as np


class Recorder:
    def __init__(self, folder, budget_gb=10):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget_gb * 1024**3)
        self.bytes = 0
        self.count = 0
        self.events = (self.folder / 'events.jsonl').open('w', encoding='utf-8', buffering=1)

    def save(self, circuit, rgb, event, extra_arrays=None):
        started = time.perf_counter()
        path = self.folder / f'frame_{self.count:07d}.npz'
        # Every graph state, including unannotated segments, is reconstructible.
        common = dict(input_indices=circuit.last_input_indices, input_values=circuit.last_input_values,
                      rgb=rgb, tick=np.int64(circuit.tick))
        if extra_arrays is not None:
            if set(extra_arrays) & (set(common) | {'state', 'state_indices', 'state_values'}):
                raise ValueError('Additional recording arrays conflict with neural snapshot fields')
            common.update(extra_arrays)
        if 'muscles' in event:
            legs = list(event['muscles'])
            common.update(
                leg_names=np.asarray(legs),
                muscle_activity=np.array([[event['muscles'][leg]['flexor_activity'], event['muscles'][leg]['extensor_activity']] for leg in legs]),
                muscle_activation=np.array([[event['muscles'][leg]['flexor_activation'], event['muscles'][leg]['extensor_activation']] for leg in legs]),
                muscle_force_N=np.array([[event['muscles'][leg]['flexor_force_N'], event['muscles'][leg]['extensor_force_N']] for leg in legs]),
                joint_torque_Nm=np.array([event['muscles'][leg]['torque_Nm'] for leg in legs]),
                joint_angle_rad=np.array([event['muscles'][leg]['angle_rad'] for leg in legs]),
            )
        # Dense state costs 4 bytes/node, whereas sparse int64+float32 costs
        # 12 bytes/active node. Deflate of tens of millions of floats blocks UI.
        active = int(np.count_nonzero(circuit.state))
        if active * 3 >= len(circuit.state):
            np.savez(path, state=circuit.state, **common)
            representation = 'dense_uncompressed'
        else:
            nonzero = np.flatnonzero(circuit.state)
            np.savez(path, state_indices=nonzero, state_values=circuit.state[nonzero], **common)
            representation = 'sparse_uncompressed'
        self.bytes += path.stat().st_size
        self.events.write(json.dumps(dict(event, frame=self.count, snapshot=path.name,
                                         nonzero_states=active, representation=representation,
                                         save_ms=(time.perf_counter()-started)*1000), ensure_ascii=False) + '\n')
        self.count += 1
        return self.bytes >= self.budget

    def close(self):
        self.events.close()


def state_blocks(snapshot, block_size=1_000_000):
    """Read old compressed sparse and new lossless sparse/dense recordings."""
    if 'state' in snapshot.files:
        state = snapshot['state']
        for start in range(0, len(state), block_size):
            block = state[start:start+block_size]
            local = np.flatnonzero(block)
            if len(local):
                yield local+start, block[local]
    else:
        yield snapshot['state_indices'], snapshot['state_values']


def state_at(snapshot, indices):
    if 'state' in snapshot.files:
        return snapshot['state'][indices]
    ids, values = snapshot['state_indices'], snapshot['state_values']
    places = np.searchsorted(ids, indices)
    result = np.zeros(len(indices), dtype=values.dtype)
    valid = np.flatnonzero(places < len(ids))
    equal = ids[places[valid]] == indices[valid]
    valid = valid[equal]
    result[valid] = values[places[valid]]
    return result

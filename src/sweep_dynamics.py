"""Replay recorded camera frames to compare gain, leak and propagation time."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd

from experiment_config import load_experiment
from full_cns_model import FullCNS


STAGES = [
    ('photoreceptor', 'ol_sensory'),
    ('optic_lobe', 'ol_intrinsic'),
    ('visual_projection', 'visual_projection'),
    ('descending', 'descending_neuron'),
    ('leg_motor', 'vnc_motor'),
]


def images(run, case):
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = [event for event in events if event['case'] == case]
    result = []
    for event in selected:
        with np.load(run / event['snapshot']) as snapshot:
            result.append(snapshot['rgb'].copy())
    if not result:
        raise ValueError(f'no recorded frames for case: {case}')
    return result


def sweep(run, gains, leaks, post_ticks):
    run = run.resolve()
    experiment = load_experiment()
    defaults = experiment['defaults']
    circuit = FullCNS(
        defaults['circuit'], glutamate=defaults['glutamate'], gain=defaults['gain'],
        leak=defaults['leak'], weight_mode=defaults['weight_mode'],
        equation=defaults['equation'], bias=defaults['bias'])
    neutral_images = images(run, 'neutral')
    bright_images = images(run, 'bright')
    onset_tick = int(round(defaults['stimulus_onset'] / defaults['neural_interval']))
    checkpoints = [onset_tick + value for value in post_ticks]
    total_ticks = max(checkpoints)
    frame = circuit.frame
    motor = frame.subclass.fillna('').isin(['fl', 'ml', 'hl'])
    indices = {}
    for name, superclass in STAGES:
        mask = frame.superclass.fillna('').eq(superclass)
        if name == 'leg_motor':
            mask &= motor
        indices[name] = frame.loc[mask, 'graph_index'].to_numpy(dtype=np.int64)

    rows = []
    for gain in gains:
        for leak in leaks:
            states = {}
            for case, recorded in [('neutral', neutral_images), ('bright', bright_images)]:
                circuit.gain = float(gain)
                circuit.leak = float(leak)
                circuit.reset()
                for tick in range(1, total_ticks + 1):
                    image = recorded[min(tick - 1, len(recorded) - 1)]
                    circuit.step(image)
                    if tick in checkpoints:
                        states[(case, tick)] = circuit.state.copy()
            for post_tick, tick in zip(post_ticks, checkpoints):
                delta = states[('bright', tick)] - states[('neutral', tick)]
                for stage, graph_indices in indices.items():
                    values = delta[graph_indices]
                    neutral_values = states[('neutral', tick)][graph_indices]
                    bright_values = states[('bright', tick)][graph_indices]
                    rows.append({
                        'gain': float(gain), 'leak': float(leak),
                        'post_stimulus_ticks': int(post_tick), 'stage': stage,
                        'signed_mean_delta': float(values.mean()),
                        'mean_absolute_neuron_delta': float(np.abs(values).mean()),
                        'maximum_absolute_neuron_delta': float(np.abs(values).max()),
                        'responding_fraction': float(np.count_nonzero(np.abs(values) > 1e-12) / len(values)),
                        'neutral_mean_absolute_state': float(np.abs(neutral_values).mean()),
                        'bright_mean_absolute_state': float(np.abs(bright_values).mean()),
                        'neutral_saturated_fraction': float(np.count_nonzero(np.abs(neutral_values) >= .99) / len(values)),
                        'bright_saturated_fraction': float(np.count_nonzero(np.abs(bright_values) >= .99) / len(values)),
                    })
            print(f'비교 완료: gain={gain:g}, leak={leak:g}', flush=True)
    table = pd.DataFrame(rows)
    output = run / 'dynamics_sweep.csv'
    table.to_csv(output, index=False, encoding='utf-8-sig')
    motor_table = table[table.stage.eq('leg_motor')].sort_values(
        'mean_absolute_neuron_delta', ascending=False)
    print(motor_table.head(20).to_string(index=False), flush=True)
    print(f'저장: {output}', flush=True)
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--gains', nargs='+', type=float, default=[.95, 1.5, 2., 3.])
    parser.add_argument('--leaks', nargs='+', type=float, default=[.65, .3, 0.])
    parser.add_argument('--post-ticks', nargs='+', type=int, default=[10, 20, 40, 60])
    args = parser.parse_args()
    if any(value < 0 for value in args.gains) or any(not 0 <= value < 1 for value in args.leaks):
        parser.error('gains must be nonnegative and leaks must be in [0,1)')
    if any(value < 1 for value in args.post_ticks):
        parser.error('post-ticks must be positive')
    sweep(args.run, args.gains, args.leaks, args.post_ticks)

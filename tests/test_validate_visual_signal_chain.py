import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from validate_visual_signal_chain import validate


def test_validator_creates_human_readable_artifacts(tmp_path):
    rows = []
    definitions = [
        ('ol_sensory', None, 'L'), ('ol_sensory', None, 'R'),
        ('ol_intrinsic', None, 'L'), ('visual_projection', None, 'R'),
        ('descending_neuron', None, 'L'), ('vnc_motor', 'fl', 'R')]
    for graph_index, (superclass, subclass, side) in enumerate(definitions):
        rows.append({'bodyId': graph_index + 100, 'graph_index': graph_index,
                     'superclass': superclass, 'subclass': subclass,
                     'rootSide': side, 'somaSide': None})
    pd.DataFrame(rows).to_feather(tmp_path / 'annotations.feather')
    effects = {'acetylcholine': 1., 'gaba': -1., 'glutamate': -1., 'histamine': -1.}
    metadata = {'parameters': {'stimulus_onset': 0.}, 'nt': {'modeled_effects': effects},
                'circuit_report': {'scope': 'photoreceptor_to_leg_motor_structural_corridor',
                                   'runtime_uses_only_saved_corridor': True,
                                   'graph_nodes_including_segments': 6, 'csr_edges': 5},
                'dynamics': {'bias': 0.0, 'matches_flynn_equation': True},
                'equation': 'leak*h + (1-leak)*tanh(gain*recurrent + sensory + bias)'}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    events = []
    for frame, (case, scale) in enumerate([('neutral', 1.), ('move_left', 2.), ('move_right', 3.)]):
        state = np.arange(1, 7, dtype=np.float32) * scale
        name = f'frame_{frame:07d}.npz'
        np.savez(tmp_path / name, state=state, input_indices=np.array([0, 1]),
                 input_values=np.array([scale, scale + .5], dtype=np.float32))
        events.append({'case': case, 'frame': frame, 'snapshot': name,
                       'input_sim_seconds': 0., 'sim_seconds': 0.})
    (tmp_path / 'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events), encoding='utf-8')

    report = validate(tmp_path)

    output = tmp_path / 'signal_chain_validation'
    assert report['verdict'] == '검토 가능'
    assert report['runtime_uses_only_visual_motor_corridor'] is True
    assert report['uses_signed_flynn_dynamics'] is True
    for name in ['report.html', 'validation_report.json', 'stage_timeseries.csv',
                 'condition_summary.csv', 'side_summary.csv',
                 'signal_chain_heatmap.svg', 'left_right_response.svg',
                 'extracted_circuit_overview.svg']:
        assert (output / name).is_file()
    assert '시각 입력' in (output / 'report.html').read_text(encoding='utf-8')

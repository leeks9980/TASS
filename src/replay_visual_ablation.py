"""Replay recorded camera frames with candidate neurons clamped to zero.

This is a counterfactual intervention inside the recorded point-neuron model. It does not
run MuJoCo and must not be interpreted as an in-vivo biological ablation result.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd

from full_cns_model import FullCNS
from visual_input_boundary import is_photoreceptor_boundary


def selected_events(run, case, episode):
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    events = [event for event in events if event['case'] == case]
    if episode is None:
        episodes = sorted({event['episode'] for event in events})
        if not episodes:
            raise ValueError(f'{case} 기록이 없습니다.')
        episode = episodes[-1]
    events = sorted((event for event in events if event['episode'] == episode), key=lambda event: event['frame'])
    if not events:
        raise ValueError(f'{case}, episode {episode} 기록이 없습니다.')
    return events, episode


def candidate_bodies(args):
    if args.bodies:
        return np.asarray(args.bodies, dtype=np.int64)
    path = args.run / 'visual_response_analysis' / 'stimulus_selectivity.csv'
    if not path.exists():
        raise FileNotFoundError('먼저 analyze_visual_responses.py를 실행하거나 --bodies를 지정하세요.')
    frame = pd.read_csv(path)
    frame = frame.loc[frame.preferred_condition.eq(args.case)]
    if args.superclass:
        frame = frame.loc[frame.superclass.fillna('').eq(args.superclass)]
    return frame.sort_values('preferred_absolute_response', ascending=False).bodyId.head(args.top).to_numpy(dtype=np.int64)


def replay(circuit, run, events, output_indices, clamp):
    circuit.reset()
    rows = []
    for event in events:
        with np.load(run / event['snapshot']) as snapshot:
            rgb = snapshot['rgb']
        circuit.step(rgb, clamp_indices=clamp)
        rows.append(circuit.state[output_indices].copy())
    return np.asarray(rows)


def summarize(label, bodies, baseline, intervention, times, onset, output_sides):
    keep = times >= onset
    delta = intervention-baseline
    rows = []
    for side in ['L', 'R', 'ALL']:
        positions = output_sides[side]
        base_series = baseline[:, positions].mean(axis=1) if len(positions) else np.zeros(len(times))
        intervention_series = intervention[:, positions].mean(axis=1) if len(positions) else np.zeros(len(times))
        effect = intervention_series-base_series
        post = effect[keep]
        peak_position = int(np.argmax(np.abs(post))) if len(post) else 0
        rows.append({'intervention': label, 'ablated_body_ids': ' '.join(map(str, bodies)),
                     'output_side': side, 'output_neuron_count': int(len(positions)),
                     'baseline_post_mean': float(base_series[keep].mean()),
                     'ablated_post_mean': float(intervention_series[keep].mean()),
                     'mean_signed_effect': float(post.mean()),
                     'mean_absolute_effect': float(np.abs(post).mean()),
                     'peak_signed_effect': float(post[peak_position]),
                     'peak_time_after_onset_s': float(times[keep][peak_position]-onset)})
    return rows


def analyze(args):
    metadata = json.loads((args.run / 'metadata.json').read_text(encoding='utf-8'))
    boundary = metadata.get('inputs', {}).get('boundary')
    if not is_photoreceptor_boundary(boundary):
        raise ValueError('이 기록은 현재 광수용체 입력 모델로 생성되지 않았습니다. 이전 L1/L2/L3 기록에는 제거 재생을 적용하지 않습니다.')
    folder = Path(metadata['circuit_folder'])
    if json.loads((folder / 'report.json').read_text(encoding='utf-8')) != metadata['circuit_report']:
        raise ValueError('기록 당시 full_cns 그래프와 현재 그래프가 다릅니다.')
    parameters = metadata['parameters']
    circuit = FullCNS(folder, glutamate=parameters.get('glutamate', 'negative'),
                      gain=float(parameters['gain']), leak=float(parameters['leak']),
                      weight_mode=parameters.get('weight_mode', 'incoming'),
                      equation=parameters.get(
                          'equation', 'leak*h + (1-leak)*tanh(gain*recurrent + sensory + bias)'),
                      bias=float(parameters.get('bias', 0.)))
    events, episode = selected_events(args.run, args.case, args.episode)
    onset = args.after_seconds
    if onset is None:
        onset = float(parameters.get('stimulus_onset', .5))
    times = np.asarray([event.get('input_sim_seconds', event['sim_seconds']) for event in events], dtype=float)
    if not np.any(times >= onset):
        raise ValueError('자극 시작 이후 프레임이 없습니다.')
    sides = circuit.frame.somaSide.fillna(circuit.frame.rootSide).fillna('').to_numpy()
    dn_mask = circuit.frame.superclass.fillna('').str.startswith('descending_neuron').to_numpy()
    dn_positions = np.flatnonzero(dn_mask)
    output_indices = circuit.annotated_indices[dn_positions]
    output_sides = {'L': np.flatnonzero(sides[dn_positions] == 'L'),
                    'R': np.flatnonzero(sides[dn_positions] == 'R'),
                    'ALL': np.arange(len(output_indices), dtype=np.int64)}
    bodies = candidate_bodies(args)
    if not len(bodies):
        raise ValueError('제거할 후보 뉴런이 없습니다.')
    indices = np.searchsorted(circuit.ids, bodies)
    valid = indices < len(circuit.ids)
    present = np.zeros(len(bodies), dtype=bool)
    present[valid] = circuit.ids[indices[valid]] == bodies[valid]
    valid = present
    if not np.all(valid):
        raise ValueError(f'전체 그래프에 없는 bodyId: {bodies[~valid].tolist()}')
    baseline = replay(circuit, args.run, events, output_indices, None)
    result_rows, curve_rows = [], []
    interventions = [('group', bodies, indices)] if args.mode == 'group' else [
        (f'body_{int(body)}', np.asarray([body]), np.asarray([index]))
        for body, index in zip(bodies, indices)]
    for label, intervention_bodies, intervention_indices in interventions:
        ablated = replay(circuit, args.run, events, output_indices, intervention_indices)
        result_rows.extend(summarize(label, intervention_bodies, baseline, ablated,
                                     times, onset, output_sides))
        for side in ['L', 'R', 'ALL']:
            positions = output_sides[side]
            baseline_series = (baseline[:, positions].mean(axis=1) if len(positions)
                               else np.zeros(len(times)))
            ablated_series = (ablated[:, positions].mean(axis=1) if len(positions)
                              else np.zeros(len(times)))
            for event, time_value, original, changed in zip(events, times, baseline_series, ablated_series):
                curve_rows.append({'intervention': label, 'output_side': side,
                                   'frame': int(event['frame']), 'time_s': time_value,
                                   'time_after_onset_s': time_value-onset,
                                   'baseline_activity': float(original),
                                   'ablated_activity': float(changed),
                                   'signed_effect': float(changed-original)})
    output = args.run / 'visual_ablation_analysis'
    output.mkdir(exist_ok=True)
    pd.DataFrame(result_rows).to_csv(output / 'ablation_effects.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(curve_rows).to_csv(output / 'ablation_timeline.csv', index=False, encoding='utf-8-sig')
    report = {'case': args.case, 'episode': episode, 'stimulus_onset_s': onset,
              'mode': args.mode, 'candidate_body_ids': bodies.tolist(),
              'output': 'all annotated descending_neuron populations, summarized by soma/root side',
              'intervention': 'candidate state clamped to zero after every neural update',
              'meaning': 'counterfactual causal effect inside this deterministic point-neuron model only',
              'limitations': ['Not biological ablation evidence.',
                              'Replays camera frames open-loop; body/camera feedback is not recomputed.',
                              'Requires a recording made with the current ol_sensory input boundary.']}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'기록 영상 기반 제거 재생 결과 저장: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--episode', type=int)
    parser.add_argument('--after-seconds', type=float)
    parser.add_argument('--bodies', type=int, nargs='+')
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--superclass')
    parser.add_argument('--mode', choices=('group', 'individual'), default='individual')
    args = parser.parse_args()
    if args.top < 1:
        parser.error('top must be positive')
    analyze(args)

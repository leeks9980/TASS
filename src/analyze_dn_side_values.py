"""Summarize fixed-body visual responses as left/right DN population values."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def neuron_side(frame):
    side = frame['rootSide'].fillna(frame['somaSide']).astype('string')
    suffix = frame['instance'].astype('string').str.extract(r'_([LR])$')[0]
    return side.fillna(suffix)


def safe_index(right, left):
    denominator = abs(right) + abs(left)
    return float((right-left) / denominator) if denominator > 0 else 0.0


def analyze(run, output):
    ranking = run / 'automatic_post_analysis' / 'dn_response_rankings'
    source_report = json.loads((ranking / 'report.json').read_text(encoding='utf-8'))
    cases = source_report['cases']
    rows = []
    vectors = {}
    side_counts = None
    for case in cases:
        frame = pd.read_csv(ranking / f'{case}.csv')
        frame = frame.sort_values('bodyId').reset_index(drop=True)
        sides = neuron_side(frame)
        vectors[case] = frame.set_index('bodyId')['signed_response']
        if side_counts is None:
            side_counts = {side: int(sides.eq(side).fillna(False).sum()) for side in ('L', 'R')}
        result = {'case': case}
        for side in ('L', 'R'):
            mask = sides.eq(side).fillna(False).to_numpy(dtype=bool)
            target = frame.loc[mask, 'target_mean'].to_numpy(dtype=float)
            reference = frame.loc[mask, 'reference_mean'].to_numpy(dtype=float)
            signed = frame.loc[mask, 'signed_response'].to_numpy(dtype=float)
            result[f'{side}_target_mean'] = float(target.mean())
            result[f'{side}_neutral_mean'] = float(reference.mean())
            result[f'{side}_signed_change'] = float(signed.mean())
            result[f'{side}_absolute_change'] = float(np.abs(signed).mean())
        result['target_lr_index'] = safe_index(result['R_target_mean'], result['L_target_mean'])
        result['change_lr_index'] = safe_index(result['R_signed_change'], result['L_signed_change'])
        result['absolute_change_lr_index'] = safe_index(
            result['R_absolute_change'], result['L_absolute_change'])
        rows.append(result)

    summary = pd.DataFrame(rows)
    matrix = pd.DataFrame(vectors).fillna(0.0)
    correlation = matrix.corr()
    distances = pd.DataFrame(index=cases, columns=cases, dtype=float)
    for left in cases:
        for right in cases:
            distances.loc[left, right] = float(np.linalg.norm(matrix[left]-matrix[right]))
    left_right_distance = (float(distances.loc['move_left', 'move_right'])
                           if {'move_left', 'move_right'} <= set(cases) else None)
    left_right_correlation = (float(correlation.loc['move_left', 'move_right'])
                              if {'move_left', 'move_right'} <= set(cases) else None)
    mirror_turn_product = None
    if {'move_left', 'move_right'} <= set(cases):
        turns = summary.set_index('case')['change_lr_index']
        mirror_turn_product = float(turns['move_left'] * turns['move_right'])

    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / 'dn_side_summary.csv', index=False, encoding='utf-8-sig')
    correlation.to_csv(output / 'condition_correlation.csv', encoding='utf-8-sig')
    distances.to_csv(output / 'condition_l2_distance.csv', encoding='utf-8-sig')
    report = {
        'schema': 1,
        'source_run': str(run.resolve()),
        'body_condition': 'fixed during the automatic recording',
        'dn_count': source_report['descending_neuron_count'],
        'side_counts': side_counts,
        'reference': source_report['reference'],
        'stimulus_onset_s': source_report['stimulus_onset_s'],
        'metrics': {
            'move_left_vs_move_right_response_correlation': left_right_correlation,
            'move_left_vs_move_right_l2_distance': left_right_distance,
            'mirror_turn_product': mirror_turn_product,
            'mirror_expectation': 'negative: opposite stimuli should produce opposite-signed lateral evidence',
        },
        'interpretation': {
            'target_lr_index': '(right target mean - left target mean) / sum of magnitudes',
            'change_lr_index': '(right neutral-subtracted mean - left) / sum of magnitudes',
            'absolute_change_lr_index': 'side imbalance in response magnitude, ignoring response sign',
        },
        'limitations': [
            'These are anatomical-side population summaries, not a validated motor command.',
            'One mean per condition cannot estimate trial-to-trial biological or model variability.',
            'High-dimensional patterns can differ even when bilateral means are similar.',
        ],
    }
    (output / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    shown = summary[['case', 'L_target_mean', 'R_target_mean',
                     'L_signed_change', 'R_signed_change', 'change_lr_index']]
    print(shown.to_string(index=False, float_format=lambda value: f'{value:.9g}'), flush=True)
    if left_right_correlation is not None:
        print(f'\nmove_left/right correlation: {left_right_correlation:.9f}', flush=True)
        print(f'move_left/right L2 distance: {left_right_distance:.9g}', flush=True)
        print(f'mirror turn product: {mirror_turn_product:.9g} '
              f'({"opposite" if mirror_turn_product < 0 else "same direction"})', flush=True)
    print(f'\nSaved: {output.resolve()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    destination = args.output or args.run / 'dn_side_validation'
    analyze(args.run, destination)

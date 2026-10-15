"""Rank descending neurons for every recorded stimulus with one snapshot pass."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from full_cns_recording import state_at


def main(run: Path, output: Path, reference: str) -> None:
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    onset = float(metadata.get('parameters', {}).get('stimulus_onset', .5))
    annotations = feather.read_feather(run / 'annotations.feather')
    superclass = annotations['superclass'].fillna('').astype(str)
    dn = annotations[superclass.str.startswith('descending_neuron')].copy()
    if dn.empty:
        raise ValueError('No descending neurons were found in annotations.feather.')
    dn = dn.drop_duplicates('graph_index').sort_values('graph_index')
    indices = dn['graph_index'].to_numpy(dtype=np.int64)

    events = [json.loads(line) for line in
              (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    cases = sorted({event['case'] for event in events if event['case'] != reference})
    sums = {case: np.zeros(len(indices), dtype=np.float64) for case in [reference] + cases}
    counts = {case: 0 for case in sums}
    eligible = [event for event in events
                if event['case'] in sums and
                float(event.get('input_sim_seconds', event['sim_seconds'])) >= onset]
    for number, event in enumerate(eligible, 1):
        case = event['case']
        time_value = float(event.get('input_sim_seconds', event['sim_seconds']))
        with np.load(run / event['snapshot']) as snapshot:
            sums[case] += state_at(snapshot, indices)
        counts[case] += 1
        if number == 1 or number % 10 == 0 or number == len(eligible):
            print(f'DN ranking snapshot {number}/{len(eligible)} ({case}, {time_value:g}s)', flush=True)

    if counts[reference] == 0:
        raise ValueError(f'No post-onset {reference} frames were found.')
    reference_mean = sums[reference] / counts[reference]
    output.mkdir(parents=True, exist_ok=True)
    report = {'reference': reference, 'stimulus_onset_s': onset,
              'descending_neuron_count': int(len(indices)), 'frame_counts': counts,
              'cases': cases}
    for case in cases:
        if counts[case] == 0:
            continue
        target_mean = sums[case] / counts[case]
        signed = target_mean - reference_mean
        result = dn.copy()
        result['target_mean'] = target_mean
        result['reference_mean'] = reference_mean
        result['signed_response'] = signed
        result['response_score'] = np.abs(signed)
        result = result.sort_values('response_score', ascending=False)
        result.to_csv(output / f'{case}.csv', index=False, encoding='utf-8-sig')
    (output / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'DN rankings saved with one snapshot pass: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', default='neutral')
    args = parser.parse_args()
    main(args.run, args.output, args.reference)

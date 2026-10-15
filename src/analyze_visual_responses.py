"""Compare every recorded visual condition with neutral and measure selectivity/timing.

This reads existing lossless snapshots only. It does not step the neural model or MuJoCo.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from full_cns_recording import state_at, state_blocks


def events_for(run):
    return [json.loads(line) for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]


def accumulate(run, events, count, destination):
    destination[:] = 0.
    for event in events:
        with np.load(run / event['snapshot']) as snapshot:
            for indices, values in state_blocks(snapshot):
                destination[indices] += values
    if events:
        destination[:] /= len(events)
    destination.flush()


def top_difference(target, reference, top, minimum):
    selected, scores = [], []
    for start in range(0, len(target), 1_000_000):
        stop = min(len(target), start+1_000_000)
        local_scores = np.abs(target[start:stop]-reference[start:stop])
        k = min(top, len(local_scores))
        if not k:
            continue
        local = np.argpartition(local_scores, len(local_scores)-k)[-k:]
        selected.append(local+start)
        scores.append(local_scores[local])
    selected, scores = np.concatenate(selected), np.concatenate(scores)
    order = np.argsort(scores)[::-1][:top]
    return selected[order][scores[order] > minimum]


def analyze(args):
    run = args.run
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    onset = args.after_seconds
    if onset is None:
        onset = float(metadata.get('parameters', {}).get('stimulus_onset', .5))
    circuit = Path(metadata['circuit_folder'])
    if json.loads((circuit / 'report.json').read_text(encoding='utf-8')) != metadata['circuit_report']:
        raise ValueError('기록 당시 full_cns 그래프와 현재 그래프가 다릅니다.')
    ids = np.load(circuit / 'body_ids.npy', mmap_mode='r')
    annotation = feather.read_feather(run / 'annotations.feather')
    all_events = events_for(run)
    usable = [event for event in all_events
              if event.get('input_sim_seconds', event['sim_seconds']) >= onset]
    reference_events = [event for event in usable if event['case'] == args.reference]
    if not reference_events:
        raise ValueError(f'{args.reference} 기준 기록이 없습니다.')
    cases = args.cases or sorted({event['case'] for event in usable if event['case'] != args.reference})
    if not cases:
        raise ValueError('비교할 자극 조건 기록이 없습니다.')
    output = run / 'visual_response_analysis'
    output.mkdir(exist_ok=True)
    reference = np.lib.format.open_memmap(output / '_reference_mean.npy', mode='w+',
                                          dtype=np.float64, shape=(len(ids),))
    target = np.lib.format.open_memmap(output / '_target_mean.npy', mode='w+',
                                       dtype=np.float64, shape=(len(ids),))
    accumulate(run, reference_events, len(ids), reference)
    union = []
    counts = {args.reference: len(reference_events)}
    for case in cases:
        condition_events = [event for event in usable if event['case'] == case]
        if not condition_events:
            raise ValueError(f'{case} 기록이 없습니다.')
        counts[case] = len(condition_events)
        accumulate(run, condition_events, len(ids), target)
        union.append(top_difference(target, reference, args.top_per_case, args.minimum_response))
    selected = np.unique(np.concatenate(union))
    if not len(selected):
        raise ValueError('minimum-response를 넘는 반응 후보가 없습니다.')

    # A second, small indexed pass gives every candidate a value for every case.
    values = {case: [] for case in [args.reference] + cases}
    times = {case: [] for case in [args.reference] + cases}
    ticks = {case: [] for case in [args.reference] + cases}
    for event in all_events:
        case = event['case']
        if case not in values:
            continue
        with np.load(run / event['snapshot']) as snapshot:
            values[case].append(state_at(snapshot, selected).astype(np.float64))
        times[case].append(float(event.get('input_sim_seconds', event['sim_seconds'])))
        ticks[case].append(int(event.get('neural_tick', event['frame'])))
    arrays = {case: np.asarray(rows) for case, rows in values.items()}
    post = {case: np.asarray(times[case]) >= onset for case in arrays}
    empty_post = [case for case, mask in post.items() if not np.any(mask)]
    if empty_post:
        raise ValueError(f'자극 시작 이후 프레임이 없는 조건: {empty_post}')
    reference_mean = arrays[args.reference][post[args.reference]].mean(axis=0)
    response = np.vstack([arrays[case][post[case]].mean(axis=0)-reference_mean for case in cases]).T
    absolute = np.abs(response)
    preferred_index = absolute.argmax(axis=1)
    best = absolute[np.arange(len(selected)), preferred_index]
    second = np.partition(absolute, -2, axis=1)[:, -2] if len(cases) > 1 else np.zeros(len(selected))
    selectivity = np.divide(best-second, best, out=np.zeros_like(best), where=best > 0)
    result = pd.DataFrame({'graph_index': selected, 'bodyId': ids[selected],
                           'preferred_condition': [cases[i] for i in preferred_index],
                           'preferred_signed_response': response[np.arange(len(selected)), preferred_index],
                           'preferred_absolute_response': best, 'selectivity_index': selectivity,
                           'reference_mean': reference_mean})
    for index, case in enumerate(cases):
        result[f'{case}_signed_response'] = response[:, index]
        result[f'{case}_absolute_response'] = absolute[:, index]
    result = result.merge(annotation.drop(columns=['graph_index']), on='bodyId', how='left', validate='one_to_one')
    result = result.sort_values(['preferred_absolute_response', 'selectivity_index'], ascending=False)
    result.to_csv(output / 'stimulus_selectivity.csv', index=False, encoding='utf-8-sig')

    timeline_indices = result.head(args.timeline_top).graph_index.to_numpy(dtype=np.int64)
    places = np.searchsorted(selected, timeline_indices)
    metric_rows, timeline_rows = [], []
    neutral_by_tick = {tick: row for tick, row in zip(ticks[args.reference], arrays[args.reference][:, places])}
    neutral_post_mean = arrays[args.reference][post[args.reference]][:, places].mean(axis=0)
    for case in cases:
        condition = arrays[case][:, places]
        for frame_index, (time_value, tick, row) in enumerate(zip(times[case], ticks[case], condition)):
            baseline = neutral_by_tick.get(tick, neutral_post_mean)
            delta = row-baseline
            if time_value >= onset:
                for candidate_index, graph_index in enumerate(timeline_indices):
                    timeline_rows.append({'case': case, 'frame_in_case': frame_index,
                                          'time_s': time_value, 'time_after_onset_s': time_value-onset,
                                          'graph_index': int(graph_index), 'bodyId': int(ids[graph_index]),
                                          'activity': row[candidate_index],
                                          'neutral_activity': baseline[candidate_index],
                                          'signed_response': delta[candidate_index]})
        case_times = np.asarray(times[case])
        keep = case_times >= onset
        case_delta = condition[keep]-np.asarray([
            neutral_by_tick.get(tick, neutral_post_mean) for tick in np.asarray(ticks[case])[keep]])
        for candidate_index, graph_index in enumerate(timeline_indices):
            series = case_delta[:, candidate_index]
            peak_index = int(np.argmax(np.abs(series)))
            peak = float(series[peak_index])
            threshold = max(args.minimum_response, .2*abs(peak))
            crossings = np.flatnonzero(np.abs(series) >= threshold)
            metric_rows.append({'case': case, 'graph_index': int(graph_index),
                                'bodyId': int(ids[graph_index]), 'peak_signed_response': peak,
                                'peak_time_after_onset_s': float(case_times[keep][peak_index]-onset),
                                'onset_time_after_stimulus_s': (float(case_times[keep][crossings[0]]-onset)
                                                                 if len(crossings) else np.nan),
                                'onset_threshold': threshold,
                                'onset_rule': 'first |delta| >= max(minimum_response, 20% of peak)'})
    pd.DataFrame(timeline_rows).to_csv(output / 'response_timeline.csv', index=False, encoding='utf-8-sig')
    metrics = pd.DataFrame(metric_rows).merge(annotation, on=['bodyId', 'graph_index'], how='left')
    metrics.to_csv(output / 'response_timing.csv', index=False, encoding='utf-8-sig')
    report = {'reference': args.reference, 'conditions': cases, 'stimulus_onset_s': onset,
              'recorded_input_boundary': metadata.get('inputs', {}).get('boundary', 'legacy or unspecified'),
              'frame_counts': counts, 'candidate_count': int(len(selected)),
              'timeline_candidate_count': int(len(timeline_indices)),
              'selectivity': '(largest absolute condition response - second largest) / largest',
              'response': 'condition post-onset mean minus neutral post-onset mean',
              'limitations': ['Observational model response, not biological causality.',
                              'Neutral matching uses neural tick when available and otherwise neutral mean.',
                              'Timing threshold is a stated heuristic, not a statistical significance test.']}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # Windows keeps mmap files locked until every mapping reference is released.
    del reference, target
    gc.collect()
    for temporary in ['_reference_mean.npy', '_target_mean.npy']:
        (output / temporary).unlink(missing_ok=True)
    print(f'다중 자극 반응 분석 저장: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', default='neutral')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--after-seconds', type=float)
    parser.add_argument('--top-per-case', type=int, default=2000)
    parser.add_argument('--timeline-top', type=int, default=200)
    parser.add_argument('--minimum-response', type=float, default=0.)
    args = parser.parse_args()
    if args.top_per_case < 1 or args.timeline_top < 1 or args.minimum_response < 0:
        parser.error('top values must be positive and minimum-response must be nonnegative')
    analyze(args)

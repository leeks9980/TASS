"""Trace activity-weighted, time-resolved paths from recorded DN responses to camera inputs.

The search follows modeled incoming contributions backward one recorded neural tick at a
time. Recurrence is preserved as time-indexed edges instead of being removed as a cycle.
"""
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
from neurotransmitter_model import resolved_neurotransmitters
from visual_input_boundary import is_photoreceptor_boundary


def load_events(run, case, episode):
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


def choose_targets(args, annotation, ids):
    if args.targets:
        return np.asarray(args.targets, dtype=np.int64)
    ranked = args.run / 'visual_response_analysis' / 'stimulus_selectivity.csv'
    if not ranked.exists():
        raise FileNotFoundError('먼저 analyze_visual_responses.py를 실행하거나 --targets bodyId를 지정하세요.')
    frame = pd.read_csv(ranked)
    dn = frame.superclass.fillna('').str.startswith('descending_neuron')
    if 'preferred_condition' in frame and args.case:
        preferred = dn & frame.preferred_condition.eq(args.case)
        if preferred.any():
            dn = preferred
    targets = frame.loc[dn].sort_values('preferred_absolute_response', ascending=False).bodyId.head(args.target_count)
    if not len(targets):
        raise ValueError('반응 순위에 하행신경 후보가 없습니다.')
    return targets.to_numpy(dtype=np.int64)


def trace(args):
    metadata = json.loads((args.run / 'metadata.json').read_text(encoding='utf-8'))
    boundary = metadata.get('inputs', {}).get('boundary')
    if not is_photoreceptor_boundary(boundary):
        raise ValueError('이 기록은 광수용체 입력 경계로 생성되지 않아 광수용체→DN 경로로 해석할 수 없습니다.')
    folder = Path(metadata['circuit_folder'])
    if json.loads((folder / 'report.json').read_text(encoding='utf-8')) != metadata['circuit_report']:
        raise ValueError('기록 당시 full_cns 그래프와 현재 그래프가 다릅니다.')
    ids = np.load(folder / 'body_ids.npy', mmap_mode='r')
    annotation = feather.read_feather(args.run / 'annotations.feather')
    annotation_lookup = annotation.set_index('bodyId')[['type', 'superclass']].to_dict('index')
    targets = choose_targets(args, annotation, ids)
    target_indices = np.searchsorted(ids, targets)
    valid = target_indices < len(ids)
    present = np.zeros(len(targets), dtype=bool)
    present[valid] = ids[target_indices[valid]] == targets[valid]
    valid = present
    if not np.all(valid):
        raise ValueError(f'전체 그래프에 없는 target bodyId: {targets[~valid].tolist()}')
    events, episode = load_events(args.run, args.case, args.episode)
    final_position = len(events)-1 if args.frame is None else next(
        (i for i, event in enumerate(events) if event['frame'] == args.frame), None)
    if final_position is None:
        raise ValueError('지정한 프레임이 선택한 case/episode에 없습니다.')
    if final_position < 1:
        raise ValueError('역추적에는 이전 신경 프레임이 필요합니다.')

    indptr = np.load(folder / 'indptr.npy', mmap_mode='r')
    indices = np.load(folder / 'indices.npy', mmap_mode='r')
    counts = np.load(folder / 'synapse_counts.npy', mmap_mode='r')
    totals = np.load(folder / 'incoming_sum.npy', mmap_mode='r')
    nt = feather.read_feather(folder / 'neurotransmitters.feather')
    names, _ = resolved_neurotransmitters(nt)
    nt_lookup = dict(zip(nt.body.astype(np.int64), names))
    effects = metadata['nt']['modeled_effects']
    gain = float(metadata['parameters']['gain'])
    raw_weights = metadata['parameters'].get('weight_mode', 'incoming') == 'raw'

    # Direct input IDs are the actual source boundary used in this recorded episode.
    source_indices = set()
    for event in events[:final_position+1]:
        with np.load(args.run / event['snapshot']) as snapshot:
            source_indices.update(int(index) for index in snapshot['input_indices'])

    frontier = [{'node': int(index), 'frame_position': final_position,
                 'path_score': 1., 'target_body': int(body)}
                for index, body in zip(target_indices, targets)]
    edge_rows, reached_rows = [], []
    visited = set()
    for depth in range(args.max_depth):
        candidates = []
        by_frame = {}
        for item in frontier:
            post = item['node']
            frame_position = item['frame_position']
            key = (post, frame_position, item['target_body'])
            if key in visited or frame_position < 1:
                continue
            visited.add(key)
            by_frame.setdefault(frame_position, []).append(item)

        # Every branch at the same depth normally reads the same previous frame.
        # Open/decompress that whole-CNS snapshot once, not once per frontier node.
        for frame_position, frame_items in by_frame.items():
            previous_event = events[frame_position-1]
            with np.load(args.run / previous_event['snapshot']) as snapshot:
                for item in frame_items:
                    post = item['node']
                    start, stop = int(indptr[post]), int(indptr[post+1])
                    pres = np.asarray(indices[start:stop], dtype=np.int64)
                    synapses = np.asarray(counts[start:stop], dtype=np.float64)
                    if not len(pres):
                        continue
                    activity = state_at(snapshot, pres).astype(np.float64)
                    signs = np.asarray([effects.get(nt_lookup.get(int(ids[pre]), ''), 1.) for pre in pres])
                    weights = synapses if raw_weights else np.divide(
                        synapses, float(totals[post]), out=np.zeros_like(synapses),
                        where=float(totals[post]) > 0)
                    contribution = gain*weights*signs*activity
                    eligible = np.flatnonzero(np.abs(contribution) >= args.minimum_contribution)
                    if not len(eligible):
                        continue
                    keep = eligible[np.argsort(np.abs(contribution[eligible]))[::-1][:args.inputs_per_node]]
                    for position in keep:
                        pre = int(pres[position])
                        edge = {'depth': depth+1, 'pre_frame': int(previous_event['frame']),
                                'post_frame': int(events[frame_position]['frame']),
                                'body_pre': int(ids[pre]), 'body_post': int(ids[post]),
                                'synapse_count': float(synapses[position]),
                                'previous_activity': float(activity[position]),
                                'assumed_sign': float(signs[position]),
                                'modeled_signal_contribution': float(contribution[position]),
                                'absolute_contribution': float(abs(contribution[position])),
                                'path_score': float(item['path_score']*abs(contribution[position])),
                                'target_body': item['target_body']}
                        pre_note = annotation_lookup.get(edge['body_pre'], {})
                        post_note = annotation_lookup.get(edge['body_post'], {})
                        edge.update(pre_type=pre_note.get('type'), pre_superclass=pre_note.get('superclass'),
                                    post_type=post_note.get('type'), post_superclass=post_note.get('superclass'))
                        edge_rows.append(edge)
                        if pre in source_indices:
                            reached_rows.append({'source_body': int(ids[pre]), 'target_body': item['target_body'],
                                                 'depth': depth+1, 'frame': int(previous_event['frame']),
                                                 'path_score': edge['path_score']})
                        else:
                            candidates.append({'node': pre, 'frame_position': frame_position-1,
                                               'path_score': edge['path_score'],
                                               'target_body': item['target_body']})
        candidates.sort(key=lambda item: item['path_score'], reverse=True)
        frontier = candidates[:args.beam_width]
        if not frontier:
            break

    output = args.run / 'visual_path_analysis'
    output.mkdir(exist_ok=True)
    edges = pd.DataFrame(edge_rows)
    reached = pd.DataFrame(reached_rows)
    edges.to_csv(output / 'activity_weighted_edges.csv', index=False, encoding='utf-8-sig')
    reached.to_csv(output / 'reached_photoreceptors.csv', index=False, encoding='utf-8-sig')
    if len(edges):
        # Preserve every discovered time-indexed recurrent/feedback edge in a compact graph export.
        nodes = np.union1d(edges.body_pre.to_numpy(dtype=np.int64), edges.body_post.to_numpy(dtype=np.int64))
        node_frame = pd.DataFrame({'bodyId': nodes}).merge(annotation.drop(columns=['graph_index']),
                                                          on='bodyId', how='left')
        node_frame.to_feather(output / 'path_neurons.feather')
    report = {'case': args.case, 'episode': episode, 'target_body_ids': targets.tolist(),
              'recorded_input_boundary': boundary,
              'start_frame': int(events[final_position]['frame']), 'max_depth': args.max_depth,
              'inputs_per_node': args.inputs_per_node, 'beam_width': args.beam_width,
              'minimum_contribution': args.minimum_contribution,
              'edge_count': int(len(edges)), 'reached_input_count': int(len(reached)),
              'recurrence_policy': 'time-indexed repeated body IDs are retained; only identical body/frame/target states are deduplicated',
              'limitations': ['Beam search can omit weak or broadly distributed paths.',
                              'Modeled contribution depends on point-neuron and transmitter assumptions.',
                              'A reached path is model signal attribution, not biological causation.']}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'시각 입력까지 역추적한 경로 저장: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--episode', type=int)
    parser.add_argument('--frame', type=int)
    parser.add_argument('--targets', type=int, nargs='+')
    parser.add_argument('--target-count', type=int, default=20)
    parser.add_argument('--max-depth', type=int, default=12)
    parser.add_argument('--inputs-per-node', type=int, default=8)
    parser.add_argument('--beam-width', type=int, default=500)
    parser.add_argument('--minimum-contribution', type=float, default=0.)
    args = parser.parse_args()
    if min(args.target_count, args.max_depth, args.inputs_per_node, args.beam_width) < 1 or args.minimum_contribution < 0:
        parser.error('counts/depth/width must be positive and minimum-contribution nonnegative')
    trace(args)

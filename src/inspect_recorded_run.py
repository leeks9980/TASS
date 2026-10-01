"""Analyze existing recordings only; no MuJoCo, neural stepping or learning."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy.sparse import csr_matrix


@contextmanager
def recorded_state(path):
    """Map uncompressed dense arrays; read old sparse data without rewriting it."""
    archive = np.load(path)
    mapping = None
    try:
        if 'state' in archive.files:
            with zipfile.ZipFile(path) as z:
                info = z.getinfo('state.npy')
                if info.compress_type == zipfile.ZIP_STORED:
                    with path.open('rb') as f:
                        f.seek(info.header_offset)
                        header = struct.unpack('<IHHHHHIIIHH', f.read(30))
                        start = info.header_offset+30+header[-2]+header[-1]
                    with z.open('state.npy') as f:
                        version = np.lib.format.read_magic(f)
                        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                        shape, fortran, dtype = read_header(f)
                        offset = start+f.tell()
                    mapping = np.memmap(path, mode='r', dtype=dtype, offset=offset, shape=shape, order='F' if fortran else 'C')
                    def at(indices):
                        return np.asarray(mapping[indices]).copy()
                else:
                    dense = archive['state']
                    def at(indices):
                        return dense[indices]
            yield archive, at
        else:
            ids, values = archive['state_indices'], archive['state_values']
            def at(indices):
                p = np.searchsorted(ids, indices)
                result = np.zeros(len(indices), dtype=values.dtype)
                valid = np.flatnonzero(p < len(ids))
                valid = valid[ids[p[valid]] == indices[valid]]
                result[valid] = values[p[valid]]
                return result
            yield archive, at
    finally:
        archive.close()
        if mapping is not None:
            mapping._mmap.close()


def stats(values):
    return {'n': len(values), 'active': int(np.count_nonzero(values)),
            'mean': float(values.mean()) if len(values) else 0.,
            'median': float(np.median(values)) if len(values) else 0.,
            'p95': float(np.quantile(values, .95)) if len(values) else 0.,
            'max': float(values.max()) if len(values) else 0.}


def events(folder):
    result = []
    for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # An actively written final partial line is not a committed frame.
            continue
    return result


def inspect(run):
    meta = json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('output_mode') == 'public_fly_body':
        from analyze_fly_recording import analyze
        return analyze(run)
    source = Path(meta['circuit_folder'])
    if json.loads((source/'report.json').read_text(encoding='utf-8')) != meta['circuit_report']:
        raise ValueError('Recorded source graph differs from current graph')
    annotation = feather.read_feather(run/'annotations.feather')
    indices = annotation.graph_index.to_numpy(dtype=np.int64)
    sides = annotation.somaSide.fillna(annotation.rootSide).fillna('')
    motor = annotation.superclass.eq('vnc_motor') & annotation.subclass.isin(['fl', 'ml', 'hl'])
    groups = {t: annotation.type.eq(t).to_numpy() for t in ['L1', 'L2', 'L3']}
    groups.update({s: annotation.superclass.eq(s).to_numpy() for s in
                   ['ol_intrinsic', 'visual_projection', 'cb_intrinsic', 'descending_neuron', 'vnc_intrinsic']})
    groups['leg_motor'] = motor.to_numpy()
    groups['leg_motor_L'] = (motor & sides.eq('L')).to_numpy()
    groups['leg_motor_R'] = (motor & sides.eq('R')).to_numpy()
    log = events(run)
    if len(log) < 2:
        raise ValueError('At least two committed frames are needed')
    out = run/'diagnosis'
    out.mkdir(exist_ok=True)
    timeline, group_rows = [], []
    first_activity = final_activity = None
    # Analyze every committed frame; a live later frame does not change this snapshot.
    for event in log:
        with recorded_state(run/event['snapshot']) as (sample, at):
            activity = at(indices)
            if first_activity is None:
                first_activity = activity.copy()
            final_activity = activity
            image = sample['rgb'].astype(np.float32)/255
            drive = sample['input_values']
            row = dict(event)
            row['rgb_mean'] = float(image.mean())
            row['drive_mean'] = float(drive.mean())
            row['drive_max'] = float(drive.max())
            row['active_annotated'] = int(np.count_nonzero(activity))
            row['mean_annotated'] = float(activity.mean())
            row['max_annotated'] = float(activity.max())
            row['left_motor_mean_recomputed'] = float(activity[groups['leg_motor_L']].mean())
            row['right_motor_mean_recomputed'] = float(activity[groups['leg_motor_R']].mean())
            row['all_states_active_count'] = event['nonzero_states']
            timeline.append(row)
            for group, mask in groups.items():
                group_rows.append(dict(frame=event['frame'], sim_seconds=event['sim_seconds'], group=group, **stats(activity[mask])))
        print(f'기존 기록 분석 {run.name}: {event["frame"]}', flush=True)
    table = pd.DataFrame(timeline)
    table.to_csv(out/'timeline.csv', index=False, encoding='utf-8-sig')
    group_table = pd.DataFrame(group_rows)
    group_table.to_csv(out/'population_activity.csv', index=False, encoding='utf-8-sig')
    columns = ['bodyId', 'graph_index', 'type', 'superclass', 'subclass', 'somaSide', 'rootSide',
               'consensus_nt', 'predicted_nt_confidence']
    neurons = annotation[columns].copy()
    neurons['first_activity'] = first_activity
    neurons['final_activity'] = final_activity
    neurons['change_from_first'] = final_activity-first_activity
    neurons.sort_values('final_activity', ascending=False).to_csv(out/'annotated_neurons.csv', index=False, encoding='utf-8-sig')
    neurons.loc[motor].sort_values('final_activity', ascending=False).to_csv(out/'motor_neurons.csv', index=False, encoding='utf-8-sig')

    last = log[-1]
    previous = next((e for e in reversed(log[:-1]) if e['episode'] == last['episode']), None)
    motor_positions = np.flatnonzero(motor.to_numpy())
    motor_graph = indices[motor_positions]
    n = meta['circuit_report']['graph_nodes_including_segments']
    raw = csr_matrix((np.load(source/'synapse_counts.npy', mmap_mode='r'),
                      np.load(source/'indices.npy', mmap_mode='r'),
                      np.load(source/'indptr.npy', mmap_mode='r')), shape=(n,n), copy=False)
    incoming = raw[motor_graph, :].tocsr()
    pres = np.unique(incoming.indices)
    if previous is None:
        pre_activity = np.zeros(len(pres), dtype=np.float32)
        old_motor = np.zeros(len(motor_graph), dtype=np.float32)
    else:
        with recorded_state(run/previous['snapshot']) as (_, at):
            pre_activity = at(pres)
            old_motor = at(motor_graph)
    graph_ids = np.load(source/'body_ids.npy', mmap_mode='r')
    nt = feather.read_feather(source/'neurotransmitters.feather')
    names = nt.ground_truth.fillna(nt.consensus_nt).fillna(nt.celltype_predicted_nt).fillna(nt.predicted_nt).fillna('').str.lower()
    nt_lookup = dict(zip(nt.body, names))
    effects = meta['nt']['modeled_effects']
    labels = [nt_lookup.get(int(x), '') for x in graph_ids[pres]]
    signs = np.array([effects.get(x, 1.) for x in labels])
    annotation_lookup = dict(zip(indices, annotation.superclass.fillna('unclassified')))
    quality = np.array([int(x) in annotation_lookup for x in pres])
    ledger = []
    for i, graph in enumerate(motor_graph):
        a, b = incoming.indptr[i:i+2]
        pre = incoming.indices[a:b]
        counts = incoming.data[a:b]
        selected = np.searchsorted(pres, pre)
        total = float(counts.sum(dtype=np.float64))
        weights = counts/total if meta['parameters'].get('weight_mode','incoming') == 'incoming' and total else counts
        contributions = weights*signs[selected]*pre_activity[selected]*meta['parameters']['gain']
        pos = float(contributions[contributions > 0].sum())
        neg = float(-contributions[contributions < 0].sum())
        net = pos-neg
        expected = meta['parameters']['leak']*old_motor[i]+(1-meta['parameters']['leak'])*max(0., np.tanh(net))
        actual = float(final_activity[motor_positions[i]])
        ledger.append({'bodyId': int(graph_ids[graph]), 'type': annotation.iloc[motor_positions[i]].type,
            'side': sides.iloc[motor_positions[i]], 'source_neurons_or_segments': len(pre),
            'total_incoming_synapses': total, 'unannotated_synapse_fraction': float(counts[~quality[selected]].sum()/total) if total else 0.,
            'positive_drive': pos, 'negative_drive': neg, 'net_drive': net,
            'inhibition_fraction_of_positive': neg/pos if pos else None,
            'prior_activity': float(old_motor[i]), 'final_activity': actual,
            'reconstructed_final': float(expected), 'reconstruction_error': abs(expected-actual)})
    ledger = pd.DataFrame(ledger)
    ledger.to_csv(out/'motor_input_balance.csv', index=False, encoding='utf-8-sig')
    finals = {group: stats(final_activity[mask]) for group, mask in groups.items()}
    top_motor = neurons.loc[motor].sort_values('final_activity', ascending=False).head(12).to_dict('records')
    summary = {
        'run': run.name, 'frames_analyzed': len(log), 'cases': sorted(set(e['case'] for e in log)),
        'output_mode': meta.get('output_mode', 'wheel_mean_proxy'),
        'episodes': sorted(set(e['episode'] for e in log)), 'first_sim_seconds': log[0]['sim_seconds'],
        'last_sim_seconds': last['sim_seconds'], 'final_path_m': last['path_m'],
        'final_motor_commands': last['motor_commands'],
        'final_motor_command_unit': last.get('motor_command_unit', 'rad/s'),
        'final_wheel_velocity': last['qvel'][-2:] if 'muscles' not in last else None,
        'final_body_x_velocity': last['qvel'][0] if 'muscles' not in last else None,
        'final_muscles': last.get('muscles'),
        'final_neutral_difference': last.get('neutral_difference'),
        'final_group_stats': finals,
        'median_compute_ms': float(table.compute_ms_excluding_record.median()),
        'median_save_ms': float(table.save_ms.median()) if 'save_ms' in table else None,
        'mn_reconstruction_max_error': float(ledger.reconstruction_error.max()),
        'mn_positive_drive_mean': float(ledger.positive_drive.mean()),
        'mn_negative_drive_mean': float(ledger.negative_drive.mean()),
        'mn_net_drive_mean': float(ledger.net_drive.mean()),
        'mn_unannotated_synapse_fraction_mean': float(ledger.unannotated_synapse_fraction.mean()),
        'mn_net_drive_nonpositive': int((ledger.net_drive <= 0).sum()),
        'top_motor_neurons': top_motor,
        'limits': ['Only recorded stimuli can be analyzed. No claim of obstacle recognition or absent motivation.',
                   'Population means are descriptive, not a causal serial pathway.',
                   'Neural activity units and motor decoder have not been biologically calibrated.',
                   'The muscle diagnostic rig has a fixed body; path length is not a locomotion result.'],
    }
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: None if pd.isna(x) else str(x)), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = inspect(args.run)
    print(json.dumps({k:v for k,v in result.items() if k not in ['top_motor_neurons']}, ensure_ascii=False, indent=2), flush=True)

"""Reconstruct incoming modeled signal contributions from a recorded update."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow as pa
import pyarrow.ipc as ipc
from full_cns_recording import state_at


def trace(args):
    meta = json.loads((args.run / 'metadata.json').read_text(encoding='utf-8'))
    folder = Path(meta['circuit_folder'])
    if json.loads((folder / 'report.json').read_text(encoding='utf-8')) != meta['circuit_report']:
        raise ValueError('실행 당시의 full_cns 데이터와 현재 데이터가 다릅니다.')
    ids = np.load(folder / 'body_ids.npy', mmap_mode='r')
    row = int(np.searchsorted(ids, args.body))
    if row >= len(ids) or int(ids[row]) != args.body:
        raise ValueError('해당 body ID가 전체 그래프에 없습니다.')
    events = [json.loads(s) for s in (args.run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    matching = [e for e in events if e['frame'] == args.frame]
    if not matching:
        raise ValueError('해당 기록 프레임이 없습니다.')
    event = matching[0]
    indptr = np.load(folder / 'indptr.npy', mmap_mode='r')
    cols = np.load(folder / 'indices.npy', mmap_mode='r')[indptr[row]:indptr[row+1]]
    counts = np.load(folder / 'synapse_counts.npy', mmap_mode='r')[indptr[row]:indptr[row+1]]
    old = np.zeros(len(cols), dtype=np.float32)
    old_post = 0.
    previous = next((e for e in events if e['frame'] == args.frame-1 and e['episode'] == event['episode']), None)
    if previous is not None:
        with np.load(args.run / previous['snapshot']) as snap:
            values = state_at(snap, np.concatenate([cols, [row]]))
            old = values[:-1]
            old_post = float(values[-1])
    nt = feather.read_feather(folder / 'neurotransmitters.feather')
    names = nt.ground_truth.fillna(nt.consensus_nt).fillna(nt.celltype_predicted_nt).fillna(nt.predicted_nt).fillna('').str.lower()
    lookup = dict(zip(nt.body, names))
    labels = [lookup.get(int(i), '') for i in ids[cols]]
    effects = meta['nt']['modeled_effects']
    signs = np.array([effects.get(n, 1.) for n in labels])
    total = float(counts.sum(dtype=np.float64))
    if meta['parameters'].get('weight_mode', 'incoming') == 'raw':
        normalized = counts
    else:
        normalized = counts / total if total else np.zeros_like(counts)
    contribution = normalized * signs * old * meta['parameters']['gain']
    frame = pd.DataFrame({'body_pre': ids[cols], 'body_post': args.body, 'synapse_count': counts,
                          'nt_label': labels, 'assumed_sign': signs, 'previous_activity': old,
                          'modeled_signal_contribution': contribution})
    frame = frame.merge(nt.rename(columns={'body': 'body_pre'}), on='body_pre', how='left', validate='many_to_one')
    annotation = feather.read_feather(args.run / 'annotations.feather')[['bodyId', 'type', 'superclass']]
    frame = frame.merge(annotation.rename(columns={'bodyId': 'body_pre', 'type': 'pre_type', 'superclass': 'pre_superclass'}), on='body_pre', how='left')
    frame['absolute_contribution'] = frame.modeled_signal_contribution.abs()
    frame = frame.sort_values('absolute_contribution', ascending=False)
    with np.load(args.run / event['snapshot']) as snapshot:
        drive = float(snapshot['input_values'][snapshot['input_indices'] == row].sum())
        actual = float(state_at(snapshot, np.array([row]))[0])
    leak = meta['parameters']['leak']
    expected = leak*old_post+(1-leak)*max(0., np.tanh(float(contribution.sum())+drive))
    out = args.run / 'analysis'
    out.mkdir(exist_ok=True)
    stem = f'trace_frame_{args.frame}_body_{args.body}'
    frame.to_csv(out / (stem+'.csv'), index=False, encoding='utf-8-sig')
    synapse_rows = None
    if args.synapses:
        print('개별 시냅스 원본 표를 순차 조회합니다. 대형 파일이므로 시간이 걸립니다.', flush=True)
        partners = ROOT / 'data' / 'male-cns-v1.0' / 'syn-partners-male-cns-v1.0-minconf-0.5.feather'
        if not partners.exists():
            raise FileNotFoundError('시냅스 파트너 표 다운로드가 필요합니다. download_all_cns.cmd를 실행하세요.')
        pieces = []
        with pa.memory_map(str(partners), 'r') as source:
            reader = ipc.open_file(source)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                mask = batch.column('body_post').to_numpy() == args.body
                if mask.any():
                    pieces.append(batch.filter(pa.array(mask)))
        if pieces:
            synapses = pa.Table.from_batches(pieces).to_pandas()
            # Preserve the actual per-presynapse uncertainty, rather than just
            # labeling every edge with its body's aggregate transmitter class.
            tbar_path = ROOT / 'data' / 'male-cns-v1.0' / 'tbar-neurotransmitters-male-cns-v1.0.feather'
            if tbar_path.exists():
                prediction_pieces = []
                pres = np.unique(synapses.body_pre.to_numpy())
                with pa.memory_map(str(tbar_path), 'r') as source:
                    reader = ipc.open_file(source)
                    for i in range(reader.num_record_batches):
                        batch = reader.get_batch(i)
                        mask = np.isin(batch.column('body').to_numpy(), pres)
                        if mask.any():
                            prediction_pieces.append(batch.filter(pa.array(mask)))
                if prediction_pieces:
                    predictions = pa.Table.from_batches(prediction_pieces).to_pandas().rename(columns={
                        'x': 'x_pre', 'y': 'y_pre', 'z': 'z_pre', 'body': 'nt_body',
                        'conf': 'nt_conf', 'sv': 'nt_sv', 'primary': 'nt_primary',
                        'major': 'nt_major', 'point_id': 'nt_point_id', 'split': 'nt_split'})
                    synapses = synapses.merge(predictions, on=['x_pre', 'y_pre', 'z_pre'], how='left', validate='many_to_one')
            synapses.to_csv(out / (stem+'_synapses.csv'), index=False, encoding='utf-8-sig')
            synapse_rows = len(synapses)
        else:
            synapse_rows = 0
        stats_path = ROOT / 'data' / 'male-cns-v1.0' / 'body-stats-male-cns-v1.0-minconf-0.5.feather'
        if stats_path.exists():
            stats_pieces = []
            bodies = np.union1d(ids[cols], [args.body])
            with pa.memory_map(str(stats_path), 'r') as source:
                reader = ipc.open_file(source)
                for i in range(reader.num_record_batches):
                    batch = reader.get_batch(i)
                    mask = np.isin(batch.column('body').to_numpy(), bodies)
                    if mask.any():
                        stats_pieces.append(batch.filter(pa.array(mask)))
            if stats_pieces:
                pa.Table.from_batches(stats_pieces).to_pandas().to_csv(out / (stem+'_body_stats.csv'), index=False, encoding='utf-8-sig')
        points_path = ROOT / 'data' / 'male-cns-v1.0' / 'syn-points-male-cns-v1.0-minconf-0.5.feather'
        if points_path.exists():
            point_pieces = []
            with pa.memory_map(str(points_path), 'r') as source:
                reader = ipc.open_file(source)
                for i in range(reader.num_record_batches):
                    batch = reader.get_batch(i)
                    mask = batch.column('body').to_numpy() == args.body
                    if mask.any():
                        point_pieces.append(batch.filter(pa.array(mask)))
            if point_pieces:
                pa.Table.from_batches(point_pieces).to_pandas().to_csv(out / (stem+'_points.csv'), index=False, encoding='utf-8-sig')
    (out / (stem+'.json')).write_text(json.dumps({'body': args.body, 'frame': args.frame,
        'previous_state': old_post, 'camera_drive': drive, 'reconstructed_state': float(expected),
        'recorded_state': actual, 'original_partner_rows': synapse_rows,
        'note': 'Model signal attribution, not experimental biological causation.'}, indent=2), encoding='utf-8')
    print(f'입력 연결별 신호 기록: {out / (stem+".csv")}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--body', type=int, required=True)
    parser.add_argument('--synapses', action='store_true', help='Also scan original partner table for synapse locations, confidence and neuropil.')
    trace(parser.parse_args())

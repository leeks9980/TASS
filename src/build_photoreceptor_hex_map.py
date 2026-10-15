"""Infer each MaleCNS photoreceptor's optic-lobe hex column from published edges.

Photoreceptors do not carry assignedOlHex1/2 in MaleCNS v1.0. Their real
outgoing connections to columnar neurons do, so the dominant postsynaptic hex
column supplies a data-derived mapping. Ambiguity is reported rather than
hidden, and receptors with no supported column are left unmapped.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

DATA = ROOT / 'data' / 'male-cns-v1.0'
DEFAULT_CIRCUIT = ROOT / 'full_cns'


def weighted_edges(path, sensory_ids, coordinate_ids):
    chunks = []
    with pa.memory_map(str(path), 'r') as source:
        reader = ipc.open_file(source)
        for number in range(reader.num_record_batches):
            batch = reader.get_batch(number)
            pre = batch.column('body_pre').to_numpy()
            post = batch.column('body_post').to_numpy()
            keep = np.isin(pre, sensory_ids) & np.isin(post, coordinate_ids)
            if np.any(keep):
                chunks.append(pd.DataFrame({
                    'bodyId': pre[keep].astype(np.int64, copy=False),
                    'body_post': post[keep].astype(np.int64, copy=False),
                    'weight': batch.column('weight').to_numpy()[keep],
                }))
            if number == 0 or (number + 1) % 10 == 0 or number + 1 == reader.num_record_batches:
                print(f'광수용체 column 연결 검색 {number + 1}/{reader.num_record_batches}', flush=True)
    if not chunks:
        raise ValueError('광수용체에서 좌표 보유 시엽 뉴런으로 가는 연결이 없습니다.')
    return pd.concat(chunks, ignore_index=True)


def build(data, circuit):
    annotations = feather.read_feather(
        data / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
    sensory = annotations[annotations.superclass.fillna('').eq('ol_sensory')].copy()
    coordinate = annotations[
        annotations.assignedOlHex1.notna() & annotations.assignedOlHex2.notna()].copy()
    if sensory.empty or coordinate.empty:
        raise ValueError('광수용체 또는 hex 좌표 보유 뉴런 주석이 없습니다.')
    edges = weighted_edges(
        data / 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
        sensory.bodyId.to_numpy(dtype=np.int64),
        coordinate.bodyId.to_numpy(dtype=np.int64))
    columns = coordinate[['bodyId', 'assignedOlHex1', 'assignedOlHex2', 'rootSide',
                          'somaSide', 'type']].rename(columns={
                              'bodyId': 'body_post', 'type': 'post_type'})
    joined = edges.merge(columns, on='body_post', how='inner', validate='many_to_one')
    joined['side'] = joined.rootSide.fillna(joined.somaSide)
    joined = joined[joined.side.isin(['L', 'R'])].copy()
    evidence = (joined.groupby(['bodyId', 'side', 'assignedOlHex1', 'assignedOlHex2'], as_index=False)
                .agg(column_weight=('weight', 'sum'),
                     supporting_targets=('body_post', 'nunique')))
    totals = evidence.groupby('bodyId').agg(
        mapped_output_weight=('column_weight', 'sum'),
        candidate_columns=('column_weight', 'size')).reset_index()
    dominant = (evidence.sort_values(['bodyId', 'column_weight'], ascending=[True, False])
                .drop_duplicates('bodyId'))
    mapping = dominant.merge(totals, on='bodyId', how='left', validate='one_to_one')
    mapping['dominant_weight_fraction'] = mapping.column_weight / mapping.mapped_output_weight
    mapping = mapping.merge(
        sensory[['bodyId', 'type', 'rootSide', 'somaSide']], on='bodyId', how='left',
        validate='one_to_one', suffixes=('_column', '_photoreceptor'))
    ids = np.load(circuit / 'body_ids.npy', mmap_mode='r')
    mapping['graph_index'] = np.searchsorted(ids, mapping.bodyId.to_numpy(dtype=np.int64))
    mapping = mapping.sort_values('bodyId').reset_index(drop=True)
    feather.write_feather(mapping, circuit / 'photoreceptor_hex_map.feather')
    missing = sensory[~sensory.bodyId.isin(mapping.bodyId)].copy()
    missing.to_csv(circuit / 'unmapped_photoreceptors.csv', index=False, encoding='utf-8-sig')
    report = {
        'schema': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'male-cns:v1.0',
        'method': 'dominant weighted outgoing connection to neurons with assignedOlHex1/2',
        'photoreceptors': int(len(sensory)), 'mapped': int(len(mapping)),
        'unmapped': int(len(missing)), 'mapped_fraction': float(len(mapping) / len(sensory)),
        'single_candidate_column': int(mapping.candidate_columns.eq(1).sum()),
        'dominant_fraction_at_least_0_8': int(mapping.dominant_weight_fraction.ge(.8).sum()),
        'side_counts': {str(k): int(v) for k, v in mapping.side.value_counts().items()},
        'limitations': [
            'Hex columns preserve retinotopic topology but are not calibrated camera angles.',
            'Dominant downstream column is an inference from published connectivity.',
            'Unmapped receptors receive no direct image input; no coordinates are invented.',
        ],
    }
    (circuit / 'photoreceptor_hex_map_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--circuit', type=Path, default=DEFAULT_CIRCUIT)
    args = parser.parse_args()
    build(args.data, args.circuit)

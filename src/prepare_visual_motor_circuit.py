"""Build an executable photoreceptor-to-leg-motor corridor from MaleCNS.

The full connectome is used only to discover structural directed paths.  The
saved runtime graph contains only nodes and edges that lie on a directed walk
from an annotated ol_sensory neuron to an annotated leg motor neuron within the
requested hop bound.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))

import numpy as np
import pyarrow.feather as feather
from scipy.sparse import coo_matrix

from extract_visual_leg_circuit import DATA, ANNOTATIONS, NT, arrays, batches, build_graph, distances


DEFAULT = ROOT / 'visual_motor_circuit'
LEG_SUBCLASSES = ['fl', 'ml', 'hl']


def prepare(output, minimum=1, max_hops=12):
    manifest_path = DATA / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Run download_connectome.py successfully first')
    output.mkdir(parents=True, exist_ok=True)
    work = output / 'work'
    work.mkdir(exist_ok=True)

    annotations = feather.read_feather(DATA / ANNOTATIONS)
    nt = feather.read_feather(DATA / NT)
    source_bodies = annotations.loc[
        annotations.superclass.fillna('').eq('ol_sensory'), 'bodyId'
    ].to_numpy(dtype=np.int64)
    target_bodies = annotations.loc[
        annotations.superclass.fillna('').eq('vnc_motor') &
        annotations.subclass.fillna('').isin(LEG_SUBCLASSES), 'bodyId'
    ].to_numpy(dtype=np.int64)
    if len(source_bodies) == 0 or len(target_bodies) == 0:
        raise ValueError('Photoreceptor or leg motor annotations are missing')

    print('전체 연결표는 경로 발견에만 사용합니다.', flush=True)
    ids, graph, raw_count, eligible_count = build_graph(minimum, work)
    sources = source_bodies[np.isin(source_bodies, ids)]
    targets = target_bodies[np.isin(target_bodies, ids)]
    if len(sources) == 0 or len(targets) == 0:
        raise ValueError('Photoreceptor or leg motor IDs are absent from the weighted graph')

    print(f'광수용체 {len(sources):,}개에서 순방향 거리 계산', flush=True)
    forward = distances(graph, np.searchsorted(ids, sources))
    print(f'다리 운동뉴런 {len(targets):,}개에서 역방향 거리 계산', flush=True)
    backward = distances(graph.transpose().tocsr(), np.searchsorted(ids, targets))
    total_hops = forward.astype(np.int64) + backward.astype(np.int64)
    keep_nodes = (forward >= 0) & (backward >= 0) & (total_hops <= max_hops)
    kept_ids = ids[keep_nodes]
    if len(kept_ids) == 0:
        raise ValueError('No visual-to-motor corridor was found for the selected hop bound')

    edge_count = 0
    for batch in batches():
        pre, post, weight = arrays(batch)
        valid = weight >= minimum
        pre, post = pre[valid], post[valid]
        source_index = np.searchsorted(ids, pre)
        target_index = np.searchsorted(ids, post)
        keep = (keep_nodes[source_index] & keep_nodes[target_index] &
                (forward[source_index].astype(np.int64) + 1 + backward[target_index] <= max_hops))
        edge_count += int(keep.sum())
    if edge_count == 0:
        raise ValueError('The selected corridor contains no edges')

    index_type = np.int32 if len(kept_ids) < np.iinfo(np.int32).max else np.int64
    rows = np.lib.format.open_memmap(work / 'rows.npy', mode='w+', dtype=index_type, shape=(edge_count,))
    cols = np.lib.format.open_memmap(work / 'cols.npy', mode='w+', dtype=index_type, shape=(edge_count,))
    values = np.lib.format.open_memmap(work / 'weights.npy', mode='w+', dtype=np.float32, shape=(edge_count,))
    offset = 0
    for batch in batches():
        pre, post, weight = arrays(batch)
        valid = weight >= minimum
        pre, post, weight = pre[valid], post[valid], weight[valid]
        source_index = np.searchsorted(ids, pre)
        target_index = np.searchsorted(ids, post)
        keep = (keep_nodes[source_index] & keep_nodes[target_index] &
                (forward[source_index].astype(np.int64) + 1 + backward[target_index] <= max_hops))
        count = int(keep.sum())
        if count:
            end = offset + count
            rows[offset:end] = np.searchsorted(kept_ids, post[keep])
            cols[offset:end] = np.searchsorted(kept_ids, pre[keep])
            values[offset:end] = weight[keep]
            offset = end
    rows.flush(); cols.flush(); values.flush()
    matrix = coo_matrix((values, (rows, cols)), shape=(len(kept_ids), len(kept_ids))).tocsr()

    np.save(output / 'body_ids.npy', kept_ids)
    np.save(output / 'indptr.npy', matrix.indptr)
    np.save(output / 'indices.npy', matrix.indices)
    np.save(output / 'synapse_counts.npy', matrix.data)
    np.save(output / 'incoming_sum.npy', np.asarray(matrix.sum(axis=1)).ravel())

    selected_annotations = annotations[annotations.bodyId.isin(kept_ids)].copy()
    selected_annotations = selected_annotations.merge(
        nt.rename(columns={'body': 'bodyId'}), on='bodyId', how='left', validate='one_to_one')
    selected_annotations['graph_index'] = np.searchsorted(
        kept_ids, selected_annotations.bodyId.to_numpy(dtype=np.int64))
    feather.write_feather(selected_annotations, output / 'annotations.feather')
    feather.write_feather(nt[nt.body.isin(kept_ids)].copy(), output / 'neurotransmitters.feather')

    kept_sources = int(np.isin(sources, kept_ids).sum())
    kept_targets = int(np.isin(targets, kept_ids).sum())
    report = {
        'dataset': 'male-cns:v1.0',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'photoreceptor_to_leg_motor_structural_corridor',
        'graph_nodes_including_segments': int(len(kept_ids)),
        'annotation_rows': int(len(selected_annotations)),
        'unannotated_graph_ids': int(len(kept_ids) - len(selected_annotations)),
        'original_connection_rows': int(raw_count),
        'eligible_connection_rows': int(eligible_count),
        'csr_edges': int(matrix.nnz),
        'filters': {'minimum_synapse_weight': int(minimum), 'maximum_path_hops': int(max_hops)},
        'source_selection': {'superclass': 'ol_sensory', 'available': int(len(source_bodies)),
                             'in_weighted_graph': int(len(sources)), 'retained': kept_sources},
        'target_selection': {'superclass': 'vnc_motor', 'subclasses': LEG_SUBCLASSES,
                             'available': int(len(target_bodies)), 'in_weighted_graph': int(len(targets)),
                             'retained': kept_targets},
        'full_connectome_used_for_path_discovery_only': True,
        'runtime_uses_only_saved_corridor': True,
        'source_manifest': json.loads(manifest_path.read_text(encoding='utf-8')),
        'limitations': [
            'This is a structural directed-path corridor, not proof of a functional circuit.',
            'The hop bound can omit longer visual-to-motor paths.',
            'Unannotated segments on retained paths remain in the runtime graph.',
            'Point-neuron dynamics and transmitter signs remain modeled assumptions.',
        ],
    }
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'runtime_nodes': len(kept_ids), 'runtime_edges': int(matrix.nnz),
                      'photoreceptors': kept_sources, 'leg_motor_neurons': kept_targets},
                     ensure_ascii=False, indent=2), flush=True)

    del matrix, graph, rows, cols, values
    gc.collect()
    for name in ['source_indices.npy', 'target_indices.npy', 'rows.npy', 'cols.npy', 'weights.npy']:
        path = work / name
        if path.exists():
            path.unlink()
    work.rmdir()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--min-weight', type=int, default=1)
    parser.add_argument('--max-hops', type=int, default=12)
    args = parser.parse_args()
    if args.min_weight < 1 or args.max_hops < 1:
        parser.error('min-weight and max-hops must be positive')
    prepare(args.output, args.min_weight, args.max_hops)

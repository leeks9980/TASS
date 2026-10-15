"""Preserve the entire published weighted graph; never select by function or hops."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy.sparse import coo_matrix

DATA = ROOT / 'data' / 'male-cns-v1.0'
DEFAULT = ROOT / 'full_cns'


def batches():
    with pa.memory_map(str(DATA / 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'), 'r') as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    # No edge threshold or anatomical exclusions. Merge chunks progressively to
    # avoid retaining a duplicate copy of the whole list of segment IDs.
    levels = []
    count = 0
    for batch in batches():
        pre = batch.column('body_pre').to_numpy()
        post = batch.column('body_post').to_numpy()
        chunk = np.unique(np.concatenate([pre, post]))
        level = 0
        while level < len(levels) and levels[level] is not None:
            chunk = np.union1d(levels[level], chunk)
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(chunk)
        else:
            levels[level] = chunk
        previous_count = count
        count += len(pre)
        if count // 4_000_000 != previous_count // 4_000_000:
            print(f'전체 ID 수집: 연결 {count:,}', flush=True)
    ids = np.empty(0, dtype=np.int64)
    for chunk in levels:
        if chunk is not None:
            ids = np.union1d(ids, chunk)
    del levels, chunk
    annotations = feather.read_feather(DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
    nt = feather.read_feather(DATA / 'body-neurotransmitters-male-cns-v1.0.feather')
    # Isolated annotated neurons are preserved too, even if no edge references them.
    ids = np.union1d(ids, annotations.bodyId.to_numpy(dtype=np.int64))
    ids = np.union1d(ids, nt.body.to_numpy(dtype=np.int64))
    np.save(folder / 'body_ids.npy', ids)
    annotations = annotations.merge(nt.rename(columns={'body': 'bodyId'}), on='bodyId', how='left', validate='one_to_one')
    annotations['graph_index'] = np.searchsorted(ids, annotations.bodyId.to_numpy())
    feather.write_feather(annotations, folder / 'annotations.feather')
    # Keep NT predictions even for IDs not present in the annotation table.
    feather.write_feather(nt, folder / 'neurotransmitters.feather')
    n = len(ids)
    index_type = np.int32 if max(n, count) < np.iinfo(np.int32).max else np.int64
    rows = np.lib.format.open_memmap(folder / 'build_rows.npy', mode='w+', dtype=index_type, shape=(count,))
    cols = np.lib.format.open_memmap(folder / 'build_cols.npy', mode='w+', dtype=index_type, shape=(count,))
    weights = np.lib.format.open_memmap(folder / 'build_weights.npy', mode='w+', dtype=np.float32, shape=(count,))
    offset = 0
    for batch in batches():
        pre, post, weight = [batch.column(k).to_numpy() for k in ['body_pre', 'body_post', 'weight']]
        end = offset + len(pre)
        rows[offset:end] = np.searchsorted(ids, post)
        cols[offset:end] = np.searchsorted(ids, pre)
        weights[offset:end] = weight
        previous_offset = offset
        offset = end
        if offset // 4_000_000 != previous_offset // 4_000_000 or offset == count:
            print(f'전체 연결 인덱싱: {offset:,}/{count:,}', flush=True)
    rows.flush(); cols.flush(); weights.flush()
    print('전체 CSR 구성 중. PC 메모리를 많이 사용할 수 있습니다.', flush=True)
    matrix = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    np.save(folder / 'indptr.npy', matrix.indptr)
    np.save(folder / 'indices.npy', matrix.indices)
    np.save(folder / 'synapse_counts.npy', matrix.data)
    np.save(folder / 'incoming_sum.npy', np.asarray(matrix.sum(axis=1)).ravel())
    report = {
        'dataset': 'male-cns:v1.0', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'graph_nodes_including_segments': n, 'annotation_rows': len(annotations),
        'unannotated_graph_ids': n - len(annotations), 'original_connection_rows': count,
        'csr_edges': int(matrix.nnz), 'filters': None, 'isolated_annotated_neurons_retained': True,
        'all_superclasses_retained': True, 'recurrence_and_self_loops_retained': True,
        'source_manifest': json.loads((DATA / 'manifest.json').read_text(encoding='utf-8')),
        'auxiliary_usage': {
            'body_stats': 'retained source; graph already contains the summed synapse counts',
            'syn_points_and_partners': 'retained source for later spatial and ROI tracing, not point-neuron dynamics',
            'tbar_nt': 'retained source; runtime uses published neuron-level aggregation, not a spatial synapse model'},
        'limitations': ['Graph IDs include unannotated segments, not necessarily complete biological neurons.',
                        'Point-neuron dynamics and robot decoding remain explicitly modeled assumptions.'],
    }
    # Completion marker written last. A failed build is never treated as complete.
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    del matrix, rows, cols, weights
    gc.collect()
    for name in ['build_rows.npy', 'build_cols.npy', 'build_weights.npy']:
        (folder / name).unlink()
    print(f'전체 그래프 저장 완료: {folder}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    prepare(args.output)

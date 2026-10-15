"""Build an annotated recurrent MaleCNS brain network, without running a simulation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy.sparse import coo_matrix, save_npz

DATA = ROOT / 'data' / 'male-cns-v1.0'
OUTPUT = ROOT / 'navigation_circuit'
CLASSES = ['ol_intrinsic', 'ol_sensory', 'visual_projection',
           'visual_projection_tbc', 'visual_centrifugal', 'cb_intrinsic',
           'descending_neuron', 'descending_neuron_tbc']


def prepare(output, minimum):
    output.mkdir(parents=True, exist_ok=True)
    annotations = feather.read_feather(DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
    frame = annotations.loc[annotations.superclass.isin(CLASSES)].sort_values('bodyId').reset_index(drop=True)
    nt = feather.read_feather(DATA / 'body-neurotransmitters-male-cns-v1.0.feather')
    frame = frame.merge(nt.rename(columns={'body': 'bodyId'}), on='bodyId', how='left', validate='one_to_one')
    ids = frame.bodyId.to_numpy(dtype=np.int64)
    if not all(frame.type.eq(t).any() for t in ['L1', 'L2', 'L3']):
        raise ValueError('L1/L2/L3 입력 주석이 없습니다.')
    rows, cols, values = [], [], []
    scanned = 0
    with pa.memory_map(str(DATA / 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'), 'r') as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            pre, post, weight = [batch.column(k).to_numpy() for k in ['body_pre', 'body_post', 'weight']]
            scanned += len(pre)
            a, b = np.searchsorted(ids, pre), np.searchsorted(ids, post)
            valid = (a < len(ids)) & (b < len(ids)) & (weight >= minimum)
            a, b, pre, post, weight = [x[valid] for x in [a, b, pre, post, weight]]
            valid = (ids[a] == pre) & (ids[b] == post)
            rows.append(b[valid].astype(np.int32))
            cols.append(a[valid].astype(np.int32))
            values.append(weight[valid].astype(np.float32))
            print(f'원본 읽기 {i + 1}/{reader.num_record_batches}: {scanned:,} 행', flush=True)
    matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(len(ids), len(ids))).tocsr()
    save_npz(output / 'weights.npz', matrix)
    feather.write_feather(frame, output / 'neurons.feather')
    report = {
        'dataset': 'male-cns:v1.0', 'nodes': len(ids), 'edges': int(matrix.nnz),
        'classes': CLASSES, 'minimum_synapses': minimum,
        'inputs': {t: int(frame.type.eq(t).sum()) for t in ['L1', 'L2', 'L3']},
        'descending_outputs': int(frame.superclass.str.startswith('descending_neuron').sum()),
        'recurrent_edges_retained': True, 'self_loops_retained': True,
        'limitations': ['Annotated brain subset, not the full CNS or the complete visual system.',
                        'Unannotated segments, VNC, other sensory populations and ascending feedback excluded.',
                        'Selection by superclass; all selected internal edges retained, without hop pruning.',
                        'No learned or biologically calibrated neuron parameters or motor decoder.'],
    }
    manifest = DATA / 'manifest.json'
    if manifest.exists():
        report['source_manifest'] = json.loads(manifest.read_text(encoding='utf-8'))
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'저장 완료: {output}\n뉴런 {len(ids):,}, 연결 {matrix.nnz:,}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--min-weight', type=int, default=5)
    args = parser.parse_args()
    if args.min_weight < 1:
        parser.error('--min-weight must be positive')
    prepare(args.output, args.min_weight)

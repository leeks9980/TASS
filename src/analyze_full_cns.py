"""Rank recorded whole-graph responses and optionally export a candidate subgraph."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy.sparse import csr_matrix, save_npz
from full_cns_recording import state_blocks


def analyze(args):
    run = args.run
    meta = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    if args.after_seconds is None:
        args.after_seconds = float(meta.get('parameters', {}).get('stimulus_onset', 2.))
    circuit = Path(meta['circuit_folder'])
    current = json.loads((circuit / 'report.json').read_text(encoding='utf-8'))
    if current != meta['circuit_report']:
        raise ValueError('회로 준비 결과가 실행 당시와 다릅니다. 실행 당시 full_cns 폴더를 복원하세요.')
    ids = np.load(circuit / 'body_ids.npy', mmap_mode='r')
    annotation = feather.read_feather(run / 'annotations.feather')
    eligible = None
    if args.superclass:
        eligible = np.sort(annotation.loc[annotation.superclass.eq(args.superclass), 'graph_index'].to_numpy())
        if not len(eligible):
            raise ValueError(f'기록 주석에 {args.superclass}가 없습니다.')
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    total = np.lib.format.open_memmap(out / 'target_sum.npy', mode='w+', dtype=np.float64, shape=(len(ids),))
    reference = np.lib.format.open_memmap(out / 'reference_sum.npy', mode='w+', dtype=np.float64, shape=(len(ids),))
    total[:] = 0
    reference[:] = 0
    target_count = reference_count = 0
    for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event.get('input_sim_seconds', event['sim_seconds']) < args.after_seconds:
            continue
        if event['case'] not in [args.case, args.reference]:
            continue
        with np.load(run / event['snapshot']) as sample:
            if event['case'] == args.case:
                for idx, value in state_blocks(sample):
                    total[idx] += value
                target_count += 1
            else:
                for idx, value in state_blocks(sample):
                    reference[idx] += value
                reference_count += 1
    if not target_count:
        raise ValueError(f'{args.case} 기록이 없습니다. GUI에서 해당 조건을 실행한 뒤 분석하세요.')
    total /= target_count
    if reference_count:
        reference /= reference_count
        comparison = f'{args.case} minus {args.reference}'
    else:
        print('neutral 기준 기록이 없습니다. 자극 선택성이 아닌 평균 활동 크기로 순위를 매깁니다.', flush=True)
        comparison = f'{args.case} activity only; no reference'
    # Chunked ranking avoids allocating another full graph-size working array.
    candidates, candidate_scores = [], []
    for start in range(0, len(ids), 1_000_000):
        stop = min(len(ids), start+1_000_000)
        score = np.abs(total[start:stop]-reference[start:stop])
        if eligible is not None:
            local = eligible[np.searchsorted(eligible, start):np.searchsorted(eligible, stop)]-start
            if not len(local):
                continue
            score = score[local]
        else:
            local = np.arange(len(score))
        k = min(args.top, len(score))
        idx = np.argpartition(score, len(score)-k)[-k:]
        candidates.append(local[idx]+start)
        candidate_scores.append(score[idx])
    candidates, scores = np.concatenate(candidates), np.concatenate(candidate_scores)
    order = np.argsort(scores)[::-1][:args.top]
    selected = candidates[order]
    selected = selected[scores[order] > args.minimum_response]
    result = pd.DataFrame({'graph_index': selected, 'bodyId': ids[selected],
                           'response_score': np.abs(total[selected]-reference[selected]),
                           'target_mean': total[selected], 'reference_mean': reference[selected]})
    result = result.merge(annotation.drop(columns=['graph_index']), on='bodyId', how='left', validate='one_to_one')
    result.to_csv(out / 'ranked_responses.csv', index=False, encoding='utf-8-sig')
    report = {'comparison': comparison, 'target_frames': target_count, 'reference_frames': reference_count,
              'selected_graph_ids': len(selected), 'top': args.top, 'post_recording_superclass_filter': args.superclass,
              'caution': 'Activity-based candidates, not causal or functional circuit proof. Body movement and temporal history can confound contrasts.'}
    if args.export:
        dest = args.export
        dest.mkdir(parents=True, exist_ok=True)
        indices = np.sort(selected)
        raw = csr_matrix((np.load(circuit / 'synapse_counts.npy', mmap_mode='r'),
                          np.load(circuit / 'indices.npy', mmap_mode='r'),
                          np.load(circuit / 'indptr.npy', mmap_mode='r')),
                         shape=(len(ids), len(ids)), copy=False)
        subset = raw[indices, :][:, indices].tocsr()
        save_npz(dest / 'weights.npz', subset)
        np.save(dest / 'body_ids.npy', ids[indices])
        nodes = pd.DataFrame({'bodyId': ids[indices], 'source_graph_index': indices,
                              'local_index': np.arange(len(indices), dtype=np.int64)})
        nodes = nodes.merge(annotation.drop(columns=['graph_index']), on='bodyId', how='left', validate='one_to_one')
        nodes.to_feather(dest / 'neurons.feather')
        result.to_feather(dest / 'ranked_neurons.feather')
        report['export'] = {'path': str(dest), 'edges': int(subset.nnz),
                            'closure': 'Induced subgraph only; inactive intermediates and necessary feedback may be missing.'}
        (dest / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'기록 기반 순위 저장: {out / "ranked_responses.csv"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--case', default='free')
    parser.add_argument('--reference', default='neutral')
    parser.add_argument('--after-seconds', type=float, default=None, help='Default: recorded stimulus onset; legacy runs 2s.')
    parser.add_argument('--top', type=int, default=1000)
    parser.add_argument('--minimum-response', type=float, default=0.)
    parser.add_argument('--superclass', help='Optional post-recording ranking scope, e.g. vnc_motor or descending_neuron.')
    parser.add_argument('--export', type=Path)
    args = parser.parse_args()
    if args.top < 1 or args.case == args.reference:
        parser.error('top must be positive and case must differ from reference')
    analyze(args)

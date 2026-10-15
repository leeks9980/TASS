"""Compare two saved activity-weighted visual trace result folders."""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


def edge_keys(frame):
    return set(zip(frame.body_pre.astype(int), frame.body_post.astype(int), frame.depth.astype(int)))


def summarize(folder):
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    edges = pd.read_csv(folder / 'activity_weighted_edges.csv')
    reached = pd.read_csv(folder / 'reached_photoreceptors.csv')
    return report, edges, reached


def main(args):
    left_report, left_edges, left_reached = summarize(args.left)
    right_report, right_edges, right_reached = summarize(args.right)
    left_keys, right_keys = edge_keys(left_edges), edge_keys(right_edges)
    union = left_keys | right_keys
    left_sources = set(left_reached.source_body.astype(int))
    right_sources = set(right_reached.source_body.astype(int))
    source_union = left_sources | right_sources
    edge_columns = ['body_pre', 'body_post', 'depth']
    edge_values = ['modeled_signal_contribution', 'path_score']
    left_edge_values = left_edges.groupby(edge_columns, as_index=False)[edge_values].sum()
    right_edge_values = right_edges.groupby(edge_columns, as_index=False)[edge_values].sum()
    paired_edges = left_edge_values.merge(
        right_edge_values, on=edge_columns, suffixes=('_left', '_right'))
    source_left = left_reached.groupby('source_body', as_index=False).path_score.sum()
    source_right = right_reached.groupby('source_body', as_index=False).path_score.sum()
    paired_sources = source_left.merge(source_right, on='source_body', suffixes=('_left', '_right'))
    def correlation(frame, left, right):
        if len(frame) < 2:
            return None
        return float(np.corrcoef(frame[left], frame[right])[0, 1])
    result = {
        'left_case': left_report['case'], 'right_case': right_report['case'],
        'target_body_ids': left_report['target_body_ids'],
        'left_edges': len(left_edges), 'right_edges': len(right_edges),
        'edge_jaccard': len(left_keys & right_keys) / len(union) if union else 1.0,
        'left_reached_rows': len(left_reached), 'right_reached_rows': len(right_reached),
        'left_unique_photoreceptors': len(left_sources),
        'right_unique_photoreceptors': len(right_sources),
        'photoreceptor_jaccard': len(left_sources & right_sources) / len(source_union) if source_union else 1.0,
        'matched_time_indexed_edges': len(paired_edges),
        'edge_contribution_correlation': correlation(
            paired_edges, 'modeled_signal_contribution_left', 'modeled_signal_contribution_right'),
        'edge_path_score_correlation': correlation(paired_edges, 'path_score_left', 'path_score_right'),
        'photoreceptor_total_path_score_correlation': correlation(
            paired_sources, 'path_score_left', 'path_score_right'),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

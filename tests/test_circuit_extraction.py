"""Small directed-graph checks independent of the downloaded dataset."""
import sys
from pathlib import Path

TEST_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEST_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(TEST_PROJECT_ROOT / '.deps'))


import unittest
import json
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import extract_visual_leg_circuit as extraction
import pyarrow as pa
import pyarrow.feather as feather
from extract_visual_leg_circuit import corridor_masks, distances, valid_edges
from scipy.sparse import coo_matrix


class CorridorTests(unittest.TestCase):
    def test_parallel_paths_cycles_and_disconnected_targets(self):
        # 0 and 6 are inputs, 4 and 7 are targets. 7 has no input path.
        # 1<->2 is a cycle, and 0->5 is an irrelevant dead end.
        pre = np.array([0, 1, 0, 3, 2, 1, 2, 0, 6])
        post = np.array([1, 4, 3, 4, 1, 2, 4, 5, 3])
        graph = coo_matrix((np.ones(len(pre), dtype=bool), (pre, post)), shape=(8, 8)).tocsr()
        forward = distances(graph, [0, 6])
        backward = distances(graph.T.tocsr(), [4, 7])
        np.testing.assert_array_equal(forward, [0, 1, 2, 1, 2, 1, 0, -1])
        nodes, edges = corridor_masks(forward, backward, pre, post)
        self.assertEqual(set(np.flatnonzero(nodes)), {0, 1, 2, 3, 4, 6})
        self.assertFalse(edges[7])
        nodes, edges = corridor_masks(forward, backward, pre, post, 2)
        self.assertNotIn(2, np.flatnonzero(nodes))
        self.assertEqual(set(zip(pre[edges], post[edges])), {(0, 1), (1, 4), (0, 3), (3, 4), (6, 3)})

    def test_empty_seeds(self):
        graph = coo_matrix((3, 3), dtype=bool).tocsr()
        np.testing.assert_array_equal(distances(graph, []), [-1, -1, -1])

    def test_weight_filter_preserves_self_connections(self):
        np.testing.assert_array_equal(valid_edges(np.array([0, 0]), np.array([0, 1]), np.array([2, 1]), 2), [True, False])

    def test_file_export_keeps_unknown_intermediate_and_reports_missing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = pd.DataFrame({
                "bodyId": [0, 2, 4, 5], "type": ["L2", "MNfl01", "MNml01", "L2"],
                "superclass": ["ol_intrinsic", "vnc_motor", "vnc_motor", "ol_intrinsic"],
                "subclass": ["", "fl", "ml", ""], "somaSide": ["L", "L", "R", "R"]})
            feather.write_feather(annotations, root / extraction.ANNOTATIONS)
            feather.write_feather(pd.DataFrame({"body": [0, 1, 2], "consensus_nt": ["gaba", "unknown", "acetylcholine"]}), root / extraction.NT)
            feather.write_feather(pd.DataFrame({"body_pre": [0, 1, 0, 1, 8], "body_post": [1, 2, 3, 1, 9], "weight": [2, 3, 1, 4, 1]}), root / extraction.WEIGHTS)
            (root / "manifest.json").write_text('{}', encoding="utf-8")
            with patch.object(extraction, "DATA", root):
                extraction.run(1, 2, root / "output")
            report = json.loads((root / "output/report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["targets_reachable_from_L2"], 1)
            self.assertEqual(report["variants"]["unbounded"]["unannotated_segments"], 1)
            neurons = feather.read_feather(root / "output/unbounded/neurons.feather")
            self.assertEqual(set(neurons.bodyId), {0, 1, 2})
            self.assertFalse(neurons.loc[neurons.bodyId.eq(1), "has_neuron_annotation"].item())
            edges = feather.read_feather(root / "output/unbounded/connections.feather")
            self.assertEqual(set(zip(edges.body_pre, edges.body_post, edges.weight)), {(0, 1, 2), (1, 2, 3), (1, 1, 4)})
            bounded = feather.read_feather(root / "output/max_2_hops/connections.feather")
            self.assertEqual(len(bounded), 2)
            source_report = pd.read_csv(root / "output/L2_sources.csv")
            self.assertFalse(source_report.loc[source_report.bodyId.eq(5), "in_graph"].item())


if __name__ == "__main__":
    unittest.main()

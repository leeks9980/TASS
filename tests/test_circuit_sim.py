"""Headless GUI-model checks. These do not start the interactive simulation."""
import sys
from pathlib import Path

TEST_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEST_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(TEST_PROJECT_ROOT / '.deps'))


import tempfile
import unittest
import circuit_sim_model as sim
import numpy as np
import pandas as pd
import pyarrow.feather as feather


class SimulatorTests(unittest.TestCase):
    def test_camera_wall_side_and_hidden_wall(self):
        image = sim.camera_image(-0.8, 1.5, True)
        self.assertGreater(image[:48].mean(), image[48:].mean())
        np.testing.assert_array_equal(sim.camera_image(0, 1, False), np.zeros(96))

    def test_transmitter_assumptions_are_explicit(self):
        frame = pd.DataFrame({"consensus_nt": ["acetylcholine", "gaba", "glutamate", None],
                              "predicted_nt": [None] * 4, "celltype_predicted_nt": [None] * 4})
        np.testing.assert_array_equal(sim.nt_signs(frame, "NT 가정 · Glutamate 미정"), [1, -1, 0, 0])
        np.testing.assert_array_equal(sim.nt_signs(frame, "NT 가정 · Glutamate −"), [1, -1, -1, 0])

    def test_real_edge_driven_side_response_and_decay(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame = pd.DataFrame({
                "bodyId": [10, 11, 20, 21, 30, 31], "type": ["L2", "L2", "IN", "IN", "MNfl01", "MNfl01"],
                "superclass": ["ol_intrinsic", "ol_intrinsic", "vnc_intrinsic", "vnc_intrinsic", "vnc_motor", "vnc_motor"],
                "subclass": ["", "", "", "", "fl", "fl"], "somaSide": ["L", "R", "L", "R", "L", "R"],
                "rootSide": [None] * 6, "from_L2_hops": [0, 0, 1, 1, 2, 2],
                "has_neuron_annotation": [True] * 6, "consensus_nt": ["acetylcholine"] * 6,
                "predicted_nt": [None] * 6, "celltype_predicted_nt": [None] * 6,
                "predicted_nt_confidence": [.8] * 6})
            feather.write_feather(frame, folder / "neurons.feather")
            feather.write_feather(pd.DataFrame({"body_pre": [10, 20, 11, 21, 30, 20],
                                                "body_post": [20, 30, 21, 31, 20, 20],
                                                "weight": [5, 10, 5, 10, 8, 7]}), folder / "connections.feather")
            model = sim.CircuitModel(folder)
            self.assertEqual(model.weights.nnz, 4)
            drive = np.concatenate([np.ones(48), np.zeros(48)])
            for _ in range(24):
                score, raw, _, _ = model.step(drive)
            self.assertGreater(score[0], .99)
            self.assertEqual(score[1], 0)
            for _ in range(24):
                score, raw, _, _ = model.step(np.zeros(96))
            self.assertLess(score[0], 1e-5)


if __name__ == "__main__":
    unittest.main()

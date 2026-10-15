import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from analyze_dn_side_values import neuron_side, safe_index


class DNSideValueTests(unittest.TestCase):
    def test_side_fallback(self):
        frame = pd.DataFrame({
            'rootSide': ['L', None, None],
            'somaSide': ['R', 'R', None],
            'instance': ['a_R', 'b_L', 'c_L'],
        })
        self.assertEqual(neuron_side(frame).tolist(), ['L', 'R', 'L'])

    def test_safe_index(self):
        self.assertEqual(safe_index(0.0, 0.0), 0.0)
        self.assertAlmostEqual(safe_index(3.0, 1.0), 0.5)


if __name__ == '__main__':
    unittest.main()

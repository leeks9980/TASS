from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from visual_input_boundary import is_photoreceptor_boundary


def test_current_hex_mapped_photoreceptor_boundary_is_accepted():
    assert is_photoreceptor_boundary(
        'connectivity-derived, hex-mapped ol_sensory photoreceptors')


def test_previous_photoreceptor_boundary_remains_accepted():
    assert is_photoreceptor_boundary(
        'annotated ol_sensory photoreceptor population')


def test_non_photoreceptor_boundary_is_rejected():
    assert not is_photoreceptor_boundary('L1/L2/L3 visual interneurons')
    assert not is_photoreceptor_boundary(None)

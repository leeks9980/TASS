from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from neurotransmitter_model import neurotransmitter_effects, resolved_neurotransmitters


def test_published_label_priority_and_provenance():
    frame = pd.DataFrame({
        'ground_truth': ['histamine', None, None, None],
        'consensus_nt': ['acetylcholine', 'gaba', None, None],
        'celltype_predicted_nt': ['gaba', 'glutamate', 'acetylcholine', None],
        'predicted_nt': ['unclear', 'unclear', 'gaba', 'dopamine'],
    })
    names, sources = resolved_neurotransmitters(frame)
    assert names.tolist() == ['histamine', 'gaba', 'acetylcholine', 'dopamine']
    assert sources.tolist() == [
        'ground_truth', 'consensus_nt', 'celltype_predicted_nt', 'predicted_nt']


def test_default_signs_include_inhibitory_photoreceptor_histamine():
    effects = neurotransmitter_effects()
    assert effects['acetylcholine'] == 1
    assert effects['gaba'] == -1
    assert effects['glutamate'] == -1
    assert effects['histamine'] == -1

"""MaleCNS neurotransmitter labels and explicit point-model sign assumptions."""

import pandas as pd


def resolved_neurotransmitters(frame):
    """Resolve the published per-body label without discarding its provenance."""
    names = frame.ground_truth.fillna(frame.consensus_nt)
    names = names.fillna(frame.celltype_predicted_nt).fillna(frame.predicted_nt)
    names = names.fillna('unclear').str.lower()

    source = pd.Series('predicted_nt', index=frame.index, dtype=object)
    source.loc[frame.celltype_predicted_nt.notna()] = 'celltype_predicted_nt'
    source.loc[frame.consensus_nt.notna()] = 'consensus_nt'
    source.loc[frame.ground_truth.notna()] = 'ground_truth'
    return names, source


def neurotransmitter_effects(glutamate='negative'):
    """Return presynaptic signs used by the simplified rate model.

    These are model-wide approximations, not receptor-resolved synaptic signs.
    """
    if glutamate not in {'positive', 'negative'}:
        raise ValueError("glutamate must be 'positive' or 'negative'")
    return {
        'acetylcholine': 1.0,
        'gaba': -1.0,
        'glutamate': -1.0 if glutamate == 'negative' else 1.0,
        'histamine': -1.0,
        # Shiu et al. whole-brain compatibility assumption. These are
        # neuromodulators and +1 must not be read as a receptor-resolved fact.
        'dopamine': 1.0,
        'octopamine': 1.0,
        'serotonin': 1.0,
    }

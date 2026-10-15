"""Names of recording input boundaries that directly drive photoreceptors."""

PHOTORECEPTOR_BOUNDARIES = frozenset({
    'annotated ol_sensory photoreceptor population',
    'connectivity-derived, hex-mapped ol_sensory photoreceptors',
})


def is_photoreceptor_boundary(boundary):
    """Return true only for a known direct ol_sensory input boundary."""
    return boundary in PHOTORECEPTOR_BOUNDARIES

"""CONIFER estimators — one engine, one estimator per forest target.

The reusable core is the compositional area-level Fay-Herriot engine (debiased-ML mean +
design-aware conformal simplex sets). ``DiameterDistribution`` is that engine pointed at
the DBH-class composition; future targets (species shares, size x species) reuse the same
engine with a different simplex.
"""
from __future__ import annotations

# Validated core, vendored under _engine/ (see _engine/__provenance__.txt).
from ._engine.stemclass import DiameterDistribution

# The engine is target-agnostic; expose it under its method-family name so siblings read
# consistently (CompositionalFH is DiameterDistribution today — one object, two names).
CompositionalFH = DiameterDistribution


class SpeciesComposition:
    """Planned CONIFER v0.3 — species shares as the composition.

    Reuses the :class:`CompositionalFH` engine with a species simplex in place of the
    DBH classes; only the target wiring differs. Not implemented in v0.1.
    """

    def __init__(self, *args, **kwargs):  # pragma: no cover - intentional stub
        raise NotImplementedError(
            "conifer.SpeciesComposition arrives in v0.3. The engine (CompositionalFH) "
            "already supports a general composition; only the species-target wiring is "
            "pending. Use conifer.DiameterDistribution today."
        )


__all__ = ["DiameterDistribution", "CompositionalFH", "SpeciesComposition"]

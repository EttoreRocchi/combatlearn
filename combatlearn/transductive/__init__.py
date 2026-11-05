"""Transductive (whole-cohort) harmonizers.

A method lands here when a condition-of-interest term is protected in the design,
which is unavailable at transform time: it runs as ``fit_transform`` on a complete
cohort and is not a scikit-learn ``Pipeline`` step. :class:`TransductiveComBat`
offers ``method='seq'`` and ``'met'`` (upstream ``full_mod=TRUE``, ``condition``
required). ``method='longitudinal'`` is deprecated since v2.4.0 and will be removed
in v3.0.0; use :class:`~combatlearn.LongitudinalComBat`.
"""

from ._transductive import TransductiveComBat

__all__ = ["TransductiveComBat"]

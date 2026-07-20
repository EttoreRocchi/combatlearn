"""Transductive (whole-cohort) harmonizers.

This tier holds one-shot harmonizers that do not meet the inductive
fit/transform contract - their benefit is realized in-sample, so they are used
as ``fit_transform`` on a complete cohort rather than fit on train and applied
to held-out data. They are deliberately not scikit-learn ``Pipeline`` steps.

Methods are exposed through a single :class:`TransductiveComBat` class with a
``method=`` selector (mirroring the inductive :class:`~combatlearn.ComBat`), so
new transductive engines are added as new ``method=`` values.
"""

from ._transductive import TransductiveComBat

__all__ = ["TransductiveComBat"]

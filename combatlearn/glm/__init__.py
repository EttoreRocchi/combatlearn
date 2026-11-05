"""Non-Gaussian GLM ComBat engines: ComBat-seq and ComBat-met.

Both fit a parametric GLM, then apply a strictly per-observation quantile map, so
they are inductive and safe inside a cross-validation pipeline. See :class:`GLMComBat`.
"""

from ._estimator import GLMComBat
from ._model import GLMComBatModel

__all__ = ["GLMComBat", "GLMComBatModel"]

API Reference
=============

Complete API documentation for combatlearn.

ComBat
------

The main scikit-learn compatible transformer for batch effect correction.

.. autoclass:: combatlearn.ComBat
   :members:
   :inherited-members: BaseEstimator, TransformerMixin, ReprHTMLMixin, _HTMLDocumentationLinkMixin, _MetadataRequester, _SetOutputMixin, object
   :show-inheritance:

   .. automethod:: __init__

NestedComBat
------------

Multi-batch-variable ComBat (Nested / OPNested / GMM ComBat) for harmonizing over
several batch variables at once.

.. autoclass:: combatlearn.NestedComBat
   :members:
   :inherited-members: BaseEstimator, TransformerMixin, ReprHTMLMixin, _HTMLDocumentationLinkMixin, _MetadataRequester, _SetOutputMixin, object
   :show-inheritance:

   .. automethod:: __init__

LongitudinalComBat
------------------

Inductive Longitudinal ComBat for repeated-measures designs: empirical-Bayes
harmonization with a per-subject random intercept. Fit on train and applied to
held-out data under a stricter precondition than :class:`~combatlearn.ComBat` -
the batch *and* the subject must have been seen at fit.

.. autoclass:: combatlearn.LongitudinalComBat
   :members:
   :inherited-members: BaseEstimator, TransformerMixin, ReprHTMLMixin, _HTMLDocumentationLinkMixin, _MetadataRequester, _SetOutputMixin, object
   :show-inheritance:

   .. automethod:: __init__

GLMComBat
---------

Inductive ComBat for non-Gaussian data: ComBat-seq (``method="seq"``) for RNA-seq
counts and ComBat-met (``method="met"``) for DNA-methylation beta-values. Both fit
a parametric GLM and then map each observation through a **strictly
per-observation** quantile map, so the fitted parameters freeze at fit and
``transform`` applies to held-out samples.

.. autoclass:: combatlearn.GLMComBat
   :members:
   :inherited-members: BaseEstimator, TransformerMixin, ReprHTMLMixin, _HTMLDocumentationLinkMixin, _MetadataRequester, _SetOutputMixin, object
   :show-inheritance:

   .. automethod:: __init__

TransductiveComBat
------------------

Whole-cohort, ``fit_transform``-only harmonizers whose requested *configuration*
cannot be frozen at fit - concretely, when a condition-of-interest / outcome term
is protected in the design, which is unavailable (and leakage-prone) at transform
time.

Its principal occupants are ``method="seq"`` and ``method="met"``: the
condition-protected ComBat-seq and ComBat-met configurations (upstream
``full_mod=TRUE``) used for differential expression and differential methylation,
which require a ``condition``. Protecting the condition is the *only* thing they
do that the inductive :class:`~combatlearn.GLMComBat` cannot; without one, use
``GLMComBat``. ``method="longitudinal"`` is deprecated in favour of the inductive
:class:`~combatlearn.LongitudinalComBat`.

.. autoclass:: combatlearn.transductive.TransductiveComBat
   :members:
   :inherited-members: BaseEstimator, TransformerMixin, ReprHTMLMixin, _HTMLDocumentationLinkMixin, _MetadataRequester, _SetOutputMixin, object
   :show-inheritance:

   .. automethod:: __init__

Inspection
----------

Functions for inspecting fitted ComBat models.

.. automodule:: combatlearn.inspection
   :members:

Metrics
-------

Functions for computing batch effect metrics.

.. automodule:: combatlearn.metrics
   :members: compute_batch_metrics

Visualization
-------------

Functions for visualizing batch effects and ComBat corrections.

.. automodule:: combatlearn.visualization
   :members: plot_transformation, plot_feature_diagnostics, plot_batch_effect_heatmap

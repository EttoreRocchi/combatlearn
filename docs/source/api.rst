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

TransductiveComBat
------------------

Whole-cohort, ``fit_transform``-only harmonizers whose benefit is realized
in-sample (currently Longitudinal ComBat).

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

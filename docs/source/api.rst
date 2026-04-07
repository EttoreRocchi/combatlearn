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

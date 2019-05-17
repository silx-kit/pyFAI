pyFAI API
=========

This chapter describes the programming interface of pyFAI, so what you can
expect after having launched *Jupyter notebook* (or ipython) and typed:

.. code-block:: python

	import pyFAI

The most important class is AzimuthalIntegrator which is an object containing
both the geometry (it inherits from Geometry, another class)
and exposes important methods (functions) like integrate1d and integrate2d.

.. toctree::
   :maxdepth: 2

   pyFAI

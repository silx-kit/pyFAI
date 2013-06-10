pyFAI API
=========

This chapter describes the programming interface of pyFAI, so what you can expect after having launched ipython and typed:
..

	import pyFAI

The most important class is AzimuthalIntegrator which is an object containing both the geometry (it inherits from Geometry, another class)
and exposes important methods (functions) like integrate1d and integrate2d.

.. toctree::
   :maxdepth: 4

   pyFAI

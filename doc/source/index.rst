.. pyFAI documentation master file, created by
   sphinx-quickstart on Mon Nov 19 13:19:53 2012.

:Author: Jérôme Kieffer
:Date: 30/10/2020
:Keywords: generic description of the geometry
:Target: General audience

Fast Azimuthal Integration using Python
=======================================

PyFAI is a python libary for azimuthal integration of X-Ray/neutron scattering data acquired with area-detectors. 
For this, images needs to be re-binned in polar coordinate systems. 
Additional tools are provided to calibrate the experimental setup, i.e. define where the detector is positionned in space considering the sample and the incident beam. 

.. figure:: img/overview.png
   :align: center
   :alt: PyFAI is about regridding image in polar space.

The sub-title of the project, *the space-folder*, is related to the expertise acquired in re-distributing the signal acquired in one geometry (often cartesian) 
into another one (often polar) while propagating properly the associated error. Unlike interpolation, the flux is conserved in those transformations, and can be 
used for any type of space transformation, including image distortion and many more.     

This documentation starts with a general descriptions of the pyFAI library.
This first chapter contains an introduction to pyFAI, what it is, what it aims at
and how it works (from the scientists' point of view).
Especially, geometry, calibration, azimuthal integration algorithms are described
and pixel splitting schemes are explained there. 
The most important part is this scheme explaining the geometry used: 

.. figure:: img/PONI.png
   :align: center
   :alt: The geometry used by pyFAI is inspired by SPD

.. toctree::
   :maxdepth: 1


   pyFAI
   usage/index
   man/scripts
   design/index
   api/modules
   operations/index
   ecosystem
   project
   changelog
   publications
   biblio
   glossary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


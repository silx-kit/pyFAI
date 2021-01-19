.. pyFAI documentation master file, created by
   sphinx-quickstart on Mon Nov 19 13:19:53 2012.

:Author: Jérôme Kieffer
:Date: 30/10/2020
:Keywords: generic description of the geometry
:Target: General audience

Fast Azimuthal Integration using Python
=======================================

PyFAI is a python libary for azimuthal integration of X-ray/neutron/electron scattering data acquired with area detectors. 
For this, images needs to be re-binned in polar coordinate systems. 
Additional tools are provided to calibrate the experimental setup, i.e. define where the detector is positioned in space considering the sample and the incident beam. 

.. figure:: img/overview.png
   :align: center
   :alt: PyFAI is about regridding image in polar space.

The core idea is to redistribute the signal acquired with the experimental geometry 
into a geometry suitable for further analysis, like Rietveld refinement for power data 
or Inverse Fourier Transform for SAXS data.    
Unlike interpolation, this re-distribution conserves the signal, its variance and 
can be used for other types of transformation like distortion correction.  

Since the alignment of the beam, the sample and the detector can never be perfect,
pyFAI tries to cope with it by calibrating their relative position using a 
reference sample material (called calibrant). 
After calibration, the geometry can be saved in a *poni-file* and used to perform azimuthal averaging
of several samples. 
The geometry used by pyFAi is described in this scheme:
     
.. figure:: img/PONI.png
   :align: center
   :alt: The geometry used by pyFAI is inspired by SPD

 


This documentation starts with a general descriptions of the pyFAI library.
This first chapter contains an introduction to pyFAI, what it is, what it aims at
and how it works (from the scientists' point of view).
Especially, geometry, calibration, azimuthal integration algorithms are described
and pixel splitting schemes are explained there. 

Follows tutorials, manual pages of applications, the description of the programming interface 
(use the search bar as it is pretty long) and the instruction on how to install the software on various platforms. 

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

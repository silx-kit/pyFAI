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



The documentation starts with a general descriptions of the pyFAI library.
This first chapter contains an introduction to pyFAI, what it is, what it aims at
and how it works (from the scientists point of view).
Especially, geometry, calibration, azimuthal integration algorithms are described
and pixel splitting schemes are explained.

Follows trainings, cookbooks and tutorials on how to use pyFAI:
Training session are conferences which were video-recorded 
and made available online.
Cookbooks explain how to use pyFAI in practical cases.
Tutorials present the usage of pyFAI within the *Jupyter* notebook environment and 
present advanced features but require a good knowledge both of python and pyFAI.
After the tutorials, all manual pages of pyFAI programs, both graphical interfaces
and scripts are described in the documentation.

The design of the programming interface (API) is then exposed before a
comprehensive description of most modules contained in pyFAI.
Some minor submodules as well as the documentation of the Cython sub-modules are
not included for concision purposes.

Installation procedures for Windows, MacOSX and Linux operating systems are then
described.

Finally other programs/projects relying on pyFAI are presented and the project is
summarized from a developer's point of view.

In appendix there are some figures about the project and its management and a list
of publication on pyFAI.

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


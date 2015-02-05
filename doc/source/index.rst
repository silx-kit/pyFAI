.. pyFAI documentation master file, created by
   sphinx-quickstart on Mon Nov 19 13:19:53 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fast Azimuthal Integration using Python
=======================================

PyFAI A Python libary for high performance azimuthal integration which can use on GPU,
which was presented at EuroScipy 2014: `the video is online <https://www.youtube.com/watch?v=QSlo_Nyzeig>`_
as well as the proceedings (soon).

This document starts with a general descriptions of the pyFAI library in the first chapter.
This first chapter contains an introduction to pyFAI, what it is, what it aims at and how it works
(for scientists). Especially, geometry, calibration, azimuthal integration algorithms are described
and pixel splitting schemes are explained.

The second chapter is the concatenation of the manual pages of all (relevant) scripts. Those are programs to be launched at the
command line allowing the treatment of a  diffraction experiment without knowing anything about Python.

The third chapter contains a comprehensive description of most Python modules contained in pyFAI.
Some minor submodules as well as the documentation of the Cython sub-modules are not included for concision purposes.
The last chapter is an appendix giving some figures about the project and its management.

TODO: Split documentation into smth for: Scientists (Quick-start) + sys-admin (install on any operating system) + programmer ...

Split into Design (specification,class diagrams) , Usage and Operation

.. toctree::
   :maxdepth: 2


   pyFAI
   usage/cookbook/index
   usage/tutorial/index
   man/scripts
   design/index
   api/modules
   operations/index
   ecosystem
   project
   biblio


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


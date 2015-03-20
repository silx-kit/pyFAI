.. pyFAI documentation master file, created by
   sphinx-quickstart on Mon Nov 19 13:19:53 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fast Azimuthal Integration using Python
=======================================

PyFAI A Python libary for high performance azimuthal integration which can use on GPU,
which was presented at EuroScipy 2014: `the video is online <https://www.youtube.com/watch?v=QSlo_Nyzeig>`_
as well as the `proceedings <http://arxiv.org/abs/1412.6367>`_.

This document starts with a general descriptions of the pyFAI library in the first chapter.
This first chapter contains an introduction to pyFAI, what it is, what it aims at and how it works
(for scientists). Especially, geometry, calibration, azimuthal integration algorithms are described
and pixel splitting schemes are explained.

Follows cookbook, tutorials on how to use pyFAI scripts, then the manual pages of all scripts.
Those are programs to be launched at the
command line allowing the treatment of a  diffraction experiment without knowing anything about Python.

The design of the programming interface is then exposed before a comprehensive description of most modules contained in pyFAI.
Some minor submodules as well as the documentation of the Cython sub-modules are not included for concision purposes.
The last chapter is an appendix giving some figures about the project and its management.

Installation procedures for Windows, MacOSX and Linux operating systems are then described.
Finally other programs/projects relying on pyFAI are presented and the project is summarized from a developer's point of view.

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


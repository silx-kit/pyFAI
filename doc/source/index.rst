.. pyFAI documentation master file, created by
   sphinx-quickstart on Mon Nov 19 13:19:53 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fast Azimuthal Integration using Python
=======================================

PyFAI is a python libary for azimuthal integration of diffraction data acquired
with 2D detectors, providing high performance thanks to GPU computing.
Most concepts have been presented at EuroScipy 2014 in this
`video <https://www.youtube.com/watch?v=QSlo_Nyzeig>`_
as well as the `proceedings <http://arxiv.org/abs/1412.6367>`_.

The documentation starts with a general descriptions of the pyFAI library.
This first chapter contains an introduction to pyFAI, what it is, what it aims at
and how it works (from the scientists point of view).
Especially, geometry, calibration, azimuthal integration algorithms are described
and pixel splitting schemes are explained.

Follows cookbook and tutorials on how to use pyFAI:
Cookbooks focus on how to use pyFAI using the various graphical interfaces or
from scripts on the command linee.
Tutorials use the *Jupyter* notebook (formerly ipython) and present the Python interface.
The first tutorial start with a quick general introduction of the notebooks,
how to process data in the general case, subsequent tutorials are more advanced
and require already a good knowledge both in Python and pyFAI..
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
   usage/cookbook/index
   usage/tutorial/index
   man/scripts
   design/index
   api/modules
   operations/index
   ecosystem
   project
   changelog
   publications
   biblio

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


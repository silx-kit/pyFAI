Installation of Python Fast Azimuthal Integration library
=========================================================

Author: Jérôme Kieffer

Date: 18/12/2014

Keywords: Installation procedure

Target: System administrators

Reference:

Abstract
--------

Installation procedure


Dependencies
------------

PyFAI is a Python library which relies on the scientific stack (numpy, scipy, matplotlib)

* Python: version 2.6, 2.7 and soon 3.2 and 3.4
* NumPy: version 1.4 or newer
* SciPy: version 0.7 or newer
* Matplotlib: verson 0.99 or newer
* FabIO: version 0.08 or newer

There are plenty of optional dependencies which will not prevent pyFAI from working
by may impair performances or prevent tools from properly working:

* h5py (to access HDF5 files)
* pyopencl (for GPU computing)
* fftw (for image analysis)
* pymca (for mask drawing)
* PyQt4 or PySide (for the graphical user interface)

Build dependencies:
-------------------

In addition to the run dependencies, pyFAI needs a C compiler.

C files are generated from Cython_ source and distributed. Cython is only needed for developing new binary modules.
If you want to generate your own C files, make sure your local Cython version supports memory-views (available from Cython v0.17 and newer).

Building procedure
------------------

::
    python setup.py build install

There are few specific options to setup.py:

* --no-cython: do not use cython (even if present) and use the C source code provided by the development team
* --no-openmp: if you compiler lacks OpenMP support
* --with-testimages: build the source distribution including all test images. Download 200MB of test images to create a self consistent tar-ball.

.. toctree::
   :maxdepth: 2

   linux
   macosx
   windows


Test suites
-----------
PyFAI comes with a test suite to ensure all core functionnalities are working as expected:

::
    python setup.py build test

Nota: to run the test an internet connection is needed as 200MB of test imagesneed to be downloaded.


.. toctree::
   :maxdepth: 2

   project

References:
-----------

:: _Cython: http://cython.org

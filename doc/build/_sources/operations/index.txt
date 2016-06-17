:Author: Jérôme Kieffer
:Date: 06/06/2016
:Keywords: Installation procedure
:Target: System administrators


Installation of Python Fast Azimuthal Integration library
=========================================================


Abstract
--------

Installation procedure for all main operating systems

Hardware requirement
--------------------

PyFAI has been tested on various hardware: i386, x86_64, PPC64le, ARM.
The main constrain may be the memory requirement: 2GB of memory is a minimal requirement to run the tests.
The program may run with less but "MemoryError" are expected (appearing sometimes as segmentation faults).
As a consequence, a 64-bits operating system with enough memory is strongly advised.

Dependencies
------------

PyFAI is a Python library which relies on the scientific stack (numpy, scipy, matplotlib)

* Python: version 2.7, 3.4 and 3.5.
  Support for 2.6, 3.2 and 3.3 has been dropped in v0.12
* NumPy: version 1.4 or newer
* SciPy: version 0.7 or newer
* Matplotlib: verson 0.99 or newer
* FabIO: version 0.08 or newer
* h5py (to access HDF5 files)

There are plenty of optional dependencies which will not prevent pyFAI from working
by may impair performances or prevent tools from properly working:


* pyopencl (for GPU computing)
* fftw (for image analysis)
* pymca (for mask drawing)
* PyQt4 or PySide (for the graphical user interface)

Build dependencies:
-------------------

In addition to the run dependencies, pyFAI needs a C compiler to build extensions.

C files are generated from cython_ source and distributed.
The distributed version correspond to OpenMP version.
Non-OpenMP version needs to be built from cython source code (especially on MacOSX).
If you want to generate your own C files, make sure your local Cython version
is sufficiently recent (>0.20).

.. _cython: http://cython.org


Building procedure
------------------

.. code ::

    python setup.py build
    pip install . --upgrade

There are few specific options to ``setup.py``:

* ``--no-cython``: Prevent Cython (even if present) to re-generate the C source code. Use the one provided by the development team.
* ``--no-openmp``: Recompiles the Cython code without OpenMP support (default under MacOSX).
* ``--openmp``: Recompiles the Cython code with OpenMP support (Default under Windows and Linux).
* ``--with-testimages``: build the source distribution including all test images. Download 200MB of test images to create a self consistent tar-ball.

Detailed installation procedure on different operating systems
--------------------------------------------------------------

.. toctree::
   :maxdepth: 2

   linux
   macosx
   windows


Test suites
-----------

PyFAI comes with a test suite to ensure all core functionalities are working as expected and numerical results are correct:

.. code::

    python setup.py build test

**Nota:** to run the test, an internet connection is needed as 200MB of test images need to be download.
You may have to set the environment variable *http_proxy* and *https_proxy*
according to the networking environment you are in.
Specifically at ESRF, please phone the hotline at 24-24 to get those information.

Environment variables
---------------------

PyFAI can use a certain number of environment variable to modify its default behavior:

* PYFAI_OPENCL: set to "0" to disable the use of OpenCL
* PYFAI_DATA: path with gui, calibrant, ...
* PYFAI_TESTIMAGES: path wit test images (if absent, they get downloaded from the internet)



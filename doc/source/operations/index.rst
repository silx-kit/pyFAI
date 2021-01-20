:Author: Jérôme Kieffer
:Date: 07/01/2021
:Keywords: Installation procedure
:Target: System administrators


Installation
============

Installation procedure for all main operating systems

Hardware requirement
--------------------

PyFAI has been tested on various hardware: i386, x86_64, PPC64le, ARM, ARM64.
The main constrain may be the memory requirement: 2GB of memory is a minimal requirement to run the tests.
The program may run with less but "MemoryError" are expected (appearing sometimes as segmentation faults).
As a consequence, a 64-bits operating system with enough memory is strongly advised.

Dependencies
------------

PyFAI is a Python library which relies on the scientific stack (numpy, scipy, matplotlib)

* Python: version 3.6 or newer (version 3.5 was dropped with 0.19)
* NumPy: version 1.12 or newer  
* SciPy: version 0.18 or newer 
* Matplotlib: verson 2.0 or newer 
* FabIO: version 0.5 or newer
* h5py: version 2.10 or newer
* silx: version 0.14 or newer

There are plenty of optional dependencies which will not prevent pyFAI from working
by may impair performances or prevent tools from properly working:


* pyopencl (for GPU computing)
* fftw (for image analysis)
* PyQt5 or PySide2 (for the graphical user interface)

Build dependencies:
-------------------

In addition to the run dependencies, pyFAI needs a C compiler to build extensions.

C files are generated from cython_ source and distributed.
The distributed version correspond to OpenMP version.
Non-OpenMP version needs to be built from cython source code (especially on MacOSX).
If you want to generate your own C files, make sure your local Cython version
is sufficiently recent (>0.21).

.. _cython: http://cython.org


Building procedure
------------------

.. code-block:: shell

    pip install -r requirements.txt
    python3 setup.py build
    pip install . --upgrade

There are few specific options to ``setup.py build``:

* ``--no-cython``: Prevent Cython (even if present) to re-generate the C source code. Use the one provided by the development team.
* ``--force-cython``: Force the re-cythonization of all binary extensions.
* ``--no-openmp``: Recompiles the Cython code without OpenMP support (default under MacOSX).
* ``--openmp``: Recompiles the Cython code with OpenMP support (Default under Windows and Linux).
* ``--with-testimages``: build the source distribution including all test images. Download 200MB of test images to create a self consistent tar-ball.

Detailed installation procedure
-------------------------------

.. toctree::
   :maxdepth: 2

   linux
   macosx
   windows


Test suites
-----------

PyFAI comes with a test suite to ensure all core functionalities are working as expected and numerical results are correct:

.. code-block:: shell

    python3 setup.py build
    python3 run_tests.py

**Nota:** to run the test, an internet connection is needed as 200MB of test images need to be download.
You may have to set the environment variable *http_proxy* and *https_proxy*
according to the networking environment you are in.
This is no more needed at the ESRF.

Environment variables
---------------------

PyFAI can use a certain number of environment variable to modify its default behavior:

* PYFAI_OPENCL: set to "0" to disable the use of OpenCL
* PYFAI_DATA: path with gui, calibrant, ...
* PYFAI_TESTIMAGES: path wit test images (if absent, they get downloaded from the internet)
* PYFAI_NO_LOGGING: Disable the configuration of any python logger in interactive mode



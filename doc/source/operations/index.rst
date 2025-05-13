:Author: Jérôme Kieffer
:Date: 28/01/2025
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

* Python: version 3.9 or newer
* NumPy: version 1.12 or newer
* SciPy: version 0.18 or newer
* Matplotlib: verson 2.0 or newer
* FabIO: version 0.5 or newer
* h5py: version 2.10 or newer
* silx: version 2 or newer

There are plenty of optional dependencies which will not prevent pyFAI from working
by may impair performances or prevent tools from properly working:


* pyopencl (for GPU computing)
* fftw (for image analysis)
* PyQt5, PyQt6 or PySide6 (for the graphical user interface)

Build dependencies:
-------------------

PyFAI v2023.01 intoduced a new build system based on `meson` with the following requirements:

* meson-python (>=0.11)
* git
* meson (>=0.64)
* ninja

The former build system was using `setup.py` files, based on setuptools and numpy.distutils, was removed with pyFAI v2023.10.

In addition to the build tools, pyFAI needs a C/C++ compiler to build extensions and cython_ (>0.29) to generate those C/C++ files.
The following compiler have been successfully tested:

* Linux: `gcc` and `clang` (both support OpenMP)
* Windows: msvc++ (supports OpenMP)
* Apple: clang modified version for Apple computers without support for OpenMP, please use OpenCL for parallelization.

.. _cython: http://cython.org


Building procedure
------------------

.. code-block:: shell

    git clone https://github.com/silx-kit/pyFAI
    cd pyFAI
    pip install -r requirements.txt
    pip install . --upgrade

or

.. code-block:: shell

    git clone https://github.com/silx-kit/pyFAI
    cd pyFAI
    pip install build --upgrade
    pip install -r requirements.txt
    python3 -m build --wheel
    pip install --pre --no-index --find-links dist pyFAI


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

    python3 run_tests.py

There are few specific options to run_tests.py:

* ``-x``: Disable all tests relative to the GUI (faster)
* ``-o``: Disable all tests relative to OpenCL (faster)
* ``-c``: Estimates the test-coverage for the project, requires the ``coverage`` package.


**Nota:** to run the test, an internet connection is needed as 160 MB of test images need to be download.
You may have to set the environment variable *http_proxy* and *https_proxy*
according to the networking environment you are in.

Environment variables
---------------------

PyFAI can use a certain number of environment variable to modify its default behavior:

* PYFAI_OPENCL: set to "0" to disable the use of OpenCL, like the ``-o`` option
* PYFAI_DATA: path with gui, calibrant, ...
* PYFAI_TESTIMAGES: path wit test images (if absent, they get downloaded from the internet)
* PYFAI_NO_LOGGING: Disable the configuration of any python logger in interactive mode

Project structure
=================

PyFAI is a library to deal with diffraction images for data reduction.
This chapter describes the project from the computer engineering point of view.

PyFAI is an open source project licensed under the GPL mainly written in Python (v2.6, 2.7) and heavily relying on the
python scientific ecosystem: numpy, scipy and matplotlib. It provides high performances image treatment thanks to cython and
OpenCL... but only a C-compiler is needed to build that.

Programming language
--------------------

PyFAI is a Python project but uses many programming languages:

* 15000 lines of Python (plus 5000 for the test)
* 11000 lines of Cython which are converted into ... C (about a million lines)
* 5000 lines of OpenCL kernels

The OpenCL code has been tested using:

* Nvidia OpenCL v1.1 on Linux, Windows (GPU device)
* Intel OpenCL v1.2 on Linux and Windows (CPU and ACC (Phi) devices)
* AMD OpenCL v1.2 on Linux and Windows (CPU and GPU device)
* Apple OpenCL v1.2 on MacOSX  (CPU and GPU)
* Beignet OpenCL v1.2 on Linux (GPU device)
* Pocl OpenCL v1.2 on Linux (CPU device)

Repository:
-----------

The project is hosted by GitHub:
https://github.com/pyFAI/pyFAI

Which provides the `issue tracker <https://github.com/kif/pyFAI/issues>`_ in addition to Git hosting.
Collaboration is done via Pull-Requests in github's web interface:

Everybody is welcome to `fork the project <https://github.com/pyFAI/pyFAI/fork>`_ and adapt it to his own needs:
CEA-Saclay, Synchrotrons Soleil, Desy and APS have already done so.
Collaboration is encouraged and new developments can be submitted and merged into the main branch
via pull-requests.

Getting help
------------

A mailing list: pyfai@esrf.fr is publicly available.
To subscribe, send an email to sympa@esrf.fr with "subscribe pyfai" as subject.
On this mailing list, you will have information about release of the software, new features available and meet
experts to help you solve issues related to azimuthal integration.
This mailing list is archived and can be consulted at:
`http://www.edna-site.org/lurker <http://www.edna-site.org/lurker/list/pyfai.en.html>`_


Run dependencies
----------------

* Python version 2.6, 2.7 3.2 or newer
* NumPy
* SciPy
* Matplotlib
* FabIO
* h5py (optional)
* pyopencl (optional)
* fftw (optional)
* pymca (optional)
* PyQt4 or PySide (for the graphical user interface)

Build dependencies:
-------------------

In addition to the run dependencies, pyFAI needs a C compiler.
There is an issue with MacOSX (v10.8 onwards) where the default compiler (Xcode5 or 6) switched from gcc 4.2 to clang which
dropped the support for OpenMP (clang v3.5 supports OpenMP under linux but not directly under MacOSX).
Multiple solution exist, pick any of those:

* Install a recent version of GCC (>=4.2)
* Use Xcode without OpenMP, using the --no-openmp flag for setup.py.

C files are generated from cython source and distributed. Cython is only needed for developing new binary modules.
If you want to generate your own C files, make sure your local Cython version supports memory-views (available from Cython v0.17 and newer),
unless your Cython files will not be compiled or used.

Building procedure
------------------

As most of the python projects:
...............................

    python setup.py build install

There are few specific options to setup.py:

* --no-cython: do not use cython (even if present) and use the C source code provided by the development team
* --no-openmp: if you compiler lacks OpenMP support, like Xcode on MacOSX
* --with-testimages: build the source distribution including all test images. Download 200MB of test images to create a self consistent tar-ball.


Test suites
-----------

To run the test an internet connection is needed as 200MB of test images will be downloaded.
............................................................................................
    python setup.py build test

Setting the environment variable http_proxy can be necessary (depending on your network):

..

   export http_proxy=http://proxy.site.org:3128

PyFAI comes with 28 test-suites (163 tests in total) representing a coverage of 65%.
This ensures both non regression over time and ease the distribution under different platforms:
pyFAI runs under Linux, MacOSX and Windows (in each case in 32 and 64 bits).
Test may not pass on computer featuring less than 2GB of memory.

.. csv-table:: Test suite coverage
   :header: "Name", "Stmts", "Miss", "Cover"
   :widths: 50, 8, 8, 8

   "pyFAI.__init__", "10", "3", "70%"
   "pyFAI.azimuthalIntegrator", "1205", "330", "73%"
   "pyFAI.blob_detection", "521", "323", "38%"
   "pyFAI.calibrant ", "196", "69", "65%"
   "pyFAI.detectors ", "993", "248", "75%"
   "pyFAI.geometry", "768", "182", "76%"
   "pyFAI.geometryRefinement", "371", "205", "45%"
   "pyFAI.gui_utils", "53", "33", "38%"
   "pyFAI.io", "421", "189", "55%"
   "pyFAI.massif", "187", "59", "68%"
   "pyFAI.ocl_azim", "307", "91", "70%"
   "pyFAI.ocl_azim_csr", "261", "55", "79%"
   "pyFAI.ocl_azim_lut", "258", "55", "79%"
   "pyFAI.opencl", "151", "44", "71%"
   "pyFAI.peak_picker ", "592", "439", "26%"
   "pyFAI.spline", "329", "220", "33%"
   "pyFAI.units ", "40", "5", "88%"
   "pyFAI.utils ", "664", "300", "55%"

Note that the test coverage tool does not count lines of Cython, nor those of OpenCL

Continuous integration is made by a home-made scripts which checks out the latest release and builds and runs the test every night.
Nightly builds are available for debian6-64 bits in:
http://www.edna-site.org/pub/debian/binary/

List of contributors in code
----------------------------

::

    $ git log  --pretty='%aN##%s' | grep -v 'Merge pull' | grep -Po '^[^#]+' | sort | uniq -c | sort -rn

As of 03/2015:
 * Jérôme Kieffer (ESRF)
 * Aurore Deschildre (ESRF)
 * Frédéric-Emmanuel Picca (Soleil)
 * Giannis Ashiotis (ESRF)
 * Dimitrios Karkoulis (ESRF)
 * Jon Wright (ESRF)
 * Zubair Nawaz (Sesame)
 * Amund Hov (ESRF)
 * Dodogerstlin @github
 * Gunthard Benecke (Desy)
 * Gero Flucke (Desy)


List of other contributors (ideas or code)
------------------------------------------

* Peter Boesecke (geometry)
* Manuel Sanchez del Rio (histogramming)
* Armando Solé (masking widget + PyMca plugin)
* Sebastien Petitdemange (Lima plugin)

List of supporters
------------------

* LinkSCEEM project: porting to OpenCL
* ESRF ID11: Provided manpower in 2012 and 2013 and beamtime
* ESRF ID13: Provided manpower in 2012, 2013, 2014, 2015 and beamtime
* ESRF ID29: provided manpower in 2013 (MX-calibrate)
* ESRF ID02: provided manpower 2014
* ESRF ID15: provide manpower 2015

PyFAI is a library to deal with diffraction images for data reduction.
This chapter describes the project from the computer engineering point of view.

Project structure
=================

PyFAI is an open source project licensed under the GPL mainly written in Python (v2.6 or 2.7) and heavily relying on the
python scientific ecosystem: numpy, scipy and matplotlib. It provides high performances image treatment thanks to cython and
OpenCL... but only a C-compiler is needed to build that.

Programming language
--------------------

PyFAI is a Python project but uses many programming languages:

* 12000 lines of Python (plus 3000 for the test)
* 8000 lines of Cython which are converted into ...
* 400000 lines of C
* 2000 lines of OpenCL kernels

Repository:
-----------

The project is hosted by GitHub:
https://github.com/kif/pyFAI

Which provides the issue tracker in addition to Git hosting.
Collaboration is done via Pull-Requests in github's web interface:

Anybody can fork the project and adapt it to his own needs (people from CEA-saclay or Synchrotron Soleil did it).
If developments are useful to other, new developments can be merged into the main branch.

Run dependencies
----------------

* Python2.6 or python2.7
* NumPy
* SciPy
* Matplotlib
* FabIO
* pyOpencl (optional)
* fftw (optional)

Build dependencies:
-------------------
In addition to the run dependencies, pyFAI needs a C compiler which supports OpenMP (gcc>=4.2, msvc, ...)
There is an issue with MacOSX (v10.8 and newer) where the default compiler switched from gcc 4.2 to clang which 
dropped the support for OpenMP. The best solution looks like to install any recent gcc and use it for compiling pyFAI.

C files are generated from cython source and distributed. Cython is only needed for developing new binary modules.
If you want to generate your own C files, make sure your local cython version supports memory-views (available from Cython v0.17 and newer).

Building procedure
------------------

As most of the python projects:
..

    python setup.py build install

Test suites
-----------

To run the test an internet connection is needed as 200MB of test images will be downloaded. 
..

    python setup.py build test

Setting the environment variable http_proxy can be necessary (depending on your network):

.. 
 
   export http_proxy=http://proxy.site.org:3128
   
PyFAI comes with 23 test-suites (132 tests in total) representing a coverage of 65%.
This ensures both non regression over time and ease the distribution under different platforms:
pyFAI runs under linux, MacOSX and Windows (in each case in 32 and 64 bits)

.. csv-table:: Test suite coverage
   :header: "Name", "Stmts", "Exec", "Cover"
   :widths: 50, 8, 8, 8

   "pyFAI/__init__            ",    "10",   "7",      "70%" 
   "pyFAI/azimuthalIntegrator ",    "1140", "879",    "77%"
   "pyFAI/detectors           ",    "441",  "289",    "65%"
   "pyFAI/blob_detection      ",    "510",  "194",    "38%"
   "pyFAI/calibrant           ",    "161",  "69",     "42%"
   "pyFAI/calibration         ",    "770",  "0",      "0%"
   "pyFAI/detectors           ",    "694",  "548",    "78%"
   "pyFAI/distortion          ",    "454",  "0",      "0%"
   "pyFAI/geometry            ",    "707",   "545",   "77%"
   "pyFAI/geometryRefinement  ",    "371",   "221",   "59%"
   "pyFAI/integrate_widget    ",    "402",   "0",     "0%"
   "pyFAI/io                  ",    "344",   "0",     "0%"
   "pyFAI/ocl_azim            ",    "306",   "215",   "70%"
   "pyFAI/ocl_azim_csr        ",    "242",   "171",   "70%"
   "pyFAI/ocl_azim_csr_dis    ",    "239",   "0",     "0%"
   "pyFAI/ocl_azim_lut        ",    "228",   "198",   "86%"
   "pyFAI/opencl              ",    "140",   "102",   "72%"
   "pyFAI/peakPicker          ",    "694",   "322",   "46%"
   "pyFAI/spline              ",    "327",   "108",   "33%"
   "pyFAI/units               ",    "40",   "35",     "87%"
   "pyFAI/utils               ",    "658",   "377",   "57%"
   "pyFAI/worker              ",    "183",   "0",     "0%"
   "pyFAI                     ",   "9061",   "4280",  "47%"
   "tests                     ",   "1890",   "1563",  "83"
   "TOTAL                     ",   "10951",   "5843", "53%"

Note that the test coverage tool does not count lines of Cython. 


Continuous integration is made by a home-made scripts which checks out the latest release and builds and runs the test every night.
Nightly builds are available for debian6-64 bits in:
http://www.edna-site.org/pub/debian/binary/

List of contributors in code
----------------------------

::

    $ git log  --pretty='%aN##%s' | grep -v 'Merge pull' | grep -Po '^[^#]+' | sort | uniq -c | sort -rn 

As of 06/2014:
 * Jérôme Kieffer (ESRF)
 * Frédéric-Emmanuel Picca (Soleil)
 * Dimitris Karkoulis (ESRF)
 * Aurore Deschildre (ESRF)
 * Giannis Ashiotis (ESRF)
 * Zubair Nawaz (Sesame)
 * Jon Wright (ESRF)
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
* ESRF ID13: Provided manpower in 2012, 2013, 2014 and beamtime
* ESRF ID29: provided manpower in 2013 (MX-calibrate)
* ESRF ID02: provide manpower 2014

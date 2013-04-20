PyFAI is a library to deal with diffraction images for data reduction.
This chapter describes the project from the computer engineering point of view.

Project structure
=================

PyFAI is an open source project licensed under the GPL mainly written in Python (v2.6 or 2.7) and heavily relying on the
python scientific ecosystem: numpy, scipy and matplotlib. It provides high perfromances image treatement thanks to cython and
OpenCL... but only a C-compiler is needed to build that.

Programming language
--------------------

PyFAI is a python project but uses many programming languages:

* 8000 lines of Python (without the tests)
* 3500 lines of cython which are converted into
* 139055 lines of C
* 880 lines of OpenCL kernels

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

C files are generated from cython source and distributed. Cython is only needed for developing new binary modules.

Building procedure
------------------

As most of the python projects:
..

    python setup.py build install

Test suites
-----------

To run the test an internet connection is needed as 200MB of test images will be downloaded
..

    python setup.py build test


PyFAI comes with 15 test-suites (98 tests in total) representing a coverage of 65%.
This ensures both non regression over time and ease the distribution under different platforms:
pyFAI runs under linux, MacOSX and Windows (in each case in 32 and 64 bits)

.. csv-table:: Supported formats
   :header: "Name", "Stmts", "Miss", "Cover"
   :widths: 50, 8, 8, 8

   "pyFAI/__init__",                 "9",     "3", "67%"
   "pyFAI/azimuthalIntegrator",   "1097",   "329", "70%"
   "pyFAI/detectors",              "271",   "105", "61%"
   "pyFAI/geometry",               "672",   "166", "75%"
   "pyFAI/geometryRefinement",     "353",   "198", "44%"
   "pyFAI/ocl_azim",               "304",    "90", "70%"
   "pyFAI/ocl_azim_lut",           "196",    "29", "85%"
   "pyFAI/opencl",                 "134",    "42", "69%"
   "pyFAI/peakPicker",             "609",   "388", "36%"
   "pyFAI/spline",                 "324",   "217", "33%"
   "pyFAI/units",                   "36",     "2", "94%"
   "pyFAI/utils",                  "494",   "196", "60%"
   "testAzimuthalIntegrator",      "189",    "75", "60%"
   "testBilinear",                  "40",     "4", "90%"
   "testDistortion",                "47",     "3", "94%"
   "testExport",                    "69",     "7", "90%"
   "testFlat",                      "88",     "9", "90%"
   "testGeometry",                  "50",     "4", "92%"
   "testGeometryRefinement",        "53",     "3", "94%"
   "testHistogram",                "153",    "14", "91%"
   "testIntegrate",                "136",    "10", "93%"
   "testMask",                      "85",    "21", "75%"
   "testOpenCL",                    "89",     "9", "90%"
   "testPeakPicking",               "70",     "6", "91%"
   "testPolarization",              "54",    "21", "61%"
   "testSaxs",                     "102",    "29", "72%"
   "testUtils",                     "80",     "4", "95%"
   "test_all",                      "47",     "1", "98%"
   "utilstest",                    "139",    "88", "37%"
   "TOTAL",                        "5990",  "2073",   "65%"



Continuous integration is made by a home-made scripts which checks out the latest release and builds and runs the test every night.
Nightly builds are available for debian6-64 bits in:
http://www.edna-site.org/pub/debian/binary/

List of contributors
--------------------

As of 04/2013 (number of commits in parenthesis):

* Jérôme Kieffer (656)
* Dimitris Karkoulis (22)
* Frédéric-Emmanuel Picca (21)
* Jonathan Wright (6)
* Amund Hov (1)

List of other contributors (ideas or code)
------------------------------------------

* Peter Boesecke (geometry)
* Manuel Sanchez del Rio (histogramming)
* Armando Solé (masking widget + PyMca plugin)
* Sebastien Petitdemange (Lima plugin)

List of supporters
------------------

* LinkSCEEM project: initial porting to OpenCL
* ESRF ID11: Provided manpower in 2012 and 2013 and beamtime
* ESRF ID13: Provided manpower in 2012 and 2013 and beamtime
* ESRF ID29: will provide manpower in  2013

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

* 10200 lines of Python (without the tests)
* 3500 lines of cython which are converted into
* 252000 lines of C
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

To run the test an internet connection is needed as 200MB of test images will be downloaded. 
..

    python setup.py build test

Setting the environment variable http_proxy can be necessary (depending on your network):

.. 
 
   export http_proxy=http://proxy.site.org:3128
   
PyFAI comes with 19 test-suites (109 tests in total) representing a coverage of 65%.
This ensures both non regression over time and ease the distribution under different platforms:
pyFAI runs under linux, MacOSX and Windows (in each case in 32 and 64 bits)

.. csv-table:: Test suite coverage
   :header: "Name", "Stmts", "Exec", "Cover"
   :widths: 50, 8, 8, 8

   "pyFAI/__init__            ",    "10",   "7",    "70%" 
   "pyFAI/azimuthalIntegrator ",    "934",   "687",    "73%"
   "pyFAI/detectors           ",    "441",   "289",    "65%"
   "pyFAI/geometry            ",    "718",   "529",    "73%"
   "pyFAI/geometryRefinement  ",    "361",   "162",    "44%"
   "pyFAI/ocl_azim            ",    "306",   "215",    "70%"
   "pyFAI/ocl_azim_lut        ",    "197",   "166",    "84%"
   "pyFAI/opencl              ",    "140",   "102",    "72%"
   "pyFAI/peakPicker          ",    "615",   "221",    "35%"
   "pyFAI/spline              ",    "327",   "108",    "33%"
   "pyFAI/units               ",    "40",   "35",    "87%"
   "pyFAI/utils               ",    "596",   "363",    "60%"
   "testAzimuthalIntegrator   ",    "233",   "156",    "66%"
   "testBilinear              ",    "39",   "35",    "89%"
   "testDetector              ",    "35",   "32",    "91%"
   "testDistortion            ",    "52",   "46",    "88%"
   "testExport                ",    "68",   "61",    "89%"
   "testFlat                  ",    "89",   "85",    "95%"
   "testGeometry              ",    "86",   "82",    "95%"
   "testGeometryRefinement    ",    "53",   "50",    "94%"
   "testHistogram             ",    "152",   "138",    "90%"
   "testIntegrate             ",    "135",   "125",    "92%"
   "testMask                  ",    "133",   "106",    "79%"
   "testOpenCL                ",    "96",   "83",    "86%"
   "testPeakPicking           ",    "83",   "75",    "90%"
   "testPolarization          ",    "53",   "32",    "60%"
   "testSaxs                  ",    "101",   "72",    "71%"
   "testUtils                 ",    "83",   "76",    "91%"
   "test_all                  ",    "48",   "47",    "97%"
   "utilstest                 ",    "168",   "84",    "50%"
   "TOTAL                     ",   "6392",   "4269",    "66%"



Continuous integration is made by a home-made scripts which checks out the latest release and builds and runs the test every night.
Nightly builds are available for debian6-64 bits in:
http://www.edna-site.org/pub/debian/binary/

List of contributors in code
----------------------------

::

    $ git log  --pretty='%aN##%s' | grep -v 'Merge pull' | grep -Po '^[^#]+' | sort | uniq -c | sort -rn 

As of 01/2014:
 * Jérôme Kieffer (ESRF)
 * Dimitris Karkoulis (ESRF)
 * Jon Wright (ESRF)
 * Frédéric-Emmanuel Picca (Soleil)
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

* LinkSCEEM project: initial porting to OpenCL
* ESRF ID11: Provided manpower in 2012 and 2013 and beamtime
* ESRF ID13: Provided manpower in 2012 and 2013 and beamtime
* ESRF ID29: provided manpower in 2013 (MX-calibrate)
* ESRF ID02: 2014

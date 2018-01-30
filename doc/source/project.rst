:Author: Jérôme Kieffer
:Date: 09/05/2017
:Keywords: Project management description
:Target: developers

Project
=======

PyFAI is a library to deal with diffraction images for data reduction.
This chapter describes the project from the computer engineering point of view.

PyFAI is an open source project licensed under the MIT license (previously
under GPL) mainly written in Python (v2.7, 3.4 or newer).
It is managed by the Silx team and is heavily relying on the
Python scientific ecosystem: numpy, scipy and matplotlib.
It provides high performances image treatment thanks to Cython and
OpenCL... but only a C-compiler is needed to build it.

Programming language
--------------------

PyFAI is a Python project but uses many programming languages:

* 23000 lines of Python (plus 5000 for the test)
* 8000 lines of Cython which are converted into ... C (about half a million lines)
* 5000 lines of OpenCL kernels

The OpenCL code has been tested using:

* Nvidia OpenCL v1.1 and v1.2 on Linux, Windows (GPU device)
* Intel OpenCL v1.2 on Linux and Windows (CPU and ACC (Phi) devices)
* AMD OpenCL v1.2 on Linux and Windows (CPU and GPU device)
* Apple OpenCL v1.2 on MacOSX  (CPU and GPU)
* Beignet OpenCL v1.2 on Linux (GPU device)
* Pocl OpenCL v1.2 on Linux (CPU device)

Repository
----------

The project is hosted on GitHub:
https://github.com/silx-kit/pyFAI

Which provides the `issue tracker <https://github.com/silx-kit/pyFAI/issues>`_ in
addition to Git hosting.
Collaboration is done via Pull-Requests in github's web interface:

Everybody is welcome to `fork the project <https://github.com/silx-kit/pyFAI/fork>`_
and adapt it to his own needs:
CEA-Saclay, Synchrotrons Soleil, Desy and APS have already done so.
Collaboration is encouraged and new developments can be submitted and merged
into the main branch via pull-requests.

Getting help
------------

A mailing list, pyfai@esrf.fr, is publicly available.
I t is the best place to ask your questions: the author and many advanced users
are there and willing to help you.
To subscribe to this mailing list, send an email to
`pyfai-subscribe@esrf.fr <mailto:pyfai-subscribe@esrf.fr>`_.

On this mailing list, you will have information about release of the software,
new features available and meet experts to help you solve issues related to
azimuthal integration and diffraction in general.
It also provides a knowledge-base of most frequently asked question which needs
to be integrated into this documentation.

The volume of email on the list remains low, so you can subscribe without being
too much spammed. As the mailing list is archived, and can be consulted at:
`http://www.edna-site.org/lurker <http://www.edna-site.org/lurker/list/pyfai.en.html>`_,
you can also check the volume of the list.

If you think you are facing a bug, the best is to
`create a new issue on the GitHub page <https://github.com/silx-kit/pyFAI/issues>`_
(you will need a GitHub account for that).

Direct contact with authors is discouraged:
pyFAI is open source software that we develop to aid the research
community in doing what they do best.
While we do enjoy doing this, we
would not be able to dream of spending nearly as much time with
pyFAI as we do if it wasn't for your support.
Interest of the scientific community (via a lively mailing list) and citation in
scientific publication for our `software <http://dx.doi.org/10.1107/S1600576715004306>`_
is one of the main criterion for ESRF management when deciding if they should
continue funding development.

Run dependencies
----------------

* Python version 2.7, 3.4, 3.5, 3.6
* NumPy
* SciPy
* Matplotlib
* FabIO
* h5py
* pyopencl (optional)
* PyQt4 or PySide (for the graphical user interface)
* Silx

Build dependencies
------------------

In addition to the run dependencies, pyFAI needs a C compiler.

There is an issue with MacOS (v10.8 onwards) where the default compiler
(Xcode5 or newer) dropped the support for OpenMP.
On this platform pyFAI will enforce the generation of C-files from Cython sources
(making Cython a build-dependency on MacOS) without support of OpenMP
(options: --no-openmp --force-cython).
On OSX, an alternative is to install a recent version of GCC (>=4.2) and to use
it for compiling pyFAI.
The options to be used then are * --force-cython --openmp*.

Otherwise, C files are which are provided with pyFAI sources are directly useable
and Cython is only needed for developing new binary modules.
If you want to generate your own C files, make sure your local Cython version
is recent enough (v0.21 and newer),
unless your Cython files will not be translated to C, nor used.

Building procedure
------------------

As most of the Python projects:
...............................

.. code-block:: shell

    python setup.py build bdist_wheel
    pip install dist/pyFAI-0.14.0*.whl --upgrade


There are few specific options to setup.py:

* --no-cython: do not use cython (even if present) and use the C source code
  provided by the development team
* --force-cython: enforce the regeneration of all C-files from cython sources
* --no-openmp: if you compiler lacks OpenMP support, like Xcode on MacOS.
* --openmp: enforce the use of OpenMP.
* --with-testimages: build the source distribution including all test images.
  Downloads 200MB of test images to create a self consistent tar-ball.


Test suites
-----------

To test the installed version of pyFAI:

.. code-block:: shell

    python run_test.py

or from python:

.. code-block:: python

    import pyFAI
    pyFAI.tests()


Some **Warning** messages are normal as the test procedure also
tests corner cases.

To run the test an internet connection is needed as 200MB of test images will be downloaded.
............................................................................................

Setting the environment variable http_proxy can be necessary (depending on your network):

.. code-block:: shell

   export http_proxy=http://proxy.site.org:3128

Especially at ESRF, the configuration of the network proxy can be obtained
by asking at the helpdesk: helpdesk@esrf.fr

To test the development version (built but not yet installed):

.. code-block:: shell

    python setup.py build test

or

.. code-block:: shell

    python setup.py build
    python run_test.py -i


PyFAI comes with 40 test-suites (338 tests in total) representing a coverage of 60%.
This ensures both non regression over time and ease the distribution under different platforms:
pyFAI runs under Linux, MacOSX and Windows (in each case in 32 and 64 bits).
Test may not pass on computer featuring less than 2GB of memory or 32 bit architectures.

**Note:**: The test coverage tool does not count lines of Cython, nor those of OpenCL.
`Anyway test coverage is a tool for developers about possible weaknesses of their
code and not a management indicator. <http://www.exampler.com/testing-com/writings/coverage.pdf>`_

.. toctree::
   :maxdepth: 1

   coverage


Continuous integration
----------------------

This software engineering practice consists in merging all developer working copies
to a shared mainline several times a day and build the whole project for multiple
targets.

On Debian 8 - Jessie
....................
Continuous integration is made by a home-made scripts which checks out the latest release and builds and runs the test every night.
`Nightly builds <http://www.silx.org/pub/debian/binary/>`_ are available for debian8-64 bits. To install them:

.. code-block:: shell

	sudo apt-get update
	sudo apt-get install pyfai

You have to accept non-signed packages because they are automatically built.

In addition some "cloud-based" tools are used to ensure a larger coverage of operating systems/environment.
They rely on a `"local wheelhouse" <http://www.silx.org/pub/wheelhouse/>`_.

Those wheels are optimized for Travis-CI, AppVeyor and ReadTheDocs, using them is not recommended as your Python configuration may differ
(and those libraries could even crash your system).

Linux
.....


`Travis provides continuous integration on Linux <https://travis-ci.org/silx-kit/pyFAI>`_,
64 bits computer with Python 2.7, 3.4 and 3.5.

The builds cannot yet be retrieved with Travis-CI, but manylinux-wheels are on the radar.

AppVeyor
........

`AppVeyor provides continuous integration on Windows <https://ci.appveyor.com/project/ESRF/pyfai>`_, 64 bits computer with Python 2.7 and 3.4.
Successful builds provide installers for pyFAI as *wheels* and *msi*, they are anonymously available as *artifacts*.
Due to the limitation of AppVeyor's build system, those installers have openMP disabled.

List of contributors in code
----------------------------

.. code-block:: shell

    git log  --pretty='%aN##%s' | grep -v 'Merge pull' | grep -Po '^[^#]+' | sort | uniq -c | sort -rn

As of 01/2018:
 * Jérôme Kieffer (ESRF)
 * Valentin Valls (ESRF)
 * Frédéric-Emmanuel Picca (Soleil)
 * Aurore Deschildre (ESRF)
 * Giannis Ashiotis (ESRF)
 * Dimitrios Karkoulis (ESRF)
 * Jon Wright (ESRF)
 * Zubair Nawaz (Sesame)
 * Dodogerstlin @github
 * Vadim Dyadkin (ESRF/SNBL)
 * Gunthard Benecke (Desy)
 * Gero Flucke (Desy)
 * Christopher J. Wright (Columbia University)
 * Sigmund Neher (GWDG)
 * Wout De Nolf (ESRF)
 * Bertrand Faure (Xenocs)
 * Thomas Vincent (ESRF)
 * Amund Hov (ESRF)

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
* ESRF ID13: Provided manpower in 2012, 2013, 2014, 2015, 2016 and beamtime
* ESRF ID29: provided manpower in 2013 (MX-calibrate)
* ESRF ID02: provided manpower 2014, 2016
* ESRF ID15: provide manpower 2015, 2016, 2017
* ESRF ID21: provide manpower 2015, 2016
* ESRF ID31: provide manpower 2016, 2017

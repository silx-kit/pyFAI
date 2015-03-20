
Author: Jérôme Kieffer

Date: 20/03/2015

Keywords: Installation procedure

Target: System administrators


Installation procedure on Windows
=================================

PyFAI is a Python library. Even if you are only interested in some tool like pyFAI-calib or pyFAI-integrate,
you need to install the complete library (for now).
This is usually performed in 3 steps: install Python, the scientific python stack and finally the pyFAI itself.

Get Python
----------

Unlike on Unix computers, Python is not available by default on Windows computers.
We recommend you to install the 64 bit version of Python from http://python.org, preferably the latest version from the 2.7 series.
Any version between 2.6, 2.7, 3.2, 3.3 or 3.4 should be OK but 2.7 is the most tested.

The 64bits version is strongly advised if your hardware and operating system supports it, as the 32 bits versions is
limited to 2GB of memory, hence unable to treat large images (4096 x 4096).
The test suite is not passing on Windows 32 bits due to the limited amount of memory available to the Python process,
nevertheless, pyFAI is running on Winodws32 bits (but not as well).

Alternative Scientific Python stacks exists, like Enthought Python Distribution, Canopy, Anaconda, PythonXY
or WinPython. They all offer most of the scientific packages already installed which makes the installation of
dependencies much easier. On the other hand, they all offer different packaging system and we cannot support all
of them. Moreover, distribution from Enthought and Continuum are not free so you should be able to get support
from those companies.

Install PIP
-----------

PIP is the package management system for Python, it connects to http://pypi.python.org,
download and install software packages from there.

PIP has revolutionize the way Python libraries are installed as it is able to select the right build for your system, or compile from the sources (Which could be tricky).

To install it, download:
https://bootstrap.pypa.io/get-pip.py
and run it:

::
   python get-pip.py

Assuming python.exe is already in your PATH.

Install the scientific stack
----------------------------

The strict dependencies for pyFAI are:

* NumPy
* SciPy
* matplotlib
* FabIO

Recommanded dependencies are:

* cython
* h5py
* pyopencl
* PyQt4
* pymca
* rfoo
* pyfftw3
* lxml

The ways

Using PIP
.........

Most of the dependencies are available via PIP:

::
   pip install numpy
   pip install scipy
   pip install matplotlib
   pip install fabio
   pip install PyQt4

Note that numpy/scipy/matplotlib are already installed in most "Scientific Python distribution"

If one of the dependency is not available as a Wheel (i.e. binary package) but only as a source package, a compiler will be required.
In this case, see the next paragraph
The generalization of Wheel packages should help and the installation of binary modules should become easier.

Using Christoph Gohlke repository
.................................

Christoph Gohlke, Laboratory for Fluorescence Dynamics, University of California, Irvine.
He is maintaining a repository for various Python extension (actually, all we need :) for Windows.
Check twice the Python version and the Windows version (win32 or win_amd64) before downloading and installing them

http://www.lfd.uci.edu/~gohlke/pythonlibs/

Moreover the libraries he provides are linked against the MKL library from Intel which
makes his packages faster then what you would get by simply recompiling them.

Install pyFAI via PIP
---------------------

The latest stable release of pyFAI should also be PIP-installable (starting at version 0.11)

::
   pip install pyFAI



Install pyFAI from sources
==========================

The sources of pyFAI are available at https://github.com/pyFAI/pyFAI/releases

In addition to the Python interpreter, you will need the C compiler compatible with your Python interpreter, for example you can find the one for Python2.7 at:
http://aka.ms/vcpython27

To upgrade the C-code in pyFAI, one needs in addition Cython:

::
   pip install cython



Author: Jérôme Kieffer

Date: 16/01/2015

Keywords: Installation procedure

Target: System administrators


Installation procedure on Windows
=================================

Get Python
----------

Unlike on Unix computers, Python is not available by default on windows computers.
We recommand you to install Python from http://python.org. The latest version from the 2.7 series
is currently recommended (But any version between 2.6 and 2.7, 3.2, 3.3 or 3.4 should be OK).

The 64bits version is suggested as the 32 bits versions is limited to 2GB of memory, hence unable to treat large images (4096 x 4096).

Alternative Scientific Python stacks exists, like Enthought Python Distribution, Canopy, Anaconda, PythonXY
or WinPython. They all offer most of the scientific packages already installed.

Install PIP
-----------

PIP is the package management system for Python, it connects to http://pypi.python.org,
download and install software packages from there.

To install it, download:
https://bootstrap.pypa.io/get-pip.py
and run it:

::
   python get-pip.py

Install the scientific stack
----------------------------

To install the scientific stack, we will massively use pip:

::
   pip install numpy
   pip install scipy
   pip install matplotlib
   pip install fabio
   pip install PyQt4

Note that numpy/scipy/matplotlib are already installed in most "Scientific Python distribution"


Install pyFAI via PIP
---------------------

The latest stable release of pyFAI is also PIP-installable

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


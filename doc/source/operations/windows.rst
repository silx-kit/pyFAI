..

  Author: Jérôme Kieffer
  Date: 27/10/2015
  Keywords: Installation procedure
  Target: System administrators


Installation procedure on Windows
=================================

PyFAI is a Python library. Even if you are only interested in some tool like pyFAI-calib or pyFAI-integrate,
you need to install the complete library (for now).
This is usually performed in 3 steps: install Python, the scientific Python stack and finally the pyFAI itself.

Get Python
----------

Unlike on Unix computers, Python is not available by default on Windows computers.
We recommend you to install the 64 bit version of `Python <http://python.org>`_, preferably the latest 64 bits version from the
`2.7 series <https://www.python.org/downloads/release/python-2710/`_>.
Any version between 2.6, 2.7, 3.2, 3.3, 3.4 and 3.5 is tested but 2.7 is the most supported.

The 64 bits version is strongly advised if your hardware and operating system supports it, as the 32 bits versions is
limited to 2 GB of memory, hence unable to treat large images (4096 x 4096).
The test suite is not passing on Windows 32 bits due to the limited amount of memory available to the Python process,
nevertheless, pyFAI is running on Winodws 32 bits (but not as well).

Alternative Scientific Python stacks exists, like `Enthought Python Distribution<https://www.enthought.com/products/epd/>`_ ,
`Canopy <https://www.enthought.com/products/canopy/>`_, `Anaconda <https://www.continuum.io/downloads>`_,
`PythonXY <https://python-xy.github.io/>`_ or `WinPython <http://winpython.github.io/>`_.
They all offer most of the scientific packages already installed which makes the installation of
dependencies much easier. On the other hand, they all offer different packaging system and we cannot support all
of them. Moreover, distribution from *Enthought* and *Continuum* are not free so you should be able to get support
from those companies.

**Nota:** each flavor of those Python distribution is incompatible with any other due to change in compiler or Python
compilation options. Mixing them is really looking for trouble.
If you want an advice on which scientific python distribution for Windows to use,
I would recommend `WinPython <http://winpython.github.io/>`_.

Install PIP
-----------

**PIP** is the package management system for Python, it connects to http://pypi.python.org,
download and install software packages from there.


PIP has revolutionize the way Python libraries are installed as it is able to select the right build for your system, or compile from the sources,
which could be extremely tricky.
If you installed python 2.7.10 or 3.4, PIP is already installed.
If **pip** is not yet installed, download `get_pip <https://bootstrap.pypa.io/get-pip.py>`_ and run it:

.. code::

   python get-pip.py

Assuming python.exe is already in your PATH.

**Nota:**  Because PIP connects to the network, the *http_proxy* and *https_proxy* environment variable may need to be set-up properly.
At ESRF, please get in contact with the hotline (24-24) to get those information.


Install the scientific stack
----------------------------

The strict dependencies for pyFAI are:

* NumPy
* SciPy
* matplotlib
* FabIO
* h5py

Recommended dependencies are:

* cython
* h5py
* pyopencl
* PyQt4
* pymca
* rfoo
* pyfftw3
* lxml

Using PIP
.........

Most of the dependencies are available via PIP::

   pip install numpy
   pip install scipy
   pip install matplotlib
   pip install fabio
   pip install PyQt4

Note that numpy/scipy/matplotlib are already installed in most "Scientific Python distribution"

If one of the dependency is not available as a Wheel (i.e. binary package) but only as a source package, a compiler will be required.
In this case, see the next paragraph
The generalization of Wheel packages should help and the installation of binary modules should become easier.

This requires a network access and correct proxy settings. For example at ESRF, one will need to set-up the environment for the proxy like this::

    set http_proxy=http://proxy.esrf.fr:3128
    set https_proxy=http://proxy.esrf.fr:3128  

One day, our beloved computing service will put in place a `transparent proxy <http://en.wikipedia.org/wiki/Proxy_server#Transparent_proxy>`_, one day, maybe.

Using Christoph Gohlke repository
.................................

Christoph Gohlke, Laboratory for Fluorescence Dynamics, University of California, Irvine.
He is maintaining a repository for various Python extension (actually, all we need :) for Windows.
Check twice the Python version and the Windows version (win32 or win_amd64) before downloading and installing them

http://www.lfd.uci.edu/~gohlke/pythonlibs/

Moreover the libraries he provides are linked against the MKL library from Intel which
makes his packages faster then what you would get by simply recompiling them.

Christopher now provides packages as wheels. To install them use PIP::

    pip install numpy*.whl

Install pyFAI via PIP
---------------------

The latest stable release of pyFAI should also be PIP-installable (starting at version 0.10.3)::

   pip install pyFAI



Install pyFAI from sources
==========================

The sources of pyFAI are available at https://github.com/pyFAI/pyFAI/releases

In addition to the Python interpreter, you will need the C compiler compatible with your Python interpreter, for example you can find the one for Python2.7 at:
http://aka.ms/vcpython27

To upgrade the C-code in pyFAI, one needs in addition Cython::

   pip install cython


Troubleshooting
===============

This section contains some tips on windows.

Side-by-side error
..................
When starting pyFAI you get a side-by-side error like::

    ImportError: DLL load failed: The application has failed to start because its 
    side-by-side configuration is incorrect. Please see the application event log or 
    use the command-line sxstrace.exe tool for more detail.

This means you are using a version of pyFAI which was compiled using the MSVC compiler (maybe not on your computer)
but the Microsoft Visual C++ Redistributable Package is missing. For Python2.7, 64bits the missing DLL can be downloaded from:: 
    http://www.microsoft.com/en-us/download/confirmation.aspx?id=2092

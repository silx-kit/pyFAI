:Author: Jérôme Kieffer
:Date: 31/05/2015
:Keywords: Installation procedure
:Target: System administrators under windows


Installation procedure on Windows
=================================

PyFAI is a Python library. Even if you are only interested in some tool like
pyFAI-calib or pyFAI-integrate, you need to install the complete library (for now).
This is usually performed in 3 steps:

#. install Python,
#. install the scientific Python stack
#. install pyFAI itself.

Get Python
----------

Unlike on Unix computers, Python is not available by default on Windows computers.
We recommend you to install the 64 bit version of `Python <http://python.org>`_,
preferably the latest 64 bits version from the
`2.7 series <https://www.python.org/downloads/release/python-2710/>`_.
But Python 3.4 and 3.5 are also very good candidates.
Python 2.6, 3.2 and 3.3 are no more supported since pyFAI v0.12.

The 64 bits version is strongly advised if your hardware and operating system
supports it, as the 32 bits versions is
limited to 2 GB of memory, hence unable to treat large images (like 4096 x 4096).
The test suite is not passing on Windows 32 bits due to the limited amount of
memory available to the Python process,
nevertheless, pyFAI is running on Winodws 32 bits (but not as well).

Alternative Scientific Python stacks exists, like
`Enthought Python Distribution <https://www.enthought.com/products/epd/>`_ ,
`Canopy <https://www.enthought.com/products/canopy/>`_,
`Anaconda <https://www.continuum.io/downloads>`_,
`PythonXY <https://python-xy.github.io/>`_ or
`WinPython <http://winpython.github.io/>`_.
They all offer most of the scientific packages already installed which makes
the installation of
dependencies much easier.
On the other hand, they all offer different packaging system and we cannot
support all of them.
Moreover, distribution from *Enthought* and *Continuum* are not free so you
should be able to get support from those companies.

**Nota:** any flavor of those Python distribution is probably incompatible with
any other due to change in compiler or Python compilation options.
Mixing them is really looking for trouble, hence strongly discouraged.
If you want an advice on which scientific python distribution for Windows to use,
I would recommend `WinPython <http://winpython.github.io/>`_.

Install PIP
-----------

**PIP** is the package management system for Python, it connects to
`the Python Package Index <http://pypi.python.org>`_,
download and install software packages from there.

PIP has revolutionize the way Python libraries are installed as it is able to
select the right build for your system, or compile them from the sources,
which could be extremely tricky otherwise.
If you installed python 2.7.10, 3.4 or newer, PIP is already installed.
If **pip** is not yet installed on your system, download
`get_pip.py <https://bootstrap.pypa.io/get-pip.py>`_ and run it:

.. code::

   python get-pip.py

Assuming python.exe is already in your PATH.

**Nota:**  Because PIP connects to the network, the *http_proxy* and *https_proxy*
environment variable may need to be set-up properly.
At ESRF, please get in contact with the hotline (24-24) to retrive those information.


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

Most of the dependencies are available via PIP:

.. code::

   pip install numpy --upgrade
   pip install scipy --upgrade
   pip install matplotlib --upgrade
   pip install fabio --upgrade
   pip install PyQt4 --upgrade

Note that numpy/scipy/matplotlib are already installed in most "Scientific Python distribution"

If one of the dependency is not available as a Wheel (i.e. binary package) but
only as a source package, a compiler will be required.
In this case, see the next paragraph.
The generalization of Wheel packages should help and the installation of binary
modules should become easier.

**Nota:** This requires a network access and correct proxy settings.
At ESRF, please get in contact with the hotline (24-24) to retrive those information.

.. code::

    set http_proxy=http://proxy.site.com:3128
    set https_proxy=http://proxy.site.com:3128


Using Christoph Gohlke repository
.................................

Christoph Gohlke is a researcher at Laboratory for Fluorescence Dynamics, University of California, Irvine.
He is maintaining a `large repository Python extension <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_ (actually, all we need :) for Windows.
Check twice the Python version and the Windows version (win32 or win_amd64) before downloading.

Moreover the libraries he provides are linked against the MKL library from Intel which
makes his packages faster then what you would get by simply recompiling them.

Christoph now provides packages as wheels.
To install them, download the wheels and use PIP:

.. code::

    pip install numpy*.whl

Alternatively, you can use the wheelhouse of the silx project:

.. code::

   pip install --trusted-host www.silx.org --find-links http://www.silx.org/pub/wheelhouse/ numpy scipy matplotlib fabio PyQt4

Install pyFAI via PIP
---------------------

The latest stable release of pyFAI should also be PIP-installable (starting at version 0.10.3):

.. code::

   pip install pyFAI --upgrade



Install pyFAI from sources
--------------------------

The sources of pyFAI are available at https://github.com/pyFAI/pyFAI/releases
the development is performed on https://github.com/kif/pyFAI

In addition to the Python interpreter, you will need *the* C compiler compatible
with your Python interpreter, for example you can find the one for Python2.7 at:
http://aka.ms/vcpython27

To upgrade the C-code in pyFAI, one needs in addition Cython:

.. code::

   pip install cython --upgrade
   python setup.py bdist_wheel
   pip install --pre --no-index --find-links dist/ pyFAI

Troubleshooting
---------------

This section contains some tips on windows.

Side-by-side error
..................
When starting pyFAI you get a side-by-side error like::

    ImportError: DLL load failed: The application has failed to start because its
    side-by-side configuration is incorrect. Please see the application event log or
    use the command-line sxstrace.exe tool for more detail.

This means you are using a version of pyFAI which was compiled using the MSVC compiler
(maybe not on your computer) but the Microsoft Visual C++ Redistributable Package is missing.
For Python2.7, 64bits the missing DLL can be downloaded from::
    http://www.microsoft.com/en-us/download/confirmation.aspx?id=2092


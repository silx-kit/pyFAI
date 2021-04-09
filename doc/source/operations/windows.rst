:Author: Jérôme Kieffer
:Date: 07/01/2021
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
`Python 3 series 64 bits <https://www.python.org/downloads/windows/>`_.

The 64 bits version is strongly advised if your hardware and operating system
supports it, as the 32 bits versions is
limited to 2 GB of memory, hence unable to treat large images (like 4096 x 4096).
The test suite is not passing on Windows 32 bits due to the limited amount of
memory available to the Python process.
PyFAI has not been tested under Windows 32 bits for a while now, this configuration is unsupported.

Alternative Scientific Python stacks exists, like
`Enthought Python Distribution <https://www.enthought.com>`_ ,
`Anaconda <https://www.anaconda.com/>`_,
`WinPython <http://winpython.github.io/>`_.
They all offer most of the scientific packages already installed which makes
the installation of dependencies much easier.
On the other hand, they all offer different packaging system and we cannot
support all of them.
Moreover, distribution from *Enthought* and *Anaconda* are not free so you
should be able to get support from those companies.

**Nota:** any flavor of those Python distribution is probably incompatible with
any other due to change in compiler or Python compilation options.
Mixing them is really looking for trouble, hence strongly discouraged.

Install PIP
-----------

**PIP** is the package management system for Python, it connects to
`the Python Package Index <http://pypi.python.org>`_,
download and install software packages from there.

PIP has revolutionize the way Python libraries are installed as it is able to
select the right build for your system, or compile them from the sources,
which could be extremely tricky otherwise.
If you installed Python compatible with pyFAI (3.6 or newer), PIP is already installed.
From now on, one expects *python.exe* to be in your **PATH**.

**Nota:**  Because PIP connects to the network, the *http_proxy* and *https_proxy*
environment variable may need to be set-up properly.
At ESRF, this is no more needed.


Install pyFAI via PIP
---------------------

.. code-block:: shell

   pip install pyFAI[gui] --upgrade

This will install:

* NumPy
* SciPy
* matplotlib
* FabIO
* h5py
* silx
* h5py
* PyQt5


Using Christoph Gohlke repository
.................................

Christoph Gohlke is a researcher at Laboratory for Fluorescence Dynamics, University of California, Irvine.
He is maintaining a `large repository Python extension <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_ (actually, all we need :) for Windows.
Check twice the Python version and the Windows version (win32 or win_amd64) before downloading.

Moreover the libraries he provides are linked against the MKL library from Intel which
makes his packages faster then what you would get by simply recompiling them.

Christoph now provides packages as wheels.
To install them, download the wheels and use PIP:

.. code-block:: shell

    pip install numpy*.whl

Alternatively, you can use the wheelhouse of the silx project:

.. code-block:: shell

   pip install --trusted-host www.silx.org --find-links http://www.silx.org/pub/wheelhouse/ numpy scipy matplotlib fabio PyQt5


Install pyFAI from sources
--------------------------

The sources of pyFAI are available at https://github.com/silx-kit/pyFAI/releases
the development is performed on https://github.com/silx-kit/pyFAI

In addition to the Python interpreter, you will need `*the* C compiler compatible
with your Python interpreter <https://wiki.python.org/moin/WindowsCompilers>`_.

To upgrade the C-code in pyFAI, one needs in addition Cython:

.. code-block:: shell

   pip install -r requirements.txt --upgrade
   python setup.py bdist_wheel
   pip install --upgrade --pre --no-index --find-links dist/ pyFAI 

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

:Author: Jérôme Kieffer
:Date: 28/01/2025
:Keywords: Installation procedure
:Target: System administrators under windows


Installation procedure on Windows
=================================

PyFAI is a Python library. Even if you are only interested in some tool like
pyFAI-calib3 or pyFAI-integrate, you need to install the complete library (for now).
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

An alternative is to use `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html>`_ packaging tool
which provides Python with the packaging tool.


Install pyFAI via PIP
---------------------

.. code-block:: shell

   pip install pyFAI[gui] --upgrade

This will install:

* NumPy
* SciPy
* Matplotlib
* FabIO
* h5py
* silx
* h5py
* PyQt6


Install pyFAI from sources
--------------------------

The sources of pyFAI are available at https://github.com/silx-kit/pyFAI

In addition to the Python interpreter, you will need *the* `C compiler compatible
with your Python interpreter <https://wiki.python.org/moin/WindowsCompilers>`_.

To upgrade the C/C++-code in pyFAI, one needs in addition Cython:

.. code-block:: shell

   pip install -r requirements.txt --upgrade
   pip install . --upgrade

Troubleshooting
---------------

This section contains some tips specific to Windows.

About Microsoft Visual Studio Compiler
......................................

PyFAI contains some extensions (binary modules) making use of assembly code ...
and the assembly used by ``MSVC`` differs from the une used by ``gcc`` or ``clang``,
thus those extensions won't build with alternative compilers.

If you wish to use pyFAI under Windows with an alternative compiler,
please open an issue in the GitHub tracker.
Solutions exist but they all represent a substential amount of work !


Side-by-side error
..................

When starting pyFAI you get a side-by-side error like::

    ImportError: DLL load failed: The application has failed to start because its
    side-by-side configuration is incorrect. Please see the application event log or
    use the command-line sxstrace.exe tool for more detail.

This means you are using a version of pyFAI which was compiled using the MSVC compiler
(maybe not on your computer) but the Microsoft Visual C++ Redistributable Package is missing.

:Author: Jérôme Kieffer
:Date: 31/01/2019
:Keywords: Installation procedure on MacOSX
:Target: System administrators

Installation procedure on MacOSX
================================

Install Python3:
----------------

To install pyFAI on an *Apple* computer you will need a scientific Python3 stack.
MacOSX provides by default Python2.7, you will need to install a recent version
of Python3 (3.5 at least, 3.7 recommanded but newer version should be OK).
Those distribution are available as *dmg* images from:
https://www.python.org/downloads/mac-osx/

After downloading, move the app into the *Applications* folder. 

Using a virtual environment:
----------------------------

.. code-block:: shell

	python3 -m venv pyfai
	source pyfai/bin/activate
	pip install pyFAI[full]

If you get an error about the local "UTF-8", try to:

.. code-block:: shell

   export LC_ALL=C

Before the installation.

Installation from sources
-------------------------

Get the sources from Github:

.. code-block:: shell

   wget https://github.com/silx-kit/pyFAI/archive/master.zip
   unzip master.zip
   cd pyFAI-master
   pip install -r requirements.txt
   python setup.py build bdist_wheel
   pip install --find-links=dist --pre --no-index --upgrade pyFAI


About OpenMP
............

OpenMP is a way to write multi-threaded code, running on multiple processors
simultaneously.
PyFAI makes heavy use of OpenMP, but there is an issue with recent versions of
MacOSX (>v10.6) where the default compiler of Apple, *Xcode*, dropped the
support for OpenMP.

There are two ways to compile pyFAI on MacOSX:

* Using *Xcode* and de-activating OpenMP in pyFAI
* Using another compiler which supports OpenMP

Using Xcode
...........

To build pyFAI from sources, a C-compiler is needed.
On an *Apple* computer, the default compiler is
`Xcode <https://developer.apple.com/xcode/>`_, and it is available for free on
the **AppStore**.
As pyFAI will deactivate OpenMP on apple computer, for this  
it needs to regenerate all Cython files without OpenMP.

The absence of OpenMP is mitigated on Apple computer by the support of OpenCL which provied parallel intgeration.

Using **gcc** or **clang**
..........................

If you want to keep the OpenMP feature (which makes the processing slightly faster),
the alternative is to install another compiler like `gcc <https://gcc.gnu.org/>`_
or `clang <http://clang.llvm.org/>`_ on your *Apple* computer.

.. code-block:: shell

    CC=gcc python setup.py build --openmp
    python setup.py bdist_wheel
    pip install --find-links=dist/ --pre --no-index --upgrade pyFAI


**Nota:** The usage of "python setup.py install" is now deprecated.
It causes much more trouble as there is no installed file tracking,
hence no way to properly un-install a package.

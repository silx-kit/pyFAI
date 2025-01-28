:Author: Jérôme Kieffer
:Date: 28/01/2025
:Keywords: Installation procedure on MacOSX
:Target: System administrators

Installation procedure on MacOSX
================================

Install Python3:
----------------

To install pyFAI on an *Apple* computer you will need a scientific Python3 stack.
MacOSX provides by default Python2.7, you will need to install a recent version
of Python3 (3.9 at least).
Those distribution are available as *dmg* images from:
`Python.org <https://www.python.org/downloads/mac-osx/>`_

After downloading, move the **app** into the *Applications* folder.

Using a virtual environment:
----------------------------

It is not adviced to use *pip* together with *sudo*. Always use a virtual environment !

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

   python3 -m venv pyfai
   source pyfai/bin/activate
   pip install build
   git clone https://github.com/silx-kit/pyFAI
   cd pyFAI
   pip install -r requirements.txt
   pip install . --upgrade

About OpenMP
............

OpenMP is a way to write multi-threaded code, running on multiple cores
simultaneously.
PyFAI makes heavy use of OpenMP, but there is an issue with recent versions of
MacOSX (>v10.6) where the default compiler of Apple, *Xcode*, dropped the
support for OpenMP.

There are two ways to compile pyFAI on MacOSX:

* Using *Xcode* which desctivates OpenMP
* Using another compiler which supports OpenMP

Using Xcode
...........

To build pyFAI from sources, a C-compiler is needed.
On an *Apple* computer, the default compiler is
`Xcode <https://developer.apple.com/xcode/>`_, and it is available for free on
the **AppStore**.
The absence of OpenMP is mitigated on Apple computer by the support of OpenCL which provied parallel intgeration.

Using **gcc** or **clang**
..........................

If you want to keep the OpenMP feature (which makes the processing slightly faster),
the alternative is to install another compiler like `gcc <https://gcc.gnu.org/>`_
or `clang <http://clang.llvm.org/>`_ on your *Apple* computer and define the environment variable ``CC``.

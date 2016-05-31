:Author: Jérôme Kieffer
:Date: 31/05/2015
:Keywords: Installation procedure on MacOSX
:Target: System administrators

Installation procedure on MacOSX
================================

Using PIP
---------

To install pyFAI on an *Apple* computer you will need a scientific Python stack.
MacOSX provides by default Python2.7 with Numpy which is a good basis.

.. code::

    sudo pip install matplotlib --upgrade
    sudo pip install scipy --upgrade
    sudo pip install fabio --upgrade
    sudo pip install h5py --upgrade
    sudo pip install cython --upgrade
    sudo pip install pyFAI --upgrade

If you get an error about the local "UTF-8", try to:

.. code::

   export LC_ALL=C

Before the installation.

Installation from sources
-------------------------

Get the sources from Github:

.. code::

   wget https://github.com/pyFAI/pyFAI/archive/master.zip
   unzip master.zip
   cd pyFAI-master


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
As pyFAI has by default OpenMP activated, and it needs to be de-activated,
one needs to regenerate all Cython files without OpenMP.

.. code::

    sudo pip install cython --upgrade
    rm pyFAI/ext/*.c
    python setup.py build --no-openmp
    python setup.py bdist_wheel
    sudo pip install --find-links=dist/ --pre --no-index --upgrade pyFAI

Using **gcc** or **clang**
..........................

If you want to keep the OpenMP feature (which makes the processing slightly faster),
the alternative is to install another compiler like `gcc <https://gcc.gnu.org/>`_
or `clang <http://clang.llvm.org/>`_ on your *Apple* computer.
As gcc & clang support OpenMP, there is no need to re-generate the cython files.

.. code::

    CC=gcc python setup.py build --openmp
    python setup.py bdist_wheel
    sudo pip install --find-links=dist/ --pre --no-index --upgrade pyFAI


**Nota:** The usage of "python setup.py install" is now deprecated.
It causes much more trouble as there is no installed file tracking,
hence no way to de-install properly a package.

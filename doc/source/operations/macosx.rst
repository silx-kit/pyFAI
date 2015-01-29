Author: Jérôme Kieffer

Date: 29/01/2015

Keywords: Installation procedure on MacOSX

Target: System administrators

Installation procedure on MacOSX
================================

To install pyFAI on an Apple computer you will need a scientific Python stack.
MacOSX provides by default Python2.7 with Numpy which is a good basis.

::
    sudo LC_ALL=C pip install matplotlib --upgrade
    sudo LC_ALL=C pip install scipy --upgrade
    sudo LC_ALL=C pip install fabio --upgrade
    sudo LC_ALL=C pip install pyFAI --upgrade


The LC_ALL=C addresses a bug in MacOSX where the locale is badly defined.

Installation from sources
-------------------------

To build pyFAI from sources, a compiler is needed. Apple provides Xcode for free:
https://developer.apple.com/xcode/

Another option is to use GCC which in addition supports OpenMP (see below)

Optional build dependencies: Cython (>v0.17) is needed to translate the source files into C code.
If Cython is present on your system, the source code will be re-generated and compiled.

::
    sudo LC_ALL=C pip install cython --upgrade

About OpenMP
------------

There is an issue with MacOSX (v10.8 onwards) where the default compiler (Xcode 5 or 6) switched from gcc 4.2 to clang and
dropped the support for OpenMP.
This is why OpenMP is by default deactivated under MacOSX. If you have installed an OpenMP-able compiler like GCC, you can re-activate it using the flag --openmp for setup.py

::
    LC_ALL=C python setup.py build --openmp
    sudo LC_ALL=C python setup.py install

    
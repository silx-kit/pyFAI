Author: Jérôme Kieffer

Date: 29/01/2015

Keywords: Installation procedure on Linux

Target: System administrators

Installation procedure on Linux
===============================

Installation procedure on Debian/Ubuntu
---------------------------------------

PyFAI has been designed and originally developed on Ubuntu 10.04 and debian6. Now it is included into debian7, 8 and any recent Ubuntu distribution.
To install it, simply use the package provided by the distribution.

::
   sudo apt-get install pyfai

To build a more recent version, pyFAI provides you a small scripts which builds a debian package and installs it. It relies on stdeb:

::
   sudo apt-get install python-stdeb cython python-fabio
   ./build-deb.sh

 If you are interested in programming in Python3, use

::
   sudo apt-get install cython3 python3-fabio
   ./build-deb.sh 3


Installation procedure on other linux distibution
-------------------------------------------------

::
    python setup.py build build_doc
    sudo python setup.py install
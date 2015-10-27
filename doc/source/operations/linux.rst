Author: Jérôme Kieffer

Date: 27/10/2015

Keywords: Installation procedure on Linux

Target: System administrators

Installation procedure on Linux
===============================

Installation procedure on Debian/Ubuntu
---------------------------------------

PyFAI has been designed and originally developed on Ubuntu 10.04 and debian6.
Now it is included into debian7, 8 and any recent Ubuntu and Mint distribution.
To install the package provided by the distribution, use:

.. code::

   sudo apt-get install pyfai

To build a more recent version, pyFAI provides you a small scripts which builds a debian package and installs it.
It relies on stdeb:

.. code::

   sudo apt-get install python-stdeb cython python-fabio
   ./build-deb.sh

 If you are interested in programming in Python3, use

.. code::

   sudo apt-get install cython3 python3-fabio
   ./build-deb.sh 3


Installation procedure on other linux distibution
-------------------------------------------------

If your distribution does not provide you pyFAI packages, using the **PIP** way is advised, via wheels packages.

.. code::
    wget https://bootstrap.pypa.io/get-pip.py
    sudo python get-pip.py
    sudo pip install wheel


Then you can install pyFAI the usual way:

.. code::

    python setup.py build test
    sudo pip install . --upgrade
    
Nota: The usage of "python setup.py install" is now deprecated.
It causes much more trouble as there is no installed file tracking, hence no way to de-install properly a package.
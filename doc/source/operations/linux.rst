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

Debian7 and Ubuntu 12.04
........................

To build a more recent version, pyFAI provides you a small scripts which builds a *debian* package and installs it.
It relies on *stdeb* and provides a single package with everything inside.
You will be prompted for your password to gain root access in order to be able to install the freshly built package.

.. code::

   sudo apt-get install python-stdeb cython python-fabio
   ./build-deb.sh

Debian8 and newer
.................

Thanks to the work of Frédéric-Emmanuel Picca, the debian package of pyFAI provides a pretty good template.

.. code::

   sudo apt-get install cython cython-dbg cython3 cython3-dbg debhelper dh-python python-all-dev python-all-dbg python-fabio python-fabio-dbg python-fftw python-h5py python-lxml python-lxml-dbg python-matplotlib python-matplotlib-dbg python-numpy python-numpy-dbg python-qt4 python-qt4-dbg python-scipy python-scipy-dbg python-sphinx python-sphinxcontrib.programoutput python-tk python-tk-dbg python3-all-dev python3-all-dbg python3-fabio python3-fabio-dbg python3-lxml python3-lxml-dbg python3-matplotlib python3-matplotlib-dbg python3-numpy python3-numpy-dbg python3-pyqt4 python3-pyqt4-dbg python3-scipy python3-scipy-dbg python3-sphinx python3-sphinxcontrib.programoutput python3-tk python3-tk-dbg 
   ./build-debian-full.sh
 
The first line is really long and defines all the dependence tree for building debian package, including debug and documentation.

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
    
**Nota:** The usage of "python setup.py install" is now deprecated.
It causes much more trouble as there is no installed file tracking, hence no way to de-install properly a package.
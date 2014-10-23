pyFAI: Fast Azimuthal Integration in Python
===========================================

pyFAI is an azimuthal integration library that tries to be fast (as fast as C
and even more using OpenCL and GPU).
It is based on histogramming of the 2theta/Q positions of each (center of)
pixel weighted by the intensity of each pixel, but parallel version use a
SparseMatrix-DenseVector multiplication.
Neighboring output bins get also a contribution of pixels next to the border
thanks to pixel splitting.
Finally pyFAI provides also tools to calibrate the experimental setup using Debye-Scherrer
rings of a reference compound.

References:
-----------

* The philosophy of pyFAI is described in the proceedings of SRI2012:
  doi:10.1088/1742-6596/425/20/202012 http://iopscience.iop.org/1742-6596/425/20/202012/
* Implementation in parallel is described in the proceedings of EPDIC13:
  PyFAI: a Python library for high performance azimuthal integration on GPU.
  doi:10.1017/S0885715613000924
* Benchmarks and optimization procedure is described in the proceedings of EuroSciPy2014:
  http://conference.scipy.org/category/euroscipy.html (accepted)


Installation
============

pyFAI can be downloaded from the http://forge.epn-campus.eu/projects/azimuthal/files.
Presently the source code has been distributed as a zip package and a compressed
tarball. Download either one and unpack it.
Developement is done on Github: https://github.com/pyFAI/pyFAI

e.g.
::
    tar xvzf pyFAI-0.10.2.tar.gz
or
::
    unzip pyFAI-0.10.2.zip

All files are unpacked into the directory pyFAI-0.10.2.
::
    cd pyFAI-0.10.2

Build it & test it. For test pyFAI downloads test images (you may have to setup a proxy configuration like export http_proxy=http://proxy.site.org:3128):
::
    python setup.py build test

and install pyFAI with
::
    sudo python setup.py install

Most likely you will need to do this with root privileges, sudo requests a passwors to do-so


The newest development version can be obtained by checking it out from the git repository.
::
    git clone https://github.com/pyFAI/pyFAI.git
    cd pyFAI
    sudo python setup.py install

If you want pyFAI to make use of your graphic card, please install pyopencl from:
http://mathema.tician.de/software/pyopencl

If you are using MS Windows you can also download a binary version packaged as executable
installation files (Chose the one corresponding to your python version).

For MacOSX users with MacOS version>10.7, the default compiler switched from gcc
to clang and dropped the OpenMP support. Three options for you:

* Use the new Xcode6 (with clang3.5) or Xcode4 (with gcc)
* Install a version of gcc (>4.2) on MacOSX
* use the "--no-openmp" option to setup.py if you are enforced to use Xcode4

Documentation
-------------

Documentation can be build using this command and Sphinx (installed on your computer):

::
    python setup.py build_doc


Dependencies
============

Python 2.6 or 2.7. Compatibility with python 3 is unchecked.
For full functionality of pyFAI the following modules need to be installed.

    * numpy 		- 	http://www.numpy.org
    * scipy 		- 	http://www.scipy.org
    * matplotlib 	- 	http://matplotlib.sourceforge.net/
    * fabio			-	http://sourceforge.net/projects/fable/files/fabio/
    * h5py			-   http://www.h5py.org/
    * pyopencl		-	http://mathema.tician.de/software/pyopencl/
    * python-qt4	-	http://www.riverbankcomputing.co.uk/software/pyqt/intro

Ubuntu and Debian Like linux distributions:
-------------------------------------------

To use pyFAI on Ubuntu/Debian the needed python modules
can be installed either through the Synaptic Package Manager (found in System -> Administration)
or using apt-get on from the command line in a terminal.
The extra ubuntu packages needed are:

    * python-numpy
    * python-scipy
    * python-matplotlib
    * python-dev
    * python-fabio
    * python-pyopencl
    * python-fftw
    * python-qt4

using apt-get these can be installed as:
::
    sudo apt-get install python-numpy python-scipy python-matplotlib  python-dev python-fabio python-pyopencl python-fftw python-qt4

MacOSX
------

You are advised to build pyFAI with the GCC compiler, clang (<3.5) laking the support of OpenMP.
If you use Xcode5, append the "--no-openmp" option to deactivate multithreading in binary modules

Windows
-------

Under 32 bits windows, pyFAI can be built using The MinGW compiler. Unfortunately,
pyFAI will be limited to small images as the memory consumption, limited to 2GB
under windows, easy easily reached. The VisualStudio C++ compiler works as well.
With 64 bits windows, the Visual Studio C++ compiler is the only one known to work correctly.

Getting help
============

A mailing-list, pyfai@esrf.fr, is available to get help on the program and how to use it.
One needs to subscribe by sending an email to sympa@esrf.fr with a subject "subscribe pyfai".


Maintainer
==========
 * Jérôme Kieffer (ESRF)

Contributors
============
 * Frédéric-Emmanuel Picca (Soleil)
 * Dimitris Karkoulis (ESRF)
 * Aurore Deschildre (ESRF)
 * Giannis Ashiotis (ESRF)
 * Zubair Nawaz (Sesame)
 * Jon Wright (ESRF)
 * Amund Hov (ESRF)
 * Dodogerstlin @github
 * Gunthard Benecke (Desy)
 * Gero Flucke (Desy)

Indirect contributors (ideas, ...):
-----------------------------------
 * Peter Boesecke
 * Manuel Sánchez del Río
 * Vicente Armando Solé
 * Brian Pauw
 * Veijo Honkimaki

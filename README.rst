pyFAI: Fast Azimuthal Integration in Python
===========================================

Main development website: https://github.com/silx-kit/pyFAI

|Build Status| |Appveyor Status| |myBinder Launcher| |RTD docs| |Zenodo DOI|

PyFAI is an azimuthal integration library that tries to be fast (as fast as C
and even more using OpenCL and GPU).
It is based on histogramming of the 2theta/Q positions of each (center of)
pixel weighted by the intensity of each pixel, but parallel version uses a
SparseMatrix-DenseVector multiplication.
Neighboring output bins get also a contribution of pixels next to the border
thanks to pixel splitting.
Finally pyFAI provides also tools to calibrate the experimental setup using Debye-Scherrer
rings of a reference compound.

References
----------

* The philosophy of pyFAI is described in the proceedings of SRI2012:
  doi:10.1088/1742-6596/425/20/202012 http://iopscience.iop.org/1742-6596/425/20/202012/
* Implementation in parallel is described in the proceedings of EPDIC13:
  PyFAI: a Python library for high performance azimuthal integration on GPU.
  doi:10.1017/S0885715613000924
* Benchmarks and optimization procedure is described in the proceedings of EuroSciPy2014:
  http://conference.scipy.org/category/euroscipy.html (accepted)

Installation
------------

With PIP
........

As most Python packages, pyFAI is available via PIP::

   pip install pyFAI[gui]

Provide the *--user* to perform an installation local to your user.
Under UNIX, you may have to run the command via *sudo* to gain root access an
perform a system wide installation. 
The best solution remaining to install the software into a vituralenv.

With conda
..........

pyFAI is also available via conda::

  conda install pyfai -c conda-forge

To install conda please see either `conda <https://conda.io/docs/install/quick.html>`_ or `Anaconda <https://www.continuum.io/downloads>`_.

From source code
................

The latest release of pyFAI can be downloaded from
`Github <https://github.com/silx-kit/pyFAI/archive/master.zip>`_.
Presently the source code has been distributed as a zip package.
Download it one and unpack it::

    unzip pyFAI-master.zip

As developement is also done on Github,
`development branch is also available <https://github.com/silx-kit/pyFAI/archive/master.zip>`_

All files are unpacked into the directory pyFAI-master::

    cd pyFAI-master

Build it & test it::

    python3 setup.py build test

For its tests, pyFAI downloads test images from the internet.
Depending on your network connection and your local network configuration,
you may have to setup a proxy configuration like this (no more needed at ESRF)::

   export http_proxy=http://proxy.site.org:3128

Finally, install pyFAI in the virtualenv after testing it::

    python3 setup.py bdist_wheel
    pip install pyFAI --pre --find-links dist --no-index --upgrade

If you prefer a local installation (only you will have access to the
installed version), use in addition the --user option::

    pip install pyFAI --pre --find-links dist --no-index --upgrade --user

The newest development version can also be obtained by checking out from the git
repository::

    git clone https://github.com/silx-kit/pyFAI.git
    cd pyFAI
    python3 setup.py build bdist_wheel
    pip install pyFAI --pre --find-links dist --no-index --upgrade
    
If you want pyFAI to make use of your graphic card, please install
`pyopencl <http://mathema.tician.de/software/pyopencl>`_

If you are using MS Windows you can also download a binary version packaged as executable
installation files (Chose the one corresponding to your python version).

For MacOSX users with MacOS version>10.7, the default compiler switched from gcc
to clang and dropped the OpenMP support. Please refer to the installation documentation ...

Documentation
-------------

Documentation can be build using this command and Sphinx (installed on your computer)::

    python3 setup.py build build_doc


Dependencies
------------

Python 3.6, ... 3.9 are well tested and officially supported.
For full functionality of pyFAI the following modules need to be installed.

* ``numpy``      - http://www.numpy.org
* ``scipy`` 	 - http://www.scipy.org
* ``matplotlib`` - http://matplotlib.sourceforge.net/
* ``fabio`` 	 - http://sourceforge.net/projects/fable/files/fabio/
* ``h5py``	     - http://www.h5py.org/
* ``pyopencl``	 - http://mathema.tician.de/software/pyopencl/
* ``pyqt5``	     - http://www.riverbankcomputing.co.uk/software/pyqt/intro
* ``silx``       - http://www.silx.org
* ``numexpr``    - https://github.com/pydata/numexpr

Those dependencies can simply be installed by::

   pip install -r requirements.txt


Ubuntu and Debian-like Linux distributions
------------------------------------------

To use pyFAI on Ubuntu/Debian the needed python modules
can be installed either through the Synaptic Package Manager
(found in System -> Administration)
or using apt-get on from the command line in a terminal::

   sudo apt-get install pyfai

The extra Ubuntu packages needed are:

* ``python3-numpy``
* ``python3-scipy``
* ``python3-matplotlib``
* ``python3-dev``
* ``python3-fabio``
* ``python3-pyopencl``
* ``python3-pyqt5``
* ``python3-silx``
* ``python3-numexpr``

using apt-get these can be installed as::

    sudo apt-get build-dep pyfai

MacOSX
------

One needs to install `Python` (>=3.6) and `Xcode` prior to start installing pyFAI. 
The compiled extension will use only one core due to the limitation of the compiler.
OpenCL is hence greately adviced on Apple systems. 
Then install the missing dependencies with `pip`::

   pip install -r requirements.txt
	

Windows
-------

Under Windows, one needs to install `Python` (>=3.6) and the Visual Studio C++ compiler.
Then install the missing dependencies with `pip`::

   pip install  -r requirements.txt

Getting help
------------

A mailing-list, pyfai@esrf.fr, is available to get help on the program and how to use it.
One needs to subscribe by sending an email to sympa@esrf.fr with a subject "subscribe pyfai".


Maintainers
-----------

* Jérôme Kieffer (ESRF)
* Valentin Valls (ESRF)

Contributors
------------

* Frédéric-Emmanuel Picca (Soleil)
* Thomas Vincent (ESRF)
* Dimitris Karkoulis (ESRF)
* Aurore Deschildre (ESRF)
* Giannis Ashiotis (ESRF)
* Zubair Nawaz (Sesame)
* Jon Wright (ESRF)
* Amund Hov (ESRF)
* Dodogerstlin @github
* Gunthard Benecke (Desy)
* Gero Flucke (Desy)

Indirect contributors (ideas...)
--------------------------------

* Peter Boesecke
* Manuel Sánchez del Río
* Vicente Armando Solé
* Brian Pauw
* Veijo Honkimaki

.. |Build Status| image:: https://travis-ci.org/silx-kit/pyFAI.svg?branch=master
   :target: https://travis-ci.org/silx-kit/pyFAI
.. |Appveyor Status| image:: https://ci.appveyor.com/api/projects/status/github/silx-kit/pyfai?svg=true
   :target: https://ci.appveyor.com/project/ESRF/pyfai
.. |myBinder Launcher| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/silx-kit/pyFAI/master?filepath=binder%2Findex.ipynb
.. |RTD docs| image:: https://readthedocs.org/projects/pyFAI/badge/?version=master
    :alt: Documentation Status
    :scale: 100%
    :target: https://pyfai.readthedocs.io/en/master/?badge=master
.. |Zenodo DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.832896.svg
   :target: https://doi.org/10.5281/zenodo.832896

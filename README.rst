pyFAI: Fast Azimuthal Integration in Python
===========================================

Main development website: https://github.com/silx-kit/pyFAI

|Github Actions| |Appveyor Status| |myBinder Launcher| |Zenodo DOI| |RTD docs|

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

* The philosophy of pyFAI is described in the proceedings of SRI2012: https://doi.org/10.1088/1742-6596/425/20/202012
* Implementation in parallel is described in the proceedings of EPDIC13: https://doi.org/10.1017/S0885715613000924
* Benchmarks and optimization procedure is described in the proceedings of EuroSciPy2014: https://doi.org/10.48550/arXiv.1412.6367
* Calibration procedures are described in J. Synch. Radiation (2020): https://doi.org/10.1107/S1600577520000776
* Application of signal separation to diffraction image compression and serial crystallography. J. Appl. Cryst. (2025): https://doi.org/10.1107/S1600576724011038

Installation
------------

With PIP
........

As most Python packages, pyFAI is available via PIP::

   pip install pyFAI[gui]

It is advised to run this in a `vitural environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments>`_ .
Provide the *--user* option to perform an installation local to your user-space (**not recommended**).
Under UNIX, you may have to run the command via *sudo* to gain root access and perform a system wide installation (which is **neither recommended**).

With conda
..........

pyFAI is also available via conda::

  conda install pyfai -c conda-forge

To install conda please see either `conda <https://conda.io/docs/install/quick.html>`_ or `Anaconda <https://www.continuum.io/downloads>`_.

From source code
................

The current development version of pyFAI can be downloaded from
`Github <https://github.com/silx-kit/pyFAI/archive/main.zip>`_.
Presently the source code has been distributed as a zip package.
Download it one and unpack it::

    unzip pyFAI-main.zip

All files are unpacked into the directory pyFAI-main::

    cd pyFAI-main

Install dependencies::

    pip install -r requirements.txt

Build it & test it::

    python3 run_tests.py

For its tests, pyFAI downloads test images from the internet.
Depending on your network connection and your local network configuration,
you may have to setup a proxy configuration like this (not needed at ESRF)::

   export http_proxy=http://proxy.site.org:3128

Finally, install pyFAI in the virtualenv after testing it::

    pip install .

The newest development version can also be obtained by checking out from the git
repository::

    git clone https://github.com/silx-kit/pyFAI.git
    cd pyFAI
    pip install .

If you want pyFAI to make use of your graphic card, please install
`pyopencl <http://mathema.tician.de/software/pyopencl>`_

Documentation
-------------

Documentation can be build using this command and Sphinx (installed on your computer)::

    python3 build-doc.py

Dependencies
------------

Python 3.9, ... 3.13 are well tested and officially supported.
For full functionality of pyFAI the following modules need to be installed.

* ``numpy``      - http://www.numpy.org
* ``scipy`` 	  - http://www.scipy.org
* ``matplotlib`` - http://matplotlib.sourceforge.net/
* ``fabio`` 	  - http://sourceforge.net/projects/fable/files/fabio/
* ``h5py``	     - http://www.h5py.org/
* ``pyopencl``	  - http://mathema.tician.de/software/pyopencl/
* ``pyside6``	  - https://wiki.qt.io/Qt_for_Python
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
* ``python3-qtpy``
* ``python3-silx``
* ``python3-numexpr``

using apt-get these can be installed as::

    sudo apt-get build-dep pyfai

MacOSX
------

One needs to manually install a recent version of `Python` (>=3.8) prior to installing pyFAI.
Apple provides only an outdated version of Python 2.7 which is now incomatible.
If you want to build pyFAI from sources, you will also need `Xcode` which is available from the Apple store.
The compiled extension will use only one core due to the limitation of the compiler.
OpenCL is hence greately adviced on Apple systems.
Then install the missing dependencies with `pip`::

   pip install -r requirements.txt


Windows
-------

Under Windows, one needs to install `Python` (>=3.8) prior to pyFAI.
The Visual Studio C++ compiler is also needed when building from sources.
Then install the missing dependencies with `pip`::

   pip install  -r requirements.txt

Getting help
------------

A mailing-list, pyfai@esrf.fr, is available to get help on the program and how to use it.
One needs to subscribe by sending an email to sympa@esrf.fr with a subject "subscribe pyfai".

Maintainers
-----------

* Jérôme Kieffer (ESRF)
* Edgar Gutierrez Fernandez (ESRF)
* Loïc Huder (ESRF)

Contributors
------------

* Valentin Valls (ESRF)
* Frédéric-Emmanuel Picca (Soleil)
* Thomas Vincent (ESRF)
* Dimitris Karkoulis (Formerly ESRF)
* Aurore Deschildre (Formerly ESRF)
* Giannis Ashiotis (Formerly ESRF)
* Zubair Nawaz (Formerly Sesame)
* Jon Wright (ESRF)
* Amund Hov (Formerly ESRF)
* Dodogerstlin @github
* Gunthard Benecke (Desy)
* Gero Flucke (Desy)
* Maciej Jankowski (ESRF)

Indirect contributors (ideas...)
--------------------------------

* Peter Boesecke
* Manuel Sánchez del Río
* Vicente Armando Solé
* Brian Pauw
* Veijo Honkimaki

.. |Github Actions| image:: https://github.com/silx-kit/pyFAI/actions/workflows/python-package.yml/badge.svg
.. |Appveyor Status| image:: https://ci.appveyor.com/api/projects/status/github/silx-kit/pyfai?svg=true
   :target: https://ci.appveyor.com/project/ESRF/pyfai
.. |myBinder Launcher| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/silx-kit/pyFAI/main?filepath=binder%2Findex.ipynb
.. |RTD docs| image:: https://readthedocs.org/projects/pyfai/badge/?version=latest
   :target: https://pyfai.readthedocs.io/en/latest/
.. |Zenodo DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.832896.svg
   :target: https://doi.org/10.5281/zenodo.832896

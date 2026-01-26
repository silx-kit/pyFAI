# pyFAI: Fast Azimuthal Integration in Python

Main development website: [https://github.com/silx-kit/pyFAI](https://github.com/silx-kit/pyFAI)

[![Github Actions](https://github.com/silx-kit/pyFAI/actions/workflows/python-package.yml/badge.svg)](https://github.com/silx-kit/pyFAI/actions)
[![Appveyor Status](https://ci.appveyor.com/api/projects/status/github/silx-kit/pyfai?svg=true)](https://ci.appveyor.com/project/ESRF/pyfai)
[![myBinder Launcher](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/silx-kit/pyFAI/main?filepath=binder%2Findex.ipynb)
[![RTD docs](https://readthedocs.org/projects/pyfai/badge/?version=latest)](https://pyfai.readthedocs.io/en/latest/)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.832896.svg)](https://doi.org/10.5281/zenodo.832896)

pyFAI is an azimuthal integration library that tries to be fast (as fast as C and even more using OpenCL and GPU).
It is based on histogramming of the 2theta/Q positions of each (center of) pixel weighted by the intensity of each pixel, but the parallel version uses a SparseMatrix-DenseVector multiplication.
Neighboring output bins also get a contribution of pixels next to the border thanks to pixel splitting.
pyFAI also provides tools to calibrate the experimental setup using Debye-Scherrer rings of a reference compound.

## References

- The philosophy of pyFAI is described in the proceedings of SRI2012: https://doi.org/10.1088/1742-6596/425/20/202012
- Implementation in parallel is described in the proceedings of EPDIC13: https://doi.org/10.1017/S0885715613000924
- Benchmarks and optimization procedure are described in the proceedings of EuroSciPy2014: https://doi.org/10.48550/arXiv.1412.6367
- Calibration procedures are described in J. Synch. Radiation (2020): https://doi.org/10.1107/S1600577520000776
- Application of signal separation to diffraction image compression and serial crystallography. J. Appl. Cryst. (2025): https://doi.org/10.1107/S1600576724011038

## Installation

### With PIP

As most Python packages, pyFAI is available via PIP:

```sh
pip install pyFAI[gui]
```

It is advised to run this in a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments).
Provide the `--user` option to perform an installation local to your user-space (**not recommended**).
Under UNIX, you may have to run the command via `sudo` to gain root access and perform a system wide installation (which is **neither recommended**).

### With conda

pyFAI is also available via conda:

```sh
conda install pyfai -c conda-forge
```

To install conda please see either [conda](https://conda.io/docs/install/quick.html) or [Anaconda](https://www.continuum.io/downloads).

### From source code

The current development version of pyFAI can be downloaded from [GitHub](https://github.com/silx-kit/pyFAI/archive/main.zip).
Presently the source code has been distributed as a zip package.

Download and unpack it:

```sh
unzip pyFAI-main.zip
cd pyFAI-main
```

Install dependencies:

```sh
pip install -r requirements.txt
```

Build and test it:

```sh
python3 run_tests.py
```

For its tests, pyFAI downloads test images from the internet. Depending on your network connection and your local network configuration, you may have to set up a proxy configuration like this (not needed at ESRF):

```sh
export http_proxy=http://proxy.site.org:3128
```

Finally, install pyFAI in the virtualenv after testing it:

```sh
pip install .
```

The newest development version can also be obtained by checking out from the git repository:

```sh
git clone https://github.com/silx-kit/pyFAI.git
cd pyFAI
pip install .
```

If you want pyFAI to make use of your graphic card, please install [pyopencl](http://mathema.tician.de/software/pyopencl).

## Documentation

Documentation can be built using this command and Sphinx (installed on your computer):

```sh
python3 build-doc.py
```

## Dependencies

Python 3.10 ... 3.14 are well tested and officially supported (thread-free is untested).

For full functionality of pyFAI, the following modules need to be installed:

- [`numpy`](http://www.numpy.org)
- [`scipy`](http://www.scipy.org)
- [`matplotlib`](http://matplotlib.sourceforge.net/)
- [`fabio`](http://sourceforge.net/projects/fable/files/fabio/)
- [`h5py`](http://www.h5py.org/)
- [`pyopencl`](http://mathema.tician.de/software/pyopencl/)
- [`pyside6`](https://wiki.qt.io/Qt_for_Python)
- [`silx`](http://www.silx.org)
- [`numexpr`](https://github.com/pydata/numexpr)

Those dependencies can simply be installed by:

```sh
pip install -r requirements.txt
```

## Ubuntu and Debian-like Linux distributions

To use pyFAI on Ubuntu/Debian, the needed Python modules can be installed either through the Synaptic Package Manager (found in System -> Administration) or using apt-get from the command line:

```sh
sudo apt-get install pyfai
```

The extra Ubuntu packages needed are:

- `python3-numpy`
- `python3-scipy`
- `python3-matplotlib`
- `python3-dev`
- `python3-fabio`
- `python3-pyopencl`
- `python3-qtpy`
- `python3-silx`
- `python3-numexpr`

Using apt-get these can be installed as:

```sh
sudo apt-get build-dep pyfai
```

## MacOSX

One needs to manually install a recent version of [Python](https://www.python.org/downloads/) (>=3.10) prior to installing pyFAI.
Apple provides only an outdated version of Python 2.7 which is now incompatible.
If you want to build pyFAI from sources, you will also need `Xcode` which is available from the Apple store.
The compiled extension will use only one core due to the limitation of the compiler.
OpenCL is hence greatly advised on Apple systems.
Then install the missing dependencies with pip:

```sh
pip install -r requirements.txt
```

## Windows

Under Windows, one needs to install [Python](https://www.python.org/downloads/) (>=3.10) prior to pyFAI.
The Visual Studio C++ compiler is also needed when building from sources.
Then install the missing dependencies with pip:

```sh
pip install -r requirements.txt
```

## Getting Help

A mailing-list, pyfai@esrf.fr, is available to get help on the program and how to use it.
One needs to subscribe by sending an email to sympa@esrf.fr with the subject "subscribe pyfai".

## Maintainers

- Jérôme Kieffer (ESRF)
- Edgar Gutierrez Fernandez (ESRF)
- Loïc Huder (ESRF)

## Contributors

Thanks to all who have contributed to pyFAI!

[![Contributors image](https://contrib.rocks/image?repo=silx-kit/pyFAI)](https://github.com/silx-kit/pyFAI/graphs/contributors)

## Indirect contributors (ideas...)

- Peter Boesecke
- Manuel Sánchez del Río
- Vicente Armando Solé
- Brian Pauw
- Veijo Honkimaki

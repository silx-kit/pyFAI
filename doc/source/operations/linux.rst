:Author: Jérôme Kieffer
:Date: 28/01/2025
:Keywords: Installation procedure on Linux
:Target: System administrators

Installation procedure on Linux
===============================

We cover first Debian-like distribution, then a generic recipie for all other
version is given.

Installation procedure on Debian/Ubuntu
---------------------------------------

PyFAI has been designed and originally developed on Ubuntu and debian-linux.
Now, the pyFAI library is included into debian, Ubuntu and
Mint distribution for example.
To install the package provided by the distribution, use:

.. code-block:: shell

   sudo apt-get install pyfai

The issue with distribution based installation is the obsolescence of the version
available.

Thanks to the work of Frédéric-Emmanuel Picca, the debian package of pyFAI
provides a pretty good template which allows continuous builds.

From silx repository
++++++++++++++++++++

You can automatically install the latest nightly built of pyFAI with:

.. code-block:: shell

   wget http://www.silx.org/pub/debian/silx.list
   wget http://www.silx.org/pub/debian/silx.pref
   sudo mv silx.list /etc/apt/sources.list.d/
   sudo mv silx.pref /etc/apt/preferences.d/
   sudo apt-get update
   sudo apt-get install pyfai

**Nota:** The nightly built packages are not signed, hence you will be prompted
to install non-signed packages.

Build from sources
++++++++++++++++++

One can also built the current development version from sources:

.. code-block:: shell

   sudo apt install git
   sudo apt-get build-dep pyfai
   git clone https://github.com/silx-kit/pyFAI
   cd pyFAI
   ./build-deb.sh --install


The first line installes all the dependences for building
*debian* package, including debug and documentation.
The build procedure last for a few minutes and you will be prompted for your
password in order to install the freshly built packages.
The *deb-*files, available in the *package* directory are backports for your local
installation.

Installation procedure on other linux distibution
-------------------------------------------------

If your distribution does not provide you pyFAI packages, using the **PIP** way
is advised, via wheels packages. First install *pip* and *wheel* and activate a
virtual environment:

.. code-block:: shell

   python3 -m venv pyfai
   source pyfai/bin/activate
   pip install pyFAI

Or you can install pyFAI from the sources:

.. code-block:: shell

   python3 -m venv pyfai
   source pyfai/bin/activate
   pip install build
   git clone https://github.com/silx-kit/pyFAI
   cd pyFAI
   pip install -r requirements.txt
   pip install . --upgrade

**Nota:** One should never use *sudo pip* as it is likely to interfer with the software installed with the system.

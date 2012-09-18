pyFAI : Fast Azimuthal Integration in Python
=====

pyFAI is an azimuthal integration library that tries to be fast (as fast as C
and even more using OpenCL) It is based on histogramming of the 2theta/Q position
of each (center of) pixel weighted by the intensity of each pixel.
Neighboring output bins get also a contribution of pixels next to the border

Installation
============
pyFAI can be downloaded from the http://forge.epn-campus.eu/projects/azimuthal/files.
Presently the source code has been distributed as a zip package and a compressed
tarball. Download either one and unpack it.
Developement is done on Github: https://github.com/kif/pyFAI

e.g.
tar xvzf pyFAI-0.7.0.tar.gz
or
unzip pyFAI-0.7.0.zip

all files are unpacked into the directory pyFAI-0.7.0. To install these do

cd pyFAI-0.7.0

and install pyFAI with

python setup.py install

most likely you will need to do this with root privileges (e.g. put sudo
in front of the command).


The newest development version can be obtained by checking it out from the git repository.

git clone https://github.com/kif/pyFAI.git
cd pyFAI
sudo python setup.py install

As pyFAI makes use of OpenCL to have some calculation done on the graphic card, it is
likely you will have to setup the position of those libraries by editing the setup.cfg file

If you are using MS Windows you also download a binary version packaged as executable
installation files (Chose the one corresponding to your python version).


Dependencies
============

Python 2.6 or 2.7. Compatiblity with python 3 is unchecked.
For full functionality of pyFAI the following modules need to be installed.

    * numpy 		- 	http://www.numpy.org
    * scipy 		- 	http://www.scipy.org
    * matplotlib 	- 	http://matplotlib.sourceforge.net/
    * fabio			-	http://sourceforge.net/projects/fable/files/fabio/

Ubuntu and Debian Like linux distributions:
-------------------------------------------
To use pyFAI on Ubuntu (a linux distribution based on Debian) the needed python modules
can be installed either through the Synaptic Package Manager (found in System -> Administration)
or using apt-get on from the command line in a terminal.
The extra ubuntu packages needed are:

    * python-numpy
    * python-scipy
    * python-matplotlib
    * python-dev

Only Fabio has to be downloaded separatly and installed
    * python-fabio (from http://sourceforge.net/projects/fable/files/fabio/)

using apt-get these can be installed as:

sudo apt-get install python-numpy python-scipy python-matplotlib  python-dev
wget http://sourceforge.net/projects/fable/files/fabio/0.0.7/squeeze/python-fabio_0.0.7-1_amd64.deb/download
sudo dpkg -i python-fabio_0.0.7-1_amd64.deb


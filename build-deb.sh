#!/bin/sh

# Script that builds a debian package from this library 
 
if [ -d /usr/lib/ccache ];
then 
   CCPATH=/usr/lib/ccache:$PATH 
else  
   CCPATH=$PATH
fi
export PYBUILD_DISABLE_python2=test
export PYBUILD_DISABLE_python3=test
export DEB_BUILD_OPTIONS=nocheck
rm -rf dist
python setup.py sdist
cd dist
tar -xzf pyFAI-*.tar.gz
cd pyFAI*

if [ $1 = 3 ]
then
  echo Using Python 2+3 
  PATH=$CCPATH  python3 setup.py --command-packages=stdeb.command sdist_dsc --with-python2=True --with-python3=True --no-python3-scripts=True bdist_deb --no-cython
  sudo dpkg -i deb_dist/python3-pyfai*.deb
else
  echo Using Python 2
	PATH=$CCPATH python setup.py --command-packages=stdeb.command bdist_deb --no-cython
fi

sudo su -c  "dpkg -i deb_dist/pyfai*.deb"
cd ../..


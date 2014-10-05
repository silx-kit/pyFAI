#!/bin/sh

# Script that builds a debian package from this library 
 
if [ -z $(which  ccache) ];
then 
   unset CC; 
else  
   export CC="ccache gcc"; 
fi
export PYBUILD_DISABLE_python2=test
export PYBUILD_DISABLE_python3=test
rm -rf dist
python setup.py sdist
cd dist
tar -xzf pyFAI-*.tar.gz
cd pyFAI*
python setup.py --command-packages=stdeb.command bdist_deb --no-cython
sudo su -c  "dpkg -i deb_dist/pyfai*.deb"
cd ../..

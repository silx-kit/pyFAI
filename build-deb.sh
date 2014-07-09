#!/bin/sh

# Script that builds a debian package from this library 
 
rm -rf dist
python setup.py sdist
cd dist
tar -xzf pyFAI-*.tar.gz
cd pyFAI*
python setup.py --command-packages=stdeb.command bdist_deb --no-cython
sudo su -c  "dpkg -i deb_dist/pyfai*.deb"
cd ../..

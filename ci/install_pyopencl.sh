#!/bin/sh
# Compile & install pyopencl
git clone https://github.com/pyopencl/pyopencl
pushd pyopencl
python  configure.py
python setup.py build bdist_wheel
pip install --pre dist/*.whl
popd

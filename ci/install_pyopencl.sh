#!/bin/bash
# Compile & install pyopencl
git clone --depth=1 --shallow-submodules  https://github.com/pyopencl/pyopencl
pushd pyopencl
pip install pybind11 mako
python3  configure.py
python3 setup.py build bdist_wheel
pip install --pre dist/*.whl
popd

#!/bin/bash
# Compile & install pyopencl
if [ -f ci/intel_opencl_icd.sh ]; then
	source ci/intel_opencl_icd.sh
    git clone --depth=1 --shallow-submodules --recurse-submodules https://github.com/pyopencl/pyopencl
    pushd pyopencl
    pip install "setuptools<60.0.0"
    pip install wheel pybind11 mako
    CL_LIBRARY_PATH=$(python3 -c "import os; print(os.environ.get('LD_LIBRARY_PATH','').split(':')[-1])")
    CL_INCLUDE_PATH=$(python3 -c "import os; print(os.environ.get('C_INCLUDE_PATH','').split(':')[-1])")
    echo CL_LIBRARY_PATH ${CL_LIBRARY_PATH}
    echo CL_INCLUDE_PATH ${CL_INCLUDE_PATH}
    python3  configure.py  --cl-inc-dir=${CL_INCLUDE_PATH} --cl-lib-dir=${CL_LIBRARY_PATH}
    python3 setup.py build bdist_wheel
    pip install --pre dist/*.whl
    popd
    rm -rf pyopencl
fi

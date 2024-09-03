#!/bin/bash
# Compile & install pyopencl
if [ -f ci/intel_opencl_icd.sh ];
then
    source ci/intel_opencl_icd.sh
    CL_LIBRARY_PATH=$(python3 -c "import os; print(os.environ.get('LD_LIBRARY_PATH','').split(':')[-1])")
    CL_INCLUDE_PATH=$(python3 -c "import os; print(os.environ.get('C_INCLUDE_PATH','').split(':')[-1])")
    echo CL_LIBRARY_PATH ${CL_LIBRARY_PATH}
    echo CL_INCLUDE_PATH ${CL_INCLUDE_PATH}
    pip install wheel pybind11 mako pyopencl
    python3 -c "import pyopencl; print(pyopencl.get_platforms())"
    pip install silx
    python3 -c "import silx.opencl; print(silx.opencl.ocl)"
    env |grep ICD
fi

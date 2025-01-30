#!/bin/bash

# Download the intel OpenCL ICD and setup the environment for using it.

URL="http://www.silx.org/pub/OpenCL/"
FILENAME="intel_opencl_icd-6.4.0.38.tar.gz"
rm -rf $FILENAME
wget -nv ${URL}${FILENAME}
tar -xzf $FILENAME

echo $(pwd)/intel_opencl_icd/icd/libintelocl.so > intel_opencl_icd/vendors/intel64.icd

export OCL_ICD_VENDORS=$(pwd)/intel_opencl_icd/vendors
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(pwd)/intel_opencl_icd/lib
export C_INCLUDE_PATH=${C_INCLUDE_PATH}:$(pwd)/intel_opencl_icd/include
#echo clinfo:
#ldd $(pwd)/intel_opencl_icd/bin/clinfo
#echo libOpenCL:
#ldd $(pwd)/intel_opencl_icd/lib/libOpenCL.so.1.0.0

#echo icd:
#for i in $(pwd)/intel_opencl_icd/icd/*.so
#do
#    echo $i
#    ldd $i
#    echo
#done
$(pwd)/intel_opencl_icd/bin/clinfo

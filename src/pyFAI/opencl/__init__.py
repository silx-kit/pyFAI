#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: python fast azimuthal integration project
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#

"""Contains all OpenCL implementation."""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "2012-2024 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/11/2024"
__status__ = "stable"

import os
import logging
import platform
import numpy
from ..utils.decorators import deprecated
logger = logging.getLogger(__name__)

import pyFAI

if not pyFAI.use_opencl:
    pyopencl = None
    ocl = None
elif os.environ.get("PYFAI_OPENCL") in ["0", "False"]:
    logger.info("Use of OpenCL has been disables from environment variable: PYFAI_OPENCL=0")
    pyopencl = None
    ocl = None
    OpenclProcessing = None
else:
    from silx.opencl import common
    ocl = common.ocl    # /!\ lasy loading of ocl at the silx level !!!
    from silx.opencl.common import pyopencl, mf, release_cl_buffers, allocate_cl_buffers, \
                                    measure_workgroup_size, kernel_workgroup_size

    from .. import resources
    resources.silx_integration()

    from silx.opencl import utils
    from silx.opencl.utils import get_opencl_code, concatenate_cl_kernel, read_cl_file
    from silx.opencl import processing
    OpenclProcessing = processing.OpenclProcessing


def get_x87_volatile_option(ctx):
        # this is running 32 bits OpenCL with POCL
        if (platform.machine() in ("i386", "i686", "x86_64") and (tuple.__itemsize__ == 4) and
                ctx.devices[0].platform.name == 'Portable Computing Language'):
            return "-DX87_VOLATILE=volatile"
        else:
            return ""

def get_compiler_options(ctx, x87_volatile=False, apple_gpu=False):
    """Provide a set of common compiler options to work around known bugs:

    :x87_volatile: set to true to declare all x87 operation as volatile, needed on PoCL x86 32bits
    :apple_gpu: redefine the cl_khr_fp64 to zero when the device is Apple GPU
                which wrongly declares fp64 compatibility. See #2339
    :return: compilation directive as string.
    """

    if x87_volatile:
        options = get_x87_volatile_option(ctx)
    else:
        options = ""
    if apple_gpu:
        fp64_support = 1 if "cl_khr_fp64" in ctx.devices[0].extensions else 0
        options += f" -D cl_khr_fp64={fp64_support}"
    return options.strip()


def dtype_converter(dtype):
    "convert a numpy dtype as a int8"
    dtype = numpy.dtype(dtype)
    if numpy.issubdtype(dtype, numpy.signedinteger):
        return numpy.int8(-dtype.itemsize)
    elif numpy.issubdtype(dtype, numpy.unsignedinteger):
        return numpy.int8(dtype.itemsize)
    else:
        return numpy.int8(8 * dtype.itemsize)

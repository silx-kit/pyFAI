# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2017 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""A module for performing the 1d, 2d and 3d median filter ...

The target is to mimic the signature of scipy.signal.medfilt and scipy.medfilt2

The first implementation targets 2D implementation where this operation is costly (~10s/2kx2k image)
"""
from __future__ import absolute_import, print_function, with_statement, division


__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "03/02/2017"
__copyright__ = "2012-2017, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import logging
import numpy
from collections import OrderedDict
from numpy import asarray
from .common import ocl, pyopencl
from .processing import EventDescription, OpenclProcessing, BufferDescription

if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")


class MedianFilter2D(OpenclProcessing):
    """A class for doing median filtering using OpenCL"""
    buffers = [
               BufferDescription("output", 1, numpy.float32, mf.WRITE_ONLY),
               BufferDescription("image_raw", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("image", 1, numpy.float32, mf.READ_WRITE),
               ]
    kernel_files = ["preprocess.cl", "bitonic.cl"]
    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.int32: "s32_to_float"}

    def __init__(self, shape, kernel_size=(3, 3),
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False
                 ):
        """Constructor of the OpenCL 2D median filtering class

        :param shape: shape of the images to treat
        :param kernel size: 2-tuple of odd values
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the outpcome of the compilation
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                          platformid=platformid, deviceid=deviceid,
                          block_size=block_size, profile=profile)
        self.shape = shape
        self.size = shape[0] * shape[1]
        self.kernel_size = kernel_size
        self.to_sort = 1 << ((self.kernel_size[0] * self.kernel_size[1] - 1).bit_length())
        self.workgroup_size = (max(1, self.to_sort // 8), 1, 1)  # 3D kernel
        self.buffers = [BufferDescription(i.name, i.size * self.size, i.dtype, i.flags)
                        for i in self.__class__.buffers]

        self.allocate_buffers()
        self.local_mem = pyopencl.LocalMemory(self.workgroup_size[0] * 32)  # 4byte per float, 8 element per thread
        OpenclProcessing.compile_kernels(self, self.kernel_files, "-D NIMAGE=%i" % self.size)
        self.set_kernel_arguments()

    def set_kernel_arguments(self):
        """Parametrize all kernel arguments
        """
        for val in self.mapping.values():
            self.cl_kernel_args[val] = OrderedDict(((i, self.cl_mem[i]) for i in ("image_raw", "image")))
        self.cl_kernel_args["medfilt2d"] = OrderedDict((("image", self.cl_mem["image"]),
                                                        ("output", self.cl_mem["output"]),
                                                        ("local", self.local_mem),
                                                        ("khs1", numpy.int32(self.kernel_size[0] // 2)),  # Kernel half-size along dim1 (lines)
                                                        ("khs2", numpy.int32(self.kernel_size[1] // 2)),  # Kernel half-size along dim2 (columns)
                                                        ("height", numpy.int32(self.shape[0])),  # Image size along dim1 (lines)
                                                        ("width", numpy.int32(self.shape[1]))))  # Image size along dim2 (columns))

    def send_buffer(self, data, dest):
        """Send a numpy array to the device, including the cast on the device if possible

        :param data: numpy array with data
        :param dest: name of the buffer as registered in the class
        """

        dest_type = numpy.dtype([i.dtype for i in self.buffers if i.name == dest][0])
        events = []
        if (data.dtype == dest_type) or (data.dtype.itemsize > dest_type.itemsize):
            copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], numpy.ascontiguousarray(data, dest_type))
            events.append(EventDescription("copy H->D %s" % dest, copy_image))
        else:
            copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], numpy.ascontiguousarray(data))
            kernel = getattr(self.program, self.mapping[data.dtype.type])
            cast_to_float = kernel(self.queue, (self.size,), None, self.cl_mem["image_raw"], self.cl_mem[dest])
            events += [EventDescription("copy H->D %s" % dest, copy_image), EventDescription("cast to float", cast_to_float)]
        if self.profile:
            self.events += events

    def medfilt2d(self, image, kernel_size=None):
        """Actually apply the median filtering on the image

        :param image: numpy array with the image
        :param kernel_size: 2-tuple if
        :return: median-filtered  2D image
        """
        events = []
        with self.sem:
            self.send_buffer(image, "image")
            mf2d = self.program.medfilt2d(self.queue,
                                          (self.workgroup_size[0], self.shape[0], self.shape[1]),
                                          self.workgroup_size, *list(self.cl_kernel_args["medfilt2d"].values()))
            events.append(EventDescription("median filter 2d", mf2d))

            result = numpy.empty(image.shape, "float32")
            ev = pyopencl.enqueue_copy(self.queue, result, self.cl_mem["output"])
            events.append(EventDescription("copy D->H output", ev))
            ev.wait()
        if self.profile:
            self.events += events
        return result



def medfilt2d(ary, kernel_size=3):
    """Median filter a 2-dimensional array.

    Apply a median filter to the `input` array using a local window-size
    given by `kernel_size` (must be odd).

    :param ary: A 2-dimensional input array.
    :param kernel_size: A scalar or a list of length 2, giving the size of the
                        median filter window in each dimension.  Elements of
                        `kernel_size` should be odd.  If `kernel_size` is a scalar,
                        then this scalar is used as the size in each dimension.
                        Default is a kernel of size (3, 3).
    :return: An array the same size as input containing the median filtered
            result.

    About the padding: the boundary value looks duplicated in scipy
    """
    image = numpy.atleast_2d(ary)
    kernel_size = asarray(kernel_size)
    if kernel_size.shape == ():
        kernel_size = np.repeat(kernel_size.item(), 2)
    for size in kernel_size:
        if (size % 2) != 1:
            raise ValueError("Each element of kernel_size should be odd.")

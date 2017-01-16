# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
OpenCL implementation of the preproc module
"""

from __future__ import absolute_import, print_function, division

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "16/01/2017"
__copyright__ = "2015, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import logging
import threading
import gc
import numpy
from .opencl import ocl, pyopencl, allocate_cl_buffers, release_cl_buffers
from .utils import concatenate_cl_kernel
if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")

try:
    from .ext.fastcrc import crc32
except:
    from zlib import crc32
logger = logging.getLogger("pyFAI.ocl_azim_csr")


class OCL_Preproc(object):

    def __init__(self, image_size=None, image_dtype=None, template=None,
                 dark=None, flat=None, solidangle=None, polarization=None, absorption=None,
                 mask=None, dummy=None, delta_dummy=None, empty=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=32, profile=False,
                 ):
        """
        :param image_size: (int) number of element of the input image 
        :param image_dtype: dtype of the input image
        :param template_image: retrieve image_size and image_dtype from example
        
        :param absorption: 
        :param mask:
        :param dummy:
        :param delta_dummy:
        :param empty: value to be assigned to bins without contribution from any pixel
        :param ctx: already existing context to use
        :param devicetype: can be "cpu","gpu","acc" or "all"
        :param block_size: the chosen size for WORKGROUP_SIZE
        :param platformid: number of the platform as given by clinfo
        :type platformid: int
        :param deviceid: number of the device as given by clinfo
        :type deviceid: int
        :param profile: store profiling elements (makes code slower)
        
        """
        self._sem = threading.Semaphore()

    def __del__(self):
        """Destructor: release all buffers
        """
        self._free_kernels()
        self._free_buffers()
        self._queue = None
        self.ctx = None
        gc.collect()

#     def __copy__(self):
#         """Shallow copy of the object
#
#         :return: copy of the object
#         """
#         return self.__class__(...)
#
#     def __deepcopy__(self, memo=None):
#         """deep copy of the object
#
#         :return: deepcopy of the object
#         """
#         if memo is None:
#             memo = {}
#         # TODO
#         return new_obj

    def _allocate_buffers(self):
        """
        Allocate OpenCL buffers required for a specific configuration

        Note that an OpenCL context also requires some memory, as well
        as Event and other OpenCL functionalities which cannot and are
        not taken into account here.  The memory required by a context
        varies depending on the device. Typical for GTX580 is 65Mb but
        for a 9300m is ~15Mb In addition, a GPU will always have at
        least 3-5Mb of memory in use.  Unfortunately, OpenCL does NOT
        have a built-in way to check the actual free memory on a
        device, only the total memory.
        """
        buffers = [
            ("output", mf.WRITE_ONLY, numpy.float32, self.size * 3),
            ("image_raw", mf.READ_ONLY, numpy.float32, self.size),
            ("image", mf.READ_WRITE, numpy.float32, self.size),
            ("dark", mf.READ_ONLY, numpy.float32, self.size),
            ("flat", mf.READ_ONLY, numpy.float32, self.size),
            ("polarization", mf.READ_ONLY, numpy.float32, self.size),
            ("solidangle", mf.READ_ONLY, numpy.float32, self.size),
        ]

        if self.size < self.BLOCK_SIZE:
            raise RuntimeError("Fatal error in _allocate_buffers. size (%d) must be >= BLOCK_SIZE (%d)\n",
                               self.size, self.BLOCK_SIZE)  # noqa
        self._cl_mem = allocate_cl_buffers(buffers, self.device, self.ctx)

    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        self._cl_mem = release_cl_buffers(self._cl_mem)

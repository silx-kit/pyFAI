# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif/pyFAI
#
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#                            Giannis Ashiotis
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__authors__ = ["Jérôme Kieffer", "Giannis Ashiotis"]
__license__ = "GPLv3"
__date__ = "04/04/2014"
__copyright__ = "2014, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os, gc, logging
import threading
import hashlib
import numpy
from .opencl import ocl, pyopencl
from .splitBBoxLUT import HistoBBox1d
from .utils import get_cl_file
if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")
try:
    from .fastcrc import crc32
except:
    from zlib import crc32
logger = logging.getLogger("pyFAI.OCLFullSplit")


class OCLFullSplit1d(object):
    def __init__(self,
                 pos,
                 bins=100,
                 pos0Range=None,
                 pos1Range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False,
                 unit="undefined",
                 workgroup_size=256,
                 devicetype="all",
                 platformid=None,
                 deviceid=None,
                 profile=False):
        
        
        self.bins = bins
        self.lut_size = 0
        self.allow_pos0_neg = allow_pos0_neg

        if len(pos.shape) == 3:
            assert pos.shape[1] == 4
            assert pos.shape[2] == 2
        elif len(pos.shape) == 4:
            assert pos.shape[2] == 4
            assert pos.shape[3] == 2
        else:
            raise ValueError("Pos array dimentions are wrong")
        self.size = pos.size/8
        self.pos = numpy.ascontiguousarray(pos.ravel(), dtype=numpy.float32)
        self.pos0Range = numpy.empty(2,dtype=numpy.float32)
        self.pos1Range = numpy.empty(2,dtype=numpy.float32)
        
        if (pos0Range is not None) and (len(pos0Range) is 2):
            self.pos0Range[0] = min(pos0Range) # do it on GPU?
            self.pos0Range[1] = max(pos0Range)
            if (not self.allow_pos0_neg) and (self.pos0Range[0] < 0):
                self.pos0Range[0] = 0.0
                if self.pos0Range[1] < 0:
                    print "Warning: Invalid 0-dim range! Using the data derived range instead"
                    self.pos0Range[1] = 0.0
            #self.pos0Range[0] = pos0Range[0]
            #self.pos0Range[1] = pos0Range[1]
        else:
            self.pos0Range[0] = 0.0
            self.pos0Range[1] = 0.0
        if (pos1Range is not None) and (len(pos1Range) is 2):
            self.pos1Range[0] = min(pos1Range) # do it on GPU?
            self.pos1Range[1] = max(pos1Range)
            #self.pos1Range[0] = pos1Range[0]
            #self.pos1Range[1] = pos1Range[1]
        else:
            self.pos1Range[0] = 0.0
            self.pos1Range[1] = 0.0
            
        if  mask is not None:
            assert mask.size == self.size
            self.check_mask = True
            self.cmask = numpy.ascontiguousarray(mask.ravel(), dtype=numpy.int8)
            if mask_checksum:
                self.mask_checksum = mask_checksum
            else:
                self.mask_checksum = crc32(mask)
        else:
            self.check_mask = False
            self.mask_checksum = None
        
        self._sem = threading.Semaphore()
        self.profile = profile
        self._cl_kernel_args = {}
        self._cl_mem = {}
        self.events = []
        self.workgroup_size = workgroup_size
        if self.size < self.workgroup_size:
            raise RuntimeError("Fatal error in workgroup size selection. Size (%d) must be >= workgroup size (%d)\n", self.size, self.workgroup_size)
        if (platformid is None) and (deviceid is None):
            platformid, deviceid = ocl.select_device(devicetype)
        elif platformid is None:
            platformid = 0
        elif deviceid is None:
            deviceid = 0
        self.platform = ocl.platforms[platformid]
        self.device = self.platform.devices[deviceid]
        self.device_type = self.device.type
        
        self.compile_options = "-D BINS=%i -D SIZE=%i -D WORKGROUP_SIZE=%i -D EPS=%e" % (self.bins, self.size, self.workgroup_size, numpy.finfo(numpy.float32).eps)
            
        if (self.device_type == "CPU") and (self.platform.vendor == "Apple"):
            logger.warning("This is a workaround for Apple's OpenCL on CPU: enforce BLOCK_SIZE=1")
            self.workgroup_size = 1
        try:
            self._ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
            if self.profile:         
                self._queue = pyopencl.CommandQueue(self._ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            else:
                self._queue = pyopencl.CommandQueue(self._ctx)
            self._calc_boundaries()
            self._calc_LUT()
        except pyopencl.MemoryError as error:
            raise MemoryError(error)

    def _calc_boundaries(self):
        """
        comments
        """
        # Check for memory and allocate
        size_of_float = numpy.dtype(numpy.float32).itemsize
        size_of_int = numpy.dtype(numpy.int32).itemsize
        
        ualloc  = (self.size * 8 * size_of_float)
        ualloc += (self.workgroup_size * 4 * size_of_float)
        ualloc += (4 * size_of_float)
        
        memory = self.device.memory

        pass
        #check memory of d_pos + d_preresult + d_minmax
        #load d_pos
        #allocate d_preresult
        #allocate d_minmax
        #run reduce1
        #run reduce2
        #save reference to d_minMax
        #free d_preresult
        
        
    def _calc_LUT(self):
        pass
        #check memory of d_pos + d_minmax + d_outMax + d_lutsize
        #allocate d_outMax
        #allocate d_lutsize
        #memset d_outMax
        #run lut1
        #run lut2
        #save d_lutsize
        #memset d_outMax
        #allocate d_data
        #allocate d_indices
        #run lut3
        #free d_pos
        #free d_minMax
        #free d_lutsize
        #run lut4
        #free d_outMax
        
    def get_platform(self):
        pass
    
    def get_queue(self):
        pass
    
    
    
# -*- coding: utf8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

__author__ = "Jerome Kieffer"
__license__ = "GPLv3"
__date__ = "18/10/2012"
__copyright__ = "2012, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os
import numpy
import pyopencl
from ocl_azim import OpenCL
from splitBBoxLUT import HistoBBox1d
ocl = OpenCL()
mf = pyopencl.mem_flags

class OCL_LUT_Integrator(object):
    def __init__(self, lut, devicetype="all", platformid=None, deviceid=None):
        """
        @param lut: array of uint32-float32 with shape (nbins, lut_size) with indexes and 
        """
        self._lut = lut
        self.bins,self.lut_size= lut.shape
        if (platformid is None) and (deviceid is None):
            platformid, deviceid = ocl.select_device(devicetype)
        elif platformid is None:
            platformid = 0
        elif deviceid is None:
            deviceid = 0
        self.platform =  ocl.platforms[platformid]
        self.device = self.platform .devices[deviceid]
        self.device_type = self.device.type
        self._ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
        self._queue = pyopencl.CommandQueue(self._ctx)
        self._program = pyopencl.Program(self._ctx, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"ocl_azim_LUT.cl")).read()).build()
                                         
        if self.device_type == "CPU":
            self._lut_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integ.lut)
        else:
            self._lut_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lut.T.copy())
        
    def integrate(self,data):
        data_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        outData_buf = pyopencl.Buffer(self._ctx, mf.WRITE_ONLY, numpy.dtype(numpy.float32).itemsize * self.bins)
        outCount_buf = pyopencl.Buffer(self._ctx, mf.WRITE_ONLY, numpy.dtype(numpy.float32).itemsize * self.bins)  
        outMerge_buf = pyopencl.Buffer(self._ctx, mf.WRITE_ONLY, numpy.dtype(numpy.float32).itemsize * self.bins)
        args = (data_buffer, numpy.uint32(self.bins), numpy.uint32(self.lut_size),self._lut_buffer,
               numpy.int32(0), numpy.float32(0), numpy.float32(0), outData_buf, outCount_buf, outMerge_buf)

        if self.device_type == "CPU":
            self._program.lut_integrate_single(self._queue, (self.bins,), (16,), *args)
        else:
            self._program.lut_integrate_lutT(self._queue, (self.bins,), (16,), *args)
        output = numpy.empty(self.bins, dtype=numpy.float32)
        pyopencl.enqueue_copy(self._queue, output, outMerge_buf).wait()
        return output
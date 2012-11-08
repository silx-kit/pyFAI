# -*- coding: utf-8 -*-
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
import threading
import hashlib
import numpy
from opencl import ocl, pyopencl
from splitBBoxLUT import HistoBBox1d
if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")

class OCL_LUT_Integrator(object):
    def __init__(self, lut, devicetype="all", platformid=None, deviceid=None, checksum=None):
        """
        @param lut: array of uint32 - float32 with shape (nbins, lut_size) with indexes and coefficients
        @param checksum: pre - calculated checksum to prevent re - calculating it :)
        """
        self._sem = threading.Semaphore()
        self._lut = lut
        if checksum:
            self.lut_checksum = checksum
        else:
            self.lut_checksum = hashlib.md5(self._lut).hexdigest()
        self.ws = 16
        self.bins, self.lut_size = lut.shape
        if (platformid is None) and (deviceid is None):
            platformid, deviceid = ocl.select_device(devicetype)
        elif platformid is None:
            platformid = 0
        elif deviceid is None:
            deviceid = 0
        self.platform = ocl.platforms[platformid]
        self.device = self.platform .devices[deviceid]
        self.device_type = self.device.type
        self.data_buffer = None
        try:
            self._ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
            self._queue = pyopencl.CommandQueue(self._ctx)
            self._program = pyopencl.Program(self._ctx, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocl_azim_LUT.cl")).read()).build()
            if self.device_type == "CPU":
                self._lut_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lut)
            else:
                self._lut_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lut.T.copy())
            self.outData_buffer = pyopencl.Buffer(self._ctx, mf.WRITE_ONLY, numpy.dtype(numpy.float32).itemsize * self.bins)
            self.outCount_buffer = pyopencl.Buffer(self._ctx, mf.WRITE_ONLY, numpy.dtype(numpy.float32).itemsize * self.bins)
            self.outMerge_buffer = pyopencl.Buffer(self._ctx, mf.WRITE_ONLY, numpy.dtype(numpy.float32).itemsize * self.bins)
            self.dark_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size=1)
            self.flat_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size=1)
            self.solidAngle_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size=1)
            self.polarization_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size=1)
        except pyopencl.MemoryError as error:
            raise MemoryError(error)

    def __del__(self):
        """
        Destructor: release all buffers
        """
        for buffer in (self.outData_buffer, self.outCount_buffer,
                       self.outMerge_buffer, self.data_buffer,
                       self.flat_buffer, self.solidAngle_buffer,
                       self.polarization_buffer, self._lut_buffer):
            if buffer is not None:
                try:
                    buffer.release()
                except LogicError:
                    pass

    def get_nr_threads(self, size=None, ws=None):
        """calculate the number of threads, multiple of workgroup-size and greater than bins"""
        if size is None:
            size = self.bins
        if ws is None:
            ws = self.ws
        r = size % ws
        if r == 0:
            return size
        else:
            return ws * (1 + size // ws)

    def integrate(self, data, dummy=None, delta_dummy=None, dark=None, flat=None, solidAngle=None, polarization=None):
        data = numpy.ascontiguousarray(data, dtype=numpy.float32)
        with self._sem:
            try:
                if dummy is not None:
                    do_dummy = 1
                    if delta_dummy == None:
                        delta_dummy = 0
                else:
                    do_dummy = 0
                    dummy = 0
                    delta_dummy = 0

                if self.data_buffer is None:
                    self.data_buffer = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size=data.nbytes)
                elif self.data_buffer.size != data.nbytes:
                    self.data_buffer.release()
                    self.data_buffer = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size=data.nbytes)
                pyopencl.enqueue_copy(self._queue, self.data_buffer, data)

                if dark is not None:
                    dark = numpy.ascontiguousarray(dark, dtype=numpy.float32)
                    do_dark = 1
                    if self.dark_buffer.size != dark.nbytes:
                        self.dark_buffer.release()
                        self.dark_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dark)
                    else:
                        pyopencl.enqueue_copy(self._queue, self.dark_buffer, dark)
                    dark = None
                else:
                    do_dark = 0
                if flat is not None:
                    flat = numpy.ascontiguousarray(flat, dtype=numpy.float32)
                    do_flat = 1
                    if self.flat_buffer.size != flat.nbytes:
                        self.flat_buffer.release()
                        self.flat_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
                    else:
                        pyopencl.enqueue_copy(self._queue, self.flat_buffer, flat)
                    flat = None
                else:
                    do_flat = 0

                if solidAngle is not None:
                    solidAngle = numpy.ascontiguousarray(solidAngle, dtype=numpy.float32)
                    do_solidAngle = 1
                    if self.solidAngle_buffer.size != solidAngle.nbytes:
                        self.solidAngle_buffer.release()
                        self.solidAngle_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=solidAngle)
                    else:
                        pyopencl.enqueue_copy(self._queue, self.solidAngle_buffer, solidAngle)
                    solidAngle = None
                else:
                    do_solidAngle = 0

                if polarization is not None:
                    polarization = numpy.ascontiguousarray(polarization, dtype=numpy.float32)
                    do_polarization = 1
                    if self.polarization_buffer.size != polarization.nbytes:
                        self.polarization_buffer.release()
                        self.polarization_buffer = pyopencl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=polarization)
                    else:
                        pyopencl.enqueue_copy(self._queue, self.polarization_buffer, polarization)
                else:
                    do_polarization = 0
                if do_polarization + do_solidAngle + do_flat + do_dark > 0:
                    args_cor = (self.data_buffer, numpy.uint32(data.size),
                                numpy.int32(do_dark), self.dark_buffer,
                                numpy.int32(do_flat), self.flat_buffer,
                                numpy.int32(do_solidAngle), self.solidAngle_buffer,
                                numpy.int32(do_polarization), self.polarization_buffer,
                                numpy.int32(do_dummy), numpy.float32(dummy), numpy.float32(delta_dummy))
                    pyopencl.enqueue_barrier(self._queue).wait()
                    self._program.corrections(self._queue, (self.get_nr_threads(data.size, 512),), (512,), *args_cor)

                args = (self.data_buffer, numpy.uint32(self.bins), numpy.uint32(self.lut_size), self._lut_buffer,
                       numpy.int32(do_dummy), numpy.float32(dummy), numpy.float32(delta_dummy), self.outData_buffer, self.outCount_buffer, self.outMerge_buffer)
                if self.device_type == "CPU":
                    self._program.lut_integrate_single(self._queue, (self.get_nr_threads(),), (self.ws,), *args)
                else:
                    self._program.lut_integrate_lutT(self._queue, (self.get_nr_threads(),), (self.ws,), *args)
                output = numpy.zeros(self.bins, dtype=numpy.float32)
                pyopencl.enqueue_barrier(self._queue).wait()
                pyopencl.enqueue_copy(self._queue, output, self.outMerge_buffer).wait()
            except pyopencl.MemoryError as error:
                raise MemoryError(error)
        return output

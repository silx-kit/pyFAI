#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
# 
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
# 
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Jerome Kieffer"
__license__ = "GPLv3+"
__date__ = "28/01/2016"
__copyright__ = "2011-2014, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
cimport numpy
import numpy
from cython.parallel import prange  
from libc.math cimport floor, ceil, fabs
import logging
import threading
import types
import os
import sys
import time
logger = logging.getLogger("pyFAI._distortionCSR")
from ..detectors import detector_factory
from ..decorators import timeit
import fabio
from . import ocl_azim_csr_dis

cdef struct lut_point:
    numpy.int32_t idx
    numpy.float32_t coef

cdef bint NEED_DECREF = sys.version_info < (2, 7) and numpy.version.version < "1.5"


cpdef inline float calc_area(float I1, float I2, float slope, float intercept) nogil:
    "Calculate the area between I1 and I2 of a line with a given slope & intercept"
    return 0.5 * (I2 - I1) * (slope * (I2 + I1) + 2 * intercept)


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline void integrate(float[:, :] box, float start, float stop, float slope, float intercept) nogil:
    "Integrate in a box a line between start and stop, line defined by its slope & intercept "
    cdef int i, h = 0
    cdef float P, dP, A, AA, dA, sign
    if start < stop:  # positive contribution
        P = ceil(start)
        dP = P - start
        if P > stop:  # start and stop are in the same unit
            A = calc_area(start, stop, slope, intercept)
            if A != 0:
                AA = fabs(A)
                sign = A / AA
                dA = (stop - start)  # always positive
                h = 0
                while AA > 0:
                    if dA > AA:
                        dA = AA
                        AA = -1
                    box[(<int> floor(start)), h] += sign * dA
                    AA -= dA
                    h += 1
        else:
            if dP > 0:
                A = calc_area(start, P, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = dP
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[(<int> floor(P)) - 1, h] += sign * dA
                        AA -= dA
                        h += 1
            # subsection P1->Pn
            for i in range((<int> floor(P)), (<int> floor(stop))):
                A = calc_area(i, i + 1, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA

                    h = 0
                    dA = 1.0
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[i, h] += sign * dA
                        AA -= dA
                        h += 1
            # Section Pn->B
            P = floor(stop)
            dP = stop - P
            if dP > 0:
                A = calc_area(P, stop, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[(<int> floor(P)), h] += sign * dA
                        AA -= dA
                        h += 1
    elif start > stop:  # negative contribution. Nota is start=stop: no contribution
        P = floor(start)
        if stop > P:  # start and stop are in the same unit
            A = calc_area(start, stop, slope, intercept)
            if A != 0:
                AA = fabs(A)
                sign = A / AA
                dA = (start - stop)  # always positive
                h = 0
                while AA > 0:
                    if dA > AA:
                        dA = AA
                        AA = -1
                    box[(<int> floor(start)), h] += sign * dA
                    AA -= dA
                    h += 1
        else:
            dP = P - start
            if dP < 0:
                A = calc_area(start, P, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[(<int> floor(P)) , h] += sign * dA
                        AA -= dA
                        h += 1
            # subsection P1->Pn
            for i in range((<int> start), (<int> ceil(stop)), -1):
                A = calc_area(i, i - 1, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = 1
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[i - 1 , h] += sign * dA
                        AA -= dA
                        h += 1
            # Section Pn->B
            P = ceil(stop)
            dP = stop - P
            if dP < 0:
                A = calc_area(P, stop, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA; AA = -1
                        box[(<int> floor(stop)), h] += sign * dA
                        AA -= dA
                        h += 1


class Distortion(object):
    """

    This class applies a distortion correction on an image.

    It is also able to apply an inversion of the correction.

    """
    def __init__(self, detector="detector", shape=None, compute_device="Host", workgroup_size=32):
        """
        @param detector: detector instance or detector name
        """
        if type(detector) in types.StringTypes:
            self.detector = detector_factory(detector)
        else:  # we assume it is a Detector instance
            self.detector = detector
        if "max_shape" in dir(self.detector):
            self.shape = self.detector.max_shape
        else:
            self.shsizeape = shape
        self.shape = tuple([int(i) for i in self.shape])
        self.bins = self.shape[0] * self.shape[1]
        self._sem = threading.Semaphore()
        self.bin_size = None
        self.lut_size = None
        self.pos = None
        self.LUT = None
        self.delta0 = self.delta1 = None  # max size of an pixel on a regular grid ...
        self.integrator = None
        self.compute_device = compute_device
        self.workgroup_size = workgroup_size

    def __repr__(self):
        return os.linesep.join(["Distortion correction for detector:",
                                self.detector.__repr__()])

    def calc_pos(self):
        if self.pos is None:
            with self._sem:
                if self.pos is None:
                    pos_corners = numpy.empty((self.shape[0] + 1, self.shape[1] + 1, 2), dtype=numpy.float64)
                    d1 = numpy.outer(numpy.arange(self.shape[0] + 1, dtype=numpy.float64), numpy.ones(self.shape[1] + 1, dtype=numpy.float64)) - 0.5
                    d2 = numpy.outer(numpy.ones(self.shape[0] + 1, dtype=numpy.float64), numpy.arange(self.shape[1] + 1, dtype=numpy.float64)) - 0.5
                    pos_corners[:, :, 0], pos_corners[:, :, 1] = self.detector.calc_cartesian_positions(d1, d2)[:2]
                    pos_corners[:, :, 0] /= self.detector.pixel1
                    pos_corners[:, :, 1] /= self.detector.pixel2
                    pos = numpy.empty((self.shape[0], self.shape[1], 4, 2), dtype=numpy.float32)
                    pos[:, :, 0, :] = pos_corners[:-1, :-1]
                    pos[:, :, 1, :] = pos_corners[:-1, 1:]
                    pos[:, :, 2, :] = pos_corners[1:, 1:]
                    pos[:, :, 3, :] = pos_corners[1:, :-1]
                    self.pos = pos
                    self.delta0 = int((numpy.ceil(pos_corners[1:, :, 0]) - numpy.floor(pos_corners[:-1, :, 0])).max())
                    self.delta1 = int((numpy.ceil(pos_corners[:, 1:, 1]) - numpy.floor(pos_corners[:, :-1, 1])).max())
        return self.pos

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def calc_LUT_size(self):
        """
        Considering the "half-CCD" spline from ID11 which describes a (1025,2048) detector,
        the physical location of pixels should go from:
        [-17.48634 : 1027.0543, -22.768829 : 2028.3689]
        We chose to discard pixels falling outside the [0:1025,0:2048] range with a lose of intensity

        We keep self.pos: pos_corners will not be compatible with systems showing non adjacent pixels (like some xpads)

        """
        cdef int i, j, k, l, shape0, shape1
        cdef numpy.ndarray[numpy.float32_t, ndim = 4] pos
        cdef int[:, :] pos0min, pos1min, pos0max, pos1max
        cdef numpy.ndarray[numpy.int32_t, ndim = 2] lut_size
        if self.pos is None:
            pos = self.calc_pos()
        else:
            pos = self.pos
        if self.lut_size is None:
            with self._sem:
                if self.lut_size is None:
                    shape0, shape1 = self.shape
                    pos0min = numpy.floor(pos[:, :, :, 0].min(axis=-1)).astype(numpy.int32).clip(0, self.shape[0])
                    pos1min = numpy.floor(pos[:, :, :, 1].min(axis=-1)).astype(numpy.int32).clip(0, self.shape[1])
                    pos0max = (numpy.ceil(pos[:, :, :, 0].max(axis=-1)).astype(numpy.int32) + 1).clip(0, self.shape[0])
                    pos1max = (numpy.ceil(pos[:, :, :, 1].max(axis=-1)).astype(numpy.int32) + 1).clip(0, self.shape[1])
                    lut_size = numpy.zeros(self.shape, dtype=numpy.int32)
                    with nogil:
                        for i in range(shape0):
                            for j in range(shape1):
                                for k in range(pos0min[i, j], pos0max[i, j]):
                                    for l in range(pos1min[i, j], pos1max[i, j]):
                                        lut_size[k, l] += 1
                    self.bin_size = lut_size.ravel()
                    self.lut_size = self.bin_size.sum()
                    return lut_size

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def calc_LUT(self):
        cdef:
            int i, j, ms, ml, ns, nl, shape0, shape1, delta0, delta1, buffer_size, i0, i1, size
            int offset0, offset1, box_size0, box_size1, bins, tmp_index  
            numpy.int32_t k, idx=0
            float A0, A1, B0, B1, C0, C1, D0, D1, pAB, pBC, pCD, pDA, cAB, cBC, cCD, cDA, area, value
            float[:, :, :, :] pos
            numpy.ndarray[numpy.int32_t, ndim = 2] outMax = numpy.zeros(self.shape, dtype=numpy.int32)
            float[:, :] buffer
            numpy.ndarray[numpy.int32_t, ndim = 1] indptr
            numpy.ndarray[numpy.int32_t, ndim = 1] indices 
            numpy.ndarray[numpy.float32_t, ndim = 1] data
            numpy.ndarray[numpy.int32_t, ndim = 1] bin_size

        shape0, shape1 = self.shape
 
        bin_size = self.bin_size

        if self.lut_size is None:
            self.calc_LUT_size()
        if self.LUT is None:
            with self._sem:
                if self.LUT is None:
                    pos = self.pos
                    
                    indices = numpy.zeros(shape=self.lut_size, dtype=numpy.int32)
                    data = numpy.zeros(shape=self.lut_size, dtype=numpy.float32)
                    
                    bins = shape0 * shape1
                    indptr = numpy.zeros(bins + 1, dtype=numpy.int32)
                    indptr[1:] = bin_size.cumsum(dtype=numpy.int32)
                    
                    indices_size = self.lut_size * sizeof(numpy.int32)
                    data_size = self.lut_size * sizeof(numpy.float32)
                    indptr_size = bins * sizeof(numpy.int32)
                    
                    logger.info("CSR matrix: %.3f MByte" % ((indices_size+data_size+indptr_size)/1.0e6))
                    buffer = numpy.empty((self.delta0, self.delta1),dtype=numpy.float32)
                    buffer_size = self.delta0 * self.delta1 * sizeof(float)
                    logger.info("Max pixel size: %ix%i; Max source pixel in target: %i" % (buffer.shape[1],buffer.shape[0], self.lut_size))
                    with nogil:
                        # i,j, idx are indices of the raw image uncorrected
                        for i in range(shape0):
                            for j in range(shape1):
                                # reinit of buffer
                                buffer[:, :] = 0
                                A0 = pos[i, j, 0, 0]
                                A1 = pos[i, j, 0, 1]
                                B0 = pos[i, j, 1, 0]
                                B1 = pos[i, j, 1, 1]
                                C0 = pos[i, j, 2, 0]
                                C1 = pos[i, j, 2, 1]
                                D0 = pos[i, j, 3, 0]
                                D1 = pos[i, j, 3, 1]
                                offset0 = (<int> floor(min(A0, B0, C0, D0)))
                                offset1 = (<int> floor(min(A1, B1, C1, D1)))
                                box_size0 = (<int> ceil(max(A0, B0, C0, D0))) - offset0
                                box_size1 = (<int> ceil(max(A1, B1, C1, D1))) - offset1
                                A0 -= <float> offset0
                                A1 -= <float> offset1
                                B0 -= <float> offset0
                                B1 -= <float> offset1
                                C0 -= <float> offset0
                                C1 -= <float> offset1
                                D0 -= <float> offset0
                                D1 -= <float> offset1
                                if B0 != A0:
                                    pAB = (B1 - A1) / (B0 - A0)
                                    cAB = A1 - pAB * A0
                                else:
                                    pAB = cAB = 0.0
                                if C0 != B0:
                                    pBC = (C1 - B1) / (C0 - B0)
                                    cBC = B1 - pBC * B0
                                else:
                                    pBC = cBC = 0.0
                                if D0 != C0:
                                    pCD = (D1 - C1) / (D0 - C0)
                                    cCD = C1 - pCD * C0
                                else:
                                    pCD = cCD = 0.0
                                if A0 != D0:
                                    pDA = (A1 - D1) / (A0 - D0)
                                    cDA = D1 - pDA * D0
                                else:
                                    pDA = cDA = 0.0
                                integrate(buffer, B0, A0, pAB, cAB)
                                integrate(buffer, A0, D0, pDA, cDA)
                                integrate(buffer, D0, C0, pCD, cCD)
                                integrate(buffer, C0, B0, pBC, cBC)
                                area = 0.5*((C0 - A0)*(D1 - B1)-(C1 - A1)*(D0 - B0))
                                for ms in range(box_size0):
                                    ml = ms + offset0
                                    if ml < 0 or ml >= shape0:
                                        continue
                                    for ns in range(box_size1):
                                        # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                                        nl = ns + offset1
                                        if nl < 0 or nl >= shape1:
                                            continue
                                        value = buffer[ms, ns] / area
                                        if value <= 0:
                                            continue
                                        k = outMax[ml,nl]
                                        tmp_index = indptr[ml*shape1+nl]
                                        indices[tmp_index+k] = idx
                                        data[tmp_index+k] = value
                                        outMax[ml,nl] = k + 1
                                idx += 1
                        #for i in range(bins):
                            #tmp_index = indptr[i]
                            #index_tmp_index = indices[tmp_index + bin_size[i] - 1]
                            #for j in range(tmp_index + bin_size[i], tmp_index + bin_size_padded[i]):
                                #indices[j] = index_tmp_index
                    self.LUT = (data, indices, indptr)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def correctHost(self, image):
        """
        Correct an image based on the look-up table calculated ...
        Calculation takes place on the Host

        @param image: 2D-array with the image
        @return: corrected 2D image
        """
        cdef int i, j, idx, size, bins
        cdef float coef, tmp
        cdef float[:] lout, lin, data
        cdef int[:] indices, indptr
        if self.LUT is None:
            self.calc_LUT()
        data = self.LUT[0]
        indices = self.LUT[1]
        indptr = self.LUT[2]
        bins = self.bins
        img_shape = image.shape
        if (img_shape[0] < self.shape[0]) or (img_shape[1] < self.shape[1]):
            new_image = numpy.zeros(self.shape, dtype=numpy.float32)
            new_image[:img_shape[0], :img_shape[1]] = image
            image = new_image
            logger.warning("Patching image as image is %ix%i and spline is %ix%i" % (img_shape[1], img_shape[0], self.shape[1], self.shape[0]))

        out = numpy.zeros(self.shape, dtype=numpy.float32)
        lout = out.ravel()
        lin = numpy.ascontiguousarray(image.ravel(), dtype=numpy.float32)
        size = lin.size

        for i in prange(bins, nogil=True, schedule="static"):
            for j in range(indptr[i], indptr[i + 1]):
                idx = indices[j]
                coef = data[j]
                if coef <= 0:
                    continue
                if idx >= size:
                    with gil:
                        logger.warning("Accessing %i >= %i !!!" % (idx, size))
                        continue
                tmp = lout[i] + lin[idx] * coef
                lout[i] = tmp 
        return out[:img_shape[0], :img_shape[1]]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def correctDevice(self, image):
        """
        Correct an image based on the look-up table calculated ...
        Calculation takes place on the device

        @param image: 2D-array with the image
        @return: corrected 2D image
        """
        if self.integrator is None:
            if self.LUT is None:
                self.calc_LUT()
            self.integrator = ocl_azim_csr_dis.OCL_CSR_Integrator(self.LUT, self.bins, "GPU", block_size=self.workgroup_size)
        img_shape = image.shape
        if (img_shape[0] < self.shape[0]) or (img_shape[1] < self.shape[1]):
            new_image = numpy.zeros(self.shape, dtype=numpy.float32)
            new_image[:img_shape[0], :img_shape[1]] = image
            image = new_image
            logger.warning("Patching image as image is %ix%i and spline is %ix%i" % (img_shape[1], img_shape[0], self.shape[1], self.shape[0]))

        out = self.integrator.integrate(image)
        out[1].shape = self.shape
        return out[1][:img_shape[0], :img_shape[1]]
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def correct(self, image):
        if self.compute_device == "Host":
            out = self.correctHost(image)
        elif self.compute_device == "Device":
            out = self.correctDevice(image)
        else:
            logger.warning("Please select a compute device (Host or Device)")
        return out
        
    def setHost(self):
        self.compute_device = "Host"
        
    def setDevice(self):
        self.compute_device = "Device"
                
    @timeit
    def uncorrect(self, image):
        """
        Take an image which has been corrected and transform it into it's raw (with loss of information)

        @param image: 2D-array with the image
        @return: uncorrected 2D image and a mask (pixels in raw image
        """
        cdef int[:] indices, indptr
        cdef float[:] data
        cdef int idx, bins
        if self.LUT is None:
            with self._sem:
                if self.LUT is None:
                    self.calc_LUT()
        out = numpy.zeros(self.shape, dtype=numpy.float32)
        mask = numpy.zeros(self.shape, dtype=numpy.int8)
        lmask = mask.ravel()
        lout = out.ravel()
        lin = image.ravel()
        
        data = self.LUT[0]
        indices = self.LUT[1]
        indptr = self.LUT[2]
        bins = self.bins
        for idx in range(bins):
            idx1 = indptr[idx]
            idx2 = indptr[idx+1]
            if idx1 == idx2:
                lmask[idx] = 1
                continue
            val = lin[idx] / data[idx1:idx2].sum()
            lout[indices[idx1:idx2]] += val * data[idx1:idx2]
                               
        return out, mask

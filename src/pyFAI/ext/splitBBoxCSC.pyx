# coding: utf-8
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping
## cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal Integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""Calculates histograms of pos0 (tth) weighted by Intensity

Splitting is done on the pixel's bounding box like fit2D,
Serial implementation based on a sparse CSC matrix multiplication
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "26/01/2024"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"
include "CSC_common.pxi"

import cython
import os
import sys
import logging
logger = logging.getLogger(__name__)
import numpy
from scipy import sparse
from ..utils import crc32
from ..utils.decorators import deprecated
from .splitBBox_common import SplitBBoxIntegrator

class HistoBBox1d(CscIntegrator, SplitBBoxIntegrator):
    """
    Now uses CSR (Compressed Sparse Raw) as input and converts it into CSC with main attributes:
    * nnz: number of non zero elements
    * data: coefficient of the matrix in a 1D vector of float32
    * indices: Column index position for the data (same size as
    * indptr: row pointer indicates the start of a given row. len nrow+1

    Nota: nnz = indptr[-1]
    """
    def __init__(self,
                 pos0,
                 delta_pos0,
                 pos1=None,
                 delta_pos1=None,
                 int bins=100,
                 pos0_range=None,
                 pos1_range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False,
                 unit="undefined",
                 empty=None,
                 bint chiDiscAtPi=True,
                 bint clip_pos1=False):
        """
        :param pos0: 1D array with pos0: tth or q_vect or r ...
        :param delta_pos0: 1D array with delta pos0: max center-corner distance
        :param pos1: 1D array with pos1: chi
        :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
        :param bins: number of output bins, 100 by default
        :param pos0_range: minimum and maximum  of the 2th range
        :param pos1_range: minimum and maximum  of the chi range
        :param mask: array (of int8) with masked pixels with 1 (0=not masked)
        :param allow_pos0_neg: enforce the q<0 is usually not possible
        :param unit: can be 2th_deg or r_nm^-1 ...
        :param empty: value for bins without contributing pixels
        :param chiDiscAtPi: tell if azimuthal discontinuity is at 0° or 180°
        :param clip_pos1: clip the azimuthal range to [-π π] (or [0 2π] depending on chiDiscAtPi), set to False to deactivate behavior
        """
        self.unit = unit
        self.space = tuple(str(u).split("_")[0] for u in unit) if isinstance(unit, (tuple, list)) else  str(unit).split("_")[0]
        SplitBBoxIntegrator.__init__(self, pos0, delta_pos0, pos1, delta_pos1,
                                     bins, pos0_range, pos1_range,
                                     mask, mask_checksum,
                                     allow_pos0_neg, chiDiscAtPi, clip_pos1=clip_pos1)


        self.delta = (self.pos0_max - self.pos0_min) / (<position_t> (self.bins))
        self.bin_centers = numpy.linspace(self.pos0_min + 0.5 * self.delta,
                                          self.pos0_max - 0.5 * self.delta,
                                          self.bins)
        csc = sparse.csr_matrix(self.calc_lut_1d().to_csr(), shape=(self.bins, self.size)).tocsc()

        #Call the constructor of the parent class
        CscIntegrator.__init__(self, (csc.data, csc.indices, csc.indptr), self.size, self.bins, empty or 0.0)

        self.lut_checksum = crc32(self.data)
        self.lut_nbytes = sum([i.nbytes for i in self.lut])

    @property
    def lut(self):
        return (self.data, self.indices, self.indptr)

    @property
    @deprecated(replacement="bin_centers", since_version="0.16", only_once=True)
    def outPos(self):
        return self.bin_centers

    @property
    def check_mask(self):
        return self.cmask is not None

################################################################################
# Bidimensionnal regrouping
################################################################################


class HistoBBox2d(CscIntegrator, SplitBBoxIntegrator):
    """
    2D histogramming with pixel splitting based on a look-up table stored in CSR format

    The initialization of the class can take quite a while (operation are not parallelized)
    but each integrate is parallelized and efficient.
    """
    @cython.boundscheck(False)
    def __init__(self,
                 pos0,
                 delta_pos0,
                 pos1,
                 delta_pos1,
                 bins=(100, 36),
                 pos0_range=None,
                 pos1_range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False,
                 unit="undefined",
                 empty=None,
                 bint chiDiscAtPi=True,
                 bint clip_pos1=True):
        """
        :param pos0: 1D array with pos0: tth or q_vect
        :param delta_pos0: 1D array with delta pos0: max center-corner distance
        :param pos1: 1D array with pos1: chi
        :param delta_pos1: 1D array with max pos1: max center-corner distance, unused !
        :param bins: number of output bins (tth=100, chi=36 by default)
        :param pos0_range: minimum and maximum  of the 2th range
        :param pos1_range: minimum and maximum  of the chi range
        :param mask: array (of int8) with masked pixels with 1 (0=not masked)
        :param allow_pos0_neg: enforce the q<0 is usually not possible
        :param unit: can be 2th_deg or r_nm^-1 ...
        :param empty: value for bins where no pixels are contributing
        :param chiDiscAtPi: tell if azimuthal discontinuity is at 0 (0° when False) or π (180° when True)
        :param clip_pos1: clip the azimuthal range to [-π π] (or [0 2π] depending on chiDiscAtPi), set to False to deactivate behavior
        """
        SplitBBoxIntegrator.__init__(self, pos0, delta_pos0, pos1, delta_pos1,
                                     bins, pos0_range, pos1_range, mask, mask_checksum, allow_pos0_neg, chiDiscAtPi,
                                     clip_pos1)
        self.unit = unit
        self.space = tuple(str(u).split("_")[0] for u in unit) if isinstance(unit, (tuple, list)) else  str(unit).split("_")[0]
        self.bin_centers = None
        self.delta0 = (self.pos0_max - self.pos0_min) / (<position_t> (self.bins[0]))
        self.delta1 = (self.pos1_max - self.pos1_min) / (<position_t> (self.bins[1]))
        self.bin_centers0 = numpy.linspace(self.pos0_min + 0.5 * self.delta0,
                                           self.pos0_max - 0.5 * self.delta0,
                                           self.bins[0])
        self.bin_centers1 = numpy.linspace(self.pos1_min + 0.5 * self.delta1,
                                           self.pos1_max - 0.5 * self.delta1,
                                           self.bins[1])
        output_size = numpy.prod(self.bins)
        csc = sparse.csr_matrix(self.calc_lut_2d().to_csr(), shape=(output_size, self.size)).tocsc()
        #Call the constructor of the parent class
        CscIntegrator.__init__(self, (csc.data, csc.indices, csc.indptr), self.size, output_size, empty or 0.0)
        self.lut_checksum = crc32(self.data)
        self.lut_nbytes = sum([i.nbytes for i in self.lut])

    @property
    def lut(self):
        return (self.data, self.indices, self.indptr)

    @property
    @deprecated(replacement="bin_centers0", since_version="0.16", only_once=True)
    def outPos0(self):
        return self.bin_centers0

    @property
    @deprecated(replacement="bin_centers1", since_version="0.16", only_once=True)
    def outPos1(self):
        return self.bin_centers1

    @property
    def check_mask(self):
        return self.cmask is not None

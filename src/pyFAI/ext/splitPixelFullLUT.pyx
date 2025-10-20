# coding: utf-8
# cython: embedsignature=True, language_level=3, binding=True
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developing
## cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal Integration
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2014-2023 European Synchrotron Radiation Facility, Grenoble, France
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

"""Full pixel Splitting implemented using Sparse-matrix Dense-Vector multiplication,
Sparse matrix represented using the LUT representation.
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "04/10/2023"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"
include "LUT_common.pxi"

import cython
import os
import sys
import logging
logger = logging.getLogger(__name__)
import numpy
from ..utils import crc32
from ..utils.decorators import deprecated
from .splitpixel_common import FullSplitIntegrator


class HistoLUT1dFullSplit(LutIntegrator, FullSplitIntegrator):
    """
    Now uses LUT representation for the integration
    """
    @cython.boundscheck(False)
    def __init__(self,
                 pos,
                 int bins=100,
                 pos0_range=None,
                 pos1_range=None,
                 mask=None,
                 mask_checksum=None,
                 allow_pos0_neg=False,
                 unit="undefined",
                 empty=None,
                 bint chiDiscAtPi=True):
        """
        :param pos: 3D or 4D array with the coordinates of each pixel point
        :param bins: number of output bins, 100 by default
        :param pos0_range: minimum and maximum  of the 2th range
        :param pos1_range: minimum and maximum  of the chi range
        :param mask: array (of int8) with masked pixels with 1 (0=not masked)
        :param allow_pos0_neg: enforce the q<0 is usually not possible
        :param unit: can be 2th_deg or r_nm^-1 ...
        :param empty: value of output bins without any contribution when dummy is None
        :param chiDiscAtPi: tell if azimuthal discontinuity is at 0° or 180°
        """
        self.unit = unit
        self.space = tuple(str(u).split("_")[0] for u in unit) if isinstance(unit, (list, tuple)) else  str(unit).split("_")[0]
        FullSplitIntegrator.__init__(self, pos, bins, pos0_range, pos1_range, mask, mask_checksum, allow_pos0_neg, chiDiscAtPi)

        self.delta = (self.pos0_max - self.pos0_min) / (<position_t> (self.bins))
        self.bin_centers = numpy.linspace(self.pos0_min + 0.5 * self.delta,
                                          self.pos0_max - 0.5 * self.delta,
                                          self.bins)

        lut = self.calc_lut_1d().to_lut()
        #Call the constructor of the parent class
        LutIntegrator.__init__(self, lut, self.pos.shape[0], empty or 0.0)

        self.lut_checksum = crc32(self.lut)
        self.lut_nbytes = self.lut.nbytes


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

class HistoLUT2dFullSplit(LutIntegrator, FullSplitIntegrator):
    """
    Now uses CSR (Compressed Sparse raw) with main attributes:
    * nnz: number of non zero elements
    * data: coefficient of the matrix in a 1D vector of float32
    * indices: Column index position for the data (same size as
    * indptr: row pointer indicates the start of a given row. len nrow+1

    Nota: nnz = indptr[-1]
    """
    def __init__(self,
                 pos,
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
        :param pos: 3D or 4D array with the coordinates of each pixel point
        :param bins: number of output bins (tth=100, chi=36 by default)
        :param pos0_range: minimum and maximum  of the 2th range
        :param pos1_range: minimum and maximum  of the chi range
        :param mask: array (of int8) with masked pixels with 1 (0=not masked)
        :param allow_pos0_neg: enforce the q<0 is usually not possible
        :param unit: can be 2th_deg or r_nm^-1 ...
        :param empty: value for bins where no pixels are contributing
        :param chiDiscAtPi: tell if azimuthal discontinuity is at 0° or 180°
        :param clip_pos1: True if azimuthal direction is periodic (chi angle), False for non periodic units
        """
        FullSplitIntegrator.__init__(self, pos, bins, pos0_range, pos1_range, mask, mask_checksum, allow_pos0_neg, chiDiscAtPi, clip_pos1)
        self.unit = unit
        self.space = tuple(str(u).split("_")[0] for u in unit) if isinstance(unit, (list, tuple)) else  str(unit).split("_")[0]
        self.bin_centers = None
        self.delta0 = (self.pos0_max - self.pos0_min) / (<position_t> (self.bins[0]))
        self.delta1 = (self.pos1_max - self.pos1_min) / (<position_t> (self.bins[1]))
        self.bin_centers0 = numpy.linspace(self.pos0_min + 0.5 * self.delta0,
                                           self.pos0_max - 0.5 * self.delta0,
                                           self.bins[0])
        self.bin_centers1 = numpy.linspace(self.pos1_min + 0.5 * self.delta1,
                                           self.pos1_max - 0.5 * self.delta1,
                                           self.bins[1])

        lut = self.calc_lut_2d().to_lut()
        #Call the constructor of the parent class
        LutIntegrator.__init__(self, lut, self.pos.shape[0], empty or 0.0)

        self.lut_checksum = crc32(self.lut)
        self.lut_nbytes = lut.nbytes

    @property
    def check_mask(self):
        return self.cmask is not None

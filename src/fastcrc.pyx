#!/usr/bin/env python
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
__date__ = "19-11-2012"
__contact__ = "Jerome.kieffer@esrf.fr"
__license__ = "GPL v3+"
__doc__ = """
Simple Cython module for doing CRC32 for checksums, possibly with SSE4 acceleration
"""

import cython
cimport numpy
import numpy

from crc32 cimport crc32 as C_crc32


def crc32(numpy.ndarray data not None):
    """
    Calculate the CRC32 checksum of a numpy array
    @param data: a numpy array
    @return unsigned integer
    """
    cdef numpy.uint32_t size = data.nbytes
    return C_crc32(<char *> data.data, size)

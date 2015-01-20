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
__doc__ = """
Inverse watershed for connecting region of high intensity
"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "20150120"
__status__ = "stable"
__license__ = "GPLv3+"

import cython
import numpy
cimport numpy

from pyFAI.bilinear import Bilinear

cdef bint get_bit(int byteval, int idx) nogil:
    return ((byteval&(1<<idx))!=0);


cdef class Region:
    cdef:
        int index, size
#        float mini=0, maxi=0
        list neighbors, border
    def __cinit__(self, int idx):
        self.index = idx
        self.neighbors = []
        self.border = [] #list of pixel indices of the border
        self.size = 0

    def __repr__(self):
        return "Region %s of size %s:\n neighbors: %s\n border: %s"%(self.index, self.size, self.neighbors, self.border)

class InverseWatershed(object):
    """
    Idea:

    * label all peaks
    * define region around those peaks which raise always to this peak
    * define the border of such region
    * search for the pass between two peaks
    * merge region with high pass between them

    """
    def __init__(self, data not None):
        self.data = numpy.ascontiguousarray(data, dtype=numpy.float32)
        self.height, self.width  = data.shape
        self.bilinear = Bilinear(data)
        self.regions = dict()
        self.labels = numpy.zeros((self.height, self.width), dtype="int32")
        self.borders = numpy.zeros((self.height, self.width), dtype="uint8")

    def init_labels(self):
        cdef:
            int i, j, width=self.width, height=self.height, idx, res
            numpy.int32_t[:,:] labels = self.labels
            dict regions = self.regions
        for i in range(height):
            for j in range(width):
                idx = j+i*width
                res = self.bilinear.cp_local_maxi(idx)
                labels[i, j] = res
                if idx == res:
                    regions[res] = Region(res)

    def init_borders(self):
        cdef:
            int i, j, width=self.width, height=self.height, idx, res
            numpy.int32_t[:,:] labels = self.labels
            numpy.uint8_t[:,:] borders = self.borders
            numpy.uint8_t neighb=0
        for i in range(height):
            for j in range(width):
                idx = j+i*width
                res =  labels[i,j]
                if res == idx:
                    # local maximum
                    continue
                if i>0 and j>0 and labels[i-1,j-1]!=res:
                    neighb |= 1
                if i>0 and labels[i-1,j]!=res:
                    neighb |= 1<<1
                if i>0 and j<width-1 and labels[i-1,j+1]!=res:
                    neighb |= 1<<2
                if j<width-1 and labels[i,j+1]!=res:
                    neighb |= 1<<3
                if i<height-1 and j<width-1 and labels[i+1,j+1]!=res:
                    neighb |= 1<<4
                if i<height-1 and labels[i+1,j]!=res:
                    neighb |= 1<<5
                if i<height-1 and j>0 and labels[i+1,j-1]!=res:
                    neighb |= 1<<6
                if j>0 and labels[i,j-1]!=res:
                    neighb |= 1<<7
                borders[i,j] = neighb
#                if neighb:

    def init_regions(self):
        cdef:
            int i, j, width=self.width, height=self.height, idx, res
            numpy.int32_t[:,:] labels = self.labels
            numpy.uint8_t[:,:] borders = self.borders
            numpy.uint8_t neighb=0
            Region region
            dict regions = self.regions
        for i in range(height):
            for j in range(width):
                idx = j+i*width
                neighb = borders[i,j]
                res =  labels[i,j]
                region = regions[res]
                region.size +=1
                if neighb==0: continue
                region.border.append(idx)
                if get_bit(neighb,1):
                    region.neighbors.append(labels[i-1,j])
                elif get_bit(neighb,3):
                    print(3,neighb,i,j,idx,labels[i,j])
                    region.neighbors.append(labels[i,j+1])
                elif get_bit(neighb,5):
                    region.neighbors.append(labels[i+1,j])
                elif get_bit(neighb,7):
                    region.neighbors.append(labels[i,j-1])
                elif get_bit(neighb,0):
                    region.neighbors.append(labels[i-1,j-1])
                elif get_bit(neighb,2):
                    region.neighbors.append(labels[i-1,j+1])
                elif get_bit(neighb,4):
                    print(4,neighb,i,j,idx,labels[i,j])
                    region.neighbors.append(labels[i+1,j+1])
                elif get_bit(neighb,6):
                    region.neighbors.append(labels[i+1,j-1])

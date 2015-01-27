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
__date__ = "26/01/2015"
__status__ = "stable"
__license__ = "GPLv3+"

import cython
import numpy
cimport numpy
import sys 
from pyFAI.bilinear import Bilinear
from pyFAI.utils import timeit
cdef bint get_bit(int byteval, int idx) nogil:
    return ((byteval&(1<<idx))!=0);


cdef class Region:
    cdef:
        int index, size, pass_to
        float mini, maxi, highest_pass
        list neighbors, border, peaks

    def __cinit__(self, int idx):
        self.index = idx
        self.neighbors = []
        self.border = [] #list of pixel indices of the border
        self.peaks=[idx]
        self.size = 0
        self.pass_to = - 1
        self.mini = - 1
        self.maxi = - 1
        self.highest_pass = -sys.maxsize

    def __repr__(self):
        return "Region %s of size %s:\n neighbors: %s\n border: %s\n"%(self.index, self.size, self.neighbors, self.border)+\
               "peaks: %s\n maxi=%s, mini=%s, pass=%s to %s"%(self.peaks, self.maxi, self.mini,self.highest_pass,self.pass_to)

    def init_values(self, float[:] flat):
        """
        Initalize the values : maxi, mini and pass both heigth and so on
        @param flat: flat view on the data (intensity)
        @return: True if there is a problem and the region should be removed
        """
        cdef:
            int i, k, imax, imin
            float mini, maxi
        self.maxi = flat[self.index]
        if  len(self.neighbors)!=len(self.border):
            print(self.index, len(self.neighbors),len(self.border))
            print(self)
            return True
        if self.neighbors:
            imax = imin = 0
            i = self.border[imax]
            val = mini = maxi = flat[i]
            for k in range(1,len(self.border)):
                i = self.border[k]
                val = flat[i]
                if val<mini:
                    mini = val
                    imin = k
                elif val>maxi:
                    maxi = val
                    imax = k
            if self.mini == - 1:
                self.mini = mini
            self.highest_pass = maxi
            self.pass_to = self.neighbors[imax]
        else:
            return True

    def get_size(self):
        return self.size
    
    def get_highest_pass(self):
        return self.highest_pass

    def get_maxi(self):
        return self.maxi
    
    def get_mini(self):
        return self.mini

    def get_pass_to(self):
        return self.pass_to

    def get_index(self):
        return self.index

    def get_borders(self):
        return self.border
    
    def get_neighbors(self):
        return self.neighbors

    def merge(self, Region other):
        """
        merge 2 regions
        """
        cdef:
            int i
            list new_neighbors=[], new_border=[]
            Region region
        if other.maxi > self.maxi:
            region = Region(other.index)
            region.maxi = other.maxi
        else:
            region = Region(self.index)
            region.maxi = self.maxi
        region.mini = min(self.mini, other.mini)
        for i in range(len(self.neighbors)):
            if self.neighbors[i] not in other.peaks:
                if self.border[i] not in new_border:
                    new_border.append(self.border[i])
                    new_neighbors.append(self.neighbors[i])
        for i in range(len(other.neighbors)):
            if other.neighbors[i] not in self.peaks:
                if other.border[i] not in new_border:
                    new_border.append(other.border[i])
                    new_neighbors.append(other.neighbors[i])
        region.neighbors = new_neighbors
        region.border = new_border
        region.peaks = self.peaks + other.peaks
        region.size = self.size + other.size
        return region


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
        self.height, self.width = data.shape
        self.bilinear = Bilinear(data)
        self.regions = dict()
        self.labels = numpy.zeros((self.height, self.width), dtype="int32")
        self.borders = numpy.zeros((self.height, self.width), dtype="uint8")

    def init(self):
        self.init_labels()
        self.init_borders()
        self.init_regions()
        self.init_pass()
        self.merge_singleton()
        self.merge_twins()
        self.merge_intense(1.0)
        print("found %s regions, after merge remains %s"%(len(self.regions),len(set(self.regions.values()))))

    @timeit
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
    @timeit
    def init_borders(self):
        cdef:
            int i, j, width=self.width, height=self.height, idx, res
            
            numpy.int32_t[:, :] labels = self.labels
            numpy.uint8_t[:, :] borders = self.borders
            numpy.uint8_t neighb
        for i in range(height):
            for j in range(width):
                neighb = 0
                idx = j+i*width
                res =  labels[i, j]
                if i>0 and j>0 and labels[i - 1, j - 1]!=res:
                    neighb |= 1
                if i>0 and labels[i - 1, j]!=res:
                    neighb |= 1<<1
                if i>0 and j<(width-1) and labels[i - 1, j + 1]!=res:
                    neighb |= 1<<2
                if j<(width-1) and labels[i, j + 1]!=res:
                    neighb |= 1<<3
                if i<(height-1) and j<(width-1) and labels[i + 1, j + 1]!=res:
                    neighb |= 1<<4
                if i<(height-1) and labels[i + 1, j]!=res:
                    neighb |= 1<<5
                if i<(height-1) and j>0 and labels[i + 1, j - 1]!=res:
                    neighb |= 1<<6
                if j>0 and labels[i, j - 1]!=res:
                    neighb |= 1<<7
                borders[i, j] = neighb

    @timeit
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
                neighb = borders[i, j]
                res = labels[i, j]
                region = regions[res]
                region.size +=1
                if neighb==0: continue
                region.border.append(idx)
                if get_bit(neighb, 1):
                    region.neighbors.append(labels[i - 1, j])
                elif get_bit(neighb, 3):
                    region.neighbors.append(labels[i, j + 1])
                elif get_bit(neighb, 5):
                    region.neighbors.append(labels[i + 1, j])
                elif get_bit(neighb, 7):
                    region.neighbors.append(labels[i, j - 1])
                elif get_bit(neighb, 0):
                    region.neighbors.append(labels[i - 1, j - 1])
                elif get_bit(neighb, 2):
                    region.neighbors.append(labels[i - 1, j + 1])
                elif get_bit(neighb, 4):
                    region.neighbors.append(labels[i + 1, j + 1])
                elif get_bit(neighb, 6):
                    region.neighbors.append(labels[i + 1, j - 1])

    @timeit
    def init_pass(self):
        cdef:
            int i, j, k, width=self.width, imax, imin
            float[:] flat = self.data.ravel()
            numpy.uint8_t neighb=0
            Region region
            dict regions = self.regions
            float val, maxi, mini
        for region in list(regions.values()):
            if region.init_values(flat):
                regions.pop(region.index)

    @timeit
    def merge_singleton(self):
        "merge single pixel region"
        cdef:
            int idx, i, j, key, key1
            Region region1, region2, region
            dict regions = self.regions
            numpy.uint8_t neighb = 0
            float ref = 0.0            
            float[:, :] data = self.data
            numpy.int32_t[:, :] labels = self.labels
            numpy.uint8_t[:, :] borders = self.borders
            int to_merge = -1
            int width = self.width
            int cnt = 0
            float[:] flat = self.data.ravel()
        for key1 in list(regions.keys()):
            region1 = regions[key1]
            if region1.maxi == region1.mini:
                to_merge = -1
                if region1.size == 1:
                    i = region1.index // width
                    j = region1.index % width                    
                    neighb = borders[i, j]
                    if get_bit(neighb, 1) and (region1.maxi == data[i - 1, j]):
                        to_merge = labels[i - 1, j]
                    elif get_bit(neighb, 3) and (region1.maxi == data[i, j + 1]):
                        to_merge = labels[i, j + 1]
                    elif get_bit(neighb, 5) and (region1.maxi == data[i + 1, j]):
                        to_merge = labels[i + 1, j]
                    elif get_bit(neighb, 7) and (region1.maxi == data[i, j - 1]):
                        to_merge = labels[i, j - 1]
                    elif get_bit(neighb, 0) and (region1.maxi == data[i - 1, j - 1]):
                        to_merge = labels[i - 1, j - 1]
                    elif get_bit(neighb, 2) and (region1.maxi == data[i - 1, j + 1]):
                        to_merge = labels[i - 1, j + 1]
                    elif get_bit(neighb, 4) and (region1.maxi == data[i + 1, j + 1]):
                        to_merge = labels[i + 1, j + 1]
                    elif get_bit(neighb, 6) and (region1.maxi == data[i + 1, j - 1]):
                        to_merge = labels[i + 1, j - 1]
                if to_merge < 0:
                    if len(region1.neighbors) == 1 or (region1.neighbors == [region1.neighbors[0]] * len(region1.neighbors)):
                        to_merge = region1.neighbors[0]
                    else:
                        to_merge = region1.neighbors[0]
                        region2 = regions[to_merge]
                        ref = region2.maxi
                        for idx in region1.neighbors[1:]:
                            region2 = regions[to_merge]
                            if region2.maxi > ref:
                                to_merge = idx
                                ref = region2.maxi
                if (to_merge < 0):
                    print("error in merging %s"%region1)
                else:
                    region2 = regions[to_merge]
                    region = region1.merge(region2)
                    region.init_values(flat)
                    for key in region.peaks:
                        regions[key] = region
                    cnt += 1
        print("Did %s merge_singleton" % cnt)

    @timeit
    def merge_twins(self):
        """
        Twins are two peak region which are best linked together:
        A -> B and B -> A
        """
        cdef:
            int i, j, k, imax, imin, key1, key2, key
            float[:] flat = self.data.ravel()
            numpy.uint8_t neighb = 0
            Region region1, region2, region
            dict regions = self.regions
            float val, maxi, mini
            bint found = True
            int width = self.width
            int cnt = 0
        for key1 in list(regions.keys()):
            region1 = regions[key1]
            key2 = region1.pass_to
            region2 = regions[key2]
            if region1 == region2:
                continue
            if (region2.pass_to in region1.peaks and region1.pass_to in region2.peaks):
                idx1 = region1.index
                idx2 = region2.index
#                 print("merge %s(%s) %s(%s)" % (idx1, idx1, key2, idx2))
                region = region1.merge(region2)
                region.init_values(flat)
                for key in region.peaks:
                    regions[key] = region
                cnt += 1
        print("Did %s merge_twins" % cnt)
        
    @timeit
    def merge_intense(self, thres=1.0):
        """
        Merge groups then (pass-mini)/(maxi-mini) >=thres
        """
        cdef:
            int key1, key2, idx1, idx2
            Region region1, region2, region
            dict regions = self.regions
            float ratio
            float[:] flat = self.data.ravel()
            int cnt = 0
        for key1 in list(regions.keys()):
            region1 = regions[key1]
            if region1.maxi == region1.mini:
                print("Error with")
                print region1
                continue
            ratio = (region1.highest_pass - region1.mini) / (region1.maxi - region1.mini)
            if ratio >= thres:
                key2 = region1.pass_to
                idx1 = region1.index
                region2 = regions[key2]
                idx2 = region2.index
#                 print("merge %s(%s) %s(%s)" % (idx1, idx1, key2, idx2))
                region = region1.merge(region2)
                region.init_values(flat)
                for key in region.peaks:
                    regions[key] = region
                cnt += 1
        print("Did %s merge_intense" % cnt)
        
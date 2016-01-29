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
__date__ = "28/01/2016"
__status__ = "stable"
__license__ = "GPLv3+"

import cython
import numpy
cimport numpy
import sys 
import logging
logger = logging.getLogger("pyFAI.ext.watershed") 
from ..decorators import timeit
from cython.parallel import prange

include "numpy_common.pxi"
include "bilinear.pxi"

cdef bint get_bit(int byteval, int idx) nogil:
    return ((byteval & (1 << idx)) != 0)


cdef class Region:
    cdef:
        readonly int index, size, pass_to
        readonly float mini, maxi, highest_pass
        readonly list neighbors, border, peaks

    def __cinit__(self, int idx):
        self.index = idx
        self.neighbors = []
        self.border = []  # list of pixel indices of the border
        self.peaks = [idx]
        self.size = 0
        self.pass_to = - 1
        self.mini = - 1
        self.maxi = - 1
        self.highest_pass = -sys.maxsize

    def __dealloc__(self):
        """Destructor"""
        self.neighbors = None
        self.border = None
        self.peaks = None

    def __repr__(self):
        return "Region %s of size %s:\n neighbors: %s\n border: %s\n" % (self.index, self.size, self.neighbors, self.border) + \
               "peaks: %s\n maxi=%s, mini=%s, pass=%s to %s" % (self.peaks, self.maxi, self.mini, self.highest_pass, self.pass_to)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init_values(self, float[:] flat):
        """
        Initialize the values : maxi, mini and pass both height and so on
        @param flat: flat view on the data (intensity)
        @return: True if there is a problem and the region should be removed
        """
        cdef:
            int i, k, imax, imin
            float mini, maxi, val
            int border_size = len(self.border)
            int neighbors_size = len(self.neighbors)
        self.maxi = flat[self.index]
        if neighbors_size != border_size:
            print(self.index, neighbors_size, border_size)
            print(self)
            return True
        if neighbors_size:
            imax = imin = 0
            i = self.border[imax]
            val = mini = maxi = flat[i]
            for k in range(1, border_size):
                i = self.border[k]
                val = flat[i]
                if val < mini:
                    mini = val
                    imin = k
                elif val > maxi:
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

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def merge(self, Region other):
        """
        merge 2 regions
        """
        cdef:
            int i
            list new_neighbors = []
            list new_border = []
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
#     cdef:
#         readonly float[:, :]  data
#         readonly size_t width, height
#         readonly dict regions
#         readonly numpy.int32_t[:, :] labels
#         readonly numpy.uint8_t[:, :] borders
#         readonly float  thres, _actual_thres
#         readonly Bilinear bilinear
    NAME = "Inverse watershed"
    VERSION = "1.0"
    
    def __init__(self, data not None, thres=1.0):
        """
        @param data: 2d image as numpy array

        """
        assert data.ndim == 2
        self.data = numpy.ascontiguousarray(data, dtype=numpy.float32)
        
        self.height, self.width = data.shape
        self.bilinear = Bilinear(data)
        self.regions = {}
        self.labels = numpy.zeros((self.height, self.width), dtype="int32")
        self.borders = numpy.zeros((self.height, self.width), dtype="uint8")
        self.thres = thres
        self._actual_thres = 2

    def __dealloc__(self):
        """destructor"""
        self.data = None
        self.bilinear = None
        self.regions = None
        self.labels = None
        self.borders = None
        self.dict = None

    def save(self, fname):
        """
        Save all regions into a HDF5 file
        """
        import h5py
        with h5py.File(fname) as h5:
            h5["NAME"] = self.NAME
            h5["VERSION"] = self.VERSION
            for i in ("data", "height", "width", "labels", "borders", "thres"):
                h5[i] = self.__getattribute__(i)
            r = h5.require_group("regions")
            
            for i in set(self.regions.values()):
                s = r.require_group(str(i.index))
                for j in ("index", "size", "pass_to", "mini", "maxi", "highest_pass", "neighbors", "border", "peaks"):
                    s[j] = i.__getattribute__(j)

    @classmethod
    def load(cls, fname):
        """
        Load data from a HDF5 file
        """
        import h5py
        with h5py.File(fname) as h5:
            assert h5["VERSION"].value == cls.VERSION
            assert h5["NAME"].value == cls.NAME
            self = cls(h5["data"].value, h5["thres"].value)
            for i in ("labels", "borders"):
                setattr(self, i, h5[i].value)
            for i in h5["regions"].values():
                r = Region(i["index"].value)
                r.size = i["size"].value
                r.pass_to = i["pass_to"].value
                r.mini = i["mini"].value
                r.maxi = i["maxi"].value
                r.highest_pass = i["highest_pass"].value
                r.neighbors = list(i["neighbors"].value)
                r.border = list(i["border"].value)
                r.peaks = list(i["peaks"].value)
                for j in r.peaks:
                    self.regions[j] = r
        return self
    
    def init(self):
        self.init_labels()
        self.init_borders()
        self.init_regions()
        self.init_pass()
#        self.merge_singleton()
#        self.merge_twins()
#        self.merge_intense(self.thres)
        logger.info("found %s regions, after merge remains %s" % (len(self.regions), len(set(self.regions.values()))))

    @timeit
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init_labels(self):
        cdef:
            int i, j, width = self.width, height = self.height, idx, res
            numpy.int32_t[:, :] labels = self.labels
            dict regions = self.regions
            Bilinear bilinear = self.bilinear
        for i in range(height):
            for j in range(width):
                idx = j + i * width
                res = bilinear.c_local_maxi(idx)
                labels[i, j] += res
                if idx == res:
                    regions[res] = Region(res) 

    @timeit 
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init_borders(self):
        cdef:
            int i, j, width = self.width, height = self.height, idx, res
            numpy.int32_t[:, :] labels = self.labels
            numpy.uint8_t[:, :] borders = self.borders
            numpy.uint8_t neighb
        for i in range(height):
            for j in range(width):
                neighb = 0
                idx = j + i * width
                res = labels[i, j]
                if (i > 0) and (j > 0) and (labels[i - 1, j - 1] != res):
                    neighb |= 1
                if (i > 0) and (labels[i - 1, j] != res):
                    neighb |= 1 << 1
                if (i > 0) and (j < (width - 1)) and (labels[i - 1, j + 1] != res):
                    neighb |= 1 << 2
                if (j < (width - 1)) and (labels[i, j + 1] != res):
                    neighb |= 1 << 3
                if (i < (height - 1)) and (j < (width - 1)) and (labels[i + 1, j + 1] != res):
                    neighb |= 1 << 4
                if (i < (height - 1)) and (labels[i + 1, j] != res):
                    neighb |= 1 << 5
                if (i < (height - 1)) and (j > 0) and (labels[i + 1, j - 1] != res):
                    neighb |= 1 << 6
                if (j > 0) and (labels[i, j - 1] != res):
                    neighb |= 1 << 7
                borders[i, j] = neighb

    @timeit
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init_regions(self):
        cdef:
            int i, j, idx, res
            numpy.int32_t[:, :] labels = self.labels
            numpy.uint8_t[:, :] borders = self.borders
            numpy.uint8_t neighb = 0
            Region region
            dict regions = self.regions
            int width = self.width
            int  height = self.height
        for i in range(height):
            for j in range(width):
                idx = j + i * width
                neighb = borders[i, j]
                res = labels[i, j]
                region = regions[res]
                region.size += 1
                if neighb == 0: 
                    continue
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
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init_pass(self):
        cdef:
            int i, j, k, imax, imin
            float[:] flat = self.data.ravel()
            numpy.uint8_t neighb = 0
            Region region
            dict regions = self.regions
            float val, maxi, mini
            int width = self.width
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
                    if len(region1.neighbors) == 0:
                        print("no neighbors: %s" % region1)
                    elif (len(region1.neighbors) == 1) or \
                         (region1.neighbors == [region1.neighbors[0]] * len(region1.neighbors)):
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
                    logger.info("error in merging %s" % region1)
                else:
                    region2 = regions[to_merge]
                    region = region1.merge(region2)
                    region.init_values(flat)
                    for key in region.peaks:
                        regions[key] = region
                    cnt += 1
        logger.info("Did %s merge_singleton" % cnt)

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
#                 logger.info("merge %s(%s) %s(%s)" % (idx1, idx1, key2, idx2))
                region = region1.merge(region2)
                region.init_values(flat)
                for key in region.peaks:
                    regions[key] = region
                cnt += 1
        logger.info("Did %s merge_twins" % cnt)
        
    @timeit
    def merge_intense(self, thres=1.0):
        """
        Merge groups then (pass-mini)/(maxi-mini) >=thres
        """
        if thres > self._actual_thres:
            logger.warning("Cannot increase threshold: was %s, requested %s. You should re-init the object." % self._actual_thres, thres)
        self._actual_thres = thres
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
                logger.error(region1)
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
        logger.info("Did %s merge_intense" % cnt)
    
    def peaks_from_area(self, mask, Imin=None, keep=None, bint refine=True, float dmin=0.0, **kwarg):
        """
        @param mask: mask of data points valid
        @param Imin: Minimum intensity for a peak 
        @param keep: Number of  points to keep
        @param refine: refine sub-pixel position
        @param dmin: minimum distance from 
        """
        cdef:
            int i, j, l, x, y, width = self.width
            numpy.uint8_t[:] mask_flat = numpy.ascontiguousarray(mask.ravel(), numpy.uint8)
            int[:] input_points = numpy.where(mask_flat)[0].astype(numpy.int32)
            numpy.int32_t[:] labels = self.labels.ravel()
            dict regions = self.regions
            Region region
            list output_points = [], intensities = [], argsort, tmp_lst, rej_lst
            set keep_regions = set()
            float[:] data = self.data.ravel() 
            double d2, dmin2
        for i in input_points:
            l = labels[i]
            region = regions[l]
            keep_regions.add(region.index)
        for i in keep_regions:
            region = regions[i]
            for j in region.peaks:
                if mask_flat[j]:
                    intensities.append(data[j])
                    x = j % self.width
                    y = j // self.width
                    output_points.append((y, x))
        if refine:
            for i in range(len(output_points)):
                output_points[i] = self.bilinear.local_maxi(output_points[i])
        if Imin or keep:
            argsort = sorted(range(len(intensities)), key=intensities.__getitem__, reverse=True)
            if Imin:
                argsort = [i for i in argsort if intensities[i] >= Imin]
            output_points = [output_points[i] for i in argsort]
            
            if dmin:
                dmin2 = dmin * dmin
            else:
                dmin2 = 0.0
            if keep and len(output_points)>keep:
                tmp_lst = output_points
                rej_lst = []
                output_points = []
                for pt in tmp_lst:
                    for pt2 in output_points:
                        d2 = (pt[0]-pt2[0])**2 + (pt[1]-pt2[1])**2
                        if d2<=dmin2:
                            rej_lst.append(pt)
                            break
                    else:
                        output_points.append(pt)
                        if len(output_points)>=keep:
                            return output_points
                output_points = (output_points+rej_lst)[:keep]
        return output_points
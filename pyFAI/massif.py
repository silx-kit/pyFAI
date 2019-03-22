#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/03/2019"
__status__ = "production"

import sys
import os
import threading
from math import ceil, sqrt
import logging
logger = logging.getLogger(__name__)
import numpy
import fabio
from scipy.ndimage import label, distance_transform_edt
from scipy.ndimage.filters import median_filter
from .utils.decorators import deprecated
from .ext.bilinear import Bilinear
from .utils import gaussian_filter, binning, unbinning, is_far_from_group
from .third_party import six

if os.name != "nt":
    WindowsError = RuntimeError


class Massif(object):
    """
    A massif is defined as an area around a peak, it is used to find neighboring peaks
    """
    TARGET_SIZE = 1024

    def __init__(self, data=None, mask=None):
        """Constructor of the class...

        :param data: 2D array or filename (discouraged)
        :param mask: array with non zero for invalid data
        """
        if isinstance(data, six.string_types) and os.path.isfile(data):
            self.data = fabio.open(data).data.astype("float32")
        elif isinstance(data, fabio.fabioimage.fabioimage):
            self.data = data.data.astype("float32")
        else:
            try:
                self.data = data.astype("float32")
            except Exception as error:
                logger.error("Unable to understand this type of data %s: %s", data, error)
        self.log_info = True
        """If true, more information is displayed in the logger relative to picking."""
        self.mask = mask
        self._cleaned_data = None
        self._bilin = Bilinear(self.data)
        self._blurred_data = None
        self._median_data = None
        self._labeled_massif = None
        self._number_massif = None
        self._valley_size = None
        self._binned_data = None
        self._reconstruct_used = None
        self.binning = None  # Binning is 2-list usually
        self._sem = threading.Semaphore()
        self._sem_label = threading.Semaphore()
        self._sem_binning = threading.Semaphore()
        self._sem_median = threading.Semaphore()

    def nearest_peak(self, x):
        """
        :param x: coordinates of the peak
        :returns: the coordinates of the nearest peak
        """
        out = self._bilin.local_maxi(x)
        if isinstance(out, tuple):
            res = out
        elif isinstance(out, numpy.ndarray):
            res = tuple(out)
        else:
            res = [int(i) for idx, i in enumerate(out) if 0 <= i < self.data.shape[idx]]
        if (len(res) != 2) or not((0 <= out[0] < self.data.shape[0]) and (0 <= res[1] < self.data.shape[1])):
            logger.warning("in nearest_peak %s -> %s", x, out)
            return
        elif (self.mask is not None) and self.mask[int(res[0]), int(res[1])]:
            logger.info("Masked pixel %s -> %s", x, out)
            return
        else:
            return res

    def calculate_massif(self, x):
        """
        defines a map of the massif around x and returns the mask
        """
        labeled = self.get_labeled_massif()
        if labeled[x[0], x[1]] > 0:  # without relabeled the background is 0 labeled.max():
            return (labeled == labeled[x[0], x[1]])

    def find_peaks(self, x, nmax=200, annotate=None, massif_contour=None, stdout=sys.stdout):
        """
        All in one function that finds a maximum from the given seed (x)
        then calculates the region extension and extract position of the neighboring peaks.

        :param Tuple[int] x: coordinates of the peak, seed for the calculation
        :param int nmax: maximum number of peak per region
        :param annotate: callback method taking number of points + coordinate as input.
        :param massif_contour: callback to show the contour of a massif with the given index.
        :param stdout: this is the file where output is written by default.
        :return: list of peaks
        """
        region = self.calculate_massif(x)
        if region is None:
            if self.log_info:
                logger.error("You picked a background point at %s", x)
            return []
        xinit = self.nearest_peak(x)
        if xinit is None:
            if self.log_info:
                logger.error("Unable to find peak in the vinicy of %s", x)
            return []
        else:
            if not region[int(xinit[0] + 0.5), int(xinit[1] + 0.5)]:
                logger.error("Nearest peak %s is not in the same region  %s", xinit, x)
                return []

            if annotate is not None:
                try:
                    annotate(xinit, x)
                except Exception as error:
                    logger.debug("Backtrace", exc_info=True)
                    logger.error("Error in annotate %i: %i %i. %s", 0, xinit[0], xinit[1], error)

        listpeaks = []
        listpeaks.append(xinit)
        cleaned_data = self.cleaned_data
        mean = cleaned_data[region].mean(dtype=numpy.float64)
        region2 = region * (cleaned_data > mean)
        idx = numpy.vstack(numpy.where(region2)).T
        numpy.random.shuffle(idx)
        nmax = min(nmax, int(ceil(sqrt(idx.shape[0]))))
        if massif_contour is not None:
            try:
                massif_contour(region)
            except (WindowsError, MemoryError) as error:
                logger.debug("Backtrace", exc_info=True)
                logger.error("Error in plotting region: %s", error)
        nbFailure = 0
        for j in idx:
            xopt = self.nearest_peak(j)
            if xopt is None:
                nbFailure += 1
                continue
            if (region2[int(xopt[0] + 0.5), int(xopt[1] + 0.5)]) and not (xopt in listpeaks):
                if stdout:
                    stdout.write("[ %4i, %4i ] --> [ %5.1f, %5.1f ] after %3i iterations %s" % (tuple(j) + tuple(xopt) + (nbFailure, os.linesep)))
                listpeaks.append(xopt)
                nbFailure = 0
            else:
                nbFailure += 1
            if (len(listpeaks) > nmax) or (nbFailure > 2 * nmax):
                break
        return listpeaks

    def peaks_from_area(self, mask, Imin=numpy.finfo(numpy.float64).min,
                        keep=1000, dmin=0.0, seed=None, **kwarg):
        """
        Return the list of peaks within an area

        :param mask: 2d array with mask.
        :param Imin: minimum of intensity above the background to keep the point
        :param keep: maximum number of points to keep
        :param kwarg: ignored parameters
        :param dmin: minimum distance to another peak
        :param seed: list of good guesses to start with
        :return: list of peaks [y,x], [y,x], ...]
        """
        all_points = numpy.vstack(numpy.where(mask)).T
        res = []
        cnt = 0
        dmin2 = dmin * dmin
        if len(all_points) > 0:
            numpy.random.shuffle(all_points)
        if seed:
            seeds = numpy.array(list(seed))
            if len(seeds) > 0:
                numpy.random.shuffle(seeds)
            all_points = numpy.concatenate((seeds, all_points))
        for idx in all_points:
            out = self.nearest_peak(idx)
            if out is not None:
                msg = "[ %3i, %3i ] -> [ %.1f, %.1f ]"
                logger.debug(msg, idx[1], idx[0], out[1], out[0])
                p0, p1 = int(round(out[0])), int(round(out[1]))
                if mask[p0, p1]:
                    if (self.data[p0, p1] > Imin) and is_far_from_group(out, res, dmin2):
                        res.append(out)
                        cnt = 0
            if len(res) >= keep or cnt > keep:
                break
            else:
                cnt += 1
        return res

    def init_valley_size(self):
        if self._valley_size is None:
            self.valley_size = max(5., max(self.data.shape) / 50.)

    @property
    def valley_size(self):
        "Defines the minimum distance between two massifs"
        if self._valley_size is None:
            self.init_valley_size()
        return self._valley_size

    @valley_size.setter
    def valley_size(self, size):
        new_size = float(size)
        if self._valley_size != new_size:
            self._valley_size = new_size
            t = threading.Thread(target=self.get_labeled_massif)
            t.start()

    @valley_size.deleter
    def valley_size(self):
        self._valley_size = None
        self._blurred_data = None

    @property
    def cleaned_data(self):
        if self.mask is None:
            return self.data
        else:
            if self._cleaned_data is None:
                idx = distance_transform_edt(self.mask,
                                             return_distances=False,
                                             return_indices=True)
                self._cleaned_data = self.data[tuple(idx)]
            return self._cleaned_data

    def get_binned_data(self):
        """
        :return: binned data
        """
        if self._binned_data is None:
            with self._sem_binning:
                if self._binned_data is None:
                    logger.info("Image size is %s", self.data.shape)
                    self.binning = []
                    for i in self.data.shape:
                        if i % self.TARGET_SIZE == 0:
                            self.binning.append(max(1, i // self.TARGET_SIZE))
                        else:
                            for j in range(i // self.TARGET_SIZE - 1, 0, -1):
                                if i % j == 0:
                                    self.binning.append(max(1, j))
                                    break
                            else:
                                self.binning.append(1)
#                    self.binning = max([max(1, i // self.TARGET_SIZE) for i in self.data.shape])
                    logger.info("Binning size is %s", self.binning)
                    self._binned_data = binning(self.cleaned_data, self.binning)
        return self._binned_data

    def get_median_data(self):
        """
        :return: a spatial median filtered image 3x3
        """
        if self._median_data is None:
            with self._sem_median:
                if self._median_data is None:
                    self._median_data = median_filter(self.cleaned_data, 3)
        return self._median_data

    def get_blurred_data(self):
        """
        :return: a blurred image
        """
        if self._blurred_data is None:
            with self._sem:
                if self._blurred_data is None:
                    logger.debug("Blurring image with kernel size: %s", self.valley_size)
                    self._blurred_data = gaussian_filter(self.get_binned_data(),
                                                         [self.valley_size / i for i in self.binning],
                                                         mode="reflect")
        return self._blurred_data

    def get_labeled_massif(self, pattern=None, reconstruct=True):
        """
        :param pattern: 3x3 matrix
        :param reconstruct: if False, split massif at masked position, else reconstruct missing part.
        :return: an image composed of int with a different value for each massif
        """
        if self._labeled_massif is None:
            with self._sem_label:
                if self._labeled_massif is None:
                    if pattern is None:
                        pattern = numpy.ones((3, 3), dtype=numpy.int8)
                    logger.debug("Labeling all massifs. This takes some time !!!")
                    massif_binarization = (self.get_binned_data() > self.get_blurred_data())
                    if (self.mask is not None) and (not reconstruct):
                            binned_mask = binning(self.mask.astype(int), self.binning, norm=False)
                            massif_binarization = numpy.logical_and(massif_binarization, binned_mask == 0)
                    self._reconstruct_used = reconstruct
                    labeled_massif, self._number_massif = label(massif_binarization,
                                                                pattern)
                    # TODO: investigate why relabel fails
                    # relabeled = relabel(labeled_massif, self.get_binned_data(), self.get_blurred_data())
                    relabeled = labeled_massif
                    self._labeled_massif = unbinning(relabeled, self.binning, False)
                    logger.info("Labeling found %s massifs.", self._number_massif)
        return self._labeled_massif

    @deprecated(reason="switch to pep8 style", replacement="init_valley_size", since_version="0.16.0")
    def initValleySize(self):
        self.init_valley_size()

    @deprecated(reason="switch to PEP8 style", replacement="get_median_data", since_version="0.16.0")
    def getMedianData(self):
        return self.get_median_data()

    @deprecated(reason="switch to PEP8 style", replacement="get_binned_data", since_version="0.16.0")
    def getBinnedData(self):
        return self.get_binned_data()

    @deprecated(reason="switch to PEP8 style", replacement="get_blurred_data", since_version="0.16.0")
    def getBluredData(self):
        return self.get_blurred_data()

    @deprecated(reason="switch to PEP8 style", replacement="get_labeled_massif", since_version="0.16.0")
    def getLabeledMassif(self, pattern=None):
        return self.get_labeled_massif(pattern)

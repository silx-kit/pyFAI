#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "15/05/2017"
__status__ = "production"

import sys
import os
import threading
from math import ceil, sqrt
import logging
logger = logging.getLogger("pyFAI.massif")
import numpy
import fabio
from scipy.ndimage import label
from scipy.ndimage.filters import median_filter

from .ext.bilinear import Bilinear
from .utils import gaussian_filter, binning, unBinning, relabel, is_far_from_group
try:
    from .third_party import six
except (ImportError, Exception):
    import six

if os.name != "nt":
    WindowsError = RuntimeError


class Massif(object):
    """
    A massif is defined as an area around a peak, it is used to find neighboring peaks
    """
    TARGET_SIZE = 1024

    def __init__(self, data=None):
        """

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
        self._bilin = Bilinear(self.data)
        self._blured_data = None
        self._median_data = None
        self._labeled_massif = None
        self._number_massif = None
        self._valley_size = None
        self._binned_data = None
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
            logger.error("in nearest_peak %s -> %s", x, out)
            return
        else:
            return res

    def calculate_massif(self, x):
        """
        defines a map of the massif around x and returns the mask
        """
        labeled = self.getLabeledMassif()
        if labeled[x[0], x[1]] != labeled.max():
            return (labeled == labeled[x[0], x[1]])

    def find_peaks(self, x, nmax=200, annotate=None, massif_contour=None, stdout=sys.stdout):
        """
        All in one function that finds a maximum from the given seed (x)
        then calculates the region extension and extract position of the neighboring peaks.
        :param x: coordinates of the peak, seed for the calculation
        :type x: tuple of integer
        :param nmax: maximum number of peak per region
        :param annotate: call back method taking number of points + coordinate as input.
        :param massif_contour: callback to show the contour of a massif with the given index.
        :param stdout: this is the file where output is written by default.
        :return: list of peaks
        """
        listpeaks = []
        region = self.calculate_massif(x)
        if region is None:
            logger.error("You picked a background point at %s", x)
            return listpeaks
        xinit = self.nearest_peak(x)
        if xinit is None:
            logger.error("Unable to find peak in the vinicy of %s", x)
            return listpeaks
        else:
            if not region[int(xinit[0] + 0.5), int(xinit[1] + 0.5)]:
                logger.error("Nearest peak %s is not in the same region  %s", xinit, x)
                return listpeaks

            if annotate is not None:
                try:
                    annotate(xinit, x)
                except Exception as error:
                    logger.error("Error in annotate %i: %i %i. %s", len(listpeaks), xinit[0], xinit[1], error)

        listpeaks.append(xinit)
        mean = self.data[region].mean(dtype=numpy.float64)
        region2 = region * (self.data > mean)
        idx = numpy.vstack(numpy.where(region2)).T
        numpy.random.shuffle(idx)
        nmax = min(nmax, int(ceil(sqrt(idx.shape[0]))))
        if massif_contour is not None:
            try:
                massif_contour(region)
            except (WindowsError, MemoryError) as error:
                logger.error("Error in plotting region: %s", error)
        nbFailure = 0
        for j in idx:
            xopt = self.nearest_peak(j)
            if xopt is None:
                nbFailure += 1
                continue
            if (region2[int(xopt[0] + 0.5), int(xopt[1] + 0.5)]) and not (xopt in listpeaks):
                stdout.write("[ %4i, %4i ] --> [ %5.1f, %5.1f ] after %3i iterations %s" % (tuple(j) + tuple(xopt) + (nbFailure, os.linesep)))
                listpeaks.append(xopt)
                nbFailure = 0
            else:
                nbFailure += 1
            if (len(listpeaks) > nmax) or (nbFailure > 2 * nmax):
                break
        return listpeaks

    def peaks_from_area(self, mask, Imin=None, keep=1000, dmin=0.0, seed=None, **kwarg):
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
                logger.debug("[ %3i, %3i ] -> [ %.1f, %.1f ]" %
                      (idx[1], idx[0], out[1], out[0]))
                p0, p1 = int(round(out[0])), int(round(out[1]))
                if mask[p0, p1]:

                    if (self.data[p0, p1] > Imin) and \
                        is_far_from_group(out, res, dmin2):
                        res.append(out)
                        cnt = 0
            if len(res) >= keep or cnt > keep:
                break
            else:
                cnt += 1
        return res

    def initValleySize(self):
        if self._valley_size is None:
            self.valley_size = max(5., max(self.data.shape) / 50.)

    def getValleySize(self):
        if self._valley_size is None:
            self.initValleySize()
        return self._valley_size

    def setValleySize(self, size):
        new_size = float(size)
        if self._valley_size != new_size:
            self._valley_size = new_size
            # self.getLabeledMassif()
            t = threading.Thread(target=self.getLabeledMassif)
            t.start()

    def delValleySize(self):
        self._valley_size = None
        self._blured_data = None
    valley_size = property(getValleySize, setValleySize, delValleySize, "Defines the minimum distance between two massifs")

    def getBinnedData(self):
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
                    self._binned_data = binning(self.data, self.binning)
        return self._binned_data

    def getMedianData(self):
        """
        :return: a spacial median filtered image
        """
        if self._median_data is None:
            with self._sem_median:
                if self._median_data is None:
                    self._median_data = median_filter(self.data, 3)
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        with open("median_data.npy", "wb") as f:
                            numpy.save(f, self._median_data)
                        # fabio.edfimage.edfimage(data=self._median_data).write("median_data.edf")
        return self._median_data

    def getBluredData(self):
        """
        :return: a blurred image
        """

        if self._blured_data is None:
            with self._sem:
                if self._blured_data is None:
                    logger.debug("Blurring image with kernel size: %s", self.valley_size)
                    self._blured_data = gaussian_filter(self.getBinnedData(), [self.valley_size / i for i in self.binning], mode="reflect")
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        fabio.edfimage.edfimage(data=self._blured_data).write("blured_data.edf")
        return self._blured_data

    def getLabeledMassif(self, pattern=None):
        """
        :return: an image composed of int with a different value for each massif
        """
        if self._labeled_massif is None:
            with self._sem_label:
                if self._labeled_massif is None:
                    if pattern is None:
                        pattern = [[1] * 3] * 3  # [[0, 1, 0], [1, 1, 1], [0, 1, 0]]#[[1] * 3] * 3
                    logger.debug("Labeling all massifs. This takes some time !!!")
                    labeled_massif, self._number_massif = label((self.getBinnedData() > self.getBluredData()), pattern)
                    logger.info("Labeling found %s massifs.", self._number_massif)
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        fabio.edfimage.edfimage(data=labeled_massif).write("labeled_massif_small.edf")
                    relabeled = relabel(labeled_massif, self.getBinnedData(), self.getBluredData())
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        fabio.edfimage.edfimage(data=relabeled).write("relabeled_massif_small.edf")
                    self._labeled_massif = unBinning(relabeled, self.binning, False)
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        fabio.edfimage.edfimage(data=self._labeled_massif).write("labeled_massif.edf")
                    logger.info("Labeling found %s massifs.", self._number_massif)
        return self._labeled_massif

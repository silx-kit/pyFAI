#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Aurore Deschildre
#                            Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

from __future__ import division, print_function, absolute_import
__authors__ = ["Aurore Deschildre", "Jérôme Kieffer"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/03/2018"
__status__ = "production"
__docformat__ = 'restructuredtext'

import os
import logging
logger = logging.getLogger(__name__)
import numpy
try:
    from .ext._convolution import gaussian_filter
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    from scipy.ndimage.filters import gaussian_filter
try:
    from .ext import _blob
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    _blob = None

try:
    from .ext import morphology
    pyFAI_morphology = True
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    from scipy.ndimage import morphology
    pyFAI_morphology = False

from .ext.bilinear import Bilinear

from math import sqrt

from .utils import binning, is_far_from_group


def image_test():
    img = numpy.zeros((128 * 4, 128 * 4))
    a = numpy.linspace(0.5, 8, 16)
    xc = [64, 64, 64, 64, 192, 192, 192, 192, 320, 320, 320, 320, 448, 448, 448, 448]
    yc = [64, 192, 320, 448, 64, 192, 320, 448, 64, 192, 320, 448, 64, 192, 320, 448]
    cpt = 0
    for sigma in a:
        img = make_gaussian(img, sigma, xc[cpt], yc[cpt])
        cpt = cpt + 1
    return img


def make_gaussian(im, sigma, xc, yc):
    size = int(8 * sigma + 1)
    if size % 2 == 0:
        size += 1
    x = numpy.arange(0, size, 1, float)
    y = x[:, numpy.newaxis] * 4
    x0 = y0 = size // 2
    gaus = numpy.exp(-4 * numpy.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
    im[xc - size / 2:xc + size / 2 + 1, yc - size / 2:yc + size / 2 + 1] = gaus
    return im


def local_max(dogs, mask=None, n_5=True):
    """
    :param dogs: 3d array with (sigma,y,x) containing difference of gaussians
    :param mask: mask out keypoint next to the mask (or inside the mask)
    :param n_5: look for a larger neighborhood
    """
    if mask is not None:
        mask = mask.astype(bool)
    ns = dogs.shape[0]
    kpma = numpy.zeros(shape=dogs.shape, dtype=numpy.uint8)
    for i in range(1, ns - 1):
        cur_dog = dogs[i]
        next_dog = dogs[i + 1]
        prev_dog = dogs[i - 1]
        slic = cur_dog[1:-1, 1:-1]
        kpm = kpma[i]
        kpm[1:-1, 1:-1] += (slic > cur_dog[:-2, 1:-1]) * (slic > cur_dog[2:, 1:-1])
        kpm[1:-1, 1:-1] += (slic > cur_dog[1:-1, :-2]) * (slic > cur_dog[1:-1, 2:])
        kpm[1:-1, 1:-1] += (slic > cur_dog[:-2, :-2]) * (slic > cur_dog[2:, 2:])
        kpm[1:-1, 1:-1] += (slic > cur_dog[2:, :-2]) * (slic > cur_dog[:-2, 2:])

        # with next DoG
        kpm[1:-1, 1:-1] += (slic > next_dog[:-2, 1:-1]) * (slic > next_dog[2:, 1:-1])
        kpm[1:-1, 1:-1] += (slic > next_dog[1:-1, :-2]) * (slic > next_dog[1:-1, 2:])
        kpm[1:-1, 1:-1] += (slic > next_dog[:-2, :-2]) * (slic > next_dog[2:, 2:])
        kpm[1:-1, 1:-1] += (slic > next_dog[2:, :-2]) * (slic > next_dog[:-2, 2:])
        kpm[1:-1, 1:-1] += (slic >= next_dog[1:-1, 1:-1])

        # with previous DoG
        kpm[1:-1, 1:-1] += (slic > prev_dog[:-2, 1:-1]) * (slic > prev_dog[2:, 1:-1])
        kpm[1:-1, 1:-1] += (slic > prev_dog[1:-1, :-2]) * (slic > prev_dog[1:-1, 2:])
        kpm[1:-1, 1:-1] += (slic > prev_dog[:-2, :-2]) * (slic > prev_dog[2:, 2:])
        kpm[1:-1, 1:-1] += (slic > prev_dog[2:, :-2]) * (slic > prev_dog[:-2, 2:])
        kpm[1:-1, 1:-1] += (slic >= prev_dog[1:-1, 1:-1])

        if n_5:
            target = 38
            slic = cur_dog[2:-2, 2:-2]

            kpm[2:-2, 2:-2] += (slic > cur_dog[:-4, 2:-2]) * (slic > cur_dog[4:, 2:-2])  # decalage horizontal
            kpm[2:-2, 2:-2] += (slic > cur_dog[2:-2, :-4]) * (slic > cur_dog[2:-2, 4:])  # decalage vertical
            kpm[2:-2, 2:-2] += (slic > cur_dog[:-4, :-4]) * (slic > cur_dog[4:, 4:])  # diagonale
            kpm[2:-2, 2:-2] += (slic > cur_dog[4:, :-4]) * (slic > cur_dog[:-4, 4:])
            kpm[2:-2, 2:-2] += (slic > cur_dog[4:, 1:-3]) * (slic > cur_dog[:-4, 1:-3])
            kpm[2:-2, 2:-2] += (slic > cur_dog[1:-3, :-4]) * (slic > cur_dog[1:-3, 4:])
            kpm[2:-2, 2:-2] += (slic > cur_dog[3:-1, :-4]) * (slic > cur_dog[3:-1, 4:])
            kpm[2:-2, 2:-2] += (slic > cur_dog[4:, 3:-1]) * (slic > cur_dog[:-4, 3:-1])

            # with next DoG
            kpm[2:-2, 2:-2] += (slic > next_dog[:-4, 2:-2]) * (slic > next_dog[4:, 2:-2])
            kpm[2:-2, 2:-2] += (slic > next_dog[2:-2, :-4]) * (slic > next_dog[2:-2, 4:])
            kpm[2:-2, 2:-2] += (slic > next_dog[:-4, :-4]) * (slic > next_dog[4:, 4:])
            kpm[2:-2, 2:-2] += (slic > next_dog[4:, :-4]) * (slic > next_dog[:-4, 4:])
            kpm[2:-2, 2:-2] += (slic > next_dog[4:, 1:-3]) * (slic > next_dog[:-4, 1:-3])
            kpm[2:-2, 2:-2] += (slic > next_dog[1:-3, :-4]) * (slic > next_dog[1:-3, 4:])
            kpm[2:-2, 2:-2] += (slic > next_dog[3:-1, :-4]) * (slic > next_dog[3:-1, 4:])
            kpm[2:-2, 2:-2] += (slic > next_dog[4:, 3:-1]) * (slic > next_dog[:-4, 3:-1])

            # with previous DoG
            kpm[2:-2, 2:-2] += (slic > prev_dog[:-4, 2:-2]) * (slic > prev_dog[4:, 2:-2])
            kpm[2:-2, 2:-2] += (slic > prev_dog[2:-2, :-4]) * (slic > prev_dog[2:-2, 4:])
            kpm[2:-2, 2:-2] += (slic > prev_dog[:-4, :-4]) * (slic > prev_dog[4:, 4:])
            kpm[2:-2, 2:-2] += (slic > prev_dog[4:, :-4]) * (slic > prev_dog[:-4, 4:])
            kpm[2:-2, 2:-2] += (slic > prev_dog[4:, 1:-3]) * (slic > prev_dog[:-4, 1:-3])
            kpm[2:-2, 2:-2] += (slic > prev_dog[1:-3, :-4]) * (slic > prev_dog[1:-3, 4:])
            kpm[2:-2, 2:-2] += (slic > prev_dog[3:-1, :-4]) * (slic > prev_dog[3:-1, 4:])
            kpm[2:-2, 2:-2] += (slic > prev_dog[4:, 3:-1]) * (slic > prev_dog[:-4, 3:-1])

        else:
            target = 14

        if mask is not None:
            kpm[mask] = 0

    return kpma == target


class BlobDetection(object):
    """
        Performs a blob detection:
        http://en.wikipedia.org/wiki/Blob_detection
        using a Difference of Gaussian + Pyramid of Gaussians

    """
    tresh = 0.6

    def __init__(self, img, cur_sigma=0.25, init_sigma=0.5, dest_sigma=1, scale_per_octave=2, mask=None):
        """
        Performs a blob detection:
        http://en.wikipedia.org/wiki/Blob_detection
        using a Difference of Gaussian + Pyramid of Gaussians

        :param img: input image
        :param cur_sigma: estimated smoothing of the input image. 0.25 correspond to no interaction between pixels.
        :param init_sigma: start searching at this scale (sigma=0.5: 10% interaction with first neighbor)
        :param dest_sigma: sigma at which the resolution is lowered (change of octave)
        :param scale_per_octave: Number of scale to be performed per octave
        :param mask: mask where pixel are not valid
        """
        # self.raw = numpy.log(img.astype(numpy.float32))
        self.raw = img.astype(numpy.float32)
        self.cur_sigma = float(cur_sigma)
        self.init_sigma = float(init_sigma)
        self.dest_sigma = float(dest_sigma)
        self.scale_per_octave = int(scale_per_octave)
        self.raw_mask = mask
        self.cur_mask = None
        self.do_mask = True
        self.mask = None
        self.grow = None
        self.data = None  # current image
        self.sigmas = None  # contains pairs of absolute sigma and relative ones...
        self.blurs = []  # different blurred images
        self.dogs = []  # different difference of gaussians
        self.dogs_init = []
        self.border_size = 5  # size of the border, unused: prefer mask
        self.keypoints = []
        self.delta = []
        self.curr_reduction = 1.0
        self.detection_started = False
        self.octave = 0
        self.raw_kp = []
        self.ref_kp = []
        self.dtype = numpy.dtype([('x', numpy.float32), ('y', numpy.float32), ('sigma', numpy.float32), ('I', numpy.float32)])
        self.bilinear = None
        self.already_blurred = []

    def __repr__(self):
        lststr = ["Blob detection, shape=%s, processed=%s." % (self.raw.shape, self.detection_started)]
        lststr.append("Sigmas: input=%.3f \t init=%.3f, dest=%.3f over %i blurs/octave" % (self.cur_sigma, self.init_sigma, self.dest_sigma, self.scale_per_octave))
        lststr.append("found %s keypoint up to now, we are at reduction %s" % (len(self.keypoints), self.curr_reduction))
        return os.linesep.join(lststr)

    def _init_mask(self):
        """
        Initialize the mask
        """
        if self.raw_mask is not None:
            self.mask = (self.raw_mask != 0).astype(numpy.int8)
        else:
            self.mask = (self.raw < 0).astype(numpy.int8)
        # mask out the border of the image
        self.mask[0, :] = 1
        self.mask[-1, :] = 1
        self.mask[:, 0] = 1
        self.mask[:, -1] = 1
        to_mask = numpy.where(self.mask)
        # always use a mask!!
        self.do_mask = True  # to_mask[0].size > 0
        if self.do_mask:
            self.raw[to_mask] = 0

            # initial grow of 4*sigma_dest ... subsequent re-grow of half
            grow = int(round(4.0 * self.dest_sigma))
            if not pyFAI_morphology:
                my, mx = numpy.ogrid[-grow:grow + 1, -grow:grow + 1]
                grow = (mx * mx + my * my) <= grow * grow
            self.cur_mask = morphology.binary_dilation(self.mask, grow)
            # subsequent grow
            grow = int(2.0 * self.dest_sigma)
            if not pyFAI_morphology:
                my, mx = numpy.ogrid[-grow:grow + 1, -grow:grow + 1]
                grow = (mx * mx + my * my) <= grow * grow
            self.grow = grow

    def _initial_blur(self):
        """
        Blur the original image to achieve the requested level of blur init_sigma
        """
        if self.init_sigma > self.cur_sigma:
            sigma = sqrt(self.init_sigma * self.init_sigma - self.cur_sigma * self.cur_sigma)
            self.data = gaussian_filter(self.raw, sigma)
        else:
            self.data = self.raw

    def _calc_sigma(self):
        """
        Calculate all sigma to blur an image within an octave
        """
        if self.data is None:
            self._initial_blur()
        previous = self.init_sigma
        incr = 0
        self.sigmas = [(previous, incr)]
        for i in range(1, self.scale_per_octave + 3):
            sigma_abs = self.init_sigma * (self.dest_sigma / self.init_sigma) ** (1.0 * i / (self.scale_per_octave))
            increase = previous * sqrt((self.dest_sigma / self.init_sigma) ** (2.0 / self.scale_per_octave) - 1.0)
            self.sigmas.append((sigma_abs, increase))
            previous = sigma_abs
        logger.debug("Sigma= %s", self.sigmas)

    def _one_octave(self, shrink=True, refine=True, n_5=False):
        """
        Return the blob coordinates for an octave

        :param shrink: perform the image shrinking after the octave processing
        :param refine: can be None, True, "SG2" and "SG4" do_SG4: perform 3point hessian calculation or Savitsky-Golay 2nd or 4th order fit.
        :param n_5: use 5 points instead of 3 in y and x to determinate if a point is a maximum

        """
        if not self.sigmas:
            self._calc_sigma()
        if self.do_mask and (self.cur_mask is None):
            self._init_mask()
        if self.do_mask and (numpy.logical_not(self.cur_mask).sum(dtype=int) == 0):
            return

        previous = self.data
        dog_shape = (len(self.sigmas) - 1,) + self.data.shape
        self.dogs = numpy.zeros(dog_shape, dtype=numpy.float32)

        idx = 0
        i = 0
        for _sigma_abs, sigma_rel in self.sigmas:
            # if self.already_blurred != [] and i < 3:
            #    sigma_rel = 0
            #    if i > 0: previous = self.already_blurred[i-1]
            if sigma_rel == 0:
                self.blurs.append(previous)
            else:
                new_blur = gaussian_filter(previous, sigma_rel)
                self.blurs.append(new_blur)
                self.dogs[idx] = previous - new_blur
                previous = new_blur
                idx += 1
            i += 1

        if self.dogs[0].shape == self.raw.shape:
            self.dogs_init = self.dogs

        if _blob:
            valid_points = _blob.local_max(self.dogs, self.cur_mask, n_5)
        else:
            valid_points = local_max(self.dogs, self.cur_mask, n_5)
        kps, kpy, kpx = numpy.where(valid_points)
        self.raw_kp.append((kps, kpy, kpx))

        if refine:
            if "startswith" in dir(refine) and refine.startswith("SG"):
                kpx, kpy, kps, _delta_s = self.refine_Hessian_SG(kpx, kpy, kps)
                l = kpx.size
                peak_val = self.dogs[(numpy.around(kps).astype(int),
                                      numpy.around(kpy).astype(int),
                                      numpy.around(kpx).astype(int))]
                valid = numpy.ones(l, dtype=bool)
            else:
                kpx, kpy, kps, peak_val, valid = self.refine_Hessian(kpx, kpy, kps)
                l = valid.sum()
                self.ref_kp.append((kps, kpy, kpx))
            print('After refinement : %i keypoints' % l)
        else:
            peak_val = self.dogs[kps, kpy, kpx]
            l = kpx.size
            valid = numpy.ones(l, bool)

        keypoints = numpy.recarray((l,), dtype=self.dtype)

        if l != 0:
            keypoints[:].x = (kpx[valid] + 0.5) * self.curr_reduction - 0.5  # Place ourselves at the center of the pixel, and back
            keypoints[:].y = (kpy[valid] + 0.5) * self.curr_reduction - 0.5  # Place ourselves at the center of the pixel, and back
            sigmas = self.init_sigma * (self.dest_sigma / self.init_sigma) ** ((kps[valid]) / (self.scale_per_octave))
            keypoints[:].sigma = (self.curr_reduction * sigmas)
            keypoints[:].I = peak_val[valid]

        if shrink:
            # shrink data so that they can be treated by next octave
            logger.debug("In shrink")
            last = self.blurs[self.scale_per_octave]
            ty, tx = last.shape
            if ty % 2 != 0 or tx % 2 != 0:
                new_tx = 2 * ((tx + 1) // 2)
                new_ty = 2 * ((ty + 1) // 2)
                new_last = numpy.zeros((new_ty, new_tx), last.dtype)
                new_last[:ty, :tx] = last
                last = new_last
                if self.do_mask:
                    new_msk = numpy.ones((new_ty, new_tx), numpy.int8)
                    new_msk[:ty, :tx] = self.cur_mask
                    self.cur_mask = new_msk
            self.data = binning(last, 2) / 4.0
            self.curr_reduction *= 2.0
            self.octave += 1
            self.blurs = []
            if self.do_mask:
                self.cur_mask = (binning(self.cur_mask, 2) > 0).astype(numpy.int8)
                self.cur_mask = morphology.binary_dilation(self.cur_mask, self.grow)

        if len(self.keypoints) == 0:
            self.keypoints = keypoints
        else:
            old_size = self.keypoints.size
            new_size = old_size + l
            new_keypoints = numpy.recarray(new_size, dtype=self.dtype)
            new_keypoints[:old_size] = self.keypoints
            new_keypoints[old_size:] = keypoints
            self.keypoints = new_keypoints

    def refine_Hessian(self, kpx, kpy, kps):
        """
        Refine the keypoint location based on a 3 point derivative, and delete
        non-coherent keypoints.

        :param kpx: x_pos of keypoint
        :param kpy: y_pos of keypoint
        :param kps: s_pos of keypoint
        :return: arrays of corrected coordinates of keypoints, values and
            locations of keypoints
        """
        curr = self.dogs[(kps, kpy, kpx)]
        nx = self.dogs[(kps, kpy, kpx + 1)]
        px = self.dogs[(kps, kpy, kpx - 1)]
        ny = self.dogs[(kps, kpy + 1, kpx)]
        py = self.dogs[(kps, kpy - 1, kpx)]
        ns = self.dogs[(kps + 1, kpy, kpx)]
        ps = self.dogs[(kps - 1, kpy, kpx)]

        nxny = self.dogs[(kps, kpy + 1, kpx + 1)]
        nxpy = self.dogs[(kps, kpy - 1, kpx + 1)]
        pxny = self.dogs[(kps, kpy + 1, kpx - 1)]
        pxpy = self.dogs[(kps, kpy - 1, kpx - 1)]

        nsny = self.dogs[(kps + 1, kpy + 1, kpx)]
        nspy = self.dogs[(kps + 1, kpy - 1, kpx)]
        psny = self.dogs[(kps - 1, kpy + 1, kpx)]
        pspy = self.dogs[(kps - 1, kpy - 1, kpx)]

        nxns = self.dogs[(kps + 1, kpy, kpx + 1)]
        nxps = self.dogs[(kps - 1, kpy, kpx + 1)]
        pxns = self.dogs[(kps + 1, kpy, kpx - 1)]
        pxps = self.dogs[(kps - 1, kpy, kpx - 1)]

        dx = (nx - px) / 2.0
        dy = (ny - py) / 2.0
        ds = (ns - ps) / 2.0
        dxx = (nx - 2.0 * curr + px)
        dyy = (ny - 2.0 * curr + py)
        dss = (ns - 2.0 * curr + ps)
        dxy = (nxny - nxpy - pxny + pxpy) / 4.0
        dxs = (nxns - nxps - pxns + pxps) / 4.0
        dsy = (nsny - nspy - psny + pspy) / 4.0
        det = -(dxs * dyy * dxs) + dsy * dxy * dxs + dxs * dsy * dxy - dss * dxy * dxy - dsy * dsy * dxx + dss * dyy * dxx
        K00 = dyy * dxx - dxy * dxy
        K01 = dxs * dxy - dsy * dxx
        K02 = dsy * dxy - dxs * dyy
        K10 = dxy * dxs - dsy * dxx
        K11 = dss * dxx - dxs * dxs
        K12 = dxs * dsy - dss * dxy
        K20 = dsy * dxy - dyy * dxs
        K21 = dsy * dxs - dss * dxy
        K22 = dss * dyy - dsy * dsy

        delta_s = -(ds * K00 + dy * K01 + dx * K02) / det
        delta_y = -(ds * K10 + dy * K11 + dx * K12) / det
        delta_x = -(ds * K20 + dy * K21 + dx * K22) / det
        peakval = curr + 0.5 * (delta_s * ds + delta_y * dy + delta_x * dx)
        mask = numpy.logical_and(numpy.logical_and(abs(delta_x) < self.tresh, abs(delta_y) < self.tresh), abs(delta_s) < self.tresh)
        return kpx + delta_x, kpy + delta_y, kps + delta_s, peakval, mask

    def refine_Hessian_SG(self, kpx, kpy, kps):
        """
        Savitzky Golay algorithm to check if a point is really the maximum
        :param kpx: x_pos of keypoint
        :param kpy: y_pos of keypoint
        :param kps: s_pos of keypoint
        :return: array of corrected keypoints

        """

        k2x = []
        k2y = []
        sigmas = []
        kds = []

        # Hessian patch 3 ordre 2
        SGX0Y0 = [-0.11111111, 0.22222222, -0.11111111, 0.22222222, 0.55555556, 0.22222222, -0.11111111, 0.22222222, -0.11111111]
        SGX1Y0 = [-0.16666667, 0.00000000, 0.16666667, -0.16666667, 0.00000000, 0.16666667, -0.16666667, 0.00000000, 0.16666667]
        SGX2Y0 = [0.16666667, -0.33333333, 0.16666667, 0.16666667, -0.33333333, 0.16666667, 0.16666667, -0.33333333, 0.16666667]
        SGX0Y1 = [-0.16666667, -0.16666667, -0.16666667, 0.00000000, 0.00000000, 0.00000000, 0.16666667, 0.16666667, 0.16666667]
        SGX1Y1 = [0.25000000, 0.00000000, -0.25000000, 0.00000000, 0.00000000, 0.00000000, -0.25000000, 0.00000000, 0.25000000]
        SGX0Y2 = [0.16666667, 0.16666667, 0.16666667, -0.33333333, -0.33333333, -0.33333333, 0.16666667, 0.16666667, 0.16666667]

        # SGX0Y0 = [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
        # SGX1Y0 = [0.0,0.0,0.0,-0.5,0.0,0.5,0.0,0.0,0.0]
        # SGX2Y0 = [0.0,0.0,0.0,0.33333333,-0.66666667,0.33333333,0.0,0.0,0.0]
        # SGX0Y1 = [0.0,-0.5,0.0,0.0,0.0,0.0,0.0,0.5,0.0]
        # SGX0Y2 = [0.0, 0.33333333 , 0.0 , 0.0 , -0.66666667,0.0, 0.0 , 0.33333333 , 0.0]

        for y, x, sigma in zip(kpy, kpx, kps):

            curr_dog = self.dogs[sigma]
            prev_dog = self.dogs[sigma - 1]
            next_dog = self.dogs[sigma + 1]

            # if (x > 1 and x < curr_dog.shape[1] - 2 and y > 1 and y < curr_dog.shape[0] - 2):

            patch3 = curr_dog[y - 1:y + 2, x - 1:x + 2]
            patch3_prev = prev_dog[y - 1:y + 2, x - 1:x + 2]
            patch3_next = next_dog[y - 1:y + 2, x - 1:x + 2]

            dx = (SGX1Y0 * patch3.ravel()).sum()
            dy = (SGX0Y1 * patch3.ravel()).sum()
            d2x = (SGX2Y0 * patch3.ravel()).sum()
            d2y = (SGX0Y2 * patch3.ravel()).sum()
            dxy = (SGX1Y1 * patch3.ravel()).sum()

            s_next = (SGX0Y0 * patch3_next.ravel()).sum()
            s = (SGX0Y0 * patch3.ravel()).sum()
            s_prev = (SGX0Y0 * patch3_prev.ravel()).sum()
            d2s = (s_next + s_prev - 2.0 * s)
            ds = (s_next - s_prev) / 2.0

            dx_next = (SGX1Y0 * patch3_next.ravel()).sum()
            dx_prev = (SGX1Y0 * patch3_prev.ravel()).sum()

            dy_next = (SGX0Y1 * patch3_next.ravel()).sum()
            dy_prev = (SGX0Y1 * patch3_prev.ravel()).sum()

            dxs = (dx_next - dx_prev) / 2.0
            dys = (dy_next - dy_prev) / 2.0

            print(dx, dy, ds)
            print(d2x, d2y, d2s, dxy, dxs, dys)

            lap = numpy.array([[d2y, dxy, dys], [dxy, d2x, dxs], [dys, dxs, d2s]])
            delta = -(numpy.dot(numpy.linalg.inv(lap), [dy, dx, ds]))
            print(y, x)
            print(delta)
            # err = numpy.linalg.norm(delta[:-1])

            if numpy.abs(delta[0]) <= self.tresh and numpy.abs(delta[1]) <= self.tresh and numpy.abs(delta[2]) <= self.tresh:
                k2x.append(x + delta[1])
                k2y.append(y + delta[0])
                sigmas.append(sigma + delta[2])
                # kds.append(delta[2])
                # kdx.append(delta[1])
                # kdy.append(delta[0])

        return numpy.asarray(k2x), numpy.asarray(k2y), numpy.asarray(sigmas), numpy.asarray(kds)

    def direction(self):
        """
        Perform and plot the two main directions of the peaks, considering their previously
        calculated scale ,by calculating the Hessian at different sizes as the combination of
        gaussians and their first and second derivatives

        """
        import pylab
        j = 0
        vals = []
        vects = []
        kpx = self.keypoints.x
        kpy = self.keypoints.y
        sigma = self.keypoints.sigma
        img = self.raw
        pylab.figure()
        pylab.imshow(img, interpolation='nearest')

        for y, x, s in zip(kpy, kpx, sigma):
            s_patch = numpy.trunc(s * 2)

            if s_patch % 2 == 0:
                s_patch += 1

            if s_patch < 3:
                s_patch = 3

            if (x > s_patch / 2 and x < img.shape[1] - s_patch / 2 - 1 and y > s_patch / 2 and y < img.shape[0] - s_patch / 2):

                patch = img[y - (s_patch - 1) / 2:y + (s_patch - 1) / 2 + 1, x - (s_patch - 1) / 2:x + (s_patch - 1) / 2 + 1]
                x_patch = numpy.arange(s_patch)
                Gx = numpy.exp(-4 * numpy.log(2) * (x_patch - numpy.median(x_patch)) ** 2 / s)
                Gy = Gx[:, numpy.newaxis]
                dGx = -Gx * 4 * numpy.log(2) / s * 2 * (x_patch - numpy.median(x_patch))
                dGy = dGx[:, numpy.newaxis]
                d2Gx = -8 * numpy.log(2) / s * ((x_patch - numpy.median(x_patch)) * dGx + Gx)
                d2Gy = d2Gx[:, numpy.newaxis]

                Hxx = d2Gx * Gy
                Hyy = d2Gy * Gx
                Hxy = dGx * dGy

                d2x = (Hxx.ravel() * patch.ravel()).sum()
                d2y = (Hyy.ravel() * patch.ravel()).sum()
                dxy = (Hxy.ravel() * patch.ravel()).sum()
                H = numpy.array([[d2y, dxy], [dxy, d2x]])
                val, vect = numpy.linalg.eig(H)

                # print 'new point'
                # print x, y
                # print val
                # print vect
                # print numpy.dot(vect[0],vect[1])
                _e = numpy.abs(val[0] - val[1]) / numpy.abs(val[0] + val[1])
                j += 1
                # print j
                # print e
                if numpy.abs(val[1]) < numpy.abs(val[0]):  # reorganisation des valeurs propres et vecteurs propres
                    val[0], val[1] = val[1], val[0]
                    vect = vect[-1::-1, :]

                pylab.annotate("",
                               xy=(x + vect[0][0] * val[0], y + vect[0][1] * val[0]),
                               xytext=(x, y),
                               arrowprops=dict(facecolor='red', shrink=0.05))

                pylab.annotate("",
                               xy=(x + vect[1][0] * val[1], y + vect[1][1] * val[1]),
                               xytext=(x, y),
                               arrowprops=dict(facecolor='red', shrink=0.05))
                pylab.plot(x, y, 'og')
                vals.append(val)
                vects.append(vect)
        return vals, vects

    def refinement(self):
        from numpy import cos, sin, arctan2, pi
        val, vect = self.direction()

        L = 0.114
        # L = 1.0

        poni1 = self.raw.shape[0] / 2.0
        poni2 = self.raw.shape[1] / 2.0
        # poni1 = 0.0599/100.0 * power(10,6)
        # poni2 = -0.07623/100.0 * power(10,6)

        d1 = self.keypoints.y - poni1
        d2 = self.keypoints.x - poni2
        rot1 = rot2 = rot3 = 0
        # rot1 = -0.22466
        # rot2 = -0.07476
        # rot3 = 0.00000005

        valy, valx = numpy.transpose(vect)[0]
        phi_exp = arctan2(valy, valx) % pi
        # print "phi exp"
        # print phi_exp * 180/ pi

        cosrot1 = cos(rot1)
        cosrot2 = cos(rot2)
        cosrot3 = cos(rot3)
        sinrot1 = sin(rot1)
        sinrot2 = sin(rot2)
        sinrot3 = sin(rot3)

        L = -L
        _dy = ((L * cosrot1 * cosrot2 + d2 * cosrot2 * sinrot1 - d1 * sinrot2) * (2 * cosrot2 * cosrot3 * (d1 * cosrot2 * cosrot3 + \
                        d2 * (cosrot3 * sinrot1 * sinrot2 - cosrot1 * sinrot3) + L * (cosrot1 * cosrot3 * sinrot2 + \
                        sinrot1 * sinrot3)) + 2 * cosrot2 * sinrot3 * (d1 * cosrot2 * sinrot3 + \
                        L * (-(cosrot3 * sinrot1) + cosrot1 * sinrot2 * sinrot3) + d2 * (cosrot1 * cosrot3 + \
                        sinrot1 * sinrot2 * sinrot3)))) / (2.*sqrt((d1 * cosrot2 * cosrot3 + d2 * (cosrot3 * sinrot1 * sinrot2 - cosrot1 * sinrot3) + \
                        L * (cosrot1 * cosrot3 * sinrot2 + sinrot1 * sinrot3)) ** 2 + (d1 * cosrot2 * sinrot3 + \
                        L * (-(cosrot3 * sinrot1) + cosrot1 * sinrot2 * sinrot3) + d2 * (cosrot1 * cosrot3 + \
                        sinrot1 * sinrot2 * sinrot3)) ** 2) * ((L * cosrot1 * cosrot2 + \
                        d2 * cosrot2 * sinrot1 - d1 * sinrot2) ** 2 + (d1 * cosrot2 * cosrot3 + \
                        d2 * (cosrot3 * sinrot1 * sinrot2 - cosrot1 * sinrot3) + L * (cosrot1 * cosrot3 * sinrot2 + sinrot1 * sinrot3)) ** 2 + \
                         (d1 * cosrot2 * sinrot3 + L * (-(cosrot3 * sinrot1) + cosrot1 * sinrot2 * sinrot3) + d2 * (cosrot1 * cosrot3 + \
                        sinrot1 * sinrot2 * sinrot3)) ** 2)) + (sinrot2 * sqrt((d1 * cosrot2 * cosrot3 + \
                        d2 * (cosrot3 * sinrot1 * sinrot2 - cosrot1 * sinrot3) + L * (cosrot1 * cosrot3 * sinrot2 + sinrot1 * sinrot3)) ** 2 + \
                         (d1 * cosrot2 * sinrot3 + L * (-(cosrot3 * sinrot1) + cosrot1 * sinrot2 * sinrot3) + d2 * (cosrot1 * cosrot3 + \
                        sinrot1 * sinrot2 * sinrot3)) ** 2)) / ((L * cosrot1 * cosrot2 + d2 * cosrot2 * sinrot1 - d1 * sinrot2) ** 2 + \
                         (d1 * cosrot2 * cosrot3 + d2 * (cosrot3 * sinrot1 * sinrot2 - cosrot1 * sinrot3) + \
                        L * (cosrot1 * cosrot3 * sinrot2 + sinrot1 * sinrot3)) ** 2 + (d1 * cosrot2 * sinrot3 + L * (-(cosrot3 * sinrot1) + \
                        cosrot1 * sinrot2 * sinrot3) + d2 * (cosrot1 * cosrot3 + sinrot1 * sinrot2 * sinrot3)) ** 2)

        _dx = ((L * cosrot1 * cosrot2 + d2 * cosrot2 * sinrot1 - d1 * sinrot2) * (2 * (cosrot3 * sinrot1 * sinrot2 - \
                        cosrot1 * sinrot3) * (d1 * cosrot2 * cosrot3 + d2 * (cosrot3 * sinrot1 * sinrot2 - cosrot1 * sinrot3) + \
                        L * (cosrot1 * cosrot3 * sinrot2 + sinrot1 * sinrot3)) + 2 * (cosrot1 * cosrot3 + \
                        sinrot1 * sinrot2 * sinrot3) * (d1 * cosrot2 * sinrot3 + L * (-(cosrot3 * sinrot1) + cosrot1 * sinrot2 * sinrot3) + \
                        d2 * (cosrot1 * cosrot3 + sinrot1 * sinrot2 * sinrot3)))) / (2.*sqrt((d1 * cosrot2 * cosrot3 + d2 * (cosrot3 * sinrot1 * sinrot2 - \
                        cosrot1 * sinrot3) + L * (cosrot1 * cosrot3 * sinrot2 + sinrot1 * sinrot3)) ** 2 + (d1 * cosrot2 * sinrot3 + \
                        L * (-(cosrot3 * sinrot1) + cosrot1 * sinrot2 * sinrot3) + d2 * (cosrot1 * cosrot3 + \
                        sinrot1 * sinrot2 * sinrot3)) ** 2) * ((L * cosrot1 * cosrot2 + d2 * cosrot2 * sinrot1 - d1 * sinrot2) ** 2 + \
                         (d1 * cosrot2 * cosrot3 + d2 * (cosrot3 * sinrot1 * sinrot2 - cosrot1 * sinrot3) + \
                        L * (cosrot1 * cosrot3 * sinrot2 + sinrot1 * sinrot3)) ** 2 + (d1 * cosrot2 * sinrot3 + \
                        L * (-(cosrot3 * sinrot1) + cosrot1 * sinrot2 * sinrot3) + \
                        d2 * (cosrot1 * cosrot3 + sinrot1 * sinrot2 * sinrot3)) ** 2)) - (cosrot2 * sinrot1 * sqrt((d1 * cosrot2 * cosrot3 + \
                        d2 * (cosrot3 * sinrot1 * sinrot2 - cosrot1 * sinrot3) + \
                        L * (cosrot1 * cosrot3 * sinrot2 + sinrot1 * sinrot3)) ** 2 + (d1 * cosrot2 * sinrot3 + L * (-(cosrot3 * sinrot1) + \
                        cosrot1 * sinrot2 * sinrot3) + d2 * (cosrot1 * cosrot3 + sinrot1 * sinrot2 * sinrot3)) ** 2)) / ((L * cosrot1 * cosrot2 + \
                        d2 * cosrot2 * sinrot1 - d1 * sinrot2) ** 2 + (d1 * cosrot2 * cosrot3 + d2 * (cosrot3 * sinrot1 * sinrot2 - \
                        cosrot1 * sinrot3) + L * (cosrot1 * cosrot3 * sinrot2 + sinrot1 * sinrot3)) ** 2 + (d1 * cosrot2 * sinrot3 + \
                        L * (-(cosrot3 * sinrot1) + cosrot1 * sinrot2 * sinrot3) + d2 * (cosrot1 * cosrot3 + sinrot1 * sinrot2 * sinrot3)) ** 2)

        phi_th = arctan2(d1, d2)
        # print "phi th"
        # print phi_th_2
        err = numpy.sum((phi_th - phi_exp) ** 2) / self.keypoints.x.size
        print("err")
        print(err)

        return val, vect

    def process(self, max_octave=None):
        """
        Perform the keypoint extraction for max_octave cycles or until all octaves have been processed.
        :param max_octave: number of octave to process
        """
        finished = False
        if self.cur_mask is None:
            self._init_mask()
        if self.data is None:
            self._initial_blur()
        if self.sigmas is None:
            self._calc_sigma()
        while not finished:
            self._one_octave(shrink=True, refine=True, n_5=True)
            if max_octave and self.octave >= max_octave:
                finished = True
            else:
                finished = (numpy.logical_not(self.cur_mask).sum(dtype=int) == 0)
        logger.warning("Blob detection found %i keypoints", len(self.keypoints))

    def nearest_peak(self, p, refine=True, Imin=None):
        """
        Return the nearest peak from a position

        :param p: input position (y,x) 2-tuple of float
        :param refine: shall the position be refined on the raw data
        :param Imin: minimum of intensity above the background
        """
        if Imin:
            valid = (self.keypoints.I >= Imin)
            kp = self.keypoints[valid]
        else:
            kp = self.keypoints
        dy = kp.y - p[0]
        dx = kp.x - p[1]
        r2 = dx * dx + dy * dy
        best_pos = r2.argmin()
        best = [kp[best_pos].y, kp[best_pos].x]
        if refine:
            if self.bilinear is None:
                self.bilinear = Bilinear(self.raw)
            best = self.bilinear.local_maxi(best)
        return best

    def peaks_from_area(self, mask, keep=None, refine=True, Imin=None, dmin=0.0, **kwargs):
        """
        Return the list of peaks within an area

        :param mask: 2d array with mask.
        :param refine: shall the position be refined on the raw data
        :param Imin: minimum of intensity above the background
        :param kwarg: ignored parameters
        :return: list of peaks [y,x], [y,x], ...]
        """
        y = numpy.round(self.keypoints.y).astype(int)
        x = numpy.round(self.keypoints.x).astype(int)
        is_inside = (mask[y, x]).astype(bool)
        dmin2 = dmin * dmin
        if is_inside.sum() == 0:
            logger.error("No keypoint that region")
            return []
        kp = self.keypoints[is_inside]
        if Imin:
            valid = kp.I >= Imin
            if valid.sum() == 0:
                logger.error("no keypoint match Intensity criteria in the region")
                valid2 = self.raw[y, x] >= Imin
                good_kp = self.keypoints[numpy.logical_and(valid2, is_inside)]
            else:
                good_kp = kp[valid]
        else:
            good_kp = kp
        # sort keypoint by intensity
        order = numpy.argsort(good_kp.I)
        order = order[-1::-1][:keep]  # keep only the most intense
        good_kp = good_kp[order]
        if keep and (len(good_kp) > keep):
            keep_kp = []
            for i in good_kp:
                if is_far_from_group(i, keep_kp, dmin2):
                    keep_kp.append(i)
                if len(keep_kp) >= keep:
                    break
            good_kp = keep_kp
        if refine:
            if self.bilinear is None:
                self.bilinear = Bilinear(self.raw)
            return [self.bilinear.local_maxi((i.y, i.x)) for i in good_kp]
        else:
            return [(i.y, i.x) for i in good_kp]

    def show_stats(self):
        """
        Shows a window with the repartition of keypoint in function of scale/intensity
        """
        if len(self.keypoints) == 0:
            logger.warning("No keypoints yet: running process before display")
            self.process()
        import pylab
        f = pylab.figure()
        ax = f.add_subplot(1, 1, 1)
        ax.plot(self.keypoints.sigma, self.keypoints.I, '.r')
        ax.set_xlabel("Sigma")
        ax.set_ylabel("Intensity")
        ax.set_title("Peak repartition")
        f.show()

    def show_neighboor(self):
        import pylab
        nghx = []
        nghy = []

        for i in range(self.keypoints.x.size):
            y, x = self.nearest_peak((self.keypoints.y[i], self.keypoints.x[i]))
            nghx.append(x)
            nghy.append(y)

        nghx = numpy.asarray(nghx)
        nghy = numpy.asarray(nghy)

        pylab.figure()
        pylab.imshow(self.raw, interpolation='nearest')
        pylab.plot(self.keypoints.x, self.keypoints.y, 'og')

        for i in range(self.keypoints.x.size):
            pylab.annotate("", xy=(nghx[i], nghy[i]),
                           xytext=(self.keypoints.x[i], self.keypoints.y[i]), arrowprops=dict(facecolor='red', shrink=0.05),)

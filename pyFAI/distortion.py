# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2013-2018 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, print_function, division, with_statement

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "03/12/2018"
__status__ = "development"

import logging
import threading
import os
import numpy
logger = logging.getLogger(__name__)
from math import ceil, floor
from . import detectors
from .opencl import ocl
if ocl:
    from .opencl import azim_lut as ocl_azim_lut
    from .opencl import azim_csr as ocl_azim_csr
else:
    ocl_azim_lut = ocl_azim_csr = None
from .third_party import six

try:
    from .ext import _distortion
    from .ext import sparse_utils
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    logger.warning("Import _distortion cython implementation failed ... pure python version is terribly slow !!!")
    _distortion = None

try:
    from scipy.sparse import linalg, csr_matrix
except IOError:
    logger.warning("Scipy is missing ... uncorrection will be handled the old way")
    linalg = None
else:
    import scipy
    v = tuple(int(i) for i in scipy.version.short_version.split(".") if i.isdigit())
    if v < (0, 11):
        logger.warning("Scipy is too old ... uncorrection will be handled the old way")
        linalg = None


class Distortion(object):
    """
    This class applies a distortion correction on an image.

    New version compatible both with CSR and LUT...
    """
    def __init__(self, detector="detector", shape=None, resize=False, empty=0,
                 mask=None, method="CSR", device=None, workgroup=8):
        """
        :param detector: detector instance or detector name
        :param shape: shape of the output image
        :param resize: allow the output shape to be different from the input shape
        :param empty: value to be given for empty bins
        :param method: "lut" or "csr", the former is faster
        :param device: Name of the device: None for OpenMP, "cpu" or "gpu" or the id of the OpenCL device a 2-tuple of integer
        :param workgroup: workgroup size for CSR on OpenCL
        """
        self._shape_out = None
        if isinstance(detector, six.string_types):
            self.detector = detectors.detector_factory(detector)
        else:  # we assume it is a Detector instance
            self.detector = detector
        self.shape_in = self.detector.shape
        if mask is not None:
            self.mask = numpy.ascontiguousarray(mask, numpy.int8)
        else:
            self.mask = numpy.ascontiguousarray(self.detector.mask, numpy.int8)
        self.resize = resize
        if shape is not None:
            self._shape_out = tuple([int(i) for i in shape])
        elif not self.resize:
            if self.detector.shape is not None:
                self._shape_out = self.detector.shape
            else:
                raise RuntimeError("You need to provide either the detector or its shape")

        self._sem = threading.Semaphore()
        self.bin_size = None
        self.max_size = None
        self.pos = None
        self.lut = None
        self.delta1 = self.delta2 = None  # max size of an pixel on a regular grid ...
        self.offset1 = self.offset2 = 0  # position of the first bin
        self.integrator = None
        self.empty = empty  # "dummy" value for empty bins
        if not method:
            self.method = "lut"
        else:
            self.method = method.lower()
        self.device = device
        if not workgroup:
            self.workgroup = 8
        else:
            self.workgroup = int(workgroup)

    def __repr__(self):
        return os.linesep.join(["Distortion correction %s on device %s for detector shape %s:" % (self.method, self.device, self._shape_out),
                                self.detector.__repr__()])

    def reset(self, method=None, device=None, workgroup=None, prepare=True):
        """
        reset the distortion correction and re-calculate the look-up table

        :param method: can be "lut" or "csr", "lut" looks faster
        :param device: can be None, "cpu" or "gpu" or the id as a 2-tuple of integer
        :param worgroup: enforce the workgroup size for CSR.
        :param prepare: set to false to only reset and not re-initialize
        """
        with self._sem:
            self.max_size = None
            self.pos = None
            self.lut = None
            self.delta1 = self.delta2 = None
            self.offset1 = self.offset2 = 0
            self.integrator = None
            if method is not None:
                self.method = method.lower()
            if device is not None:
                self.device = device
            if workgroup is not None:
                self.workgroup = int(workgroup)
        if prepare:
            self.calc_init()

    @property
    def shape_out(self):
        """
        Calculate/cache the output shape

        :return: output shape
        """
        if self._shape_out is None:
            self.calc_pos()
        return self._shape_out

    def calc_pos(self, use_cython=True):
        """Calculate the pixel boundary position on the regular grid

        :return: pixel corner positions (in pixel units) on the regular grid
        :rtype: ndarray of shape (nrow, ncol, 4, 2)
        """
        if self.delta1 is None:
            with self._sem:
                if self.delta1 is None:
                    # TODO: implement equivalent in Cython
                    if _distortion and use_cython:
                        self.pos, self.delta1, self.delta2, shape_out, offset = _distortion.calc_pos(self.detector.get_pixel_corners(), self.detector.pixel1, self.detector.pixel2, self._shape_out)
                        if self._shape_out is None:
                            self.offset1, self.offset2 = offset
                            self._shape_out = shape_out
                    else:
                        pixel_size = numpy.array([self.detector.pixel1, self.detector.pixel2], dtype=numpy.float32)
                        # make it a 4D array
                        pixel_size.shape = 1, 1, 1, 2
                        pixel_size.strides = 0, 0, 0, pixel_size.strides[-1]
                        self.pos = self.detector.get_pixel_corners()[..., 1:] / pixel_size
                        if self._shape_out is None:
                            # if defined, it is probably because resize=False
                            corner_pos = self.pos.view()
                            corner_pos.shape = -1, 2
                            pos1_min, pos2_min = corner_pos.min(axis=0)
                            pos1_max, pos2_max = corner_pos.max(axis=0)
                            self._shape_out = (int(ceil(pos1_max - pos1_min)),
                                               int(ceil(pos2_max - pos2_min)))
                            self.offset1, self.offset2 = pos1_min, pos2_min
                        pixel_delta = self.pos.view()
                        pixel_delta.shape = -1, 4, 2
                        self.delta1, self.delta2 = ((numpy.ceil(pixel_delta.max(axis=1)) - numpy.floor(pixel_delta.min(axis=1))).max(axis=0)).astype(int)
        return self.pos

    def calc_size(self, use_cython=True):
        """Calculate the number of pixels falling into every single bin and

        :return: max of pixel falling into a single bin

        Considering the "half-CCD" spline from ID11 which describes a (1025,2048) detector,
        the physical location of pixels should go from:
        [-17.48634 : 1027.0543, -22.768829 : 2028.3689]
        We chose to discard pixels falling outside the [0:1025,0:2048] range with a lose of intensity
        """
        if self.pos is None:
            pos = self.calc_pos()
        else:
            pos = self.pos
        if self.max_size is None:
            with self._sem:
                if self.max_size is None:
                    if _distortion and use_cython:
                        self.bin_size = _distortion.calc_size(self.pos, self._shape_out, self.mask, (self.offset1, self.offset2))
                    else:
                        mask = self.mask
                        pos0min = (numpy.floor(pos[:, :, :, 0].min(axis=-1) - self.offset1).astype(numpy.int32)).clip(0, self._shape_out[0])
                        pos1min = (numpy.floor(pos[:, :, :, 1].min(axis=-1) - self.offset2).astype(numpy.int32)).clip(0, self._shape_out[1])
                        pos0max = (numpy.ceil(pos[:, :, :, 0].max(axis=-1) - self.offset1 + 1).astype(numpy.int32)).clip(0, self._shape_out[0])
                        pos1max = (numpy.ceil(pos[:, :, :, 1].max(axis=-1) - self.offset2 + 1).astype(numpy.int32)).clip(0, self._shape_out[1])
                        self.bin_size = numpy.zeros(self._shape_out, dtype=numpy.int32)
                        for i in range(self.shape_in[0]):
                            for j in range(self.shape_in[1]):
                                if (mask is not None) and mask[i, j]:
                                    continue
                                self.bin_size[pos0min[i, j]:pos0max[i, j], pos1min[i, j]:pos1max[i, j]] += 1
                    self.max_size = self.bin_size.max()
        return self.bin_size

    def calc_init(self):
        """Initialize all arrays
        """
        self.calc_pos()
        self.calc_size()
        self.calc_LUT()
        if ocl and self.device is not None:
            if "lower" in dir(self.device):
                self.device = self.device.lower()
                if self.method == "lut":
                    self.integrator = ocl_azim_lut.OCL_LUT_Integrator(self.lut,
                                                                      self._shape_out[0] * self._shape_out[1],
                                                                      devicetype=self.device)
                else:
                    self.integrator = ocl_azim_csr.OCL_CSR_Integrator(self.lut,
                                                                      self._shape_out[0] * self._shape_out[1],
                                                                      devicetype=self.device,
                                                                      block_size=self.workgroup)
            else:
                if self.method == "lut":
                    self.integrator = ocl_azim_lut.OCL_LUT_Integrator(self.lut,
                                                                      self._shape_out[0] * self._shape_out[1],
                                                                      platformid=self.device[0],
                                                                      deviceid=self.device[1])
                else:
                    self.integrator = ocl_azim_csr.OCL_CSR_Integrator(self.lut,
                                                                      self._shape_out[0] * self._shape_out[1],
                                                                      platformid=self.device[0], deviceid=self.device[1],
                                                                      block_size=self.workgroup)

    def calc_LUT(self, use_common=True):
        """Calculate the Look-up table

        :return: look up table either in CSR or LUT format depending on serl.method
        """
        if self.pos is None:
            self.calc_pos()

        if self.max_size is None and not use_common:
            self.calc_size()
        if self.lut is None:
            with self._sem:
                if self.lut is None:
                    mask = self.mask
                    if _distortion:
                        if use_common:
                            self.lut = _distortion.calc_sparse(self.pos, self._shape_out, max_pixel_size=(self.delta1, self.delta2), format=self.method)
                        else:
                            if self.method == "lut":
                                self.lut = _distortion.calc_LUT(self.pos, self._shape_out, self.bin_size, max_pixel_size=(self.delta1, self.delta2))
                            else:
                                self.lut = _distortion.calc_CSR(self.pos, self._shape_out, self.bin_size, max_pixel_size=(self.delta1, self.delta2))
                    else:
                        lut = numpy.recarray(shape=(self._shape_out[0], self._shape_out[1], self.max_size), dtype=[("idx", numpy.uint32), ("coef", numpy.float32)])
                        lut[:, :, :].idx = 0
                        lut[:, :, :].coef = 0.0
                        outMax = numpy.zeros(self._shape_out, dtype=numpy.uint32)
                        idx = 0
                        buffer_ = numpy.empty((self.delta1, self.delta2))
                        quad = Quad(buffer_)
                        for i in range(self._shape_out[0]):
                            for j in range(self._shape_out[1]):
                                if (mask is not None) and mask[i, j]:
                                    continue
                                # i,j, idx are indexes of the raw image uncorrected
                                quad.reinit(*list(self.pos[i, j, :, :].ravel()))
                                # print(self.pos[i, j, 0, :], self.pos[i, j, 1, :], self.pos[i, j, 2, :], self.pos[i, j, 3, :]
                                try:
                                    quad.populate_box()
                                except Exception as error:
                                    print("error in quad.populate_box of pixel %i, %i: %s" % (i, j, error))
                                    print("calc_area_vectorial", quad.calc_area_vectorial())
                                    print(self.pos[i, j, 0, :], self.pos[i, j, 1, :], self.pos[i, j, 2, :], self.pos[i, j, 3, :])
                                    print(quad)
                                    raise
                #                box = quad.get_box()
                                for ms in range(quad.get_box_size0()):
                                    ml = ms + quad.get_offset0()
                                    if ml < 0 or ml >= self._shape_out[0]:
                                        continue
                                    for ns in range(quad.get_box_size1()):
                                        # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                                        nl = ns + quad.get_offset1()
                                        if nl < 0 or nl >= self._shape_out[1]:
                                            continue
                                        val = quad.get_box(ms, ns)
                                        if val <= 0:
                                            continue
                                        k = outMax[ml, nl]
                                        lut[ml, nl, k].idx = idx
                                        lut[ml, nl, k].coef = val
                                        outMax[ml, nl] = k + 1
                                idx += 1
                        lut.shape = (self._shape_out[0] * self._shape_out[1]), self.max_size
                        self.lut = lut
        return self.lut

    def correct(self, image, dummy=None, delta_dummy=None):
        """
        Correct an image based on the look-up table calculated ...

        :param image: 2D-array with the image
        :param dummy: value suggested for bad pixels
        :param delta_dummy: precision of the dummy value
        :return: corrected 2D image
        """
        if image.ndim == 2:
            if _distortion:
                image = _distortion.resize_image_2D(image, self.shape_in)
            else:
                logger.error("The image shape (%s) is not the same as the detector (%s). Adapting shape ...", image.shape, self.shape_in)
                new_img = numpy.zeros(self.shape_in, dtype=image.dtype)
                common_shape = [min(i, j) for i, j in zip(image.shape, self.shape_in)]
                new_img[:common_shape[0], :common_shape[1]] = image[:common_shape[0], :common_shape[1]]
                image = new_img
        else:  # assume 2d+nchanel
            if _distortion:
                image = _distortion.resize_image_3D(image, self.shape_in)
            else:
                assert image.ndim == 3, "image is 3D"
                shape_in0, shape_in1 = self.shape_in
                shape_img0, shape_img1, nchan = image.shape
                if not ((shape_img0 == shape_in0) and (shape_img1 == shape_in1)):
                    new_image = numpy.zeros((shape_in0, shape_in1, nchan), dtype=numpy.float32)
                    if shape_img0 < shape_in0:
                        if shape_img1 < shape_in1:
                            new_image[:shape_img0, :shape_img1, :] = image
                        else:
                            new_image[:shape_img0, :, :] = image[:, :shape_in1, :]
                    else:
                        if shape_img1 < shape_in1:
                            new_image[:, :shape_img1, :] = image[:shape_in0, :, :]
                        else:
                            new_image[:, :, :] = image[:shape_in0, :shape_in1, :]
                    logger.warning("Patching image of shape %ix%i on expected size of %ix%i",
                                   shape_img1, shape_img0, shape_in1, shape_in0)
                image = new_image
        if self.device:
            if self.integrator is None:
                self.calc_init()
            out = self.integrator.integrate(image)[1]
        else:
            if self.lut is None:
                self.calc_LUT()
            if _distortion is not None:
                out = _distortion.correct(image, self.shape_in, self._shape_out, self.lut,
                                          dummy=dummy or self.empty, delta_dummy=delta_dummy)
            else:
                if self.method == "lut":
                    big = image.ravel().take(self.lut.idx) * self.lut.coef
                    out = big.sum(axis=-1)
                elif self.method == "csr":
                    big = self.lut[0] * image.ravel().take(self.lut[1])
                    indptr = self.lut[2]
                    out = numpy.zeros(indptr.size - 1)
                    for i in range(indptr.size - 1):
                        out[i] = big[indptr[i]:indptr[i + 1]].sum()
        try:
            if image.ndim == 2:
                out.shape = self._shape_out
            else:
                for ds in out:
                    if ds.ndim == 2:
                        ds.shape = self._shape_out
                    else:
                        ds.shape = self._shape_out + ds.shape[2:]

        except ValueError as _err:
            logger.error("Requested in_shape=%s out_shape=%s and ", self.shape_in, self.shape_out)
            raise
        return out

    def uncorrect(self, image, use_cython=False):
        """
        Take an image which has been corrected and transform it into it's raw (with loss of information)

        :param image: 2D-array with the image
        :return: uncorrected 2D image

        Nota: to retrieve the input mask on can do:

        >>> msk =  dis.uncorrect(numpy.ones(dis._shape_out)) <= 0
        """
        assert image.shape == self._shape_out
        if self.lut is None:
            self.calc_LUT()
        if (linalg is not None) and (use_cython is False):
            if self.method == "lut":
                csr = csr_matrix(sparse_utils.LUT_to_CSR(self.lut))
            else:
                csr = csr_matrix(self.lut)
            res = linalg.lsmr(csr, image.ravel())
            out = res[0].reshape(self.shape_in)
        else:  # This is deprecated and does not work with resise=True
            if self.method == "lut":
                if _distortion is not None:
                    out, _mask = _distortion.uncorrect_LUT(image, self.shape_in, self.lut)
                else:
                    out = numpy.zeros(self.shape_in, dtype=numpy.float32)
                    lout = out.ravel()
                    lin = image.ravel()
                    tot = self.lut.coef.sum(axis=-1)
                    for idx in range(self.lut.shape[0]):
                        t = tot[idx]
                        if t <= 0:
                            continue
                        val = lin[idx] / t
                        lout[self.lut[idx].idx] += val * self.lut[idx].coef
            elif self.method == "csr":
                if _distortion is not None:
                    out, _mask = _distortion.uncorrect_CSR(image, self.shape_in, self.lut)
            else:
                raise NotImplementedError()
        return out


class Quad(object):
    """
    Quad modelisation.

    .. image:: ../img/quad_model.svg
        :alt: Modelization of the quad
    """
    def __init__(self, buffer):
        self.box = buffer
        self.A0 = self.A1 = None
        self.B0 = self.B1 = None
        self.C0 = self.C1 = None
        self.D0 = self.D1 = None
        self.offset0 = self.offset1 = None
        self.box_size0 = self.box_size1 = None
        self.pAB = self.pBC = self.pCD = self.pDA = None
        self.cAB = self.cBC = self.cCD = self.cDA = None
        self.area = None

    def get_idx(self, i, j):
        pass

    def get_box(self, i, j):
        return self.box[i, j]

    def get_offset0(self):
        return self.offset0

    def get_offset1(self):
        return self.offset1

    def get_box_size0(self):
        return self.box_size0

    def get_box_size1(self):
        return self.box_size1

    def reinit(self, A0, A1, B0, B1, C0, C1, D0, D1):
        self.box[:, :] = 0.0
        self.A0 = A0
        self.A1 = A1
        self.B0 = B0
        self.B1 = B1
        self.C0 = C0
        self.C1 = C1
        self.D0 = D0
        self.D1 = D1
        self.offset0 = int(floor(min(self.A0, self.B0, self.C0, self.D0)))
        self.offset1 = int(floor(min(self.A1, self.B1, self.C1, self.D1)))
        self.box_size0 = int(ceil(max(self.A0, self.B0, self.C0, self.D0))) - self.offset0
        self.box_size1 = int(ceil(max(self.A1, self.B1, self.C1, self.D1))) - self.offset1
        self.A0 -= self.offset0
        self.A1 -= self.offset1
        self.B0 -= self.offset0
        self.B1 -= self.offset1
        self.C0 -= self.offset0
        self.C1 -= self.offset1
        self.D0 -= self.offset0
        self.D1 -= self.offset1
        self.pAB = self.pBC = self.pCD = self.pDA = None
        self.cAB = self.cBC = self.cCD = self.cDA = None
        self.area = None

    def __repr__(self):
        return os.linesep.join(["offset %i,%i size %i, %i" % (self.offset0, self.offset1, self.box_size0, self.box_size1), "box: %s" % self.box[:self.box_size0, :self.box_size1]])

    def init_slope(self):
        if self.pAB is None:
            if self.B0 == self.A0:
                self.pAB = numpy.inf
            else:
                self.pAB = (self.B1 - self.A1) / (self.B0 - self.A0)
            if self.C0 == self.B0:
                self.pBC = numpy.inf
            else:
                self.pBC = (self.C1 - self.B1) / (self.C0 - self.B0)
            if self.D0 == self.C0:
                self.pCD = numpy.inf
            else:
                self.pCD = (self.D1 - self.C1) / (self.D0 - self.C0)
            if self.A0 == self.D0:
                self.pDA = numpy.inf
            else:
                self.pDA = (self.A1 - self.D1) / (self.A0 - self.D0)
            self.cAB = self.A1 - self.pAB * self.A0
            self.cBC = self.B1 - self.pBC * self.B0
            self.cCD = self.C1 - self.pCD * self.C0
            self.cDA = self.D1 - self.pDA * self.D0

    def calc_area_AB(self, I1, I2):
        if numpy.isfinite(self.pAB):
            return 0.5 * (I2 - I1) * (self.pAB * (I2 + I1) + 2 * self.cAB)
        else:
            return 0

    def calc_area_BC(self, J1, J2):
        if numpy.isfinite(self.pBC):
            return 0.5 * (J2 - J1) * (self.pBC * (J1 + J2) + 2 * self.cBC)
        else:
            return 0

    def calc_area_CD(self, K1, K2):
        if numpy.isfinite(self.pCD):
            return 0.5 * (K2 - K1) * (self.pCD * (K2 + K1) + 2 * self.cCD)
        else:
            return 0

    def calc_area_DA(self, L1, L2):

        if numpy.isfinite(self.pDA):
            return 0.5 * (L2 - L1) * (self.pDA * (L1 + L2) + 2 * self.cDA)
        else:
            return 0

    def calc_area_old(self):
        if self.area is None:
            if self.pAB is None:
                self.init_slope()
            self.area = -(self.calc_area_AB(self.A0, self.B0) +
                          self.calc_area_BC(self.B0, self.C0) +
                          self.calc_area_CD(self.C0, self.D0) +
                          self.calc_area_DA(self.D0, self.A0))
        return self.area

    def calc_area_vectorial(self):
        if self.area is None:
            self.area = numpy.cross([self.C0 - self.A0, self.C1 - self.A1], [self.D0 - self.B0, self.D1 - self.B1]) / 2.0
        return self.area
    calc_area = calc_area_vectorial

    def populate_box(self):
        if self.pAB is None:
            self.init_slope()
        self.integrateAB(self.B0, self.A0, self.calc_area_AB)
        self.integrateAB(self.A0, self.D0, self.calc_area_DA)
        self.integrateAB(self.D0, self.C0, self.calc_area_CD)
        self.integrateAB(self.C0, self.B0, self.calc_area_BC)
        if (self.box / self.calc_area()).min() < 0:
            print(self.box)
            self.box[:, :] = 0
            print("AB")
            self.integrateAB(self.B0, self.A0, self.calc_area_AB)
            print(self.box)
            self.box[:, :] = 0
            print("DA")
            self.integrateAB(self.A0, self.D0, self.calc_area_DA)
            print(self.box)
            self.box[:, :] = 0
            print("CD")
            self.integrateAB(self.D0, self.C0, self.calc_area_CD)
            print(self.box)
            self.box[:, :] = 0
            print("BC")
            self.integrateAB(self.C0, self.B0, self.calc_area_BC)
            print(self.box)
            print(self)
            raise RuntimeError()
        self.box /= self.calc_area_vectorial()

    def integrateAB(self, start, stop, calc_area):
        h = 0
#        print(start, stop, calc_area(start, stop)
        if start < stop:  # positive contribution
            P = ceil(start)
            dP = P - start
#            print("Integrate", start, P, stop, calc_area(start, stop)
            if P > stop:  # start and stop are in the same unit
                A = calc_area(start, stop)
                if A != 0:
                    AA = abs(A)
                    sign = A / AA
                    dA = (stop - start)  # always positive
#                    print(AA, sign, dA
                    h = 0
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        self.box[int(floor(start)), h] += sign * dA
                        AA -= dA
                        h += 1
            else:
                if dP > 0:
                    A = calc_area(start, P)
                    if A != 0:
                        AA = abs(A)
                        sign = A / AA
                        h = 0
                        dA = dP
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[int(floor(P)) - 1, h] += sign * dA
                            AA -= dA
                            h += 1
                # subsection P1->Pn
                for i in range(int(floor(P)), int(floor(stop))):
                    A = calc_area(i, i + 1)
                    if A != 0:
                        AA = abs(A)
                        sign = A / AA

                        h = 0
                        dA = 1.0
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[i, h] += sign * dA
                            AA -= dA
                            h += 1
                # Section Pn->B
                P = floor(stop)
                dP = stop - P
                if dP > 0:
                    A = calc_area(P, stop)
                    if A != 0:
                        AA = abs(A)
                        sign = A / AA
                        h = 0
                        dA = abs(dP)
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[int(floor(P)), h] += sign * dA
                            AA -= dA
                            h += 1
        elif start > stop:  # negative contribution. Nota is start=stop: no contribution
            P = floor(start)
            if stop > P:  # start and stop are in the same unit
                A = calc_area(start, stop)
                if A != 0:
                    AA = abs(A)
                    sign = A / AA
                    dA = (start - stop)  # always positive
                    h = 0
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        self.box[int(floor(start)), h] += sign * dA
                        AA -= dA
                        h += 1
            else:
                dP = P - start
                if dP < 0:
                    A = calc_area(start, P)
                    if A != 0:
                        AA = abs(A)
                        sign = A / AA
                        h = 0
                        dA = abs(dP)
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[int(floor(P)), h] += sign * dA
                            AA -= dA
                            h += 1
                # subsection P1->Pn
                for i in range(int(start), int(ceil(stop)), -1):
                    A = calc_area(i, i - 1)
                    if A != 0:
                        AA = abs(A)
                        sign = A / AA
                        h = 0
                        dA = 1
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[i - 1, h] += sign * dA
                            AA -= dA
                            h += 1
                # Section Pn->B
                P = ceil(stop)
                dP = stop - P
                if dP < 0:
                    A = calc_area(P, stop)
                    if A != 0:
                        AA = abs(A)
                        sign = A / AA
                        h = 0
                        dA = abs(dP)
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[int(floor(stop)), h] += sign * dA
                            AA -= dA
                            h += 1

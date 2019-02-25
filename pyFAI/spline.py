# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Written 2009-12-22 by Jérôme Kieffer
# Copyright (C) 2009-2016  European Synchrotron Radiation Facility
#                          Grenoble, France
#
#    Principal authors: Jérôme Kieffer  (jerome.kieffer@esrf.fr)
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

"""This is piece of software aims at manipulating spline files
describing for geometric corrections of the 2D detectors using cubic-spline.

Mainly used at ESRF with FReLoN CCD camera.
"""

from __future__ import print_function, division

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@esrf.eu"
__license__ = "MIT"
__date__ = "25/02/2019"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import os
import time
import numpy
import logging
import scipy.optimize
import scipy.interpolate

logger = logging.getLogger(__name__)

try:
    # multithreaded version in Cython: about 2x faster on large array evaluation
    from .ext import _bispev as fitpack
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    from scipy.interpolate import fitpack


class Spline(object):
    """
    This class is a python representation of the spline file

    Those file represent cubic splines for 2D detector distortions and
    makes heavy use of fitpack (dierckx in netlib) --- A Python-C
    wrapper to FITPACK (by P. Dierckx). FITPACK is a collection of
    FORTRAN programs for curve and surface fitting with splines and
    tensor product splines.  See
    _http://www.cs.kuleuven.ac.be/cwis/research/nalag/research/topics/fitpack.html
    or _http://www.netlib.org/dierckx/index.html
    """

    def __init__(self, filename=None):
        """
        This is the constructor of the Spline class.

        :param filename: name of the ascii file containing the spline
        :type filename: str
        """
        self.splineOrder = 3  # This is the default, so cubic splines
        self.lenStrFloat = 14  # by default one float is 14 char in ascii
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.xDispArray = None
        self.yDispArray = None
        self.xSplineKnotsX = []
        self.xSplineKnotsY = []
        self.xSplineCoeff = []
        self.ySplineKnotsX = []
        self.ySplineKnotsY = []
        self.ySplineCoeff = []
        self.pixelSize = None  # 2-tuple of float
        self.grid = None
        self.filename = None  # string
        if filename is not None:
            self.read(filename)

    def __repr__(self):
        lst = ["Array size: x= %s - %s\ty= %s - %s" %
               (self.xmin, self.xmax, self.ymin, self.ymax)]
        lst.append("Pixel size = %s microns, Grid spacing = %s" %
                   (self.pixelSize, self.grid))
        lst.append("X-Displacement spline %i X_knots, %i Y_knots and %i coef: "
                   "should be (X_knot-1-X_order)*(Y_knot-1-Y_order)" % (len(self.xSplineKnotsX),
                                                                        len(self.xSplineKnotsY),
                                                                        len(self.xSplineCoeff)))
        lst.append("Y-Displacement spline %i X_knots, %i Y_knots and %i coef: "
                   "should be (X_knot-1-X_order)*(Y_knot-1-Y_order)" % (len(self.ySplineKnotsX),
                                                                        len(self.ySplineKnotsY),
                                                                        len(self.ySplineCoeff)))
        return os.linesep.join(lst)

    def __copy__(self):
        """:return: Shallow copy of the spline"""
        unmutable = "splineOrder", "lenStrFloat", "xmin", "ymin", "xmax", "ymax", "filename", "pixelSize", "grid"
        arrays = "xDispArray", "yDispArray"
        lists = "xSplineKnotsX", "xSplineKnotsY", "xSplineCoeff", "ySplineKnotsX", "ySplineKnotsY", "ySplineCoeff"
        new = self.__class__()
        for key in unmutable + arrays + lists:
            new.__setattr__(key, self.__getattribute__(key))
        return new

    def __deepcopy__(self, memo=None):
        """:return: deep copy of the spline"""
        unmutable = "splineOrder", "lenStrFloat", "xmin", "ymin", "xmax", "ymax", "filename", "pixelSize", "grid"
        arrays = "xDispArray", "yDispArray"
        lists = "xSplineKnotsX", "xSplineKnotsY", "xSplineCoeff", "ySplineKnotsX", "ySplineKnotsY", "ySplineCoeff"

        if memo is None:
            memo = {}
        new = self.__class__()
        memo[id(self)] = new
        for key in unmutable:
            old_value = self.__getattribute__(key)
            memo[id(old_value)] = old_value
            new.__setattr__(key, old_value)
        for key in arrays:
            old_value = self.__getattribute__(key)
            if (old_value is None) or (old_value is False):
                new_value = old_value
            elif "copy" in dir(old_value):
                new_value = old_value.copy()
            else:
                new_value = 1 * old_value
            memo[id(old_value)] = new_value
            new.__setattr__(key, new_value)
        for key in lists:
            old_value = self.__getattribute__(key)
            new_value = old_value[:]
            memo[id(old_value)] = new_value
            new.__setattr__(key, new_value)
        return new

    def zeros(self, xmin=0.0, ymin=0.0, xmax=2048.0, ymax=2048.0,
              pixSize=None):
        """
        Defines a spline file with no ( zero ) displacement.

        :param xmin: minimum coordinate in x, usually zero
        :type xmin: float
        :param xmax: maximum coordinate in x (+1) usually 2048
        :type xmax: float
        :param ymin: minimum coordinate in y, usually zero
        :type ymin: float
        :param ymax: maximum coordinate y (+1) usually 2048
        :type ymax: float
        :param pixSize: size of the pixel
        :type pixSize: float
        """
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.xDispArray = numpy.zeros((int(xmax - xmin + 1),
                                       int(ymax - ymin + 1)))
        self.yDispArray = numpy.zeros((int(xmax - xmin + 1),
                                       int(ymax - ymin + 1)))
        if pixSize:
            self.pixelSize = pixSize

    def zeros_like(self, other):
        """
        Defines a spline file with no ( zero ) displacement with the
        same shape as the other one given.

        :param other: another Spline instance
        :type other: Spline instance
        """
        self.zeros(self, other.xmin, other.ymin, other.xmax, other.ymax)

    def read(self, filename):
        """
        read an ascii spline file from file

        :param filename: file containing the cubic spline distortion file
        :type filename: str
        """
        if not os.path.isfile(filename):
            raise IOError("Spline File does not exist %s" % filename)
        self.filename = filename
        with open(filename) as opened_file:
            stringSpline = [i.rstrip() for i in opened_file]
        try:
            indexLine = 0
            for oneLine in stringSpline:
                stripedLine = oneLine.strip().upper()
                if stripedLine == "VALID REGION":
                    data = stringSpline[indexLine + 1]
                    self.xmin = float(data[self.lenStrFloat * 0:self.lenStrFloat * 1])
                    self.ymin = float(data[self.lenStrFloat * 1:self.lenStrFloat * 2])
                    self.xmax = float(data[self.lenStrFloat * 2:self.lenStrFloat * 3])
                    self.ymax = float(data[self.lenStrFloat * 3:self.lenStrFloat * 4])
                elif stripedLine == "GRID SPACING, X-PIXEL SIZE, Y-PIXEL SIZE":
                    data = stringSpline[indexLine + 1]
                    self.grid = float(data[:self.lenStrFloat])
                    self.pixelSize = \
                        (float(data[self.lenStrFloat:self.lenStrFloat * 2]),
                         float(data[self.lenStrFloat * 2:self.lenStrFloat * 3]))
                elif stripedLine == "X-DISTORTION":
                    data = stringSpline[indexLine + 1]
                    [splineKnotsXLen, splineKnotsYLen] = \
                        [int(i) for i in data.split()]
                    databloc = []
                    for line in stringSpline[indexLine + 2:]:
                        if len(line) > 0:
                            for i in range(len(line) // self.lenStrFloat):
                                databloc.append(float(line[i * self.lenStrFloat: (i + 1) * self.lenStrFloat]))
                        else:
                            break
                    self.xSplineKnotsX = numpy.array(databloc[:splineKnotsXLen], dtype=numpy.float32)
                    self.xSplineKnotsY = numpy.array(databloc[splineKnotsXLen:splineKnotsXLen + splineKnotsYLen], dtype=numpy.float32)
                    self.xSplineCoeff = numpy.array(databloc[splineKnotsXLen + splineKnotsYLen:], dtype=numpy.float32)
                elif stripedLine == "Y-DISTORTION":
                    data = stringSpline[indexLine + 1]
                    [splineKnotsXLen, splineKnotsYLen] = [int(i) for i in data.split()]
                    databloc = []
                    for line in stringSpline[indexLine + 2:]:
                        if len(line) > 0:
                            for i in range(len(line) // self.lenStrFloat):
                                databloc.append(float(line[i * self.lenStrFloat:(i + 1) * self.lenStrFloat]))
                        else:
                            break
                    self.ySplineKnotsX = numpy.array(databloc[:splineKnotsXLen], dtype=numpy.float32)
                    self.ySplineKnotsY = numpy.array(databloc[splineKnotsXLen:splineKnotsXLen + splineKnotsYLen], dtype=numpy.float32)
                    self.ySplineCoeff = numpy.array(databloc[splineKnotsXLen + splineKnotsYLen:], dtype=numpy.float32)
                # Keep this at the end
                indexLine += 1
        except Exception:
            logger.error("Error while reading file", exc_info=True)
            raise IOError("Spline File parsing error: %s" % (filename))

    def comparison(self, ref, verbose=False):
        """
        Compares the current spline distortion with a reference

        :param Spline ref: another spline file
        :param bool verbose: print or not pylab plots
        :return: True or False depending if the splines are the same or not
        :rtype: bool
        """
        self.spline2array()
        ref.spline2array()
        deltax = (self.xDispArray - ref.xDispArray)
        deltay = (self.yDispArray - ref.yDispArray)
        histX = numpy.histogram(deltax.reshape(deltax.size), bins=100)
        histY = numpy.histogram(deltay.reshape(deltay.size), bins=100)
        histXdr = (histX[1][1:] + histX[1][:-1]) / 2.0
        histYdr = (histY[1][1:] + histY[1][:-1]) / 2.0
        histXmax = histXdr[histX[0].argmax()]
        histYmax = histYdr[histY[0].argmax()]
        maxErrX = abs(deltax).max()
        maxErrY = abs(deltay).max()
        curvX = scipy.interpolate.interp1d(histXdr, histX[0] - histX[0].max() / 2.0)
        curvY = scipy.interpolate.interp1d(histYdr, histY[0] - histY[0].max() / 2.0)
        fFWHM_X = scipy.optimize.bisect(curvX, histXmax, histXdr[-1]) - scipy.optimize.bisect(curvX, histXdr[0], histXmax)
        fFWHM_Y = scipy.optimize.bisect(curvY, histYmax, histYdr[-1]) - scipy.optimize.bisect(curvY, histYdr[0], histYmax)
        logger.info("Analysis of the difference between two splines")
        logger.info("Maximum error in X= %.3f pixels,\t in Y= %.3f pixels.", maxErrX, maxErrY)
        logger.info("Maximum of histogram in X= %.3f pixels,\t in Y= %.3f pixels.", histXmax, histYmax)
        logger.info("Mean of histogram in X= %.3f pixels,\t in Y= %.3f pixels.", deltax.mean(), deltay.mean())
        logger.info("FWHM in X= %.3f pixels,\t in Y= %.3f pixels.", fFWHM_X, fFWHM_Y)

        if verbose:
            import pylab
            pylab.plot(histXdr, histX[0], label="error in X")
            pylab.plot(histYdr, histY[0], label="error in Y")
            pylab.legend()
            pylab.show()
        return ((fFWHM_X < 0.05) and (fFWHM_Y < 0.05) and
                (maxErrX < 0.5) and (maxErrY < 0.5) and
                (deltax.mean() < 0.01) and(deltay.mean() < 0.01) and
                (histXmax < 0.01) and (histYmax < 0.01))

    def spline2array(self, timing=False):
        """
        Calculates the displacement matrix using fitpack
        bisplev(x, y, tck, dx = 0, dy = 0)

        :param timing: profile the calculation or not
        :type timing: bool

        :return: xDispArray, yDispArray
        :rtype: 2-tuple of ndarray

        Evaluate a bivariate B-spline and its derivatives. Return a
        rank-2 array of spline function values (or spline derivative
        values) at points given by the cross-product of the rank-1
        arrays x and y. In special cases, return an array or just a
        float if either x or y or both are floats.
        """
        if self.xDispArray is None:
            x_1d_array = numpy.arange(self.xmin, self.xmax + 1)
            y_1d_array = numpy.arange(self.ymin, self.ymax + 1)
            startTime = time.time()
            self.xDispArray = fitpack.bisplev(x_1d_array, y_1d_array,
                                              [self.xSplineKnotsX,
                                               self.xSplineKnotsY,
                                               self.xSplineCoeff,
                                               self.splineOrder,
                                               self.splineOrder],
                                              dx=0, dy=0).transpose()
            intermediateTime = time.time()
            self.yDispArray = fitpack.bisplev(x_1d_array, y_1d_array,
                                              [self.ySplineKnotsX,
                                               self.ySplineKnotsY,
                                               self.ySplineCoeff,
                                               self.splineOrder,
                                               self.splineOrder],
                                              dx=0, dy=0).transpose()
            if timing:
                logger.info("Timing for: X-Displacement spline evaluation: %.3f sec,"
                            " Y-Displacement Spline evaluation:  %.3f sec." %
                            ((intermediateTime - startTime),
                             (time.time() - intermediateTime)))
        return self.xDispArray, self.yDispArray

    def splineFuncX(self, x, y, list_of_points=False):
        """
        Calculates the displacement matrix using fitpack for the X
        direction on the given grid.

        :param x: points of the grid in the x direction
        :type x: ndarray
        :param y: points of the grid  in the y direction
        :type y: ndarray
        :param list_of_points: if true, consider the zip(x,y) instead of the of the square array
        :return: displacement matrix for the X direction
        :rtype: ndarray
        """
        if x.ndim == 2:
            if abs(x[1:, :] - x[:-1, :] - numpy.zeros((x.shape[0] - 1, x.shape[1]))).max() < 1e-6:
                x = x[0]
                y = y[:, 0]
            elif abs(x[:, 1:] - x[:, :-1] - numpy.zeros((x.shape[0], x.shape[1] - 1))).max() < 1e-6:
                x = x[:, 0]
                y = y[0]
        if list_of_points and x.ndim == 1 and len(x) == len(y):
            size = len(x)
            if size > 1:
                x_order = x.argsort()
                y_order = y.argsort()
                x = x[x_order]
                y = y[y_order]
                x_unordered = numpy.zeros(size, dtype=numpy.int32)
                y_unordered = numpy.zeros(size, dtype=numpy.int32)
                x_unordered[x_order] = numpy.arange(size)
                y_unordered[y_order] = numpy.arange(size)
        x_disp_array = fitpack.bisplev(x, y,
                                       [self.xSplineKnotsX,
                                        self.xSplineKnotsY,
                                        self.xSplineCoeff,
                                        self.splineOrder,
                                        self.splineOrder],
                                       dx=0, dy=0)
        if list_of_points and x.ndim == 1:
            if size > 1:
                return x_disp_array[x_unordered, y_unordered]
            else:
                return numpy.array([x_disp_array])
        else:
            return x_disp_array.T

    def splineFuncY(self, x, y, list_of_points=False):
        """
        calculates the displacement matrix using fitpack for the Y
        direction

        :param x: points in the x direction
        :type x: ndarray
        :param y: points in the y direction
        :type y: ndarray
        :param list_of_points: if true, consider the zip(x,y) instead of the of the square array
        :return: displacement matrix for the Y direction
        :rtype: ndarray
        """
        if x.ndim == 2:
            if abs(x[1:, :] - x[:-1, :] - numpy.zeros((x.shape[0] - 1, x.shape[1]))).max() < 1e-6:
                x = x[0]
                y = y[:, 0]
            elif abs(x[:, 1:] - x[:, :-1] - numpy.zeros((x.shape[0], x.shape[1] - 1))).max() < 1e-6:
                x = x[:, 0]
                y = y[0]

        if list_of_points and x.ndim == 1 and len(x) == len(y):
            size = len(x)
            if size > 1:
                x_order = x.argsort()
                y_order = y.argsort()
                x = x[x_order]
                y = y[y_order]
                x_unordered = numpy.zeros(size, dtype=numpy.int32)
                y_unordered = numpy.zeros(size, dtype=numpy.int32)
                x_unordered[x_order] = numpy.arange(size)
                y_unordered[y_order] = numpy.arange(size)

        y_disp_array = fitpack.bisplev(x, y,
                                       [self.ySplineKnotsX,
                                        self.ySplineKnotsY,
                                        self.ySplineCoeff,
                                        self.splineOrder,
                                        self.splineOrder],
                                       dx=0, dy=0)
        if list_of_points and x.ndim == 1:
            if size > 1:
                return y_disp_array[x_unordered, y_unordered]
            else:
                return numpy.array([y_disp_array])
        else:
            return y_disp_array.T

    def array2spline(self, smoothing=1000, timing=False):
        """
        Calculates the spline coefficients from the displacements
        matrix using fitpack.

        :param smoothing: the greater the smoothing, the fewer the number of knots remaining
        :type smoothing: float
        :param timing: print the profiling of the calculation
        :type timing: bool
        """
        self.xmin = 0.0
        self.ymin = 0.0
        self.xmax = self.xDispArray.shape[1] - 1.0
        self.ymax = self.yDispArray.shape[0] - 1.0

        if timing:
            startTime = time.time()

        xRectBivariateSpline = scipy.interpolate.fitpack2.RectBivariateSpline(
            numpy.arange(self.xmax + 1.0),
            numpy.arange(self.ymax + 1.0),
            self.xDispArray.transpose(),
            s=smoothing)

        if timing:
            intermediateTime = time.time()

        yRectBivariateSpline = scipy.interpolate.fitpack2.RectBivariateSpline(
            numpy.arange(self.xmax + 1.0),
            numpy.arange(self.ymax + 1.0),
            self.yDispArray.transpose(),
            s=smoothing)

        if timing:
            logger.info("X-Displ evaluation= %.3f sec, Y-Displ evaluation=  %.3f sec.",
                        intermediateTime - startTime, time.time() - intermediateTime)

        xknots = xRectBivariateSpline.get_knots()
        self.xSplineKnotsX = xknots[0]
        self.xSplineKnotsY = xknots[1]
        self.xSplineCoeff = xRectBivariateSpline.get_coeffs()
        yknots = yRectBivariateSpline.get_knots()
        self.ySplineKnotsX = yknots[0]
        self.ySplineKnotsY = yknots[1]
        self.ySplineCoeff = yRectBivariateSpline.get_coeffs()

        logger.debug("x-coefs len=%i %s",
                     len(self.xSplineCoeff),
                     self.xSplineCoeff)
        logger.debug("y-coefs len=%i %s",
                     len(self.ySplineCoeff),
                     yknots)
        logger.debug("x-knots x:%i y:%i",
                     len(self.xSplineKnotsX),
                     len(self.xSplineKnotsY))
        logger.debug("y-knots x:%i y:%i",
                     len(self.ySplineKnotsX),
                     len(self.ySplineKnotsY))

        logger.debug("Residual x=%s, y=%s",
                     xRectBivariateSpline.get_residual(),
                     yRectBivariateSpline.get_residual())

    def writeEDF(self, basename):
        """
        save the distortion matrices into a couple of files called
        basename-x.edf and basename-y.edf

        :param basename: base of the name used to save the data
        :type basename: str
        """
        try:
            from fabio.edfimage import edfimage
        except ImportError:
            logger.error("You will need the Fabio library available"
                         " from the Fable sourceforge")
            return
        self.spline2array()

        edfDispX = edfimage(data=self.xDispArray.astype("float32"), header={})
        edfDispY = edfimage(data=self.yDispArray.astype("float32"), header={})
        edfDispX.write(basename + "-x.edf", force_type="float32")
        edfDispY.write(basename + "-y.edf", force_type="float32")

    def write(self, filename):
        """
        save the cubic spline in an ascii file usable with Fit2D or
        SPD

        :param filename: name of the file containing the cubic spline distortion file
        :type filename: str
        """

        lst = ["SPATIAL DISTORTION SPLINE INTERPOLATION COEFFICIENTS",
               "",
               "  VALID REGION",
               "%14.7E%14.7E%14.7E%14.7E" % (self.xmin, self.ymin, self.xmax, self.ymax),
               "",
               "  GRID SPACING, X-PIXEL SIZE, Y-PIXEL SIZE",
               "%14.7E%14.7E%14.7E" % (self.grid, self.pixelSize[0], self.pixelSize[1]),
               "",
               "  X-DISTORTION",
               "%6i%6i" % (len(self.xSplineKnotsX), len(self.xSplineKnotsY))]
        txt = ""
        for i, val in enumerate(self.xSplineKnotsX):
            txt += "%14.7E" % val
            if i % 5 == 4:
                lst.append(txt)
                txt = ""
        if txt:
            lst.append(txt)
            txt = ""
        for i, val in enumerate(self.xSplineKnotsY):
            txt += "%14.7E" % val
            if i % 5 == 4:
                lst.append(txt)
                txt = ""
        if txt:
            lst.append(txt)
            txt = ""
        for i, val in enumerate(self.xSplineCoeff):
            txt += "%14.7E" % self.xSplineCoeff[i]
            if i % 5 == 4:
                lst.append(txt)
                txt = ""
        if txt:
            lst.append(txt)
            txt = ""
        lst.append("")
        lst.append("  Y-DISTORTION\n%6i%6i" % (len(self.ySplineKnotsX),
                                               len(self.ySplineKnotsY)))
        for i, val in enumerate(self.ySplineKnotsX):
            txt += "%14.7E" % val
            if i % 5 == 4:
                lst.append(txt)
                txt = ""
        if txt:
            lst.append(txt)
            txt = ""
        for i, val in enumerate(self.ySplineKnotsY):
            txt += "%14.7E" % val
            if i % 5 == 4:
                lst.append(txt)
                txt = ""
        if txt:
            lst.append(txt)
            txt = ""
        for i, val in enumerate(self.ySplineCoeff):
            txt += "%14.7E" % val
            if i % 5 == 4:
                lst.append(txt)
                txt = ""
        if txt:
            lst.append(txt)
            txt = ""
        lst.append("")
        with open(filename, "w") as fil:
            fil.write("\n".join(lst))

    def tilt(self, center=(0.0, 0.0), tiltAngle=0.0, tiltPlanRot=0.0,
             distanceSampleDetector=1.0, timing=False):
        """
        The tilt method apply a virtual tilt on the detector, the
        point of tilt is given by the center

        :param center: position of the point of tilt, this point will not be moved.
        :type center: 2-tuple of floats
        :param tiltAngle: the value of the tilt in degrees
        :type tiltAngle: float in the range [-90:+90] degrees
        :param tiltPlanRot: the rotation of the tilt plan with the Ox axis (0 deg for y axis invariant, 90 deg for x axis invariant)
        :type tiltPlanRot: Float in the range [-180:180]
        :param distanceSampleDetector: the distance from sample to detector in meter (along the beam, so distance from sample to center)
        :type distanceSampleDetector: float

        :return: tilted Spline instance
        :rtype: Spline
        """
        if self.xDispArray is None:
            if self.filename is None:
                self.zeros()
            else:
                self.read(self.filename)
        logger.info(u"center=%s, tilt=%s, tiltPlanRot=%s, distanceSampleDetector=%sm, pixelSize=%sµm", center, tiltAngle, tiltPlanRot, distanceSampleDetector, self.pixelSize)
        if timing:
            startTime = time.time()
        distance = 1.0e6 * distanceSampleDetector  # from meters to microns
        cosb = numpy.cos(numpy.radians(tiltPlanRot))
        sinb = numpy.sin(numpy.radians(tiltPlanRot))
        cosf = numpy.cos(numpy.radians(tiltAngle))
        sinf = numpy.sin(numpy.radians(tiltAngle))

        # x and y are tilted in C/Fortran representation
        def compute_x(_, j):
            return j - center[0] - 0.5

        def compute_y(i, _):
            return i - center[1] - 0.5

        iPos = numpy.fromfunction(compute_x,
                                  (int(self.ymax - self.ymin + 1),
                                   int(self.xmax - self.xmin + 1)))
        jPos = numpy.fromfunction(compute_y,
                                  (int(self.ymax - self.ymin + 1),
                                   int(self.xmax - self.xmin + 1)))

        xPos = (iPos + self.xDispArray) * self.pixelSize[0]
        yPos = (jPos + self.yDispArray) * self.pixelSize[1]

        tiltArrayX = distance * (xPos * (cosf * cosb * cosb + sinb * sinb) + yPos * (cosf * cosb * sinb - cosb * sinb)) / \
            (distance + xPos * sinf * cosb + yPos * sinf * sinb) / self.pixelSize[0] - iPos
        tiltArrayY = distance * (xPos * (cosf * sinb * cosb - cosb * sinb) + yPos * (cosf * sinb * sinb + cosb * cosb)) / \
            (distance + xPos * sinf * cosb + yPos * sinf * sinb) / self.pixelSize[1] - jPos
        tiltedSpline = Spline()
        tiltedSpline.pixelSize = self.pixelSize
        tiltedSpline.grid = self.grid
        tiltedSpline.xDispArray = tiltArrayX
        tiltedSpline.yDispArray = tiltArrayY
        # tiltedSpline.array2spline(smoothing=1e-6, timing=True)
        if timing:
            logger.info("Time for the generation of the distorted spline: %.3f sec", time.time() - startTime)
        return tiltedSpline

    def getDetectorSize(self):
        """Returns the size of the detector.

        :rtype: Tuple[int,int]
        :return: Size y then x
        """
        return int(self.ymax - self.ymin), int(self.xmax - self.xmin)

    def setPixelSize(self, pixelSize):
        """
        Sets the size of the pixel from a 2-tuple of floats expressed
        in meters.

        :param: pixel size in meter
        :type pixelSize: 2-tuple of float
        """
        if len(pixelSize) == 2:
            self.pixelSize = (pixelSize[0] * 1.0e6, pixelSize[1] * 1.0e6)

    def getPixelSize(self):
        """
        Return the size of the pixel from as a 2-tuple of floats expressed
        in meters.

        :return: the size of the pixel from a 2D detector
        :rtype: 2-tuple of floats expressed in meter.

        """
        return (self.pixelSize[0] * 1.0e-6, self.pixelSize[1] * 1.0e-6)

    def bin(self, binning=None):
        """
        Performs the binning of a spline (same camera with different binning)

        :param binning: binning factor as integer or 2-tuple of integers
        :type: int or (int, int)

        """
        if "__len__" in dir(binning):
            binX, binY = float(binning[0]), float(binning[1])
        else:
            binX = binY = float(binning)
        self.xSplineKnotsX /= binX
        self.xSplineKnotsY /= binY
        self.ySplineKnotsX /= binX
        self.ySplineKnotsY /= binY
        self.pixelSize = (binX * self.pixelSize[0], binY * self.pixelSize[1])
        self.xmax = self.xmax / binX
        self.ymax = self.ymax / binY
        self.xSplineCoeff /= binX
        self.ySplineCoeff /= binY
        self.xDispArray = None
        self.yDispArray = None

    def correct(self, pos):
        delta1 = fitpack.bisplev(pos[1], pos[0], [self.xSplineKnotsX,
                                                  self.xSplineKnotsY,
                                                  self.xSplineCoeff,
                                                  self.splineOrder,
                                                  self.splineOrder],
                                 dx=0, dy=0)

        delta0 = fitpack.bisplev(pos[1], pos[0], [self.ySplineKnotsX,
                                                  self.ySplineKnotsY,
                                                  self.ySplineCoeff,
                                                  self.splineOrder,
                                                  self.splineOrder],
                                 dx=0, dy=0)
        return delta0 + pos[0], delta1 + pos[1]

    def flipud(self, fit=True):
        """Flip the spline upside-down

        :param bool fit: set to False to disable fitting of the coef,
                    or provide a value for the smoothing factor
        :return: new spline object
        """
        self.spline2array()
        other = self.__class__()
        other.xmin = self.xmin
        other.ymin = self.ymin
        other.xmax = self.xmax
        other.ymax = self.ymax
        other.xDispArray = numpy.flipud(self.xDispArray)
        other.yDispArray = -numpy.flipud(self.yDispArray)
        other.pixelSize = self.pixelSize
        other.grid = self.grid
        if fit is not False:
            if fit is True:
                other.array2spline()
            else:
                other.array2spline(fit)
        return other

    def fliplr(self, fit=True):
        """Flip the spline horizontally

        :param bool fit: set to False to disable fitting of the coef,
            or provide a value for the smoothing factor
        :return: new spline object
        """
        self.spline2array()
        other = self.__class__()
        other.xmin = self.xmin
        other.ymin = self.ymin
        other.xmax = self.xmax
        other.ymax = self.ymax
        other.xDispArray = -numpy.fliplr(self.xDispArray)
        other.yDispArray = numpy.fliplr(self.yDispArray)
        other.pixelSize = self.pixelSize
        other.grid = self.grid
        if fit is not False:
            if fit is True:
                other.array2spline()
            else:
                other.array2spline(fit)
        return other

    def fliplrud(self, fit=True):
        """Flip the spline upside-down and horizontally

        :param bool fit: set to False to disable fitting of the coef,
            or provide a value for the smoothing factor
        :return: new spline object
        """
        self.spline2array()
        other = self.__class__()
        other.xmin = self.xmin
        other.ymin = self.ymin
        other.xmax = self.xmax
        other.ymax = self.ymax
        other.xDispArray = -numpy.flipud(numpy.fliplr(self.xDispArray))
        other.yDispArray = -numpy.flipud(numpy.fliplr(self.yDispArray))
        other.pixelSize = self.pixelSize
        other.grid = self.grid
        if fit is not False:
            if fit is True:
                other.array2spline()
            else:
                other.array2spline(fit)
        return other

#!/usr/bin/env python
# -*- coding: utf8 -*-
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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "23/08/2012"
__status__ = "beta"

import os, threading, logging
import numpy


#from utils import timeit
logger = logging.getLogger("pyFAI.refinment2D")
from pyFAI.geometry import Geometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from PyMca import SGModule

class Refinment2D(object):
    """
    refine the parameters from image itself ...
    """
    def __init__(self, img, ai=None):
        """
        @param: raw image we are working on
        @param: ai azimuhal integrator we are working on
        """
        self.img = img
        if ai is None:
            self.ai = AzimuthalIntegrator()
        else:
            self.ai = ai

    def get_shape(self):
        return self.img.shape
    shape = property(get_shape)

    def reconstruct(self, tth, I):
        """
        Reconstruct a perfect image according to 2th / I given in input
          
        @param tth: 2 theta array
        @param I: intensity array
        """
        return numpy.interp(self.ai.twoThetaArray(self.shape), tth , I)

    def diff_tth_X(self, dx=0.1):
        f = self.ai.getFit2D()
        fp = f.copy()
        fm = f.copy()
        fm["centerX"] -= dx / 2.0
        fp["centerX"] += dx / 2.0
        ap = AzimuthalIntegrator()
        am = AzimuthalIntegrator()
        ap.setFit2D(**fp)
        am.setFit2D(**fm)
        dtthX = (ap.twoThetaArray(self.shape) - am.twoThetaArray(self.shape)) / dx
        tth, I = self.ai.xrpd(self.img, max(self.shape))
        from PyMca import SGModule
        dI = SGModule.getSavitzkyGolay(I, npoints=5, degree=2, order=1) / (tth[1] - tth[0])
        dImg = self.reconstruct(tth, dI)
        return (dtthX * dImg).sum()

    def diff_tth_tilt(self, dx=0.1):
        f = self.ai.getFit2D()
        fp = f.copy()
        fm = f.copy()
        fm["tilt"] -= dx / 2.0
        fp["tilt"] += dx / 2.0
        ap = AzimuthalIntegrator()
        am = AzimuthalIntegrator()
        ap.setFit2D(**fp)
        am.setFit2D(**fm)
        dtthX = (ap.twoThetaArray(self.shape) - am.twoThetaArray(self.shape)) / dx
        tth, I = self.ai.xrpd(self.img, max(self.shape))
        from PyMca import SGModule
        dI = SGModule.getSavitzkyGolay(I, npoints=5, degree=2, order=1) / (tth[1] - tth[0])
        dImg = self.reconstruct(tth, dI)
        return (dtthX * dImg).sum()


    def diff_Fit2D(self, axis="all", dx=0.1):
        tth, I = self.ai.xrpd(self.img, max(self.shape))
        dI = SGModule.getSavitzkyGolay(I, npoints=5, degree=2, order=1) / (tth[1] - tth[0])
        dImg = self.reconstruct(tth, dI)
        f = self.ai.getFit2D()
        tth2d_ref = self.ai.twoThetaArray(self.shape)

        keys = ["centerX", "centerY", "tilt", "tiltPlanRotation"]
        if axis != "all":
            keys = [i for i in keys if i == axis]
        grad = {}
        for key in keys:
            fp = f.copy()
            fp[key] += dx
            ap = AzimuthalIntegrator()
            ap.setFit2D(**fp)
            dtth = (ap.twoThetaArray(self.shape) - self.ai.twoThetaArray(self.shape)) / dx
            grad[key] = (dtth * dImg).sum()
        if axis == "all":
            return grad
        else:
            return grad[axis]

    def scan_centerX(self, width=1.0, points=10):
        f = self.ai.getFit2D()
        out = []
        for x in numpy.linspace(f["centerX"] - width / 2.0, f["centerX"] + width / 2.0, points):
            ax = AzimuthalIntegrator()
            fx = f.copy()
            fx["centerX"] = x
            ax.setFit2D(**fx)
#            print ax
            ref = Refinment2D(self.img, ax)
            res = ref.diff_tth_X()
            print "x= %.3f mean= %e" % (x, res)
            out.append(res)
        return numpy.linspace(f["centerX"] - width / 2.0, f["centerX"] + width / 2.0, points), out

    def scan_tilt(self, width=1.0, points=10):
        f = self.ai.getFit2D()
        out = []
        for x in numpy.linspace(f["tilt"] - width / 2.0, f["tilt"] + width / 2.0, points):
            ax = AzimuthalIntegrator()
            fx = f.copy()
            fx["tilt"] = x
            ax.setFit2D(**fx)
#            print ax
            ref = Refinment2D(self.img, ax)
            res = ref.diff_tth_tilt()
            print "x= %.3f mean= %e" % (x, res)
            out.append(res)
        return numpy.linspace(f["tilt"] - width / 2.0, f["tilt"] + width / 2.0, points), out

    def scan_Fit2D(self, width=1.0, points=10, axis="tilt", dx=0.1):
        logger.info("Scanning along axis %s" % axis)
        f = self.ai.getFit2D()
        out = []
        meas_pts = numpy.linspace(f[axis] - width / 2.0, f[axis] + width / 2.0, points)
        for x in meas_pts:
            ax = AzimuthalIntegrator()
            fx = f.copy()
            fx[axis] = x
            ax.setFit2D(**fx)
            ref = Refinment2D(self.img, ax)
            res = ref.diff_Fit2D(axis=axis, dx=dx)
            print "x= %.3f mean= %e" % (x, res)
            out.append(res)
        return meas_pts, out


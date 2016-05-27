#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif/pyFAI
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
__date__ = "29/09/2014"
__status__ = "beta"

import logging
import numpy

logger = logging.getLogger("pyFAI.refinment2D")

#from utils import timeit
from .azimuthalIntegrator import AzimuthalIntegrator
from PyMca import SGModule


class Refinment2D(object):
    """
    refine the parameters from image itself ...
    (Jerome est-ce que tu peux elaborer un petit peu plus ???)
    """
    def __init__(self, img, ai=None):
        """
        @param img: raw image we are working on
        @type img: ndarray
        @param ai: azimuhal integrator we are working on
        @type ai: pyFAI.azimuthalIntegrator.AzimutalIntegrator
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
        Reconstruct a perfect image according to 2th / I given in
        input

        @param tth: 2 theta array
        @type tth: ndarray
        @param I: intensity array
        @type I: ndarray

        @return: a reconstructed image
        @rtype: ndarray
        """
        return numpy.interp(self.ai.twoThetaArray(self.shape), tth, I)

    def diff_tth_X(self, dx=0.1):
        """
        Jerome peux-tu décrire de quoi il retourne ???

        @param dx: ???
        @type: float ???

        @return: ???
        @rtype: ???
        """
        f = self.ai.getFit2D()
        fp = f.copy()
        fm = f.copy()
        fm["centerX"] -= dx / 2.0
        fp["centerX"] += dx / 2.0
        ap = AzimuthalIntegrator()
        am = AzimuthalIntegrator()
        ap.setFit2D(**fp)
        am.setFit2D(**fm)
        dtthX = (ap.twoThetaArray(self.shape) - am.twoThetaArray(self.shape))\
            / dx
        tth, I = self.ai.xrpd(self.img, max(self.shape))
        dI = SGModule.getSavitzkyGolay(I, npoints=5, degree=2, order=1)\
            / (tth[1] - tth[0])
        dImg = self.reconstruct(tth, dI)
        return (dtthX * dImg).sum()

    def diff_tth_tilt(self, dx=0.1):
        """
        idem ici ???

        @param dx: ???
        @type dx: float ???

        @return: ???
        @rtype: ???
        """
        f = self.ai.getFit2D()
        fp = f.copy()
        fm = f.copy()
        fm["tilt"] -= dx / 2.0
        fp["tilt"] += dx / 2.0
        ap = AzimuthalIntegrator()
        am = AzimuthalIntegrator()
        ap.setFit2D(**fp)
        am.setFit2D(**fm)
        dtthX = (ap.twoThetaArray(self.shape) - am.twoThetaArray(self.shape))\
            / dx
        tth, I = self.ai.xrpd(self.img, max(self.shape))
        dI = SGModule.getSavitzkyGolay(I, npoints=5, degree=2, order=1)\
            / (tth[1] - tth[0])
        dImg = self.reconstruct(tth, dI)
        return (dtthX * dImg).sum()

    def diff_Fit2D(self, axis="all", dx=0.1):
        """
        ???

        @param axis: ???
        @type axis: ???
        @param dx: ???
        @type dx: ???

        @return: ???
        @rtype: ???
        """
        tth, I = self.ai.xrpd(self.img, max(self.shape))
        dI = SGModule.getSavitzkyGolay(I, npoints=5, degree=2, order=1)\
            / (tth[1] - tth[0])
        dImg = self.reconstruct(tth, dI)
        f = self.ai.getFit2D()
        tth2d_ref = self.ai.twoThetaArray(self.shape)  # useless variable ???

        keys = ["centerX", "centerY", "tilt", "tiltPlanRotation"]
        if axis != "all":
            keys = [i for i in keys if i == axis]
        grad = {}
        for key in keys:
            fp = f.copy()
            fp[key] += dx
            ap = AzimuthalIntegrator()
            ap.setFit2D(**fp)
            dtth = (ap.twoThetaArray(self.shape)
                    - self.ai.twoThetaArray(self.shape)) / dx
            grad[key] = (dtth * dImg).sum()
        if axis == "all":
            return grad
        else:
            return grad[axis]

    def scan_centerX(self, width=1.0, points=10):
        """
        ???

        @param width: ???
        @type width: float ???
        @param points: ???
        @type points: int ???

        @return: ???
        @rtype: ???
        """
        f = self.ai.getFit2D()
        out = []
        for x in numpy.linspace(f["centerX"] - width / 2.0,
                                f["centerX"] + width / 2.0,
                                points):
            ax = AzimuthalIntegrator()
            fx = f.copy()
            fx["centerX"] = x
            ax.setFit2D(**fx)
#            print ax
            ref = Refinment2D(self.img, ax)
            res = ref.diff_tth_X()
            print("x= %.3f mean= %e" % (x, res))
            out.append(res)
        return numpy.linspace(f["centerX"] - width / 2.0,
                              f["centerX"] + width / 2.0,
                              points), out

    def scan_tilt(self, width=1.0, points=10):
        """
        ???

        @param width: ???
        @type width: float ???
        @param points: ???
        @type points: int ???

        @return: ???
        @rtype: ???
        """
        f = self.ai.getFit2D()
        out = []
        for x in numpy.linspace(f["tilt"] - width / 2.0,
                                f["tilt"] + width / 2.0,
                                points):
            ax = AzimuthalIntegrator()
            fx = f.copy()
            fx["tilt"] = x
            ax.setFit2D(**fx)
#            print ax
            ref = Refinment2D(self.img, ax)
            res = ref.diff_tth_tilt()
            print("x= %.3f mean= %e" % (x, res))
            out.append(res)
        return numpy.linspace(f["tilt"] - width / 2.0,
                              f["tilt"] + width / 2.0,
                              points), out

    def scan_Fit2D(self, width=1.0, points=10, axis="tilt", dx=0.1):
        """
        ???

        @param width: ???
        @type width: float ???
        @param points: ???
        @type points: int ???
        @param axis: ???
        @type axis: str ???
        @param dx: ???
        @type dx: float ???

        @return: ???
        @rtype: ???
        """
        logger.info("Scanning along axis %s" % axis)
        f = self.ai.getFit2D()
        out = []
        meas_pts = numpy.linspace(f[axis] - width / 2.0,
                                  f[axis] + width / 2.0,
                                  points)
        for x in meas_pts:
            ax = AzimuthalIntegrator()
            fx = f.copy()
            fx[axis] = x
            ax.setFit2D(**fx)
            ref = Refinment2D(self.img, ax)
            res = ref.diff_Fit2D(axis=axis, dx=dx)
            print("x= %.3f mean= %e" % (x, res))
            out.append(res)
        return meas_pts, out

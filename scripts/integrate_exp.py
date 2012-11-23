#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration 
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/05/2011"

import os, sys, gc, time
import numpy
from numpy import sin, cos, arccos, sqrt, floor, ceil, radians, degrees, pi
import fabio
import matplotlib
import pylab

#sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), "plugins"))
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

if __name__ == "__main__":
    paramFile = None
    processFile = []
    for param in sys.argv[1:]:
        if param.startswith("-p="):
            paramFile = param.split("=", 1)[1]
        elif os.path.isfile(param):
            processFile.append(param)
    if paramFile and processFile:
        integrator = AzimuthalIntegrator()
        integrator.setChiDiscAtZero()
        integrator.load(paramFile)
        print integrator
        for oneFile in processFile:
            sys.stdout.write("Integrating %s --> " % oneFile)
            outFile = os.path.splitext(oneFile)[0] + ".xy"
            azimFile = os.path.splitext(oneFile)[0] + ".azim"
            data = fabio.open(oneFile).data.astype("float32")
            t0 = time.time()
            tth, I = integrator.xrpd_halfSplitPixel(data=data, nbPt=1495, filename=outFile, correctSolidAngle=False)
            t1 = time.time()
#            integrator.xrpd2(data, 1000, 360, azimFile)
            print "%s\t 1D took  %.3fs, 2D took %.3fs" % (outFile, t1 - t0, time.time() - t1)
            print "raw int: %s ; integrated: %s " % (data.sum() / data.size, I.sum())

    else:
        print("Usage:")
        print(" $ python integrate.py -p=param.poni file.edf file2.edf file3.edf")


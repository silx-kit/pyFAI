#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/11/2011"
__doc__ = """ 
saxs_integrate is the Saxs script of pyFAI that allows data reduction for Small Angle Scattering.

Parameters:
 -p=param.poni         PyFAI parameter file 
 -w=9.31e-11           wavelength (in meter)
 -m=mask.edf           mask image
 -d=-2                 dummy value for dead pixels
 -dd=-1.1              delta dummy 

 -h                    print help and exit
 
    Usage:
python saxs_integrate.py -p=param.poni -w=0.154e-9 file.edf file2.edf file3.edf
"""#IGNORE:W0622
import os, sys, time
import fabio

#sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), "plugins"))
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator



if __name__ == "__main__":
    paramFile = None
    processFile = []
    wavelength = None
    dummy = None
    delta_dummy = None
    mask = None
    integrator = AzimuthalIntegrator()
    for param in sys.argv[1:]:
        if param.startswith("-p="):
            paramFile = param.split("=", 1)[1]
        elif param.startswith("-w="):
            wavelength = float(param.split("=", 1)[1])
        elif param.startswith("-d="):
            dummy = float(param.split("=", 1)[1])
        elif param.startswith("-dd="):
            delta_dummy = float(param.split("=", 1)[1])
        elif param.startswith("-m="):
            mask = param.split("=", 1)[1]
        elif param.startswith("-h"):
            print(__doc__)
            sys.exit(1)
        elif os.path.isfile(param):
            processFile.append(param)

    if paramFile and processFile:
        integrator.load(paramFile)
        if wavelength is not None:
            integrator.wavelength = wavelength
        print integrator
        if mask is not None:
            mask = 1 - fabio.open(mask).data
        for oneFile in processFile:
            t0 = time.time()
            fabioFile = fabio.open(oneFile)
            if fabioFile.nframes > 1:
                for meth in ["BBox", "cython", "numpy"]:
                    outFile = os.path.splitext(oneFile)[0] + "_" + meth + ".dat"
                    integrator.saxs(data=fabioFile.data.astype("float32"),
                                nbPt=min(fabioFile.data.shape),
                                dummy=dummy,
                                delta_dummy=delta_dummy,
                                mask=mask,
                                variance=fabioFile.next().data.astype("float32"),
                                filename=outFile,
                                method=meth)
            else:
                for meth in ["BBox", "cython", "numpy"]:
                    outFile = os.path.splitext(oneFile)[0] + "_" + meth + ".dat"
                    integrator.saxs(data=fabioFile.data,
                                nbPt=min(fabioFile.data.shape),
                                dummy=dummy,
                                delta_dummy=delta_dummy,
                                mask=mask,
                                filename=outFile,
                                method=meth)
            t1 = time.time()
        print("Integration took %6.3fs %s --> %s" % (t1 - t0, oneFile, outFile))

    else:
        print(__doc__)


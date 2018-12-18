#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer <Jerome.Kieffer@ESRF.eu>
#             Picca Frédéric-Emmanuel <picca@synchrotron-soleil.fr>
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
#
"""Integrate 2D images into SAXS patterns. Also used in PDF measurements"""
__author__ = "Jerome Kieffer, Picca Frédéric-Emmanuel"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/12/2018"
__status__ = "production"

import os
import sys
import time
import fabio
import logging
from pyFAI import date, version as pyFAI_version
from pyFAI import units
from pyFAI import utils
from pyFAI.method_registry import IntegrationMethod
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
hc = units.hc

logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

from argparse import ArgumentParser


def main():
    usage = "pyFAI-saxs [options] -n 1000 -p ponifile file1.edf file2.edf ..."
    version = "PyFAI-saxs version %s from %s " % (pyFAI_version, date)
    description = """Azimuthal integration for SAXS users."""
    epilog = """pyFAI-saxs is the SAXS script of pyFAI that allows data
    reduction (azimuthal integration) for Small Angle Scattering with output
    axis in q space."""
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("-v", "--version", action='version', version=version)
    parser.add_argument("args", metavar="FILE", type=str, nargs='+',
                        help="Image files to integrate")
    parser.add_argument("-p", dest="ponifile",
                        type=str, default=None,
                        help="PyFAI parameter file (.poni)")
    parser.add_argument("-n", "--npt", dest="npt",
                        type=int, default=None,
                        help="Number of points in radial dimension")
    parser.add_argument("-w", "--wavelength", dest="wavelength", type=float,
                        help="wavelength of the X-Ray beam in Angstrom", default=None)
    parser.add_argument("-e", "--energy", dest="energy", type=float,
                        help="energy of the X-Ray beam in keV (hc=%skeV.A)" %
                        hc, default=None)
    parser.add_argument("-u", "--dummy", dest="dummy",
                        type=float, default=None,
                        help="dummy value for dead pixels")
    parser.add_argument("-U", "--delta_dummy", dest="delta_dummy",
                        type=float, default=None,
                        help="delta dummy value")
    parser.add_argument("-m", "--mask", dest="mask",
                        type=str, default=None,
                        help="name of the file containing the mask image")
    parser.add_argument("-d", "--dark", dest="dark",
                        type=str, default=None,
                        help="name of the file containing the dark current")
    parser.add_argument("-f", "--flat", dest="flat",
                        type=str, default=None,
                        help="name of the file containing the flat field")
#    parser.add_argument("-b", "--background", dest="background",
#                      type="string", default=None,
#                      help="name of the file containing the background")
    parser.add_argument("-P", "--polarization", dest="polarization_factor",
                        type=float, default=None,
                        help="Polarization factor, from -1 (vertical) to +1 (horizontal), \
                      default is None for no correction, synchrotrons are around 0.95")
    parser.add_argument("--error-model", dest="error_model",
                        type=str, default=None,
                        help="Error model to use. Currently on 'poisson' is implemented ")
    parser.add_argument("--unit", dest="unit",
                        type=str, default="q_nm^-1",
                        help="unit for the radial dimension: can be q_nm^-1, q_A^-1, 2th_deg, \
                          2th_rad or r_mm")
    parser.add_argument("--ext", dest="ext",
                        type=str, default=".dat",
                        help="extension of the regrouped filename (.dat)")
    parser.add_argument("--method", dest="method",
                        type=str, default=None,
                        help="Integration method ")

    options = parser.parse_args()
    if len(options.args) < 1:
        logger.error("incorrect number of arguments")
    to_process = utils.expand_args(options.args)

    if options.ponifile and to_process:
        integrator = AzimuthalIntegrator.sload(options.ponifile)

        if to_process:
            first = to_process[0]
            fabimg = fabio.open(first)
            integrator.detector.guess_binning(fabimg.data)
        if options.wavelength:
            integrator.wavelength = options.wavelength * 1e-10
        elif options.energy:
            integrator.wavelength = hc / options.energy * 1e-10
        if options.mask and os.path.exists(options.mask):  # override with the command line mask
            integrator.maskfile = options.mask
        if options.dark and os.path.exists(options.dark):  # set dark current
            integrator.darkcurrent = fabio.open(options.dark).data
        if options.flat and os.path.exists(options.flat):  # set Flat field
            integrator.flatfield = fabio.open(options.flat).data
        if options.method:
            method = options.method
        else:
            if len(to_process) > 5:
                method = IntegrationMethod.select_method(1, "full", "csr")[0]
            else:
                method = IntegrationMethod.select_method(1, "full", "histogram")[0]
        print(integrator)
        print("Mask: %s\tMethod: %s" % (integrator.maskfile, method))

        for afile in to_process:
            sys.stdout.write("Integrating %s --> " % afile)
            outfile = os.path.splitext(afile)[0] + options.ext
            t0 = time.time()
            fimg = fabio.open(afile)
            t1 = time.time()
            if fimg.nframes > 1:
                integrator.integrate1d(data=fimg.data,
                                       npt=options.npt or min(fimg.data.shape),
                                       dummy=options.dummy,
                                       delta_dummy=options.delta_dummy,
                                       filename=outfile,
                                       variance=fimg.next().data,
                                       method=method,
                                       unit=options.unit,
                                       error_model=options.error_model,
                                       polarization_factor=options.polarization_factor,
                                       metadata=fimg.header
                                       )
            else:
                integrator.integrate1d(data=fimg.data,
                                       npt=options.npt or min(fimg.data.shape),
                                       dummy=options.dummy,
                                       delta_dummy=options.delta_dummy,
                                       filename=outfile,
                                       method=method,
                                       unit=options.unit,
                                       error_model=options.error_model,
                                       polarization_factor=options.polarization_factor,
                                       metadata=fimg.header)
            t2 = time.time()

            msg = "%s,\t reading: %.3fs\t 1D integration: %.3fs."
            print(msg % (outfile, t1 - t0, t2 - t1))


if __name__ == "__main__":
    main()

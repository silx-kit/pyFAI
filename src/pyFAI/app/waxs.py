#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
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
"""Integrate 2D images into powder diffraction patterns"""
__author__ = "Jerome Kieffer, Picca Frédéric-Emmanuel"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "13/05/2025"
__status__ = "production"

import os
import sys
import time
from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
try:
    import hdf5plugin  # noqa
except ImportError:
    logger.debug("Unable to load hdf5plugin, backtrace:", exc_info=True)
import fabio
from fabio.fabioutils import exists as fabio_exists

from .. import date as pyFAI_date, version as pyFAI_version, units, utils
from ..average import average_dark
from ..integrator.azimuthal import AzimuthalIntegrator
from ..method_registry import IntegrationMethod
hc = units.hc


def main(args=None):
    usage = "pyFAI-waxs [options] -p ponifile file1.edf file2.edf ..."
    version = "pyFAI-waxs version %s from %s" % (pyFAI_version, pyFAI_date)
    description = "Azimuthal integration for powder diffraction."
    epilog = """pyFAI-waxs is the script of pyFAI that allows data reduction
    (azimuthal integration) for Wide Angle Scattering to produce X-Ray Powder
    Diffraction Pattern with output axis in 2-theta space.
    """
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
    parser.add_argument("-P", "--polarization", dest="polarization_factor",
                        type=float, default=None,
                        help="Polarization factor, from -1 (vertical) to +1 (horizontal), \
                          default is None for no correction, synchrotrons are around 0.95")

#    parser.add_argument("-b", "--background", dest="background",
#                      type=str, default=None,
#                      help="name of the file containing the background")
    parser.add_argument("--error-model", dest="error_model",
                        type=str, default=None,
                        help="Error model to use. Currently on 'poisson' is implemented ")
    parser.add_argument("--unit", dest="unit",
                        type=str, default="2th_deg",
                        help="unit for the radial dimension: can be q_nm^-1, q_A^-1, 2th_deg, \
                          2th_rad or r_mm")
    parser.add_argument("--ext", dest="ext",
                        type=str, default=".xy",
                        help="extension of the regrouped filename (.xy) ")
    parser.add_argument("--method", dest="method",
                        type=str, default=None,
                        help="Integration method ")
    parser.add_argument("--multi", dest="multiframe",  # type=bool,
                        default=False, action="store_true",
                        help="Average out all frame in a file before integrating extracting variance, otherwise treat every single frame")
    parser.add_argument("--average", dest="average", type=str,
                        default="mean",
                        help="Method for averaging out: can be 'mean' (default), 'min', 'max' or 'median")
    parser.add_argument("--do-2D", dest="do_2d",
                        default=False, action="store_true",
                        help="Perform 2D integration in addition to 1D")

    options = parser.parse_args(args)
    if len(options.args) < 1:
        logger.error("incorrect number of arguments")

    to_process = utils.expand_args(options.args)

    if options.ponifile and to_process:
        integrator = AzimuthalIntegrator.sload(options.ponifile)

        if options.method:
            method1d = method2d = options.method
        else:
            if len(to_process) > 5:
                method1d = IntegrationMethod.select_method(1, "full", "csr")[0]
                method2d = IntegrationMethod.select_method(2, "full", "csr")[0]
            else:
                method1d = IntegrationMethod.select_method(1, "full", "histogram")[0]
                method2d = IntegrationMethod.select_method(2, "full", "histogram")[0]
        if to_process:
            first = to_process[0]
            with fabio.open(first) as fabimg:
                integrator.detector.guess_binning(fabimg.data)

        if options.wavelength:
            integrator.wavelength = options.wavelength * 1e-10
        elif options.energy:
            integrator.wavelength = hc / options.energy * 1e-10
        if options.mask and fabio_exists(options.mask):  # override with the command line mask
            integrator.maskfile = options.mask
        if options.dark and fabio_exists(options.dark):  # set dark current
            with fabio.open(options.dark) as fimg:
                integrator.darkcurrent = fimg.data
        if options.flat and fabio_exists(options.flat):  # set Flat field
            with fabio.open(options.flat) as fimg:
                integrator.flatfield = fimg.data

        # print(integrator)
        # print("Mask: %s\tMethods: %s / %s" % (integrator.maskfile, method1d, method2d))
        for afile in to_process:
            sys.stdout.write("Integrating %s --> " % afile)
            outfile = os.path.splitext(afile)[0] + options.ext
            azimFile = os.path.splitext(afile)[0] + ".azim"
            t0 = time.perf_counter()
            with fabio.open(afile) as fabimg:
                if options.multiframe:
                    data = average_dark([fabimg.getframe(i).data for i in range(fabimg.nframes)], center_method=options.average)
                else:
                    data = fabimg.data
                t1 = time.perf_counter()
                integrator.integrate1d(data,
                                       options.npt or min(fabimg.data.shape),
                                       filename=outfile,
                                       dummy=options.dummy,
                                       delta_dummy=options.delta_dummy,
                                       method=method1d,
                                       unit=options.unit,
                                       error_model=options.error_model,
                                       polarization_factor=options.polarization_factor,
                                       metadata=fabimg.header
                                       )
                t2 = time.perf_counter()
                if options.do_2d:
                    integrator.integrate2d(data,
                                           options.npt or min(fabimg.data.shape),
                                           360,
                                           filename=azimFile,
                                           dummy=options.dummy,
                                           delta_dummy=options.delta_dummy,
                                           method=method2d,
                                           unit=options.unit,
                                           error_model=options.error_model,
                                           polarization_factor=options.polarization_factor,
                                           metadata=fabimg.header
                                           )
                    t3 = time.perf_counter()
                    print(f"{outfile}\t reading: {t1-t0:.3f}s\t 1D integration: {t2-t1:.3f}s,\t 2D integration {t3-t2:.3f}s.")
                else:
                    print(f"{outfile}\t reading: {t1-t0:.3f}s\t 1D integration: {t2 - t1:%.3f}s.")



if __name__ == "__main__":
    main()

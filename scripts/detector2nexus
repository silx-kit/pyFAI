#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/kif/pyFAI
#
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer <Jerome.Kieffer@ESRF.eu>
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
"""
detector2nexus is a small utility that converts a detector description into a NeXus detector
useable by other pyFAI utilities
"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/03/2015"
__status__ = "development"

import os
import sys
import fabio
import logging
import pyFAI
import pyFAI.utils
import numpy
try:
    from argparse import ArgumentParser
except ImportError:
    from pyFAI.third_party.argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector")


def main():
    usage = "detector2nexus [options] [options] -o nxs.h5"
    version = "detector2nexus version %s from %s" % (pyFAI.version, pyFAI.date)
    description = """
    Convert a complex detector definition (multiple modules, possibly in 3D)
    into a single NeXus detector definition together with the mask (and much more in the future)
    """
    epilog = """
    This summarizes detector2nexus
    """
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("-V", "--version", action='version', version=version)
    parser.add_argument("-o", "--output", dest="output",
                        type=str, default=None,
                        help="Output nexus file, unless detector_name.h5")
    parser.add_argument("-n", "--name", dest="name",
                        type=str, default=None,
                        help="name of the detector")
    parser.add_argument("-m", "--mask", dest="mask",
                        type=str, default=None,
                        help="mask corresponding to the detector")
    parser.add_argument("-D", "--detector", dest="detector", type=str,
                        default="Detector",
                        help="Base detector name (see documentation of pyFAI.detectors")
    parser.add_argument("-s", "--splinefile", dest="splinefile", type=str,
                        default=None,
                        help="Geometric distortion file from FIT2D")
    parser.add_argument("-dx", "--x-corr", dest="dx", type=str, default=None,
                        help="Geometric correction for pilatus")
    parser.add_argument("-dy", "--y-corr", dest="dy", type=str, default=None,
                        help="Geometric correction for pilatus")
    parser.add_argument("-p", "--pixel", dest="pixel", type=str, default=None,
                        help="pixel size (comma separated): x,y")
    parser.add_argument("-S", "--shape", dest="shape", type=str, default=None,
                        help="shape of the detector (comma separated): x,y")
    parser.add_argument("-d", "--dark", dest="dark", type=str, default=None,
                        help="Dark noise to be subtracted")
    parser.add_argument("-f", "--flat", dest="flat", type=str, default=None,
                        help="Flat field correction")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                        help="switch to verbose/debug mode")
#     parser.add_argument("args", metavar='FILE', type=str, nargs='+',
#                         help="Files to be processed")

    options = parser.parse_args()

    if options.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    detector = pyFAI.detectors.detector_factory(options.detector)
    if options.output:
        output = options.output
    else:
        output = detector.name + ".h5"

    if options.mask:
        mask = fabio.open(options.mask).data.astype(bool)
        if detector.mask is None:
            detector.mask = mask
        else:
            detector.mask = numpy.logical_or(mask, detector.mask)

    if options.flat:
        detector.flat = fabio.open(options.flat).data
    if options.dark:
        detector.dark = fabio.open(options.dark).data
    if options.splinefile:
        detector.set_splineFile(options.splinefile)
    else:
        if options.pixel:
            p = options.pixel.split(",")
            psize = float(p[0])
            if len(p) == 1:
                detector.pixel1 = psize
                detector.pixel2 = psize
            else:
                detector.pixel1 = float(p[1])
                detector.pixel2 = psize
        if options.shape:
            p = options.shape.split(",")
            psize = int(p[0])
            if len(p) == 1:
                detector.shape = psize, psize
            else:
                detector.shape = int(p[1]), psize
        if options.dx and options.dy:
            dx = fabio.open(options.dx).data
            dy = fabio.open(options.dy).data
            # pilatus give displaceemt in percent of pixel ...
            if ".cbf" in options.dx:
                dx *= 0.01
            if ".cbf" in options.dy:
                dy *= 0.01
            detector.set_dx(dx)
            detector.set_dy(dy)
    detector.save(output)

if __name__ == "__main__":
    main()

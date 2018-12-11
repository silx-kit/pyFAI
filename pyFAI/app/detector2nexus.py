#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#
#    Copyright (C) 2014-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer <Jerome.Kieffer@ESRF.eu>
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

"""Converts a detector description into a NeXus detector usable by other pyFAI utilities"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/10/2018"
__status__ = "development"

import sys
import logging
import numpy
import fabio
import pyFAI
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
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
    parser.add_argument("--dx", "--x-corr", dest="dx", type=str, default=None,
                        help="Geometric correction for pilatus")
    parser.add_argument("--dy", "--y-corr", dest="dy", type=str, default=None,
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
    # parser.add_argument("args", metavar='FILE', type=str, nargs='+',
    #                     help="Files to be processed")

    argv = sys.argv
    # hidden backward compatibility for -dx and -dy
    # A short option only expect a single char
    argv = ["-" + a if a.startswith("-dx") else a for a in argv]
    argv = ["-" + a if a.startswith("-dy") else a for a in argv]
    print(argv)
    options = parser.parse_args(args=argv[1:])

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

#!/usr/bin/env python
# coding: UTF-8
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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


"""extracts the mask from an Eiger master file."""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/10/2018"
__satus__ = "development"


import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger("eiger-mask")
import fabio

import pyFAI
from argparse import ArgumentParser

try:
    import h5py
except ImportError:
    logger.debug("h5py is not available", exc_info=True)
    h5py = None


def extract_mask(infile):
    """
    Retrieve the mask from an Eiger master file.

    :param infile: name of the Eiger master file
    """
    h = h5py.File(infile, "r")
    entry = h["entry"]
    instrument = entry["instrument"]
    detector = instrument["detector"]
    detectorSpecific = detector["detectorSpecific"]
    return detectorSpecific["pixel_mask"].value


def main():
    description = "A tool to extract the mask from an Eiger detector file."
    version = "eiger-mask version %s from %s" % (pyFAI.version, pyFAI.date)
    epilog = None
    if h5py is None:
        epilog = "Python h5py module is missing. It have to be installed to use this application"
    parser = ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("-V", "--version", action='version', version=version)
    parser.add_argument('input_file', help='Input file. Must be an HDF5 file.')
    parser.add_argument('output_file', nargs="?", help='Output file. It can be an msk, tif, or an edf file.')
    options = parser.parse_args()

    if h5py is None:
        logger.error("Python h5py module is expected to use this script")
        sys.exit(1)

    infile = os.path.abspath(options.input_file)
    if options.output_file is not None:
        outfile = options.output_file
    else:
        outfile = os.path.splitext(infile)[0] + "_mask.edf"

    mask = extract_mask(infile)
    if outfile.endswith("msk"):
        fabio.fit2dmaskimage.fit2dmaskimage(data=mask).write(outfile)
    elif outfile.endswith("tif"):
        fabio.tifimage.tifimage(data=mask).write(outfile)
    else:
        fabio.edfimage.edfimage(header={"data_file": infile}, data=mask).write(outfile)


if __name__ == "__main__":
    main()

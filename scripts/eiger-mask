#!/usr/bin/env python
# coding: UTF-8
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
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

"""
eiger-mask

A tool for extracting the mask from an Eiger master file.

$ eiger-mask master.h5 
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/12/2016"
__satus__ = "development"


import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eiger-mask")
import numpy
import fabio

try:
    from argparse import ArgumentParser
except ImportError:
    from pyFAI.third_party.argparse import ArgumentParser

try:
    import h5py
except ImportError:
    h5py = None


def extract_mask(infile):
    """
    Retrieve the mask from an Eiger master file.

    @param infile: name of the Eiger master file
    """
    h = h5py.File(infile, "r")
    entry = h["entry"]
    instrument = entry["instrument"]
    detector = instrument["detector"]
    detectorSpecific = detector["detectorSpecific"]
    return detectorSpecific["pixel_mask"].value

if __name__ == "__main__":
    description = "A tool to extract the mask from an Eiger detector file."
    epilog = None
    if h5py is None:
        epilog = "Python h5py module is missing. It have to be installed to use this application"
    parser = ArgumentParser(description=description, epilog=epilog)
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

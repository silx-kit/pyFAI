#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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


"""CLI interface for diffraction tomography data reduction"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/10/2018"
__satus__ = "Production"

import logging
import os
import glob
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger("diff_tomo")
from pyFAI import version as PyFAI_VERSION
from pyFAI import date as PyFAI_DATE
from pyFAI.diffmap import DiffMap
from argparse import ArgumentParser

from pyFAI.third_party import six
urlparse = six.moves.urllib.parse.urlparse


class DiffTomo(DiffMap):
    """
    Basic class for diffraction tomography experiment using pyFAI.

    """
    def __init__(self, nTrans=1, nRot=1, nDiff=1000):
        """
        Constructor of the class

        :param nTrans: number of translations
        :param nRot: number of translations
        :param nDiff: number of points in diffraction pattern
        """
        DiffMap.__init__(self, npt_slow=nRot, npt_fast=nTrans, npt_rad=nDiff)
        self.hdf5path = "diff_tomo/data/sinogram"
        self.experiment_title = "Diffraction Tomography"
        self.slow_motor_name = "rotation"
        self.fast_motor_name = "translation"

    def parse(self, with_config=False):
        """
        parse options from command line
        """
        description = """Azimuthal integration for diffraction tomography.

Diffraction tomography is an experiment where 2D diffraction patterns are recorded
while performing a 2D scan, one (the slowest) in rotation around the sample center
and the other (the fastest) along a translation through the sample.
Diff_tomo is a script (based on pyFAI and h5py) which allows the reduction of this
4D dataset into a 3D dataset containing the rotations angle (hundreds), the translation step (hundreds)
and the many diffraction angles (thousands).
The resulting dataset can be opened using the PyMca ROItool
where the 1d dataset has to be selected as last dimension.
The output file aims at being NeXus compliant.

This tool can be used for mapping experiments if one considers the slow scan
direction as the rotation. but the *diff_map* tool provides in addition a graphical
user interface.
        """
        epilog = """If the number of files is too large, use double quotes "*.edf" """
        usage = """diff_tomo [options] -p ponifile imagefiles*"""
        version = "diff_tomo from pyFAI  version %s: %s" % (PyFAI_VERSION, PyFAI_DATE)
        parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
        parser.add_argument("-V", "--version", action='version', version=version)
        parser.add_argument("args", metavar="FILE", help="List of files to calibrate", nargs='+')
        parser.add_argument("-o", "--output", dest="outfile",
                            help="HDF5 File where processed sinogram was saved, by default diff_tomo.h5",
                            metavar="FILE", default="diff_tomo.h5")
        parser.add_argument("-v", "--verbose",
                            action="store_true", dest="verbose", default=False,
                            help="switch to verbose/debug mode, defaut: quiet")
        parser.add_argument("-P", "--prefix", dest="prefix",
                            help="Prefix or common base for all files",
                            metavar="FILE", default="", type=str)
        parser.add_argument("-e", "--extension", dest="extension",
                            help="Process all files with this extension",
                            default="")
        parser.add_argument("-t", "--nTrans", dest="nTrans",
                            help="number of points in translation. Mandatory", default=None)
        parser.add_argument("-r", "--nRot", dest="nRot",
                            help="number of points in rotation. Mandatory", default=None)
        parser.add_argument("-c", "--nDiff", dest="nDiff",
                            help="number of points in diffraction powder pattern, Mandatory",
                            default=None)
        parser.add_argument("-d", "--dark", dest="dark", metavar="FILE",
                            help="list of dark images to average and subtract",
                            default=None)
        parser.add_argument("-f", "--flat", dest="flat", metavar="FILE",
                            help="list of flat images to average and divide",
                            default=None)
        parser.add_argument("-m", "--mask", dest="mask", metavar="FILE",
                            help="file containing the mask, no mask by default", default=None)
        parser.add_argument("-p", "--poni", dest="poni", metavar="FILE",
                            help="file containing the diffraction parameter (poni-file), Mandatory",
                            default=None)
        parser.add_argument("-O", "--offset", dest="offset",
                            help="do not process the first files", default=None)
        parser.add_argument("-g", "--gpu", dest="gpu", action="store_true",
                            help="process using OpenCL on GPU ", default=False)
        parser.add_argument("-S", "--stats", dest="stats", action="store_true",
                            help="show statistics at the end", default=False)

        options = parser.parse_args()
        args = options.args

        if options.verbose:
            logger.setLevel(logging.DEBUG)
        self.hdf5 = options.outfile
        if options.dark:
            dark_files = [os.path.abspath(urlparse(f).path)
                          for f in options.dark.split(",")
                          if os.path.isfile(urlparse(f).path)]
            if dark_files:
                self.dark = dark_files
            else:
                raise RuntimeError("No such dark files")

        if options.flat:
            flat_files = [os.path.abspath(urlparse(f).path)
                          for f in options.flat.split(",")
                          if os.path.isfile(urlparse(f).path)]
            if flat_files:
                self.flat = flat_files
            else:
                raise RuntimeError("No such flat files")

        self.use_gpu = options.gpu
        self.inputfiles = []
        for fn in args:
            f = urlparse(fn).path
            if os.path.isfile(f) and f.endswith(options.extension):
                self.inputfiles.append(os.path.abspath(f))
            elif os.path.isdir(f):
                self.inputfiles += [os.path.abspath(os.path.join(f, g)) for g in os.listdir(f) if g.endswith(options.extension) and g.startswith(options.prefix)]
            else:
                self.inputfiles += [os.path.abspath(f) for f in glob.glob(f)]
        self.inputfiles.sort(key=self.to_tuple)
        if not self.inputfiles:
            raise RuntimeError("No input files to process, try --help")
        if options.mask:
            mask = urlparse(options.mask).path
            if os.path.isfile(mask):
                logger.info("Reading Mask file from: %s", mask)
                self.mask = os.path.abspath(mask)
            else:
                logger.warning("No such mask file %s", mask)
        if options.poni:
            if os.path.isfile(options.poni):
                logger.info("Reading PONI file from: %s", options.poni)
                self.poni = options.poni
            else:
                logger.warning("No such poni file %s", options.poni)

        if options.nTrans is not None:
            self.npt_fast = int(options.nTrans)
        if options.nRot is not None:
            self.npt_slow = int(options.nRot)
        if options.nDiff is not None:
            self.npt_rad = int(options.nDiff)
        if options.offset is not None:
            self.offset = int(options.offset)
        else:
            self.offset = 0
        self.stats = options.stats
        return options


def main():
    dt = DiffTomo()
    dt.parse()
    dt.setup_ai()
    dt.makeHDF5()
    dt.process()
    dt.show_stats()


if __name__ == "__main__":
    main()

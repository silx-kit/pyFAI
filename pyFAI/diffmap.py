# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import absolute_import, print_function, division

"""Module with GUI for diffraction mapping experiments"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "14/12/2018"
__status__ = "development"
__docformat__ = 'restructuredtext'

import os
import time
import posixpath
import sys
import collections
import glob
import logging
logger = logging.getLogger(__name__)
import numpy
import fabio
import json

from .opencl import ocl
from .units import to_unit
from .third_party import six
from . import version as PyFAI_VERSION, date as PyFAI_DATE, load
from .io import Nexus, get_isotime
from argparse import ArgumentParser
urlparse = six.moves.urllib.parse.urlparse

DIGITS = [str(i) for i in range(10)]
Position = collections.namedtuple('Position', 'index, rot, trans')


class DiffMap(object):
    """
    Basic class for diffraction mapping experiment using pyFAI
    """
    def __init__(self, npt_fast=0, npt_slow=1, npt_rad=1000, npt_azim=None):
        """Constructor of the class DiffMap for diffraction mapping

        :param npt_fast: number of translations
        :param npt_slow: number of translations
        :param npt_rad: number of points in diffraction pattern (radial dimension)
        :param npt_azim:  number of points in diffraction pattern (azimuthal dimension)
        """
        self.npt_fast = npt_fast
        self.npt_slow = npt_slow
        self.npt_rad = npt_rad
        self.slow_motor_name = "slow"
        self.fast_motor_name = "fast"
        self.offset = 0
        self.poni = None
        self.ai = None
        self.dark = None
        self.flat = None
        self.mask = None
        self.I0 = None
        self.hdf5 = None
        self.hdf5path = "diff_map/data/map"
        self.group = None
        self.dataset = None
        self.inputfiles = []
        self.timing = []
        self.method = "csr"
        self.unit = to_unit("2th_deg")
        self.stats = False
        self._idx = -1
        self.processed_file = []
        self.nxs = None
        self.experiment_title = "Diffraction Mapping"

    def __repr__(self):
        return "%s experiment with ntp_slow: %s ntp_fast: %s, npt_diff: %s" % \
            (self.experiment_title, self.npt_slow, self.npt_fast, self.npt_rad)

    @staticmethod
    def to_tuple(name):
        """
        Extract numbers as tuple:

        to_tuple("slice06/IRIS4_1_14749.edf")
        --> (6, 4, 1, 14749)

        :param name: input string, often a filename
        """
        res = []
        cur = ""
        for c in name:
            if c in DIGITS:
                cur = cur + c
            elif cur:
                res.append(cur)
                cur = ""
        return tuple(int(i) for i in res)

    def parse(self, with_config=False):
        """
        parse options from command line: setup the object

        :return: dictionary able to setup a DiffMapWidget
        """
        description = """Azimuthal integration for diffraction imaging.

Diffraction mapping is an experiment where 2D diffraction patterns are recorded
while performing a 2D scan.

Diff_map is a graphical application (based on pyFAI and h5py) which allows the reduction of this
4D dataset into a 3D dataset containing the two motion dimensions
and the many diffraction angles (thousands). The resulting dataset can be opened using PyMca roitool
where the 1d dataset has to be selected as last dimension.
This result file aims at being NeXus compliant.

This tool can be used for diffraction tomography experiment as well, considering the slow scan direction as the rotation.
        """
        epilog = """Bugs: Many, see hereafter:
1)If the number of files is too large, use double quotes "*.edf"
2)There is a known bug on Debian7 where importing a large number of file can
take much longer than the integration itself: consider passing files in the
command line
        """
        usage = """diff_map [options] -p ponifile imagefiles*
If the number of files is too large, use double quotes like "*.edf" """
        version = "diff_tomo from pyFAI  version %s: %s" % (PyFAI_VERSION, PyFAI_DATE)
        parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
        parser.add_argument("-V", "--version", action='version', version=version)
        parser.add_argument("args", metavar="FILE", help="List of files to integrate. Mandatory without GUI", nargs='*')
        parser.add_argument("-o", "--output", dest="outfile",
                            help="HDF5 File where processed map will be saved. Mandatory without GUI",
                            metavar="FILE", default=None)
        parser.add_argument("-v", "--verbose",
                            action="store_true", dest="verbose", default=False,
                            help="switch to verbose/debug mode, default: quiet")
        parser.add_argument("-P", "--prefix", dest="prefix",
                            help="Prefix or common base for all files",
                            metavar="FILE", default="", type=str)
        parser.add_argument("-e", "--extension", dest="extension",
                            help="Process all files with this extension",
                            default="")
        parser.add_argument("-t", "--fast", dest="fast",
                            help="number of points for the fast motion. Mandatory without GUI", default=None)
        parser.add_argument("-r", "--slow", dest="slow",
                            help="number of points for slow motion. Mandatory without GUI", default=None)
        parser.add_argument("-c", "--npt", dest="npt_rad",
                            help="number of points in diffraction powder pattern. Mandatory without GUI",
                            default=None)
        parser.add_argument("-d", "--dark", dest="dark", metavar="FILE",
                            help="list of dark images to average and subtract (comma separated list)",
                            default=None)
        parser.add_argument("-f", "--flat", dest="flat", metavar="FILE",
                            help="list of flat images to average and divide (comma separated list)",
                            default=None)
        parser.add_argument("-m", "--mask", dest="mask", metavar="FILE",
                            help="file containing the mask, no mask by default", default=None)
        parser.add_argument("-p", "--poni", dest="poni", metavar="FILE",
                            help="file containing the diffraction parameter (poni-file), Mandatory without GUI",
                            default=None)
        parser.add_argument("-O", "--offset", dest="offset",
                            help="do not process the first files", default=None)
        parser.add_argument("-g", "--gpu", dest="gpu", action="store_true",
                            help="process using OpenCL on GPU ", default=False)
        parser.add_argument("-S", "--stats", dest="stats", action="store_true",
                            help="show statistics at the end", default=False)
        parser.add_argument("--gui", dest="gui", action="store_true",
                            help="Use the Graphical User Interface", default=True)
        parser.add_argument("--no-gui", dest="gui", action="store_false",
                            help="Do not use the Graphical User Interface", default=True)
        parser.add_argument("--config", dest="config", default=None,
                            help="provide a JSON configuration file")
        options = parser.parse_args()
        args = options.args
        if (options.config is not None) and os.path.exists(options.config):
            with open(options.config, "r") as fd:
                config = json.loads(fd.read())
        else:
            config = {}
        if "ai" not in config:
            config["ai"] = {}
        if options.verbose:
            logger.setLevel(logging.DEBUG)
        if options.outfile:
            self.hdf5 = options.outfile
            config["output_file"] = self.hdf5,
        if options.dark:
            dark_files = [os.path.abspath(urlparse(f).path)
                          for f in options.dark.split(",")
                          if os.path.isfile(urlparse(f).path)]
            if dark_files:
                self.dark = dark_files
                config["ai"]["dark_current"] = ",".join(dark_files)
                config["ai"]["do_dark"] = True
            else:
                raise RuntimeError("No such dark files")

        if options.flat:
            flat_files = [os.path.abspath(urlparse(f).path)
                          for f in options.flat.split(",")
                          if os.path.isfile(urlparse(f).path)]
            if flat_files:
                self.flat = flat_files
                config["ai"]["flat_field"] = ",".join(flat_files)
                config["ai"]["do_flat"] = True
            else:
                raise RuntimeError("No such flat files")

        if ocl and options.gpu:
            self.method = "csr_ocl_%i,%i" % ocl.select_device(type="gpu")
            config["ai"]["do_OpenCL"] = True
            config["ai"]["method"] = self.method

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
        config["input_data"] = [(i, None) for i in self.inputfiles]

        if options.mask:
            mask = urlparse(options.mask).path
            if os.path.isfile(mask):
                logger.info("Reading Mask file from: %s", mask)
                self.mask = os.path.abspath(mask)
                config["ai"]["mask_file"] = self.mask
                config["ai"]["do_mask"] = True
            else:
                logger.warning("No such mask file %s", mask)
        if options.poni:
            if os.path.isfile(options.poni):
                logger.info("Reading PONI file from: %s", options.poni)
                self.poni = options.poni
                config["ai"]["poni"] = self.poni
            else:
                logger.warning("No such poni file %s", options.poni)
        if options.fast is not None:
            self.npt_fast = int(options.fast)
            config["fast_motor_points"] = self.npt_fast
        if options.slow is not None:
            self.npt_slow = int(options.slow)
            config["slow_motor_points"] = self.npt_slow
        if options.npt_rad is not None:
            self.npt_rad = int(options.npt_rad)
            config["ai"]["nbpt_rad"] = self.npt_rad,
        if options.offset is not None:
            self.offset = int(options.offset)
            config["offset"] = self.offset,
        else:
            self.offset = 0
        self.stats = options.stats

        if with_config:
            if "do_2D" not in config["ai"]:
                config["ai"]["do_2D"] = False
            if "do_solid_angle" not in config["ai"]:
                config["ai"]["do_solid_angle"] = True
            if "unit" not in config["ai"]:
                config["ai"]["unit"] = "2th_deg"
            if "experiment_title" not in config:
                config["experiment_title"] = self.experiment_title
            if "fast_motor_name" not in config:
                config["fast_motor_name"] = self.fast_motor_name
            if "slow_motor_name" not in config:
                config["slow_motor_name"] = self.slow_motor_name
            return options, config
        return options

    def makeHDF5(self, rewrite=False):
        """
        Create the HDF5 structure if needed ...
        """
        import h5py
        dtype = h5py.special_dtype(vlen=six.text_type)

        if self.hdf5 is None:
            raise RuntimeError("No output HDF5 file provided")

        logger.info("Initialization of HDF5 file")
        if os.path.exists(self.hdf5) and rewrite:
            os.unlink(self.hdf5)

        spath = self.hdf5path.split("/")
        assert len(spath) > 2
        nxs = Nexus(self.hdf5, mode="w")
        entry = nxs.new_entry(entry=spath[0], program_name="pyFAI", title="diffmap")
        grp = entry
        for subgrp in spath[1:-2]:
            grp = nxs.new_class(grp, name=subgrp, class_type="NXcollection")

        processgrp = nxs.new_class(grp, "pyFAI", class_type="NXprocess")
        processgrp["program"] = numpy.array([i for i in sys.argv], dtype=dtype)
        processgrp["version"] = PyFAI_VERSION
        processgrp["date"] = get_isotime()
        if self.mask:
            processgrp["maskfile"] = self.mask
        if self.flat:
            processgrp["flatfiles"] = numpy.array([i for i in self.flat], dtype=dtype)
        if self.dark:
            processgrp["darkfiles"] = numpy.array([i for i in self.dark], dtype=dtype)
        processgrp["inputfiles"] = numpy.array([i for i in self.inputfiles], dtype=dtype)
        if self.poni is not None:
            processgrp["PONIfile"] = self.poni

        processgrp["dim0"] = self.npt_slow
        processgrp["dim0"].attrs["axis"] = self.slow_motor_name
        processgrp["dim1"] = self.npt_fast
        processgrp["dim1"].attrs["axis"] = self.fast_motor_name
        processgrp["dim2"] = self.npt_rad
        processgrp["dim2"].attrs["axis"] = "diffraction"
        for k, v in self.ai.getPyFAI().items():
            processgrp[k] = v

        self.group = nxs.new_class(grp, name=spath[-2], class_type="NXdata")

        if posixpath.basename(self.hdf5path) in self.group:
            self.dataset = self.group[posixpath.basename(self.hdf5path)]
        else:
            self.dataset = self.group.create_dataset(
                name=posixpath.basename(self.hdf5path),
                shape=(self.npt_slow, self.npt_fast, self.npt_rad),
                dtype="float32",
                chunks=(1, self.npt_fast, self.npt_rad),
                maxshape=(None, None, self.npt_rad))
            self.dataset.attrs["signal"] = "1"
            self.dataset.attrs["interpretation"] = "spectrum"
            self.dataset.attrs["axes"] = str(self.unit).split("_")[0]
            self.dataset.attrs["creator"] = "pyFAI"
            self.dataset.attrs["long_name"] = str(self)
        self.nxs = nxs

    def setup_ai(self):
        print("Setup of Azimuthal integrator ...")
        if self.poni:
            self.ai = load(self.poni)
        else:
            logger.error(("Unable to setup Azimuthal integrator:"
                          " no poni file provided"))
            raise RuntimeError("You must provide poni a file")
        if self.dark:
            self.ai.set_darkfiles(self.dark)
        if self.flat:
            self.ai.set_flatfiles(self.flat)
        if self.mask is not None:
            self.ai.detector.set_maskfile(self.mask)

    def init_ai(self):
        """Force initialization of azimuthal intgrator

        :return: radial position array
        """
        if not self.ai:
            self.setup_ai()
        if not self.group:
            self.makeHDF5(rewrite=False)
        if self.ai.detector.shape:
            # shape of detector undefined: reading the first image to guess it
            shape = self.ai.detector.shape
        else:
            fimg = fabio.open(self.inputfiles[0])
            shape = fimg.data.shape
        data = numpy.empty(shape, dtype=numpy.float32)
        print("Initialization of the Azimuthal Integrator using method %s" % self.method)
        # enforce initialization of azimuthal integrator
        print(self.ai)
        tth, _I = self.ai.integrate1d(data, self.npt_rad,
                                      method=self.method, unit=self.unit)
        if self.dataset is None:
            self.makeHDF5()
        space, unit = str(self.unit).split("_")
        if space not in self.group:
            self.group[space] = tth
            self.group[space].attrs["axes"] = 3
            self.group[space].attrs["unit"] = unit
            self.group[space].attrs["long_name"] = self.unit.label
            self.group[space].attrs["interpretation"] = "scalar"
        return tth

    def show_stats(self):
        if not self.stats:
            return
        try:
            from .gui.matplotlib import pyplot
        except ImportError:
            logger.error("Unable to start matplotlib for display")
            return
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(self.timing, 500, facecolor='green', alpha=0.75)
        ax.set_xlabel('Execution time (seconds)')
        ax.set_ylabel('Occurence')
        ax.set_title("Execution time")
        ax.grid(True)
        fig.show()
        six.moves.input("Enter to quit")

    def get_pos(self, filename=None, idx=None):
        """
        Calculate the position in the sinogram of the file according
        to it's number

        :param filename: name of current frame
        :param idx: index of current frame
        :return: namedtuple: index, rot, trans
        """
        #         n = int(filename.split(".")[0].split("_")[-1]) - (self.offset or 0)
        if idx is None:
            n = self.inputfiles.index(filename) - self.offset
        else:
            n = idx - self.offset
        return Position(n, n // self.npt_fast, n % self.npt_fast)

    def process_one_file(self, filename):
        """
        :param filename: name of the input filename
        :param idx: index of file
        """
        if self.ai is None:
            self.setup_ai()
        if self.dataset is None:
            self.makeHDF5()

        t = time.time()
        fimg = fabio.open(filename)
        self.process_one_frame(fimg.data)
        if fimg.nframes > 1:
            for i in range(fimg.nframes - 1):
                fimg = fimg.next()
                self.process_one_frame(fimg.data)
        t -= time.time()
        print("Processing %30s took %6.1fms (%i frames)" %
              (os.path.basename(filename), -1000.0 * t, fimg.nframes))
        self.timing.append(-t)
        self.processed_file.append(filename)

    def process_one_frame(self, frame):
        """
        :param frame: 2d numpy array with an image to process
        """
        self._idx += 1
        pos = self.get_pos(None, self._idx)
        shape = self.dataset.shape
        if pos.rot + 1 > shape[0]:
            self.dataset.resize((pos.rot + 1, shape[1], shape[2]))
        elif pos.index < 0 or pos.rot < 0 or pos.trans < 0:
            return

        _tth, I = self.ai.integrate1d(frame, self.npt_rad, safe=False,
                                      method=self.method, unit=self.unit)
        self.dataset[pos.rot, pos.trans, :] = I

    def process(self):
        if self.dataset is None:
            self.makeHDF5()
        self.init_ai()
        t0 = time.time()
        # self._idx = -1
        for f in self.inputfiles:
            self.process_one_file(f)
        tot = time.time() - t0
        cnt = self._idx + 1
        print(("Execution time for %i frames: %.3fs;"
               " Average execution time: %.1fms") %
              (cnt, tot, 1000. * tot / cnt))
        self.nxs.close()

    def get_use_gpu(self):
        return ("gpu" in self.method)

    def set_use_gpu(self, value):
        self.method = "csr_ocl_gpu" if value else "csr"

    use_gpu = property(get_use_gpu, set_use_gpu)

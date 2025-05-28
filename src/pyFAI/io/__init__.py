# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module for "high-performance" writing in either 1D with Ascii ,
or 2D with FabIO or even nD with n varying from  2 to 4 using HDF5

Stand-alone module which tries to offer interface to HDF5 via H5Py and
capabilities to write EDF or other formats using fabio.

Can be imported without h5py but then limited to fabio & ascii formats.

TODO:

- Add monitor to HDF5
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/05/2025"
__status__ = "production"
__docformat__ = 'restructuredtext'

import json
import logging
import numpy
import os
import posixpath
import sys
import threading
import time
from collections import OrderedDict
import __main__ as main
from .integration_config import WorkerConfig
from ._json import UnitEncoder
from ..utils import StringTypes, fully_qualified_name
from .. import units
from .. import version
from .. import containers

logger = logging.getLogger(__name__)
try:
    import fabio
except ImportError:
    fabio = None
    logger.error("fabio module missing")

from .nexus import get_isotime, from_isotime, is_hdf5, Nexus, h5py, save_NXcansas, save_NXmonpd
# Activating compression has an important performance penalty and,
# as we are saving Float32, the compression obtained is far from optimal
#
# # If compression is activated, thetest pyFAI.test.test_io is likely to fail
#
# try:
#     import hdf5plugin
# except ImportError:
#     CMP = {}
# else:
#     CMP = hdf5plugin.Bitshuffle()
CMP = {}


class Writer(object):
    """
    Abstract class for writers.
    """
    CONFIG_ITEMS = ["filename", "dirname", "extension", "subdir", "hpath"]

    def __init__(self, filename=None, extension=None):
        """
        Constructor of the class
        """
        # FIXME: Writer interface should have no constructor arguments
        #     (that's specific to each writers)
        self.filename = filename
        if filename is not None and os.path.exists(filename):
            logger.warning("Destination file %s exists", filename)
        self._sem = threading.Semaphore()
        self.dirname = None
        self.subdir = None
        self.extension = extension
        self.fai_cfg = {}
        self.lima_cfg = {}

    def __repr__(self):
        return f"Generic writer on file {self.filename}"

    def init(self, fai_cfg=None, lima_cfg=None):
        """
        Creates the directory that will host the output file(s)
        :param fai_cfg: configuration for worker
        :param lima_cfg: configuration for acquisition
        """

        with self._sem:
            if fai_cfg is not None:
                self.fai_cfg = fai_cfg
            if lima_cfg is not None:
                self.lima_cfg = lima_cfg
            if self.filename is not None:
                dirname = os.path.dirname(self.filename)
                if dirname and not os.path.exists(dirname):
                    try:
                        os.makedirs(dirname)
                    except Exception as err:
                        logger.info("Problem while creating directory %s: %s", dirname, err)

    def flush(self, *arg, **kwarg):
        """
        To be implemented
        """
        pass

    def write(self, data):
        """
        To be implemented
        """
        pass

    def setJsonConfig(self, json_config=None):
        """
        Sets the JSON configuration
        """

        if type(json_config) in StringTypes:
            if os.path.isfile(json_config):
                with open(json_config, "r") as f:
                    config = json.load(f)
            else:
                config = json.loads(json_config)
        else:
            config = dict(json_config)
        for k, v in config.items():
            if k in self.CONFIG_ITEMS:
                self.__setattr__(k, v)


class HDF5Writer(Writer):
    """
    Class allowing to write HDF5 Files.
    """
    CONFIG = "configuration"
    DATASET_NAME = "data"

    MODE_ERROR = "error"
    MODE_DELETE = "delete"
    MODE_APPEND = "append"
    MODE_OVERWRITE = "overwrite"

    def __init__(self, filename, hpath=None, entry_template=None, fast_scan_width=None, append_frames=False, mode=MODE_ERROR):
        """
        Constructor of an HDF5 writer:

        :param str filename: name of the file
        :param str hpath: Name of the entry group that will contains the NXprocess.
        :param str entry_template: Formattable template to create a new entry (if hpath is not specified)
        :param int fast_scan_width: set it to define the width of
        """
        Writer.__init__(self, filename)
        if entry_template is None:
            entry_template = "entry_{num:04}"
        self._entry_template = entry_template

        self.hpath = hpath

        self.fast_scan_width = None
        if fast_scan_width is not None:
            try:
                self.fast_scan_width = int(fast_scan_width)
            except ValueError:
                pass
        self.nxs = None
        self.entry_grp = None
        self.process_grp = None
        self.nxdata_grp = None
        self.intensity_ds = None
        self.error_ds = None
        self.stored_input = set()

        self.config_grp = None
        self.radial_ds = None
        self.azimuthal_ds = None
        self.has_radial_values = False
        self.has_azimuthal_values = False
        self.has_error_values = False
        self.chunk = None
        self.shape = None
        self.ndim = None
        self.do2D = None
        self._current_frame = None
        self._append_frames = append_frames
        self._mode = mode

    def __repr__(self):
        return f"HDF5 writer on file {self.filename}:{self.hpath} {'' if self.intensity_ds else 'un'}initialized"

    def _require_main_entry(self, mode):
        """
        Create and return the main entry used to store the data processing.

        Update `self.hpath` is needed.

        Load and return the entry while will contains the data processing.

        According to modes, this function will delete and recreate the file,
        or append the data to a new entry, or overwrite the existing entry.
        """
        if h5py is None:
            logger.error("No h5py library, no chance")
            raise RuntimeError("No h5py library, no chance")

        try:
            if mode == self.MODE_DELETE:
                self.nxs = Nexus(self.filename, mode="w", creator="pyFAI")
            else:
                self.nxs = Nexus(self.filename, mode="a", creator="pyFAI")
        except IOError:  # typically a corrupted HDF5 file !
            if mode == self.MODE_DELETE:
                logger.error("File can't be read. File %s deleted.", self.filename)
                os.unlink(self.filename)
                self.nxs = Nexus(self.filename, mode="w", creator="pyFAI")
            else:
                raise

        name = self.hpath or self._entry_template.split("_")[0]
        if "/" in name:
            logger.error("NXentry should always be at the NXroot level. Please consider using a NXsubentry then")
            force_name = True
        else:
            force_name = False

        try:
            entry = self.nxs.new_entry(entry=name, program_name="pyFAI",
                                       title=None, force_name=force_name)
        except TypeError:  # object already exists
            nb_entries = len(self.nxs.get_entries())
            entry_base = self.hpath or self._entry_template.split("_")[0]
            entry_name = "%s_%04i" % (entry_base, nb_entries)
            if mode == self.MODE_OVERWRITE:
                del self.nxs.h5[entry_name]
                entry = self.nxs.new_entry(entry=entry_name, force_name=True,
                                           program_name="pyFAI", title=None)
            elif mode == self.MODE_ERROR:
                raise IOError("Entry name %s::%s already exists" % (self.filename, entry_name))
            elif mode == self.MODE_APPEND:
                while entry_name in self.nxs.h5:
                    nb_entries += 1
                    entry_name = "%s_%04i" % (entry_base, nb_entries)
                entry = self.nxs.new_entry(entry=entry_name, force_name=True,
                                           program_name="pyFAI", title=None)
            else:
                raise RuntimeError()
        entry["program_name"].attrs["version"] = version
        self.hpath = entry.name
        return entry

    def init(self, fai_cfg=None, lima_cfg=None):
        """
        Initializes the HDF5 file for writing
        :param fai_cfg: the configuration of the worker as a dictionary|WorkerConfig
        :param lima_cfg: the configuration of the acquisition made by LIMA as a dictionary
        """
        logger.debug("Init")
        if not isinstance(fai_cfg, WorkerConfig):
            fai_cfg = WorkerConfig.from_dict(fai_cfg)
        Writer.init(self, fai_cfg, lima_cfg)
        with self._sem:
            if logger.isEnabledFor(logging.DEBUG):
                fai_cfg.save("fai_cfg.debug.json")
                with open("lima_cfg.debug.json", "w") as w:
                    w.write(json.dumps(self.lima_cfg, indent=4, cls=UnitEncoder))

            self.fai_cfg.nbpt_rad = 1000 if self.fai_cfg.nbpt_rad is None else self.fai_cfg.nbpt_rad

            self.entry_grp = self._require_main_entry(self._mode)

            self.nxs.h5.attrs["default"] = self.entry_grp.name.strip("/")

            self.process_grp = self.nxs.new_class(self.entry_grp, "integrate", class_type="NXprocess")
            self.process_grp["program"] = getattr(main, '__file__', u'pyFAI')
            self.process_grp["version"] = version
            self.process_grp["date"] = get_isotime()
            self.process_grp["sequence_index"] = 1


            self.nxdata_grp = self.nxs.new_class(self.process_grp, "results", "NXdata")
            self.nxdata_grp.attrs["signal"] = self.DATASET_NAME
            self.process_grp.attrs["default"] = posixpath.relpath(self.nxdata_grp.name, self.process_grp.name)
            self.entry_grp.attrs["default"] = posixpath.relpath(self.nxdata_grp.name, self.entry_grp.name)

            self.config_grp = self.nxs.new_class(self.process_grp, self.CONFIG, "NXnote")
            self.config_grp.attrs["desc"] = "PyFAI worker configuration"
            self.config_grp["type"] = "text/json"
            self.config_grp["data"] = json.dumps(self.fai_cfg.as_dict(), indent=2, separators=(",\r\n", ": "))

            unit = self.fai_cfg.unit
            if unit is None:
                unit = self.fai_cfg.unit = units.TTH_DEG
            rad_name = unit.space
            rad_unit = unit.unit_symbol

            self.radial_ds = self.nxdata_grp.require_dataset("radial", (self.fai_cfg.nbpt_rad,), numpy.float32)
            self.radial_ds.attrs["unit"] = rad_unit
            self.radial_ds.attrs["interpretation"] = "scalar"
            self.radial_ds.attrs["name"] = rad_name
            self.radial_ds.attrs["long_name"] = "Diffraction radial direction %s (%s)" % (rad_name, rad_unit)

            if self.fai_cfg.do_2D:
                self.azimuthal_ds = self.nxdata_grp.require_dataset("chi", (self.fai_cfg.nbpt_azim,), numpy.float32)
                self.azimuthal_ds.attrs["unit"] = "deg"
                self.azimuthal_ds.attrs["interpretation"] = "scalar"
                self.azimuthal_ds.attrs["long_name"] = "Azimuthal angle χ (degree)"
                self.nxdata_grp["title"] = "2D azimuthaly integrated data"
            else:
                self.nxdata_grp["title"] = "Azimuthaly integrated data"

            if self.fast_scan_width:
                self.fast_motor = self.entry_grp.require_dataset("fast", (self.fast_scan_width,), numpy.float32)
                self.fast_motor.attrs["long_name"] = "Fast motor position"
                self.fast_motor.attrs["interpretation"] = "scalar"
                if self.fai_cfg.do_2D:
                    chunk = 1, self.fast_scan_width, self.fai_cfg.nbpt_azim, self.fai_cfg.nbpt_rad
                    self.ndim = 4
                    axis_definition = [".", "fast", "chi", "radial"]
                else:
                    chunk = 1, self.fast_scan_width, self.fai_cfg.nbpt_rad
                    self.ndim = 3
                    axis_definition = [".", "fast", "radial"]
            else:
                if self.fai_cfg.do_2D:
                    axis_definition = [".", "chi", "radial"]
                    chunk = 1, self.fai_cfg["nbpt_azim"], self.fai_cfg.nbpt_rad
                    self.ndim = 3
                else:
                    axis_definition = [".", "radial"]
                    chunk = 1, self.fai_cfg.nbpt_rad
                    self.ndim = 2

            utf8vlen_dtype = h5py.special_dtype(vlen=str)
            self.nxdata_grp.attrs["axes"] = numpy.array(axis_definition, dtype=utf8vlen_dtype)

            if self.DATASET_NAME in self.nxdata_grp:
                del self.nxdata_grp[self.DATASET_NAME]
            shape = list(chunk)
            if self.lima_cfg.get("number_of_frames", 0) > 0:
                if self.fast_scan_width is not None:
                    shape[0] = 1 + self.lima_cfg["number_of_frames"] // self.fast_scan_width
                else:
                    shape[0] = self.lima_cfg["number_of_frames"]
            dtype = self.lima_cfg.get("dtype")
            if dtype is None:
                dtype = numpy.float32
            else:
                dtype = numpy.dtype(dtype)
            self.chunk = tuple(chunk)
            self.shape = tuple(shape)
            self.intensity_ds = self._require_dataset(self.DATASET_NAME, dtype=dtype)
            name = "Mapping " if self.fast_scan_width else "Scanning "
            name += "2D" if self.fai_cfg.do_2D else "1D"
            name += " experiment"
            self.entry_grp["title"] = name

    def flush(self, radial=None, azimuthal=None):
        """
        Update some data like axis units and so on.

        :param radial: position in radial direction
        :param  azimuthal: position in azimuthal direction
        """
        with self._sem:
            if not (self.nxs and self.nxs.h5):
                raise RuntimeError('No opened file')
            if radial is not None:
                if radial.shape == self.radial_ds.shape:
                    self.radial_ds[:] = radial
                else:
                    logger.warning("Unable to assign radial axis position")
            if azimuthal is not None:
                if azimuthal.shape == self.azimuthal_ds.shape:
                    self.azimuthal_ds[:] = azimuthal
                else:
                    logger.warning("Unable to assign azimuthal axis position")
            self.nxs.flush()

    def close(self):
        logger.debug("Close")
        if self.nxs:
            self.flush()
            with self._sem:
                # Remove any links to HDF5 file
                self.entry_grp = None
                self.nxdata_grp = None
                self.config_grp = None
                self.process_grp = None
                self.intensity_ds = None
                self.error_ds = None
                self.radial_ds = None
                self.azimuthal_ds = None
                self.fast_motor = None
                # Close the file
                self.nxs.close()
                self.nxs = None

    def write(self, data, index=None):
        """
        Minimalistic method to limit the overhead.
        :param data: array with intensities or tuple (2th,I) or (I,2th,chi)
        """
        if index is None:
            if self._append_frames:
                if self._current_frame is None:
                    self._current_frame = 0
                else:
                    self._current_frame = self._current_frame + 1
                index = self._current_frame
            else:
                index = 0
        logger.debug("Write frame %s", index)
        radial = None
        azimuthal = None
        error = None
        if isinstance(data, containers.Integrate1dResult):
            intensity = data.intensity
            radial = data.radial
            error = data.sigma
        elif isinstance(data, containers.Integrate2dResult):
            intensity = data.intensity
            radial = data.radial
            azimuthal = data.azimuthal
            error = data.sigma
        elif isinstance(data, numpy.ndarray):
            intensity = data
        elif isinstance(data, (list, tuple)):
            n = len(data)
            if n == 2:
                radial, intensity = data
            elif n == 3:
                if data[0].ndim == 2:
                    intensity, radial, azimuthal = data
                else:
                    radial, intensity, error = data
        with self._sem:
            if self.intensity_ds is None:
                logger.warning("Writer not initialized !")
                return
            if error is not None and self.error_ds is None:
                self.error_ds = self._require_dataset(self.DATASET_NAME + "_errors", dtype=error.dtype)

            if self.fast_scan_width:
                index0, index1 = (index // self.fast_scan_width, index % self.fast_scan_width)
                if index0 >= self.intensity_ds.shape[0]:
                    self.intensity_ds.resize(index0 + 1, axis=0)
                    if error is not None:
                        self.error_ds.resize(index0 + 1, axis=0)
                self.intensity_ds[index0, index1] = data
                if error is not None:
                    self.error_ds [index0, index1] = error
            else:
                if index >= self.intensity_ds.shape[0]:
                    self.intensity_ds.resize(index + 1, axis=0)
                    if error is not None:
                        self.error_ds.resize(index + 1, axis=0)
                self.intensity_ds[index] = intensity
                if error is not None:
                    self.error_ds [index] = error

            if (not self.has_azimuthal_values) and \
               (azimuthal is not None) and \
               self.azimuthal_ds is not None:
                self.azimuthal_ds[:] = azimuthal
            if (not self.has_azimuthal_values) and \
               (azimuthal is not None) and \
               self.azimuthal_ds is not None:
                self.azimuthal_ds[:] = azimuthal
                self.has_azimuthal_values = True
            if (not self.has_radial_values) and \
               (radial is not None) and \
               self.radial_ds is not None:
                self.radial_ds[:] = radial
                self.has_radial_values = True

    def _require_dataset(self, name, dtype):
        """Returns the dataset to store data/error ."""

        if self.do2D:
            result = self.nxdata_grp.require_dataset(name,
                                                     shape=self.shape,
                                                     dtype=dtype,
                                                     chunks=self.chunk,
                                                     maxshape=(None,) + self.chunk[1:],
                                                     **CMP)
            result.attrs["interpretation"] = u"image"
        else:
            result = self.nxdata_grp.require_dataset(name,
                                                     shape=self.shape,
                                                     dtype=dtype,
                                                     chunks=self.chunk,
                                                     maxshape=(None,) + self.chunk[1:])

            result.attrs["interpretation"] = u"spectrum"
        return result

    def set_hdf5_input_dataset(self, dataset):
        "record the input dataset with an external link"
        if not isinstance(dataset, h5py.Dataset):
            return
        if not (self.nxs and self.nxs.h5 and self.entry_grp):
            return
        id_ = id(dataset)
        if id_ in self.stored_input:
            return
        else:
            self.stored_input.add(id_)
        # Process 0: measurement group
        if "measurement" in self.entry_grp:
            measurement_grp = self.entry_grp["measurement"]
        else:
            measurement_grp = self.nxs.new_class(self.entry_grp, "measurement", "NXdata")
        here = os.path.dirname(os.path.abspath(self.nxs.filename))
        there = os.path.abspath(dataset.file.filename)
        name = "images_%04i" % len(self.stored_input)
        measurement_grp[name] = h5py.ExternalLink(os.path.relpath(there, here), dataset.name)
        if "signal" not in measurement_grp.attrs:
            measurement_grp.attrs["signal"] = name


class DefaultAiWriter(Writer):

    def __init__(self, filename, engine=None):
        """Constructor of the historical writer of azimuthalIntegrator.

        :param filename: name of the output file
        :param ai: integrator, should provide make_headers method.
        """
        super(DefaultAiWriter, self).__init__(filename, engine)
        self._filename = filename
        self._engine = engine
        self._already_written = False

    def init(self, fai_cfg=None, lima_cfg=None):
        pass

    def set_filename(self, filename):
        """
        Define the filename while will be used
        """
        self._filename = filename
        self._already_written = False

    def make_headers(self, hdr="#", has_mask=None, has_dark=None, has_flat=None,
                     polarization_factor=None, normalization_factor=None,
                     metadata=None):
        """
        :param hdr: string used as comment in the header
        :type hdr: str
        :param has_dark: save the darks filenames (default: no)
        :type has_dark: bool
        :param has_flat: save the flat filenames (default: no)
        :type has_flat: bool
        :param polarization_factor: the polarization factor
        :type polarization_factor: float

        :return: the header
        :rtype: str
        """
        if "make_headers" in dir(self._engine):
            header_lst = self._engine.make_headers("list")
        else:
            header = str(self._engine)
            if "\n" in header:
                header_lst = header.split()
            else:
                header_lst = [header]

        header_lst += [""
                       f"Mask applied: {has_mask}",
                       f"Dark current applied: {has_dark}",
                       f"Flat field applied: {has_flat}",
                       f"Polarization factor: {polarization_factor}",
                       f"Normalization factor: {normalization_factor}"]

        if metadata is not None:
            header_lst += ["", "Headers of the input frame:"]
            header_lst += [i.strip() for i in json.dumps(metadata, indent=2, cls=UnitEncoder).split("\n")]
        header = "\n".join([f"{hdr} {i}" for i in header_lst])

        return header

    def save1D(self, filename, dim1, I, error=None, dim1_unit="2th_deg",
               has_mask=None, has_dark=False, has_flat=False,
               polarization_factor=None, normalization_factor=None, metadata=None):
        """This method save the result of a 1D integration as ASCII file.

        :param filename: the filename used to save the 1D integration
        :type filename: str
        :param dim1: the x coordinates of the integrated curve
        :type dim1: numpy.ndarray
        :param I: The integrated intensity
        :type I: numpy.mdarray
        :param error: the error bar for each intensity
        :type error: numpy.ndarray or None
        :param dim1_unit: the unit of the dim1 array
        :type dim1_unit: pyFAI.units.Unit
        :param has_mask: a mask was used
        :param has_dark: a dark-current was applied
        :param has_flat: flat-field was applied
        :param polarization_factor: the polarization factor
        :type polarization_factor: float, None
        :param normalization_factor: the monitor value
        :type normalization_factor: float, None
        :param metadata: JSON serializable dictionary containing the metadata
        """
        dim1_unit = units.to_unit(dim1_unit)
        with open(filename, "w") as f:
            f.write(self.make_headers(has_mask=has_mask, has_dark=has_dark,
                                      has_flat=has_flat,
                                      polarization_factor=polarization_factor,
                                      normalization_factor=normalization_factor,
                                      metadata=metadata))
            try:
                f.write("\n# --> %s\n" % (filename))
            except UnicodeError:
                f.write("\n# --> %s\n" % (filename.encode("utf8")))
            if error is None:
                f.write("#%14s %14s\n" % (dim1_unit, "I "))
                f.write("\n".join(["%14.6e  %14.6e" % (t, i) for t, i in zip(dim1, I)]))
            else:
                f.write("#%14s  %14s  %14s\n" %
                        (dim1_unit, "I ", "sigma "))
                f.write("\n".join(["%14.6e  %14.6e %14.6e" % (t, i, s) for t, i, s in zip(dim1, I, error)]))
            f.write("\n")

    def save2D(self, filename, I, dim1, dim2, error=None,
               dim1_unit="2th_deg", dim2_unit="chi_deg",
               has_mask=None, has_dark=False, has_flat=False,
               polarization_factor=None, normalization_factor=None,
               metadata=None, format_="edf"):
        """This method save the result of a 2D integration.

        :param filename: the filename used to save the 2D histogram
        :type filename: str
        :param dim1: the 1st coordinates of the histogram
        :type dim1: numpy.ndarray
        :param dim1: the 2nd coordinates of the histogram
        :type dim1: numpy.ndarray
        :param I: The integrated intensity
        :type I: numpy.mdarray
        :param error: the error bar for each intensity
        :type error: numpy.ndarray or None
        :param dim1_unit: the unit of the dim1 array
        :type dim1_unit: pyFAI.units.Unit
        :param dim2_unit: the unit of the dim2 array
        :type dim2_unit: pyFAI.units.Unit
        :param has_mask: a mask was used
        :param has_dark: a dark-current was applied
        :param has_flat: flat-field was applied
        :param polarization_factor: the polarization factor
        :type polarization_factor: float, None
        :param normalization_factor: the monitor value
        :type normalization_factor: float, None
        :param metadata: JSON serializable dictionary containing the metadata
        :param format_: file-format to be used (FabIO format)
        """
        if fabio is None:
            raise RuntimeError("FabIO module is needed to save images")
        if dim2_unit == "chi_deg" and isinstance(dim1_unit, (tuple, list)) and len(dim1_unit) == 2:
            dim1_unit, dim2_unit = (units.to_unit(i) for i in dim1_unit)
        else:
            dim1_unit = units.to_unit(dim1_unit)
            dim2_unit = units.to_unit(dim2_unit)

        # Remove \n and \t)
        engine_info = " ".join(str(self._engine).split())
        header = OrderedDict()
        header["Engine"] = engine_info

        if "make_headers" in dir(self._engine):
            header.update(self._engine.make_headers("dict"))

        header[f"{dim1_unit.name}_min"] = str(dim1.min())
        header[f"{dim1_unit.name}_max"] = str(dim1.max())
        header[f"{dim2_unit.name}_min"] = str(dim2.min())
        header[f"{dim2_unit.name}_min"] = str(dim2.max())

        header["has_mask_applied"] = str(has_mask)
        header["has_dark_correction"] = str(has_dark)
        header["has_flat_correction"] = str(has_flat)
        header["polarization_factor"] = str(polarization_factor)
        header["normalization_factor"] = str(normalization_factor)

        if metadata is not None:
            blacklist = ['HEADERID', 'IMAGE', 'BYTEORDER', 'DATATYPE', 'DIM_1',
                         'DIM_2', 'DIM_3', 'SIZE']
            for key, value in metadata.items():
                if key.upper() in blacklist or key in header:
                    continue
                else:
                    header[key] = value
        try:
            des_format = fabio.fabioformats.factory(format_ + "image")
            img = des_format.__class__(data=I, header=header)

            if error is not None:
                try:
                    img.append_frame(data=error, header={"EDF_DataBlockID": "1.Image.Error"})
                except Exception:
                    logger.warning("Multi-frame format needed to save errors, saving as %s", img)
            img.write(filename)
        except IOError:
            logger.error("IOError while writing %s", filename)

    def write(self, data):
        """
        Minimalistic method to limit the overhead.

        :param data: array with intensities or tuple (2th,I) or (I,2th,chi)\
        :type data: Integrate1dResult, Integrate2dResult
        """

        if self._already_written:
            raise Exception("This file format do not support multi frame. You have to change the filename.")
        self._already_written = True

        if isinstance(data, containers.Integrate1dResult):
            self.save1D(filename=self._filename,
                        dim1=data.radial,
                        I=data.intensity,
                        error=data.sigma,
                        dim1_unit=data.unit,
                        has_mask=data.has_mask_applied,
                        has_dark=data.has_dark_correction,
                        has_flat=data.has_flat_correction,
                        polarization_factor=data.polarization_factor,
                        normalization_factor=data.normalization_factor,
                        metadata=data.metadata)

        elif isinstance(data, containers.Integrate2dResult):
            self.save2D(filename=self._filename,
                        I=data.intensity,
                        dim1=data.radial,
                        dim2=data.azimuthal,
                        error=data.sigma,
                        dim1_unit=data.unit,
                        has_mask=data.has_mask_applied,
                        has_dark=data.has_dark_correction,
                        has_flat=data.has_flat_correction,
                        polarization_factor=data.polarization_factor,
                        normalization_factor=data.normalization_factor,
                        metadata=data.metadata)
        else:
            raise Exception("Unsupported data type: %s" % type(data))

    def flush(self):
        pass

    def close(self):
        pass


class AsciiWriter(Writer):
    """
    Ascii file writer (.xy or .dat)
    """

    def __init__(self, filename=None, prefix="fai_", extension=".dat"):
        """

        """
        Writer.__init__(self, filename, extension)
        self.header = None
        if os.path.isdir(filename):
            self.directory = filename
        else:
            self.directory = os.path.dirname(filename)
        self.prefix = prefix
        self.index_format = "%04i"
        self.start_index = 0

    def __repr__(self):
        return "Ascii writer on file %s" % (self.filename)

    def init(self, fai_cfg=None, lima_cfg=None):
        """
        Creates the directory that will host the output file(s)

        """
        Writer.init(self, fai_cfg, lima_cfg)
        with self._sem:
            header_lst = ["", "== Detector =="]
            if "detector" in self.fai_cfg:
                header_lst.append("Detector: %s" % self.fai_cfg["detector"])
            if "splineFile" in self.fai_cfg:
                header_lst.append("SplineFile: %s" % self.fai_cfg["splineFile"])
            if "pixel1" in self.fai_cfg:
                header_lst.append("PixelSize: %.3e, %.3e m" % (self.fai_cfg["pixel1"], self.fai_cfg["pixel2"]))
            if "mask_file" in self.fai_cfg:
                header_lst.append("MaskFile: %s" % (self.fai_cfg["mask_file"]))

            header_lst.append("== pyFAI calibration ==")
            if "poni1" in self.fai_cfg:
                header_lst.append("PONI: %.3e, %.3e m" % (self.fai_cfg["poni1"], self.fai_cfg["poni2"]))
            if "dist" in self.fai_cfg:
                header_lst.append("Distance Sample to Detector: %s m" % self.fai_cfg["dist"])
            if "rot1" in self.fai_cfg:
                header_lst.append("Rotations: %.6f %.6f %.6f rad" % (self.fai_cfg["rot1"], self.fai_cfg["rot2"], self.fai_cfg["rot3"]))
            if "wavelength" in self.fai_cfg:
                header_lst.append("Wavelength: %s" % self.fai_cfg["wavelength"])
            if "dark_current" in self.fai_cfg:
                header_lst.append("Dark current: %s" % self.fai_cfg["dark_current"])
            if "flat_field" in self.fai_cfg:
                header_lst.append("Flat field: %s" % self.fai_cfg["flat_field"])
            if "polarization_factor" in self.fai_cfg:
                header_lst.append("Polarization factor: %s" % self.fai_cfg["polarization_factor"])
            header_lst.append("")
            if "error_model" in self.fai_cfg:
                header_lst.append("%14s %14s %s" % (self.fai_cfg["unit"], "I", "sigma"))
            else:
                header_lst.append("%14s %14s" % (self.fai_cfg["unit"], "I"))
#            header_lst.append("")
            self.header = os.linesep.join([""] + ["# " + i for i in header_lst] + [""])
        self.prefix = lima_cfg.get("prefix", self.prefix)
        self.index_format = lima_cfg.get("index_format", self.index_format)
        self.start_index = lima_cfg.get("start_index", self.start_index)
        if not self.subdir:
            self.directory = lima_cfg.get("directory", self.directory)
        elif self.subdir.startswith("/"):
            self.directory = self.subdir
        else:
            self.directory = os.path.join(lima_cfg.get("directory", self.directory), self.subdir)
        if not os.path.exists(self.directory):
            logger.warning("Output directory: %s does not exist,creating it", self.directory)
            try:
                os.makedirs(self.directory)
            except Exception as error:
                logger.info("Problem while creating directory %s: %s", self.directory, error)

    def write(self, data, index=0):
        filename = os.path.join(self.directory, self.prefix + (self.index_format % (self.start_index + index)) + self.extension)
        if filename:
            with open(filename, "w") as f:
                f.write("# Processing time: %s%s" % (get_isotime(), self.header))
                numpy.savetxt(f, data)


class FabioWriter(Writer):
    """
    Image file writer based on FabIO

    """

    def __init__(self, filename=None, extension=None, directory="", prefix=None, index_format="_%04d", start_index=0, fabio_class=None):
        """Constructor of the class

        :param filename:
        :param extension:
        :param prefix: basename of the file
        :param index_format: "_%04s" gives "_0001" for example
        :param start_index: often 0 or 1
        :param fabio_class: type of file to write
        """
        Writer.__init__(self, filename, extension)
        self.header = {}
        self.directory = directory
        self.prefix = prefix
        self.index_format = index_format
        self.start_index = start_index
        self.index = self.start_index
        if fabio is None:
            raise RuntimeError("FabIO module is needed to save images")

        if fabio_class is None and self.extension:
            if self.extension[0] == ".":
                extension = self.extension[1:]
            else:
                extension = self.extension
                self.extension = "." + extension
            classes = fabio.fabioformats.get_classes_from_extension(extension)
            if classes:
                self.fabio_class = classes[0]
            else:
                raise RuntimeError(f"Unable to determine FabioClass from extrension {extension}")
        else:
            self.fabio_class = fabio_class

        if  self.fabio_class is None:
            raise RuntimeError("Unable to define the FabIO class to save images")

    def __repr__(self):
        return f"{self.fabio_class.__name__} writer on file {self.directory}/{self.prefix}{self.index_format}{self.extension}"

    def init(self, fai_cfg=None, lima_cfg=None, directory="pyFAI"):
        """
        Creates the directory that will host the output file(s)

        """
        Writer.init(self, fai_cfg, lima_cfg)
        with self._sem:
            # dim1_unit = units.to_unit(fai_cfg.get("unit", "r_mm"))
            _header_keys = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3",
                            ]
            _header = {"dist": str(fai_cfg.get("dist")),
                       "poni1": str(fai_cfg.get("poni1")),
                       "poni2": str(fai_cfg.get("poni2")),
                       "rot1": str(fai_cfg.get("rot1")),
                       "rot2": str(fai_cfg.get("rot2")),
                       "rot3": str(fai_cfg.get("rot3")),
                       # "chi_min": str(fai_cfg.get("chi_min")),
                       # "chi_max": str(fai_cfg.get("chi_max")),
                       # dim1_unit.REPR + "_min": str(fai_cfg.get("dist")),
                       # dim1_unit.REPR + "_max": str(fai_cfg.get("dist")),
                       # "pixelX": str(fai_cfg.get("dist")),  # this is not a bug ... most people expect dim1 to be X
                       # "pixelY": str(fai_cfg.get("dist")),  # this is not a bug ... most people expect dim2 to be Y
                       # "polarization_factor": str(fai_cfg.get("dist")),
                       # "normalization_factor":str(fai_cfg.get("dist")),
                       }

#            if self.splineFile:
#                header["spline"] = str(self.splineFile)
#
#            if dark is not None:
#                if self.darkfiles:
#                    header["dark"] = self.darkfiles
#                else:
#                    header["dark"] = 'unknown dark applied'
#            if flat is not None:
#                if self.flatfiles:
#                    header["flat"] = self.flatfiles
#                else:
#                    header["flat"] = 'unknown flat applied'
#            f2d = self.getFit2D()
#            for key in f2d:
#                header["key"] = f2d[key]
        self.prefix = fai_cfg.get("prefix", "")
        self.index_format = fai_cfg.get("index_format", "%04i")
        self.start_index = fai_cfg.get("start_index", 0)
        if not self.subdir:
            self.directory = directory
        elif self.subdir.startswith("/"):
            self.directory = self.subdir
        else:
            self.directory = os.path.join(directory, self.subdir)
        self.directory = fai_cfg.get("directory", self.directory)
        if not os.path.exists(self.directory):
            logger.warning("Output directory: %s does not exist,creating it", self.directory)
            try:
                os.makedirs(self.directory)
            except Exception as error:
                logger.info("Problem while creating directory %s: %s", self.directory, error)

    def write(self, data, index=None, header=None):
        """
        :param data: 2d array to save
        :param index: index of the file
        :param header:
        """
        if index is None:
            index = self.index_format % (self.start_index + self.index)
            self.index += 1
        else:
            index = self.index_format % index
        filename = self.prefix + index + self.extension
        if self.directory:
            filename = os.path.join(self.directory, filename)
        if header is None:
            header = self.header
        else:
            _header = self.header.copy()
            _header.update(header)
            header = _header
        if filename:
            fimg = self.fabio_class(data=data, header=header)
            fimg.write(filename)
            fimg.close()
        return filename

    def close(self):
        pass


def save_integrate_result(filename, result, title="title", sample="sample", instrument="beamline"):
    """Dispatcher for saving in different formats
    """
    if filename.endswith(".nxs"):
        raise RuntimeError("Implement Nexus writer")
    elif filename.endswith(".xrdml"):
        from .xrdml import save_xrdml
        save_xrdml(filename, result)
    else:
        writer = DefaultAiWriter(filename, result.poni)
        writer.write(result)

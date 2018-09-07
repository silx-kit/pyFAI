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

"""Module for "high-performance" writing in either 1D with Ascii ,
or 2D with FabIO or even nD with n varying from  2 to 4 using HDF5

Stand-alone module which tries to offer interface to HDF5 via H5Py and
capabilities to write EDF or other formats using fabio.

Can be imported without h5py but then limited to fabio & ascii formats.

TODO:

- Add monitor to HDF5
"""


from __future__ import absolute_import, print_function, division

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/09/2018"
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

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict


from .utils import StringTypes, fully_qualified_name
from . import units
from . import version


logger = logging.getLogger(__name__)
try:
    import h5py
except ImportError as error:
    h5py = None
    logger.error("h5py module missing")
else:
    try:
        h5py._errors.silence_errors()
    except AttributeError:  # old h5py
        pass
try:
    import fabio
except ImportError:
    fabio = None
    logger.error("fabio module missing")


def get_isotime(forceTime=None):
    """
    :param forceTime: enforce a given time (current by default)
    :type forceTime: float
    :return: the current time as an ISO8601 string
    :rtype: string
    """
    if forceTime is None:
        forceTime = time.time()
    localtime = time.localtime(forceTime)
    gmtime = time.gmtime(forceTime)
    tz_h = localtime.tm_hour - gmtime.tm_hour
    tz_m = localtime.tm_min - gmtime.tm_min
    return "%s%+03i:%02i" % (time.strftime("%Y-%m-%dT%H:%M:%S", localtime), tz_h, tz_m)


def from_isotime(text, use_tz=False):
    """
    :param text: string representing the time is iso format
    """
    if len(text) == 1:
        # just in case someone sets as a list
        text = text[0]
    try:
        text = text.decode("ascii")
    except:
        text = str(text)
    if len(text) < 19:
        logger.warning("Not a iso-time string: %s", text)
        return
    base = text[:19]
    if use_tz and len(text) == 25:
        sgn = 1 if text[:19] == "+" else -1
        tz = 60 * (60 * int(text[20:22]) + int(text[23:25])) * sgn
    else:
        tz = 0
    return time.mktime(time.strptime(base, "%Y-%m-%dT%H:%M:%S")) + tz


def is_hdf5(filename):
    """
    Check if a file is actually a HDF5 file

    :param filename: this file has better to exist
    """
    signature = [137, 72, 68, 70, 13, 10, 26, 10]
    if not os.path.exists(filename):
        raise IOError("No such file %s" % (filename))
    with open(filename, "rb") as f:
        raw = f.read(8)
    sig = [ord(i) for i in raw] if sys.version_info[0] < 3 else [int(i) for i in raw]
    return sig == signature


class Writer(object):
    """
    Abstract class for writers.
    """
    CONFIG_ITEMS = ["filename", "dirname", "extension", "subdir", "hpath"]

    def __init__(self, filename=None, extension=None):
        """
        Constructor of the class
        """
        self.filename = filename
        if os.path.exists(filename):
            logger.warning("Destination file %s exists", filename)
        self._sem = threading.Semaphore()
        self.dirname = None
        self.subdir = None
        self.extension = extension
        self.fai_cfg = {}
        self.lima_cfg = {}

    def __repr__(self):
        return "Generic writer on file %s" % (self.filename)

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
    CONFIG = "pyFAI"
    DATASET_NAME = "data"

    def __init__(self, filename, hpath="data", fast_scan_width=None):
        """
        Constructor of an HDF5 writer:

        :param filename: name of the file
        :param hpath: name of the group: it will contain data (2-4D dataset), [tth|q|r] and pyFAI, group containing the configuration
        :param fast_scan_width: set it to define the width of
        """
        Writer.__init__(self, filename)
        self.hpath = hpath
        self.fast_scan_width = None
        if fast_scan_width is not None:
            try:
                self.fast_scan_width = int(fast_scan_width)
            except ValueError:
                pass
        self.hdf5 = None
        self.group = None
        self.dataset = None
        self.pyFAI_grp = None
        self.radial_values = None
        self.azimuthal_values = None
        self.error_values = None
        self.has_radial_values = False
        self.has_azimuthal_values = False
        self.has_error_values = False
        self.chunk = None
        self.shape = None
        self.ndim = None

    def __repr__(self):
        return "HDF5 writer on file %s:%s %sinitialized" % (self.filename, self.hpath, "" if self._initialized else "un")

    def init(self, fai_cfg=None, lima_cfg=None):
        """
        Initializes the HDF5 file for writing
        :param fai_cfg: the configuration of the worker as a dictionary
        """
        logger.debug("in init")
        Writer.init(self, fai_cfg, lima_cfg)
        with self._sem:
            if logger.isEnabledFor(logging.DEBUG):
                # TODO: this is Debug statement
                open("fai_cfg.debug.json", "w").write(json.dumps(self.fai_cfg, indent=4))
                open("lima_cfg.debug.json", "w").write(json.dumps(self.lima_cfg, indent=4))
            self.fai_cfg["nbpt_rad"] = self.fai_cfg.get("nbpt_rad", 1000)
            if h5py:
                try:
                    self.hdf5 = h5py.File(self.filename)
                except IOError:  # typically a corrupted HDF5 file !
                    os.unlink(self.filename)
                    self.hdf5 = h5py.File(self.filename)
            else:
                logger.error("No h5py library, no chance")
                raise RuntimeError("No h5py library, no chance")
            self.group = self.hdf5.require_group(self.hpath)
            self.group.attrs["NX_class"] = numpy.string_("NXentry")
            self.pyFAI_grp = self.hdf5.require_group(posixpath.join(self.hpath, self.CONFIG))
            self.pyFAI_grp.attrs["desc"] = numpy.string_("PyFAI worker configuration")
            for key, value in self.fai_cfg.items():
                if value is None:
                    continue
                try:
                    self.pyFAI_grp[key] = value
                except Exception as e:
                    logger.error("Unable to set %s: %s", key, value)
                    logger.debug("Backtrace", exc_info=True)
                    raise RuntimeError(e.args[0])
            rad_name, rad_unit = str(self.fai_cfg.get("unit", "2th_deg")).split("_", 1)
            self.radial_values = self.group.require_dataset(rad_name, (self.fai_cfg["nbpt_rad"],), numpy.float32)
            if self.fai_cfg.get("nbpt_azim", 0) > 1:
                self.azimuthal_values = self.group.require_dataset("chi", (self.fai_cfg["nbpt_azim"],), numpy.float32)
                self.azimuthal_values.attrs["unit"] = numpy.string_("deg")
                self.azimuthal_values.attrs["interpretation"] = numpy.string_("scalar")
                self.azimuthal_values.attrs["long name"] = numpy.string_("Azimuthal angle")

            self.radial_values.attrs["unit"] = numpy.string_(rad_unit)
            self.radial_values.attrs["interpretation"] = numpy.string_("scalar")
            self.radial_values.attrs["long name"] = numpy.string_("diffraction radial direction")
            if self.fast_scan_width:
                self.fast_motor = self.group.require_dataset("fast", (self.fast_scan_width,), numpy.float32)
                self.fast_motor.attrs["long name"] = numpy.string_("Fast motor position")
                self.fast_motor.attrs["interpretation"] = numpy.string_("scalar")
                self.fast_motor.attrs["axis"] = numpy.string_("1")
                self.radial_values.attrs["axis"] = numpy.string_("2")
                if self.azimuthal_values is not None:
                    chunk = 1, self.fast_scan_width, self.fai_cfg["nbpt_azim"], self.fai_cfg["nbpt_rad"]
                    self.ndim = 4
                    self.azimuthal_values.attrs["axis"] = numpy.string_("3")
                else:
                    chunk = 1, self.fast_scan_width, self.fai_cfg["nbpt_rad"]
                    self.ndim = 3
            else:
                self.radial_values.attrs["axis"] = numpy.string_("1")
                if self.azimuthal_values is not None:
                    chunk = 1, self.fai_cfg["nbpt_azim"], self.fai_cfg["nbpt_rad"]
                    self.ndim = 3
                    self.azimuthal_values.attrs["axis"] = numpy.string_("2")
                else:
                    chunk = 1, self.fai_cfg["nbpt_rad"]
                    self.ndim = 2

            if self.DATASET_NAME in self.group:
                del self.group[self.DATASET_NAME]
            shape = list(chunk)
            if self.lima_cfg.get("number_of_frames", 0) > 0:
                if self.fast_scan_width is not None:
                    shape[0] = 1 + self.lima_cfg["number_of_frames"] // self.fast_scan_width
                else:
                    shape[0] = self.lima_cfg["number_of_frames"]
            dtype = self.lima_cfg.get("dtype") or self.fai_cfg.get("dtype")
            if dtype is None:
                dtype = numpy.float32
            else:
                dtype = numpy.dtype(dtype)
            self.dataset = self.group.require_dataset(self.DATASET_NAME, shape, dtype=dtype, chunks=chunk,
                                                      maxshape=(None,) + chunk[1:])
            if self.fai_cfg.get("nbpt_azim", 0) > 1:
                self.dataset.attrs["interpretation"] = numpy.string_("image")
            else:
                self.dataset.attrs["interpretation"] = numpy.string_("spectrum")
            self.dataset.attrs["signal"] = numpy.string_("1")
            self.chunk = chunk
            self.shape = chunk
            name = "Mapping " if self.fast_scan_width else "Scanning "
            name += "2D" if self.fai_cfg.get("nbpt_azim", 0) > 1 else "1D"
            name += " experiment"
            self.group["title"] = numpy.string_(name)
            self.group["program"] = numpy.string_("PyFAI")
            self.group["start_time"] = numpy.string_(get_isotime())

    def flush(self, radial=None, azimuthal=None):
        """
        Update some data like axis units and so on.

        :param radial: position in radial direction
        :param  azimuthal: position in azimuthal direction
        """
        with self._sem:
            if not self.hdf5:
                raise RuntimeError('No opened file')
            if radial is not None:
                if radial.shape == self.radial_values.shape:
                    self.radial_values[:] = radial
                else:
                    logger.warning("Unable to assign radial axis position")
            if azimuthal is not None:
                if azimuthal.shape == self.azimuthal_values.shape:
                    self.azimuthal_values[:] = azimuthal
                else:
                    logger.warning("Unable to assign azimuthal axis position")
            self.hdf5.flush()

    def close(self):
        logger.debug("In close")
        if self.hdf5:
            self.flush()
            with self._sem:
                self.hdf5.close()
                self.hdf5 = None

    def write(self, data, index=0):
        """
        Minimalistic method to limit the overhead.
        :param data: array with intensities or tuple (2th,I) or (I,2th,chi)
        """
        logger.debug("In write, index %s", index)
        radial = None
        azimuthal = None
        if isinstance(data, numpy.ndarray):
            I = data
        elif isinstance(data, (list, tuple)):
            n = len(data)
            if n == 2:
                radial, I = data
            elif n == 3:
                if data[0].ndim == 2:
                    I, radial, azimuthal = data
                else:
                    radial, I, _error = data
        with self._sem:
            if self.dataset is None:
                logger.warning("Writer not initialized !")
                return
            if self.fast_scan_width:
                index0, index1 = (index // self.fast_scan_width, index % self.fast_scan_width)
                if index0 >= self.dataset.shape[0]:
                    self.dataset.resize(index0 + 1, axis=0)
                self.dataset[index0, index1] = data
            else:
                if index >= self.dataset.shape[0]:
                    self.dataset.resize(index + 1, axis=0)
                self.dataset[index] = I
            if (not self.has_azimuthal_values) and \
               (azimuthal is not None) and \
               self.azimuthal_values is not None:
                self.azimuthal_values[:] = azimuthal
            if (not self.has_azimuthal_values) and \
               (azimuthal is not None) and \
               self.azimuthal_values is not None:
                self.azimuthal_values[:] = azimuthal
                self.has_azimuthal_values = True
            if (not self.has_radial_values) and \
               (radial is not None) and \
               self.radial_values is not None:
                self.radial_values[:] = radial
                self.has_radial_values = True


class DefaultAiWriter(Writer):

    def __init__(self, filename, engine=None):
        """Constructor of the historical writer of azimuthalIntegrator.

        :param filename: name of the output file
        :param ai: integrator, should provide make_headers method.
        """
        self._filename = filename
        self._engine = engine
        self._already_written = False

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
            header_lst = self._engine.make_headers()
        else:
            header_lst = [str(self._engine), ""]

        header_lst += ["Mask applied: %s" % has_mask,
                       "Dark current applied: %s" % has_dark,
                       "Flat field applied: %s" % has_flat,
                       "Polarization factor: %s" % polarization_factor,
                       "Normalization factor: %s" % normalization_factor]

        if metadata is not None:
            header_lst += ["", "Headers of the input frame:"]
            header_lst += [i.strip() for i in json.dumps(metadata, indent=2).split("\n")]
        header = "\n".join(["%s %s" % (hdr, i) for i in header_lst])

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

    def save2D(self, filename, I, dim1, dim2, error=None, dim1_unit="2th_deg",
               has_mask=None, has_dark=False, has_flat=False,
               polarization_factor=None, normalization_factor=None,
               metadata=None):
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
        :param has_mask: a mask was used
        :param has_dark: a dark-current was applied
        :param has_flat: flat-field was applied
        :param polarization_factor: the polarization factor
        :type polarization_factor: float, None
        :param normalization_factor: the monitor value
        :type normalization_factor: float, None
        :param metadata: JSON serializable dictionary containing the metadata
        """
        if fabio is None:
            raise RuntimeError("FabIO module is needed to save EDF images")
        dim1_unit = units.to_unit(dim1_unit)

        # Remove \n and \t)
        engine_info = " ".join(str(self._engine).split())
        header = OrderedDict()
        header["Engine"] = engine_info

        if "make_headers" in dir(self._engine):
            header.update(self._engine.make_headers("dict"))

        header[dim1_unit.name + "_min"] = str(dim1.min())
        header[dim1_unit.name + "_max"] = str(dim1.max())

        header["chi_min"] = str(dim2.min())
        header["chi_max"] = str(dim2.max())

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
            img = fabio.edfimage.edfimage(data=I.astype("float32"),
                                          header=header)

            if error is not None:
                img.appendFrame(data=error, header={"EDF_DataBlockID": "1.Image.Error"})
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

        if fully_qualified_name(data) == 'pyFAI.containers.Integrate1dResult':
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

        elif fully_qualified_name(data) == 'pyFAI.containers.Integrate2dResult':
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
            if "do_poisson" in self.fai_cfg:
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

    TODO !!!
    """
    def __init__(self, filename=None):
        """

        """
        Writer.__init__(self, filename)
        self.header = None
        self.directory = None
        self.prefix = None
        self.index_format = "%04i"
        self.start_index = 0
        self.fabio_class = None
        if fabio is None:
            raise RuntimeError("FabIO module is needed to save images")

    def __repr__(self):
        return "Image writer on file %s" % (self.filename)

    def init(self, fai_cfg=None, lima_cfg=None):
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


class Nexus(object):
    """
    Writer class to handle Nexus/HDF5 data

    Manages:

    - entry

        - pyFAI-subentry

            - detector

    TODO: make it thread-safe !!!
    """

    def __init__(self, filename, mode="r"):
        """
        Constructor

        :param filename: name of the hdf5 file containing the nexus
        :param mode: can be r or a
        """
        self.filename = os.path.abspath(filename)
        self.mode = mode
        if not h5py:
            logger.error("h5py module missing: NeXus not supported")
            raise RuntimeError("H5py module is missing")
        pre_existing = os.path.exists(self.filename)
        if pre_existing and self.mode == "r":
            self.h5 = h5py.File(self.filename, mode=self.mode)
        else:
            self.h5 = h5py.File(self.filename)
        self.to_close = []

        if not pre_existing:
            self.h5.attrs["NX_class"] = "NXroot"
            self.h5.attrs["file_time"] = get_isotime()
            self.h5.attrs["file_name"] = self.filename
            self.h5.attrs["HDF5_Version"] = h5py.version.hdf5_version
            self.h5.attrs["creator"] = self.__class__.__name__

    def close(self):
        """
        close the filename and update all entries
        """
        if self.mode!="r":
            end_time = get_isotime()
            for entry in self.to_close:
                entry["end_time"] = end_time
            self.h5.attrs["file_update_time"] = get_isotime()
        self.h5.close()

    # Context manager for "with" statement compatibility
    def __enter__(self, *arg, **kwarg):
        return self

    def __exit__(self, *arg, **kwarg):
        self.close()

    def get_entry(self, name):
        """
        Retrieves an entry from its name

        :param name: name of the entry to retrieve
        :return: HDF5 group of NXclass == NXentry
        """
        for grp_name in self.h5:
            if grp_name == name:
                grp = self.h5[grp_name]
                if isinstance(grp, h5py.Group) and \
                   ("start_time" in grp) and  \
                   self.get_attr(grp, "NX_class") == "NXentry":
                        return grp

    def get_entries(self):
        """
        retrieves all entry sorted the latest first.

        :return: list of HDF5 groups
        """
        entries = [(grp, from_isotime(self.h5[grp + "/start_time"].value))
                   for grp in self.h5
                   if isinstance(self.h5[grp], h5py.Group) and
                   ("start_time" in self.h5[grp]) and
                   self.get_attr(self.h5[grp], "NX_class") == "NXentry"]
        entries.sort(key=lambda a: a[1], reverse=True)  # sort entries in decreasing time
        return [self.h5[i[0]] for i in entries]

    def find_detector(self, all=False):
        """
        Tries to find a detector within a NeXus file, takes the first compatible detector

        :param all: return all detectors found as a list
        """
        result = []
        for entry in self.get_entries():
            for instrument in self.get_class(entry, "NXsubentry") + self.get_class(entry, "NXinstrument"):
                for detector in self.get_class(instrument, "NXdetector"):
                    if all:
                        result.append(detector)
                    else:
                        return detector
        return result

    def new_entry(self, entry="entry", program_name="pyFAI",
                  title="description of experiment",
                  force_time=None, force_name=False):
        """
        Create a new entry

        :param entry: name of the entry
        :param program_name: value of the field as string
        :param title: value of the field as string
        :param force_time: enforce the start_time (as string!)
        :param force_name: force the entry name as such, without numerical suffix.
        :return: the corresponding HDF5 group
        """

        if not force_name:
            nb_entries = len(self.get_entries())
            entry = "%s_%04i" % (entry, nb_entries)
        entry_grp = self.h5.require_group(entry)
        self.h5.attrs["default"] = entry
        entry_grp.attrs["NX_class"] = numpy.string_("NXentry")
        entry_grp["title"] = numpy.string_(title)
        entry_grp["program_name"] = numpy.string_(program_name)
        if force_time:
            entry_grp["start_time"] = numpy.string_(force_time)
        else:
            entry_grp["start_time"] = numpy.string_(get_isotime())
        self.to_close.append(entry_grp)
        return entry_grp

    def new_instrument(self, entry="entry", instrument_name="id00",):
        """
        Create an instrument in an entry or create both the entry and the instrument if
        """
        if not isinstance(entry, h5py.Group):
            entry = self.new_entry(entry)
        return self.new_class(entry, instrument_name, "NXinstrument")
#        howto external link
        # myfile['ext link'] = h5py.ExternalLink("otherfile.hdf5", "/path/to/resource")

    def new_class(self, grp, name, class_type="NXcollection"):
        """
        create a new sub-group with  type class_type
        :param grp: parent group
        :param name: name of the sub-group
        :param class_type: NeXus class name
        :return: subgroup created
        """
        sub = grp.require_group(name)
        sub.attrs["NX_class"] = numpy.string_(class_type)
        return sub

    def new_detector(self, name="detector", entry="entry", subentry="pyFAI"):
        """
        Create a new entry/pyFAI/Detector

        :param detector: name of the detector
        :param entry: name of the entry
        :param subentry: all pyFAI description of detectors should be in a pyFAI sub-entry
        """
        entry_grp = self.new_entry(entry)
        pyFAI_grp = self.new_class(entry_grp, subentry, "NXsubentry")
        pyFAI_grp["definition_local"] = numpy.string_("pyFAI")
        pyFAI_grp["definition_local"].attrs["version"] = numpy.string_(version)
        det_grp = self.new_class(pyFAI_grp, name, "NXdetector")
        return det_grp

    def get_class(self, grp, class_type="NXcollection"):
        """
        return all sub-groups of the given type within a group

        :param grp: HDF5 group
        :param class_type: name of the NeXus class
        """
        coll = [grp[name] for name in grp
                if isinstance(grp[name], h5py.Group) and
                self.get_attr(grp[name], "NX_class") == class_type]
        return coll

    def get_data(self, grp, class_type="NXdata"):
        """
        return all dataset of the the NeXus class NXdata

        :param grp: HDF5 group
        :param class_type: name of the NeXus class
        """
        coll = [grp[name] for name in grp
                if isinstance(grp[name], h5py.Dataset) and
                self.get_attr(grp[name], "NX_class") == class_type]
        return coll

    def deep_copy(self, name, obj, where="/", toplevel=None, excluded=None, overwrite=False):
        """
        perform a deep copy:
        create a "name" entry in self containing a copy of the object

        :param where: path to the toplevel object (i.e. root)
        :param  toplevel: firectly the top level Group
        :param excluded: list of keys to be excluded
        :param overwrite: replace content if already existing
        """
        if (excluded is not None) and (name in excluded):
            return
        if not toplevel:
            toplevel = self.h5[where]
        if isinstance(obj, h5py.Group):
            if name not in toplevel:
                grp = toplevel.require_group(name)
                for k, v in obj.attrs.items():
                        grp.attrs[k] = v
        elif isinstance(obj, h5py.Dataset):
            if name in toplevel:
                if overwrite:
                    del toplevel[name]
                    logger.warning("Overwriting %s in %s", toplevel[name].name, self.filename)
                else:
                    logger.warning("Not overwriting %s in %s", toplevel[name].name, self.filename)
                    return
            toplevel[name] = obj.value
            for k, v in obj.attrs.items():
                toplevel[name].attrs[k] = v

    @classmethod
    def get_attr(cls, dset, name, default=None):
        """Return the attribute of the dataset

        Handles the ascii -> unicode issue in python3 #275

        :param dset: a HDF5 dataset (or a group)
        :param name: name of the attribute
        :param default: default value to be returned
        :return: attribute value decoded in python3 or default
        """
        dec = default
        if name in dset.attrs:
            raw = dset.attrs[name]
            if (sys.version_info[0] > 2) and ("decode" in dir(raw)):
                dec = raw.decode()
            else:
                dec = raw
        return dec

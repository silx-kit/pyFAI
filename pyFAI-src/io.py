# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif
#
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



__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/11/2013"
__status__ = "beta"
__docformat__ = 'restructuredtext'
__doc__ = """

Module for "high-performance" writing in either 1D with Ascii , or 2D with FabIO 
or even nD with n varying from  2 to 4 using HDF5 

Stand-alone module which tries to offer interface to HDF5 via H5Py and 
capabilities to write EDF or other formats using fabio.

Can be imported without h5py but then limited to fabio & ascii formats.

TODO:
* add monitor to HDF5
"""
import sys
import os
import time
import threading
import logging
logger = logging.getLogger("pyFAI.io")
import types
import numpy
import posixpath
import json
try:
    import h5py
except ImportError:
    h5py = None
    logger.debug("h5py is missing")

import fabio
from . import units

def getIsoTime(forceTime=None):
    """
    @param forceTime: enforce a given time (current by default)
    @type forceTime: float
    @return: the current time as an ISO8601 string
    @rtype: string  
    """
    if forceTime is None:
        forceTime = time.time()
    localtime = time.localtime(forceTime)
    gmtime = time.gmtime(forceTime)
    tz_h = localtime.tm_hour - gmtime.tm_hour
    tz_m = localtime.tm_min - gmtime.tm_min
    return "%s%+03i:%02i" % (time.strftime("%Y-%m-%dT%H:%M:%S", localtime), tz_h, tz_m)

class Writer(object):
    """
    Abstract class for writers. 
    """
    CONFIG_ITEMS = ["filename", "dirname", "extension", "subdir", "hpath"]
    def __init__(self, filename=None, extension=None):
        """
        
        """
        self.filename = filename
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
        @param fai_cfg: configuration for worker
        @param lima_cfg: configuration for acquisition 
        """

        with self._sem:
            if fai_cfg is not None:
                self.fai_cfg = fai_cfg
            if lima_cfg is not None:
                self.lima_cfg = lima_cfg
            if self.filename is not None:
                dirname = os.path.dirname(self.filename)
                if not os.path.exists(dirname):
                    try:
                        os.makedirs(dirname)
                    except Exception as err:
                        logger.info("Problem while creating directory %s: %s" % (dirname, err))
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

        if type(json_config) in types.StringTypes:
            if os.path.isfile(json_config):
                config = json.load(open(json_config, "r"))
            else:
                 config = json.loads(json_config)
        else:
            config = dict(json_config)
        for k, v in  config.items():
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
        
        @param filename: name of the file
        @param hpath: name of the group: it will contain data (2-4D dataset), [tth|q|r] and pyFAI, group containing the configuration
        @param fast_scan_width: set it to define the width of 
        """
        Writer.__init__(self, filename)
        self.hpath = hpath
        self.fast_scan_width = None
        if fast_scan_width is not None:
            try:
                self.fast_scan_width = int(fast_scan_width)
            except:
                pass
        self.hdf5 = None
        self.group = None
        self.dataset = None
        self.pyFAI_grp = None
        self.radial_values = None
        self.azimuthal_values = None
        self.chunk = None
        self.shape = None
        self.ndim = None

    def __repr__(self):
        return "HDF5 writer on file %s:%s %sinitialized" % (self.filename, self.hpath, "" if self._initialized else "un")

    def init(self, fai_cfg=None, lima_cfg=None):
        """
        Initializes the HDF5 file for writing
        @param fai_cfg: the configuration of the worker as a dictionary
        """
        Writer.init(self, fai_cfg, lima_cfg)
        with self._sem:
            #TODO: this is Debug statement
            open("fai_cfg.json", "w").write(json.dumps(self.fai_cfg, indent=4))
            open("lima_cfg.json", "w").write(json.dumps(self.lima_cfg, indent=4))
            self.fai_cfg["nbpt_rad"] = self.fai_cfg.get("nbpt_rad", 1000)
            if h5py:
                try:
                    self.hdf5 = h5py.File(self.filename)
                except IOError: #typically a corrupted HDF5 file !
                    os.unlink(self.filename)
                    self.hdf5 = h5py.File(self.filename)
            else:
                logger.error("No h5py library, no chance")
                raise RuntimeError("No h5py library, no chance")
            self.group = self.hdf5.require_group(self.hpath)
            self.group.attrs["NX_class"] = "NXentry"
            self.pyFAI_grp = self.hdf5.require_group(posixpath.join(self.hpath, self.CONFIG))
            self.pyFAI_grp.attrs["desc"] = "PyFAI worker configuration"
            for key, value in self.fai_cfg.items():
                if value is None:
                    continue
                try:
                    self.pyFAI_grp[key] = value
                except:
                    print("Unable to set %s: %s" % (key, value))
                    self.close()
                    sys.exit(1)
            rad_name, rad_unit = str(self.fai_cfg.get("unit", "2th_deg")).split("_", 1)
            self.radial_values = self.group.require_dataset(rad_name, (self.fai_cfg["nbpt_rad"],), numpy.float32)
            if self.fai_cfg.get("nbpt_azim", 0) > 1:
                self.azimuthal_values = self.group.require_dataset("chi", (self.fai_cfg["nbpt_azim"],), numpy.float32)
                self.azimuthal_values.attrs["unit"] = "deg"
                self.radial_values.attrs["interpretation"] = "scalar"
                self.radial_values.attrs["long name"] = "Azimuthal angle"

            self.radial_values.attrs["unit"] = rad_unit
            self.radial_values.attrs["interpretation"] = "scalar"
            self.radial_values.attrs["long name"] = "diffraction radial direction"
            if self.fast_scan_width:
                self.fast_motor = self.group.require_dataset("fast", (self.fast_scan_width,) , numpy.float32)
                self.fast_motor.attrs["long name"] = "Fast motor position"
                self.fast_motor.attrs["interpretation"] = "scalar"
                self.fast_motor.attrs["axis"] = "1"
                self.radial_values.attrs["axis"] = "2"
                if self.azimuthal_values is not None:
                    chunk = 1, self.fast_scan_width, self.fai_cfg["nbpt_azim"], self.fai_cfg["nbpt_rad"]
                    self.ndim = 4
                    self.azimuthal_values.attrs["axis"] = "3"
                else:
                    chunk = 1, self.fast_scan_width, self.fai_cfg["nbpt_rad"]
                    self.ndim = 3
            else:
                self.radial_values.attrs["axis"] = "1"
                if self.azimuthal_values is not None:
                    chunk = 1, self.fai_cfg["nbpt_azim"], self.fai_cfg["nbpt_rad"]
                    self.ndim = 3
                    self.azimuthal_values.attrs["axis"] = "2"
                else:
                    chunk = 1, self.fai_cfg["nbpt_rad"]
                    self.ndim = 2

            if self.DATASET_NAME in self.group:
                del self.group[self.DATASET_NAME]
            shape = list(chunk)
            if self.lima_cfg.get("number_of_frames", 0) > 0:
                if self.fast_scan_width is not None:
                    size[0] = 1 + self.lima_cfg["number_of_frames"] // self.fast_scan_width
                else:
                    size[0] = self.lima_cfg["number_of_frames"]
            self.dataset = self.group.require_dataset(self.DATASET_NAME, shape, dtype=numpy.float32, chunks=chunk,
                                                      maxshape=(None,) + chunk[1:])
            if self.fai_cfg.get("nbpt_azim", 0) > 1:
                self.dataset.attrs["interpretation"] = "image"
            else:
                self.dataset.attrs["interpretation"] = "spectrum"
            self.dataset.attrs["signal"] = "1"
            self.chunk = chunk
            self.shape = chunk
            name = "Mapping " if self.fast_scan_width else "Scanning "
            name += "2D" if self.fai_cfg.get("nbpt_azim", 0) > 1 else "1D"
            name += " experiment"
            self.group["title"] = name
            self.group["program"] = "PyFAI"
            self.group["start_time"] = getIsoTime()



    def flush(self, radial=None, azimuthal=None):
        """
        Update some data like axis units and so on.
        
        @param radial: position in radial direction
        @param  azimuthal: position in azimuthal direction
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
        with self._sem:
            if self.hdf5:
                self.flush()
                self.hdf5.close()
                self.hdf5 = None

    def write(self, data, index=0):
        """
        Minimalistic method to limit the overhead.
        """
        with self._sem:
            if self.dataset is None:
                logger.warning("Writer not initialized !")
                return
            if self.azimuthal_values is None:
                data = data[:, 1] #take the second column only aka I
            if self.fast_scan_width:
                index0, index1 = (index // self.fast_scan_width, index % self.fast_scan_width)
                if index0 >= self.dataset.shape[0]:
                    self.dataset.resize(index0 + 1, axis=0)
                self.dataset[index0, index1] = data
            else:
                if index >= self.dataset.shape[0]:
                    self.dataset.resize(index + 1, axis=0)
                self.dataset[index] = data

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
            headerLst = ["", "== Detector =="]
            if "detector" in self.fai_cfg:
                headerLst.append("Detector: %s" % self.fai_cfg["detector"])
            if "splineFile" in self.fai_cfg:
                headerLst.append("SplineFile: %s" % self.fai_cfg["splineFile"])
            if  "pixel1" in self.fai_cfg:
                headerLst.append("PixelSize: %.3e, %.3e m" % (self.fai_cfg["pixel1"], self.fai_cfg["pixel2"]))
            if "mask_file" in self.fai_cfg:
                headerLst.append("MaskFile: %s" % (self.fai_cfg["mask_file"]))

            headerLst.append("== pyFAI calibration ==")
            if "poni1" in self.fai_cfg:
                headerLst.append("PONI: %.3e, %.3e m" % (self.fai_cfg["poni1"], self.fai_cfg["poni2"]))
            if "dist" in self.fai_cfg:
                headerLst.append("Distance Sample to Detector: %s m" % self.fai_cfg["dist"])
            if "rot1" in self.fai_cfg:
                headerLst.append("Rotations: %.6f %.6f %.6f rad" % (self.fai_cfg["rot1"], self.fai_cfg["rot2"], self.fai_cfg["rot3"]))
            if "wavelength" in self.fai_cfg:
                headerLst.append("Wavelength: %s" % self.fai_cfg["wavelength"])
            if "dark_current" in self.fai_cfg:
                headerLst.append("Dark current: %s" % self.fai_cfg["dark_current"])
            if "flat_field" in self.fai_cfg:
                headerLst.append("Flat field: %s" % self.fai_cfg["flat_field"])
            if "polarization_factor" in self.fai_cfg:
                headerLst.append("Polarization factor: %s" % self.fai_cfg["polarization_factor"])
            headerLst.append("")
            if "do_poisson" in self.fai_cfg:
                headerLst.append("%14s %14s %s" % (self.fai_cfg["unit"], "I", "sigma"))
            else:
                headerLst.append("%14s %14s" % (self.fai_cfg["unit"], "I"))
#            headerLst.append("")
            self.header = os.linesep.join([""] + ["# " + i for i in headerLst] + [""])
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
            logger.warning("Output directory: %s does not exist,creating it" % self.directory)
            try:
                os.makedirs(self.directory)
            except Exception as error:
                logger.info("Problem while creating directory %s: %s" % (self.directory, error))


    def write(self, data, index=0):
        filename = os.path.join(self.directory, self.prefix + (self.index_format % (self.start_index + index)) + self.extension)
        if filename:
            with open(filename, "w") as f:
                f.write("# Processing time: %s%s" % (getIsoTime(), self.header))
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

    def __repr__(self):
        return "Image writer on file %s" % (self.filename)

    def init(self, fai_cfg=None, lima_cfg=None):
        """
        Creates the directory that will host the output file(s) 
        
        """
        Writer.init(self, fai_cfg, lima_cfg)
        with self._sem:
            dim1_unit = units.to_unit(fai_cfg["unit"])
            header_keys = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3",
                           "chi_min", "chi_max",
                           dim1_unit.REPR + "_min",
                           dim1_unit.REPR + "_max",
                           "pixelX", "pixelY",
                           "dark", "flat", "polarization_factor", "normalization_factor"]
            header = {"dist": str(fai_cfg.get("dist")),
                      "poni1": str(fai_cfg.get("poni1")),
                      "poni2": str(fai_cfg.get("poni2")),
                      "rot1": str(fai_cfg.get("rot1")),
                      "rot2": str(fai_cfg.get("rot1")),
                      "rot3": str(fai_cfg.get("dist")),
                      "chi_min": str(fai_cfg.get("dist")),
                      "chi_max": str(fai_cfg.get("dist")),
                      dim1_unit.REPR + "_min": str(fai_cfg.get("dist")),
                      dim1_unit.REPR + "_max": str(fai_cfg.get("dist")),
                      "pixelX": str(fai_cfg.get("dist")),  # this is not a bug ... most people expect dim1 to be X
                      "pixelY": str(fai_cfg.get("dist")),  # this is not a bug ... most people expect dim2 to be Y
                      "polarization_factor": str(fai_cfg.get("dist")),
                      "normalization_factor":str(fai_cfg.get("dist")),
                      }

            if self.splineFile:
                header["spline"] = str(self.splineFile)

            if dark is not None:
                if self.darkfiles:
                    header["dark"] = self.darkfiles
                else:
                    header["dark"] = 'unknown dark applied'
            if flat is not None:
                if self.flatfiles:
                    header["flat"] = self.flatfiles
                else:
                    header["flat"] = 'unknown flat applied'
            f2d = self.getFit2D()
            for key in f2d:
                header["key"] = f2d[key]
        self.prefix = prefix
        self.index_format = index_format
        self.start_index = start_index
        if not self.subdir:
            self.directory = directory
        elif self.subdir.startswith("/"):
            self.directory = self.subdir
        else:
            self.directory = os.path.join(directory, self.subdir)
        if not os.path.exists(self.directory):
            logger.warning("Output directory: %s does not exist,creating it" % self.directory)
            try:
                os.makedirs(self.directory)
            except Exception as error:
                logger.info("Problem while creating directory %s: %s" % (self.directory, error))


    def write(self, data, index=0):
        filename = os.path.join(self.directory, self.prefix + (self.index_format % (self.start_index + index)) + self.extension)
        if filename:
            with open(filename, "w") as f:
                f.write("# Processing time: %s%s" % (getIsoTime(), self.header))
                numpy.savetxt(f, data)

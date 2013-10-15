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
__date__ = "15/10/2013"
__status__ = "beta"
__docformat__ = 'restructuredtext'
__doc__ = """
Stand-alone module which tries to offer interface to HDF5 via H5Py.

Can be imported without h5py but completely useless.
"""
import os
import time
import threading
import logging
logger = logging("pyFAI.hdf5")
import types
import numpy
import posixpath

try:
    import h5py
except ImportError:
    h5py = None
    logger.debug("h5py is missing")

class HDF5Writer(object):
    """
    Class allowing to write HDF5 Files.
    
    """
    CONFIG = "pyFAI"
    DATA = "data"
    def __init__(self, filename, hpath="data", fast_scan_width=None):
        """
        Constructor of an HDF5 writer:
        
        @param filename: name of the file
        @param hpath: name of the group: it will contain data (2-4D dataset), [tth|q|r] and pyFAI, group containing the configuration
        @param fast_scan_width: set it to define the width of 
        """
        self.filename = filename
        self.hpath = hpath
        self.fast_scan_width = fast_scan_width
        self.hdf5 = None
        self.group = None
        self.dataset = None
        self.pyFAI_grp = None
        self.radial_values = None
        self.azimuthal_values = None
        self.chunk = None
        self.shape = None
        self.ndim = None
    def init(self, config={}):
        """
        Initializes the HDF5 file for writing
        @param config: the configuration of the worker as a dictionnary
        """
        dirname = os.path.dirname(self.filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname, mode)
        if h5py:
            self.hdf5 = h5py.File(self.filename)
        else:
            logger.error("No h5py library, no chance")
            raise RuntimeError("No h5py library, no chance")
        self.group = self.hdf5.require_group(self.hpath)
        self.group.attrs["NX_class"] = "NXentry"
        self.pyFAI_grp = self.hdf5.require_group(posixpath.join(self.hpath, self.CONFIG))
        self.pyFAI_grp.attrs["desc"] = "PyFAI worker configuration"
        for key, value in config.items():
            self.pyFAI_grp[key] = value
        if config.get("nbpt_azim", 0) > 1:
            self.azimuthal_values = self.group.require_dataset("chi", config["nbpt_azim"], numpy.float32)
            self.azimuthal_values.attrs["unit"] = "deg"
            self.radial_values.attrs["interpretation"] = "scalar"
            self.radial_values.attrs["long name"] = "Azimuthal angle"

        rad_name,rad_unit=str(config.get("unit","2th_deg")).split("_",1)
        self.radial_values = self.group.require_dataset(rad_name, config.get("nbpt_rad", 1000), numpy.float32)
        self.radial_values.attrs["unit"] = rad_unit
        self.radial_values.attrs["interpretation"] = "scalar"
        self.radial_values.attrs["long name"] = "diffraction radial direction"
        if self.fast_scan_width:
            self.fast_motor = self.group.require_dataset("fast", self.fast_scan_width , numpy.float32)
            self.fast_motor.attrs["long name"] = "Fast motor position"
            self.fast_motor.attrs["interpretation"] = "scalar"
            self.fast_motor.attrs["axis"] = "1"
            self.radial_values.attrs["axis"] = "2"
            if self.azimuthal_values is not None:
                chunk = 1, self.fast_scan_width, config["nbpt_azim"], config["nbpt_rad"]
                self.ndim = 4
                self.azimuthal_values.attrs["axis"] = "3"
            else:
                chunk = 1, self.fast_scan_width, config["nbpt_rad"]
                self.ndim = 3
        else:
            self.radial_values.attrs["axis"] = "1"
            if self.azimuthal_values is not None:
                chunk = 1, config["nbpt_azim"], config["nbpt_rad"]
                self.ndim = 3
                self.azimuthal_values.attrs["axis"] = "2"
            else:
                chunk = 1, config["nbpt_rad"]
                self.ndim = 2

        if self.DATA in self.group:
            del self.group[self.DATA]
        self.dataset = self.group.require_dataset(self.data, chunk, dtype=numpy.float32, chunks=chunk,
                                                  maxshape=(None,) + chunk[1:])
        if config.get("nbpt_azim", 0) > 1:
            self.dataset.attrs["interpretation"] = "image"
        else:
            self.dataset.attrs["interpretation"] = "spectrum"
        self.dataset.attrs["signal"] = "1"
        self.chunk = chunk
        self.shape = chunk


    def flush(self, radial=None, azimuthal=None):
        """
        Update some data like axis units and so on.
        
        @param radial: position in radial direction
        @param  azimuthal: position in azimuthal direction
        """
        if not self.hdf5:
            raise RuntimeError('No opened file')
        if radial:
            if radial.shape == self.radial_values.shape:
                self.radial_values[:] = radial
            else:
                logger.warning("Unable to assign radial axis position")
        if azimuthal:
            if azimuthal.shape == self.azimuthal_values.shape:
                self.azimuthal_values[:] = azimuthal
            else:
                logger.warning("Unable to assign azimuthal axis position")
        self.hdf5.flush()

    def close(self):
        if self.hdf5:
            self.flush()
            self.hdf5.close()
            self.hdf5 = None

    def write(self, data, index=0):
        """
        Minimalistic method to limit the overhead.
        """
        if self.fast_scan_width:
            index0, index1 = (index // self.fast_scan_width, index % self.fast_scan_width)
            if index[0] >= self.dataset.shape[0]:
                self.dataset.resize(index[0] + 1, axis=0)
            self.dataset[index0, index1] = data
        else:
            if index >= self.dataset.shape[0]:
                self.dataset.resize(index + 1, axis=0)
            self.dataset[index] = data

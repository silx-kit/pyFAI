#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
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
__date__ = "06/11/2012"
__status__ = "beta"

import os, logging
logger = logging.getLogger("pyFAI.opencl")

try:
    import pyopencl
except ImportError:
    logger.error("Unable to import pyOpenCl. Please install it from: http://pypi.python.org/pypi/pyopencl")
    pyopencl = None


class Device(object):
    """
    Simple class that contains the structure of an OpenCL device
    """
    def __init__(self, name=None, type=None, version=None, driver_version=None, extensions=None, memory=None):
        self.name = name
        self.type = type
        self.version = version
        self.driver_version = driver_version
        self.extensions = extensions.split()
        self.memory = memory

    def __repr__(self):
        return "%s" % self.name

class Platform(object):
    """
    Simple class that contains the structure of an OpenCL platform
    """
    def __init__(self, name=None, vendor=None, version=None, extensions=None):
        self.name = name
        self.vendor = vendor
        self.version = version
        self.extensions = extensions.split()
        self.devices = []

    def __repr__(self):
        return "%s" % self.name

    def add_device(self, device):
        self.devices.append(device)


class OpenCL(object):
    """
    Simple class that wraps the structure ocl_tools_extended.h
    """
    platforms = []
    if pyopencl:
        for platform in pyopencl.get_platforms():
            pypl = Platform(platform.name, platform.vendor, platform.version, platform.extensions)
            for device in platform.get_devices():
                ####################################################
                # Nvidia does not report int64 atomics (we are using) ...
                # this is a hack around as any nvidia GPU with double-precision supports int64 atomics
                ####################################################
                extensions = device.extensions
                if (pypl.vendor == "NVIDIA Corporation") and ('cl_khr_fp64' in extensions):
                                extensions += ' cl_khr_int64_base_atomics cl_khr_int64_extended_atomics'
                if device.type == 4:
                    devtype = "GPU"
                elif device.type == 2:
                    devtype = "CPU"
                elif device.type == 8:
                    devtype = "ACC"
                else:
                    devtype = "DEF"

                pydev = Device(device.name, devtype, device.version, device.driver_version, extensions, device.global_mem_size)
                pypl.add_device(pydev)

            platforms.append(pypl)
        del platform, device, pypl, devtype, extensions, pydev



    def __repr__(self):
        out = ["OpenCL devices:"]
        for platformid, platform in enumerate(self.platforms):
            out.append("[%s] %s: " % (platformid, platform.name) + ", ".join(["(%s,%s) %s" % (platformid, deviceid, dev.name) for deviceid, dev in enumerate(platform.devices)]))
        return os.linesep.join(out)


    def select_device(self, type="ALL", memory=None, extensions=[]):
        """
        @param type: "gpu" or "cpu" or "all" ....
        @param memory: minimum amount of memory (int)
        @param extensions: list of extensions to be present

        """
        type = type.upper()
        for platformid, platform in enumerate(self.platforms):
            for deviceid, device in enumerate(platform.devices):
                if (type in ["ALL", "DEF"]) or (device.type == type):
                    if (memory is None) or (memory <= device.memory):
                        found = True
                        for ext in extensions:
                            if ext not in device.extensions:
                                found = False
                        if found:
                            return platformid, deviceid
        return None

if pyopencl:
    ocl = OpenCL()
else:
    ocl = None

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
                devtype = pyopencl.device_type.to_string(device.type)
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

    def create_context(self, devicetype="ALL", useFp64=False, platformid=None, deviceid=None):
        """ 
        Choose a device and initiate a context. 
        
        Devicetypes can be GPU,gpu,CPU,cpu,DEF,ACC,ALL.
        Suggested are GPU,CPU. 
        For each setting to work there must be such an OpenCL device and properly installed.
        E.g.: If Nvidia driver is installed, GPU will succeed but CPU will fail. The AMD SDK kit is required for CPU via OpenCL.
        @param devicetype: string in ["cpu","gpu", "all", "acc"]
        @param useFp64: boolean specifying if double precision will be used
        @param platformid: integer
        @param devid: integer
        @return: OpenCL context on the selected device
        """
        if (platformid is not None) and (deviceid is not None):
            platformid = int(platformid)
            deviceid = int(deviceid)
        else:
            if useFp64:
                ids = ocl.select_device(type=devicetype, extensions=["cl_khr_int64_base_atomics"])
            else:
                ids = ocl.select_device(type=devicetype)
            if ids:
                platformid = ids[0]
                deviceid = ids[1]
        if (platformid is not None) and  (deviceid is not None):
            ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
        else:
            logger.warn("Last chance to get an OpenCL device ... probably not the one requested")
            ctx = pyopencl.create_some_context(interactive=False)
        return ctx

if pyopencl:
    ocl = OpenCL()
else:
    ocl = None

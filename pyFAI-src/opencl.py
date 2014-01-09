#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
__date__ = "21/11/2013"
__status__ = "stable"

import os, logging
logger = logging.getLogger("pyFAI.opencl")

try:
    import pyopencl
#    from pyFAI.opencl import ocl
except ImportError:
    logger.warning("Unable to import pyOpenCl. Please install it from: http://pypi.python.org/pypi/pyopencl")
    pyopencl = None

FLOP_PER_CORE = { "GPU": 64, # GPU, Fermi at least perform 64 flops per cycle/multicore, G80 were at 24 or 48 ...
                  "CPU": 4,  # CPU, at least intel's have 4 operation per cycle
                  "ACC": 8}  # ACC: the Xeon-phi (MIC) appears to be able to process 8 Flops per hyperthreaded-core
NVIDIA_FLOP_PER_CORE = {(1, 0): 24,  # Guessed !
                         (1, 1): 24,  # Measured on G98 [Quadro NVS 295]
                         (1, 2): 24,  # Guessed !
                         (1, 3): 24,  # measured on a GT285 (GT200)
                         (2, 0): 64,  # Measured on a 580 (GF110)
                         (2, 1): 96,  # Measured on Quadro2000 GF106GL
                         (3, 0): 384,  # Guessed!
                         (3, 5): 384}  # Measured on K20
AMD_FLOP_PER_CORE = 160  # Measured on a M7820 10 core, 700MHz 1120GFlops

class Device(object):
    """
    Simple class that contains the structure of an OpenCL device
    """
    def __init__(self, name="None", dtype=None, version=None, driver_version=None,
                 extensions="", memory=None, available=None,
                 cores=None, frequency=None, flop_core=None, idx=0):
        self.name = name.strip()
        self.type = dtype
        self.version = version
        self.driver_version = driver_version
        self.extensions = extensions.split()
        self.memory = memory
        self.available = available
        self.cores = cores
        self.frequency = frequency
        self.id = idx
        if not flop_core:
            flop_core = FLOP_PER_CORE.get(dtype, 1)
        if cores and frequency:
            self.flops = cores * frequency * flop_core
        else:
            self.flops = flop_core


    def __repr__(self):
        return "%s" % self.name

class Platform(object):
    """
    Simple class that contains the structure of an OpenCL platform
    """
    def __init__(self, name="None", vendor="None", version=None, extensions=None, idx=0):
        self.name = name.strip()
        self.vendor = vendor.strip()
        self.version = version
        self.extensions = extensions.split()
        self.devices = []
        self.id = idx

    def __repr__(self):
        return "%s" % self.name

    def add_device(self, device):
        self.devices.append(device)

    def get_device(self, key):
        """
        Return a device according to key

        @param key: identifier for a device, either it's id (int) or it's name
        @type key: int or str
        """
        out = None
        try:
            devid = int(key)
        except ValueError:
            for a_dev in self.devices:
                if a_dev.name == key:
                    out = a_dev
        else:
            if len(self.devices) > devid > 0:
                out = self.devices[devid]
        return out


class OpenCL(object):
    """
    Simple class that wraps the structure ocl_tools_extended.h
    """
    platforms = []
    if pyopencl:
        for idx, platform in enumerate(pyopencl.get_platforms()):
            pypl = Platform(platform.name, platform.vendor, platform.version, platform.extensions, idx)
            for idd, device in enumerate(platform.get_devices()):
                ####################################################
                # Nvidia does not report int64 atomics (we are using) ...
                # this is a hack around as any nvidia GPU with double-precision supports int64 atomics
                ####################################################
                extensions = device.extensions
                if (pypl.vendor == "NVIDIA Corporation") and ('cl_khr_fp64' in extensions):
                                extensions += ' cl_khr_int64_base_atomics cl_khr_int64_extended_atomics'
                devtype = pyopencl.device_type.to_string(device.type).upper()
                if len(devtype) > 3:
                    devtype = devtype[:3]
                if (pypl.vendor == "NVIDIA Corporation") and (devtype == "GPU") and "compute_capability_major_nv" in dir(device):
                    comput_cap = device.compute_capability_major_nv, device.compute_capability_minor_nv
                    flop_core = NVIDIA_FLOP_PER_CORE.get(comput_cap, min(NVIDIA_FLOP_PER_CORE.values()))
                elif (pypl.vendor == "Advanced Micro Devices, Inc.") and (devtype == "GPU"):
                    flop_core = AMD_FLOP_PER_CORE
                elif devtype == "CPU":
                    flop_core = FLOP_PER_CORE.get(devtype, 1)
                else:
                     flop_core = 1
                pydev = Device(device.name, devtype, device.version, device.driver_version, extensions,
                               device.global_mem_size, bool(device.available), device.max_compute_units,
                               device.max_clock_frequency, flop_core, idd)
                pypl.add_device(pydev)
            platforms.append(pypl)
        del platform, device, pypl, devtype, extensions, pydev


    def __repr__(self):
        out = ["OpenCL devices:"]
        for platformid, platform in enumerate(self.platforms):
            out.append("[%s] %s: " % (platformid, platform.name) + ", ".join(["(%s,%s) %s" % (platformid, deviceid, dev.name) for deviceid, dev in enumerate(platform.devices)]))
        return os.linesep.join(out)

    def get_platform(self, key):
        """
        Return a platform according

        @param key: identifier for a platform, either an Id (int) or it's name
        @type key: int or str
        """
        out = None
        try:
            platid = int(key)
        except ValueError:
            for a_plat in self.platforms:
                if a_plat.name == key:
                    out = a_plat
        else:
            if len(self.platforms) > platid > 0:
                out = self.platforms[platid]
        return out

    def select_device(self, dtype="ALL", memory=None, extensions=[], best=True, **kwargs):
        """
        Select a device based on few parameters (at the end, keep the one with most memory)

        @param type: "gpu" or "cpu" or "all" ....
        @param memory: minimum amount of memory (int)
        @param extensions: list of extensions to be present
        @param best: shall we look for the
        """
        if "type" in kwargs:
            dtype = kwargs["type"].upper()
        else:
            dtype = dtype.upper()
        if len(dtype) > 3:
            dtype = dtype[:3]
        best_found = None
        for platformid, platform in enumerate(self.platforms):
            for deviceid, device in enumerate(platform.devices):
                if (dtype in ["ALL", "DEF"]) or (device.type == dtype):
                    if (memory is None) or (memory <= device.memory):
                        found = True
                        for ext in extensions:
                            if ext not in device.extensions:
                                found = False
                        if found:
                            if not best:
                                return platformid, deviceid
                            else:
                                if not best_found:
                                    best_found = platformid, deviceid, device.flops
                                elif best_found[2] < device.flops:
                                    best_found = platformid, deviceid, device.flops
        if best_found:
            return  best_found[0], best_found[1]

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
                ids = ocl.select_device(dtype=devicetype)
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


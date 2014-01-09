#!/usr/bin/env python
#coding: utf8

from __future__ import division, print_function, with_statement

"""
Tango device server for setting up pyFAI distortion correction.
This is a stand-alond device server


"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/11/2013"
__status__ = "development"
__docformat__ = 'restructuredtext'

import os
import sys
import threading
import logging
import gc
import time
import traceback
logger = logging.getLogger("pyfai.distortionDS")
# set loglevel at least at INFO
if logger.getEffectiveLevel() > logging.INFO:
    logger.setLevel(logging.INFO)
from os.path import dirname
cwd = dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(cwd, "build", "lib.linux-x86_64-2.6"))
import pyFAI
if sys.version < (2, 7):
    from pyFAI.argparse import ArgumentParser
else:
    from argparse import ArgumentParser

try:
    from pyFAI.fastcrc import crc32
except ImportError:
    from zlib import crc32

import fabio
try:
    import pyopencl
    import pyFAI.ocl_azim_lut
except ImportError:
    pyopencl = None
    logger.warning("Unable to import pyopencl, will use OpenMP (if available)")
import PyTango
import numpy
if sys.version > (3, 0):
    from queue import Queue
else:
    from Queue import Queue
try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
except ImportError:
    print("No socket opened for debugging -> please install rfoo if you want to debug online")
else:
    print("rfoo installed, you can debug online with rconsole or rfoo-rconsole")

class DistortionDS(PyTango.Device_4Impl) :
    """
    Tango device server exposed to configure online distortion correction
    """
    DISTORTION_TASK_NAME = 'DistortionCorrectionTask'
    def __init__(self, cl, name):
        self.__Task = None
        self.get_device_properties(self.get_device_class())
        BasePostProcess.__init__(self, cl, name)
        DistortionCorrectionDeviceServer.init_device(self)

        self.__pyFAISink = None
        self.__subdir = None
        self.__extension = None
        self.__spline_filename = None
        self.__darkcurrent_filename = None
        self.__flatfield_filename = None


    def set_state(self, state) :
        """
        Switch on or off the LImA plugin
        """
        if(state == PyTango.DevState.OFF) :
            if (self.__Task):
                self.__Task = None
        elif(state == PyTango.DevState.ON) :
            if not self.__Task:
                try:
                    pass
                except:

                    traceback.print_exc()
                    return
        PyTango.Device_4Impl.set_state(self, state)

    def setDarkcurrentImage(self, filepath):
        """
        @param imagefile: filename with the path to the dark image
        """

        self.__darkcurrent_filename = filepath
        if(self.__pyFAISink) :
            self.__pyFAISink.setDarkcurrentFile(filepath)
        self.push_change_event("DarkCurrent", filepath)

    def setFlatfieldImage(self, filepath):
        """
        @param filepath: filename with the path to the flatfield image
        """
        self.__flatfield_filename = filepath
        if(self.__pyFAISink) :
            self.__pyFAISink.setFlatfieldFile(filepath)
        self.push_change_event("FlatField", filepath)

    def setSplineFile(self, filepath):
        """
        @param filepath: filename with the path to the spline distortion file
        """

        self.__spline_filename = filepath
        if(self.__pyFAISink) :
            self.__pyFAISink.setSplineFile(filepath)
        self.push_change_event("SplineFile", filepath)

    def setProcessedSubdir(self, filepath):
        """
        Directory  (relative or absolute) for processed data
        """
        self.__subdir = filepath
        if self.__pyFAISink:
            self.__pyFAISink.setSubdir(self.__subdir)

    def setProcessedExt(self, ext):
        """
        Extension for prcessed data files
        """
        self.__extension = ext
        if self.__pyFAISink:
            self.__pyFAISink.setExtension(self.__extension)

    def Reset(self) :
        """
        Force the reinitialization
        """
        self.__pyFAISink = SinkPyFAI(splinefile=self.__spline_filename,
                                     darkfile=self.__darkcurrent_filename,
                                     flatfile=self.__flatfield_filename)
        self.__Task.setSinkTask(self.__pyFAISink)

    def read_Parameters(self, attr):
        """
        Called  when reading the "Parameters" attribute
        """
#        logger.warning("in AzimuthalIntegrationDeviceServer.read_Parameters")
        if self.__pyFAISink:
            attr.set_value(self.__pyFAISink.__repr__())
        else:
            attr.set_value("No pyFAI Sink processlib active for the moment")

    def read_SplineFile(self, attr):
        attr.set_value(str(self.__spline_filename))

    def read_DarkCurrent(self, attr):
        attr.set_value(str(self.__darkcurrent_filename))

    def read_FlatField(self, attr):
        attr.set_value(str(self.__flatfield_filename))

class DistortionDSClass(PyTango.DeviceClass) :
        #        Class Properties
    class_property_list = {
        }


    #    Device Properties
    device_property_list = {
        }


    #    Command definitions
    cmd_list = {
        'setDarkcurrentImage':
        [[PyTango.DevString, "Full path of darkCurrent image file"],
         [PyTango.DevVoid, ""]],

        'setFlatfieldImage':
        [[PyTango.DevString, "Full path of flatField image file"],
         [PyTango.DevVoid, ""]],

        'setSplineFile':
        [[PyTango.DevString, "Full path of spline distortion file"],
         [PyTango.DevVoid, ""]],

        'setProcessedSubdir':
        [[PyTango.DevString, "Sub-directory with processed data"],
         [PyTango.DevVoid, ""]],

        'setProcessedExt':
        [[PyTango.DevString, "Extension of processed data files"],
         [PyTango.DevVoid, ""]],

        'Start':
        [[PyTango.DevVoid, ""],
         [PyTango.DevVoid, ""]],
        'Stop':
        [[PyTango.DevVoid, ""],
         [PyTango.DevVoid, ""]],
        'Reset':
        [[PyTango.DevVoid, ""],
         [PyTango.DevVoid, ""]],
        }


    #    Attribute definitions
    attr_list = {
        'RunLevel':
            [[PyTango.DevLong,
            PyTango.SCALAR,
            PyTango.READ_WRITE]],
        'SplineFile':
            [[PyTango.DevString,
            PyTango.SCALAR,
            PyTango.READ]],
        'DarkCurrent':
            [[PyTango.DevString,
            PyTango.SCALAR,
            PyTango.READ]],
        'FlatField':
            [[PyTango.DevString,
            PyTango.SCALAR,
            PyTango.READ]],
        'Parameters':
            [[PyTango.DevString,
            PyTango.SCALAR,
            PyTango.READ]],

#        'delete_dark_after_read':
#        [[PyTango.DevBoolean,
#          PyTango.SCALAR,
#          PyTango.READ_WRITE]],
        }
#------------------------------------------------------------------
#    AzimuthalIntegratorDeviceServerClass Constructor
#------------------------------------------------------------------
    def __init__(self, name):
        PyTango.DeviceClass.__init__(self, name)
        self.set_type(name)

_control_ref = None
def set_control_ref(control_class_ref):
    global _control_ref
    _control_ref = control_class_ref

def get_tango_specific_class_n_device() :
    return DistortionCorrectionDeviceServerClass, DistortionCorrectionDeviceServer


if __name__ == '__main__':
    logger.info("Starting PyFAI distortion correction device server")
    ltangoParam = ["DistortionDS"]
    parser = ArgumentParser(description="PyFAI geometry distortion correction Tango device server")
    parser.add_argument("-d", "--debug", action="store_true", dest="verbose", default=False,
                      help="switch to verbose/debug mode into python (-v for the tango part)")
#    parser.add_argument("-n","--ncpu", dest="ncpu", type=int, default=1,
#                        help="number of worker to use in parallel" )
    parser.add_argument("to_tango", nargs='*')
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    ltangoParam += args.to_tango
    try:
        logger.debug("Tango parameters: %s" % ltangoParam)
        py = PyTango.Util(ltangoParam)
        py.add_TgClass(DistortionDSClass, DistortionDS, 'DistortionDS')
        U = py.instance() #PyTango.Util.instance()
        U.server_init()
        U.server_run()
    except PyTango.DevFailed as e:
        EDVerbose.ERROR('PyTango --> Received a DevFailed exception: %s' % e)
        sys.exit(-1)
    except Exception as e:
        EDVerbose.ERROR('PyTango --> An unforeseen exception occurred....%s' % e)
        sys.exit(-1)

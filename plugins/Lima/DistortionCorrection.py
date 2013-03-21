#!/usr/bin/env python
#coding: utf8
"""
Tango device server for setting up pyFAI azimuthal integrator in a LImA ProcessLib.

Destination path:
Lima/tango/plugins/DistortionCorrection
"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/03/2013"
__status__ = "beta"
__docformat__ = 'restructuredtext'

import os
import sys
import threading
import logging
logger = logging.getLogger("lima.tango.pyfai")
# set loglevel at least at INFO
if logger.getEffectiveLevel() > logging.INFO:
    logger.setLevel(logging.INFO)
from os.path import dirname
cwd = dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(cwd, "build", "lib.linux-x86_64-2.6"))
import pyFAI
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
from Lima import Core
from Utils import BasePostProcess

class PyFAISink(Core.Processlib.SinkTaskBase):
    """
    This is a processlib Sink: it takes an image as input and writes a file to disk but returns nothing
    """
    def __init__(self, splinefile=None, darkfile=None, flatfile=None, extraheader=None):
        """
        @param splinefile: File with the description of the distortion as a cubic spline
        @param darkfile: image with the dark current
        @param flatfile: image with the flat field correction
        @param extraheader: dictionary with additional static header for EDF files
        """
        Core.Processlib.SinkTaskBase.__init__(self)

        self._sem = threading.Semaphore()
        if extraheader:
            self.header = extraheader
        else:
            self.header = {}
        self.splinefile = self.dis = self.det = self.ocl_integrator = None
        self.darkfile = self.darkcurrent = self.darkcurrent_crc = None
        self.flatfile = self.flatfield = self.flatfield_crc = None
        self.subdir = ""
        self.extension = ".cor"
        self.binning = (1,1)
        self.shape = (None, None)
        self.setSplineFile(splinefile)
        self.setDarkcurrentFile(darkfile)
        self.setFlatfieldFile(flatfile)

    def __repr__(self):
        lstout = ["Spline filename: %s" % self.splinefile,
                "Dark current image: %s" % self.darkfile,
                "Flat field image: %s" % self.flatfile,
                "Binning factor: %s, %s"%self.binning,
                "Image shape: %s, %s" % self.shape,
                "Directory: %s, \tExtension: %s" % (self.subdir, self.extension)]
        return os.linesep.join(lstout)

    def process(self, data) :
        """
        Process a frame
        @param data: a LImA frame with member .buffer (a numpy array) and .frameNumber (an int)
        """
        ctControl = _control_ref()
        saving = ctControl.saving()
        sav_parms = saving.getParameters()
        directory = sav_parms.directory
        if not self.subdir:
            directory = sav_parms.directory
        elif self.subdir.startswith("/"):
            directory = self.subdir
        else:
            directory = os.path.join(sav_parms.directory, self.subdir)

        if not os.path.exists(directory):
            logger.error("Ouput directory does not exist !!!  %s" % directory)

            try:
                os.makedirs(directory)
            except: #No luck withthreads
                pass
        if not self.extension:
            self.extension = ".cor"
        elif (self.extension == ".edf") and (self.subdir == ""):
            logger.warning("Modify extenstion to .cor.edf to prevent overwriting raw files !!!")
            self.extension = ".cor.edf"

        prefix = sav_parms.prefix
        nextNumber = sav_parms.nextNumber
        indexFormat = sav_parms.indexFormat
        output = os.path.join(directory, prefix + indexFormat % (nextNumber + data.frameNumber) + self.extension)
        header = self.header.copy()
        header["index"] = nextNumber + data.frameNumber
        if pyopencl and self.ocl_integrator:
            out = self.ocl_integrator.integrate(data.buffer, dark=self.darkcurrent, flat=self.flatfield,
                                                dark_checksum=self.darkcurrent_crc, flat_checksum=self.flatfield_crc)[1]
        else:
            data = numpy.ascontiguousarray(data.buffer, dtype=numpy.float32)
            if self.darkcurrent is not None:
                data -= self.darkcurrent
            if self.flatfield is not None:
                data /= self.flatfield
            if self.dis:
                out = self.dis.correct(data)
            else:
                out = data
        edf = fabio.edfimage.edfimage(data=out, header=header)
        edf.write(output)



    def setDarkcurrentFile(self, imagefile):
        """
        @param imagefile: filename with the path to the dark image
        """
        with self._sem:
            if imagefile:
                self.darkfile = imagefile
                try:
                    self.darkcurrent = numpy.ascontiguousarray(fabio.open(imagefile).data, numpy.float32)
                except Exception as error:
                    logger.warning("setDarkcurrentFile: Unable to read file %s: %s" % (imagefile, error))
                else:
                    self.darkcurrent_crc = crc32(self.darkcurrent)
                    self.header["darkcurrent"] = imagefile
            else:
                self.darkfile = self.darkcurrent = self.darkcurrent_crc = None
                self.header["darkcurrent"] = "None"

    def setFlatfieldFile(self, imagefile):
        """
        @param imagefile: filename with the path to the flatfield image
        """
        with self._sem:
            if imagefile:
                self.flatfile = imagefile
                try:
                    self.flatfield = numpy.ascontiguousarray(fabio.open(imagefile).data, numpy.float32)
                except Exception as error:
                    logger.warning("setFlatfieldFile: Unable to read file %s: %s" % (imagefile, error))
                else:
                    self.flatfield_crc = crc32(self.flatfield)
                    self.header["flatfield"] = imagefile
            else:
                self.flatfile = self.flatfield = self.flatfield_crc = None
                self.header["flatfield"] = "None"

    def setSplineFile(self, splineFile):
        """
        @param imagefile: filename with the path to the spline distortion file
        """
        with self._sem:
            if not splineFile or not os.path.exists(str(splineFile)):
                self.splinefile = None
                self.det = None
                self.dis = None
                self.header["splinefile"] = "None"
            else:
                logger.info("start config ...")
                self.det = pyFAI.detectors.FReLoN(splineFile)
                self.dis = pyFAI._distortion.Distortion(self.det)
                self.reset()
                self.header["splinefile"] = splineFile

    def setSubdir(self, path):
        """
        Set the relative or absolute path for processed data
        """
        self.subdir = path

    def setExtension(self, ext):
        """
        enforce the extension of the processed data file written
        """
        if ext:
            self.extension = ext
        else:
            self.extension = None

    def calc_LUT(self):
        """
        This is the "slow" calculation of the Look-up table that can be spown in another thread
        (especially to avoid Tango from timing out)
        """
        with self._sem:
            if self.dis:
                self.dis.calc_LUT_size()
                self.dis.calc_LUT()
                if pyopencl:
                    self.ocl_integrator = pyFAI.ocl_azim_lut.OCL_LUT_Integrator(self.dis.LUT, self.dis.shape[0] * self.dis.shape[1])
            else:
                self.splinefile = None
                self.det = None
                self.dis = None
                self.header["splinefile"] = "None"

    def reset(self):
        """
        Recalculate the lookup table in another thread
        """
        threading.Thread(target=self.calc_LUT, name="calc_LUT").start()


class DistortionCorrectionDeviceServer(BasePostProcess) :
    """
    Tango device server exposed to configure the LImA plugin
    """
    DISTORTION_TASK_NAME = 'DistortionCorrectionTask'
    Core.DEB_CLASS(Core.DebModApplication, 'DistortionCorrection')
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
                ctControl = _control_ref()
                extOpt = ctControl.externalOperation()
                extOpt.delOp(self.DISTORTION_TASK_NAME)
        elif(state == PyTango.DevState.ON) :
            if not self.__Task:
                try:
                    ctControl = _control_ref()
                    extOpt = ctControl.externalOperation()
                    self.__Task = extOpt.addOp(Core.USER_SINK_TASK,
                                                         self.DISTORTION_TASK_NAME,
                                                         self._runLevel)
                    if not self.__pyFAISink:
                        self.__pyFAISink = PyFAISink(splinefile=self.__spline_filename,
                                                     darkfile=self.__darkcurrent_filename,
                                                     flatfile=self.__flatfield_filename)
                    self.__Task.setSinkTask(self.__pyFAISink)
                except:
                    import traceback
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
        self.__pyFAISink = PyFAISink(splinefile=self.__spline_filename,
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

class DistortionCorrectionDeviceServerClass(PyTango.DeviceClass) :
        #        Class Properties
    class_property_list = {
        }


    #    Device Properties
    device_property_list = {
        }


    #    Command definitions
    cmd_list = {
        'setDarkcurrentImage':
        [[PyTango.DevString, "Full path of darkcurrent image file"],
         [PyTango.DevVoid, ""]],

        'setFlatfieldImage':
        [[PyTango.DevString, "Full path of flatfield image file"],
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

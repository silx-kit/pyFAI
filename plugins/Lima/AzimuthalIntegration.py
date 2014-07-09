#!/usr/bin/env python
# coding: utf-8
"""
Tango device server for setting up pyFAI azimuthal integrator in a LImA ProcessLib.

Destination path:
Lima/tango/plugins/AzimuthalIntegration
"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/04/2013"
__status__ = "beta"
__docformat__ = 'restructuredtext'


import threading
import logging
logger = logging.getLogger("lima.tango.pyfai")
# set loglevel at least at INFO
if logger.getEffectiveLevel() > logging.INFO:
    logger.setLevel(logging.INFO)
import sys
import os, json, distutils.util
from os.path import dirname
try:
    import pyFAI
except ImportError:
    cwd = dirname(dirname(dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.join(cwd, "build", "lib.%s-%i.%i" % (distutils.util.get_platform(), sys.version_info[0], sys.version_info[1])))
    import pyFAI

import PyTango
import numpy
from Lima import Core
from Utils import BasePostProcess
import fabio

class SinkPyFAI(Core.Processlib.SinkTaskBase):
    def __init__(self, azimuthalIntgrator=None, shapeIn=(2048, 2048), shapeOut=(360, 500), unit="r_mm"):
        """
        @param azimuthalIntgrator: pyFAI.AzimuthalIntegrator instance
        @param shapeIn: image size in input
        @param shapeOut: Integrated size: can be (1,2000) for 1D integration
        @param unit: can be "2th_deg, r_mm or q_nm^-1 ...
        """
        Core.Processlib.SinkTaskBase.__init__(self)
        if azimuthalIntgrator is None:
            self.ai = pyFAI.AzimuthalIntegrator()
        else:
            self.ai = azimuthalIntgrator
        self.nbpt_azim, self.nbpt_rad = shapeOut
        self.unit = pyFAI.units.to_unit(unit)
        self.polarization = None
        self.dummy = None
        self.delta_dummy = None
        self.correct_solid_angle = True
        self.dark_current_image = None
        self.flat_field_image = None
        self.mask_image = None
        self.subdir = ""
        self.extension = None
        self.do_poisson = None
#        self.do
        try:
            self.shapeIn = (camera.getFrameDim.getHeight(), camera.getFrameDim.getWidth())
        except Exception as error:
            logger.error("default on shapeIn %s: %s" % (shapeIn, error))
            self.shapeIn = shapeIn

    def __repr__(self):
        """
        pretty print of myself
        """
        lstout = ["Azimuthal Integrator:", self.ai.__repr__(),
                "Input image shape: %s" % list(self.shapeIn),
                "Number of points in radial direction: %s" % self.nbpt_rad,
                "Number of points in azimuthal direction: %s" % self.nbpt_azim,
                "Unit in radial dimension: %s" % self.unit.REPR,
                "Correct for solid angle: %s" % self.correct_solid_angle,
                "Polarization factor: %s" % self.polarization,
                "Dark current image: %s" % self.dark_current_image,
                "Flat field image: %s" % self.flat_field_image,
                "Mask image: %s" % self.mask_image,
                "Dummy: %s,\tDelta_Dummy: %s" % (self.dummy, self.delta_dummy),
                "Directory: %s, \tExtension: %s" % (self.subdir, self.extension)]
        return os.linesep.join(lstout)

    def do_2D(self):
        return self.nbpt_azim > 1

    def reset(self):
        """
        this is just to force the integrator to initialize
        """
        print "did a reset"
        self.ai.reset()
        # print self.__repr__()

    def reconfig(self, shape=(2048, 2048)):
        """
        this is just to force the integrator to initialize with a given input image shape
        """
        self.shapeIn = shape
        self.ai.reset()

        if self.do_2D():
            threading.Thread(target=self.ai.integrate2d,
                                 name="init2d",
                                 args=(numpy.zeros(self.shapeIn, dtype=numpy.float32),
                                        self.nbpt_rad, self.nbpt_azim),
                                 kwargs=dict(method="lut", unit=self.unit)
                                 ).start()
        else:
            threading.Thread(target=self.ai.integrate1d,
                                 name="init1d",
                                 args=(numpy.zeros(self.shapeIn, dtype=numpy.float32),
                                        self.nbpt_rad),
                                 kwargs=dict(method="lut", unit=self.unit)
                                 ).start()


    def process(self, data) :
        """
        Process a frame
        """
        kwarg = {"unit": self.unit,
                 "dummy": self.dummy,
                 "delta_dummy": self.delta_dummy,
                 "method": "lut",
                 "polarization_factor":self.polarization,
                 # "filename": None,
                 "safe": True,
                 "data": data.buffer,
                 }
        ctControl = _control_ref()
        saving = ctControl.saving()
        sav_parms = saving.getParameters()
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
            except:  # No luck withthreads
                pass

#        directory = sav_parms.directory
        prefix = sav_parms.prefix
        nextNumber = sav_parms.nextNumber
        indexFormat = sav_parms.indexFormat
        kwarg["filename"] = os.path.join(directory, prefix + indexFormat % (nextNumber + data.frameNumber))
        if self.do_2D():
            kwarg["npt_rad"] = self.nbpt_rad
            kwarg["npt_azim"] = self.nbpt_azim
            if self.extension:
                kwarg["filename"] += self.extension
            else:
                kwarg["filename"] += ".azim"
        else:
            kwarg["npt"] = self.nbpt_rad
            if self.extension:
                kwarg["filename"] += self.extension
            else:
                kwarg["filename"] += ".xy"
        if self.do_poisson:
            kwarg["error_model"] = "poisson"
        else:
            kwarg["error_model"] = "None"

        try:
            if self.do_2D():
                self.ai.integrate2d(**kwarg)
            else:
                self.ai.integrate1d(**kwarg)
        except:
            print data.buffer.shape, data.buffer.size
            print self.ai
            print self.ai._lut_integrator
            print self.ai._lut_integrator.size
            raise
        # return rData

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

    def setDarkcurrentFile(self, imagefile):
        self.ai.set_darkfiles(imagefile)
        self.dark_current_image = imagefile

    def setFlatfieldFile(self, imagefile):
        self.ai.set_flatfiles(imagefile)
        self.flat_field_image = imagefile

    def setJsonConfig(self, jsonconfig):
        print("start config ...")
        if os.path.isfile(jsonconfig):
            config = json.load(open(jsonconfig, "r"))
        else:
            config = json.loads(jsonconfig)
        if "poni" in config:
            poni = config["poni"]
            if poni and os.path.isfile(poni):
                self.ai = pyFAI.load(poni)

        detector = config.get("detector", "detector")
        self.ai.detector = pyFAI.detectors.detector_factory(detector)

        if "wavelength" in config:
            wavelength = config["wavelength"]
            try:
                fwavelength = float(wavelength)
            except ValueError:
                logger.error("Unable to convert wavelength to float: %s" % wavelength)
            else:
                if fwavelength <= 0 or fwavelength > 1e-6:
                    logger.warning("Wavelength is in meter ... unlikely value %s" % fwavelength)
                self.ai.wavelength = fwavelength

        splineFile = config.get("splineFile")
        if splineFile and os.path.isfile(splineFile):
            self.ai.detector.splineFile = splineFile
        self.ai.pixel1 = float(config.get("pixel1", 1))
        self.ai.pixel2 = float(config.get("pixel2", 1))
        self.ai.dist = config.get("dist", 1)
        self.ai.poni1 = config.get("poni1", 0)
        self.ai.poni2 = config.get("poni2", 0)
        self.ai.rot1 = config.get("rot1", 0)
        self.ai.rot2 = config.get("rot2", 0)
        self.ai.rot3 = config.get("rot3", 0)


        if config.get("chi_discontinuity_at_0"):
            self.ai.setChiDiscAtZero()

        mask_file = config.get("mask_file")
        do_mask = config.get("do_mask")
        if mask_file and os.path.exists(mask_file) and do_mask:
            try:
                mask = fabio.open(mask_file).data
            except Exception as error:
                logger.error("Unable to load mask file %s, error %s" % (mask_file, error))
            else:
                self.ai.mask = mask
                self.mask_image = os.path.abspath(mask_file)

        self.ai.set_darkfiles([i.strip() for i in config.get("dark_current", "").split(",")
                               if os.path.isfile(i.strip())])
        self.ai.set_flatfiles([i.strip() for i in config.get("flat_field", "").split(",")
                               if os.path.isfile(i.strip())])
        self.dark_current_image = self.ai.darkfiles
        self.flat_field_image = self.ai.flatfiles
        if config.get("do_2D"):
            self.nbpt_azim = int(config.get("azim_pt"))
        else:
            self.nbpt_azim = 1
        if config.get("rad_pt"):
            self.nbpt_rad = int(config.get("rad_pt"))
        self.unit = pyFAI.units.to_unit(config.get("unit", pyFAI.units.TTH_DEG))
        self.do_poisson = config.get("do_poisson")
        if config.get("do_polarization"):
            self.polarization = config.get("polarization")
        else:
            self.polarization = None
        logger.info(self.ai.__repr__())
        self.reset()
        # For now we do not calculate the LUT as the size of the input image is unknown


class AzimuthalIntegrationDeviceServer(BasePostProcess) :
    AZIMUTHAL_TASK_NAME = 'AzimuthalIntegrationTask'
    Core.DEB_CLASS(Core.DebModApplication, 'AzimuthalIntegration')
    def __init__(self, cl, name):
        self.__azimuthalIntegratorTask = None
        self.__jsonConfig = None
        self.__pyFAISink = None
        self.get_device_properties(self.get_device_class())
        self.__extension = ""
        self.__subdir = None
        BasePostProcess.__init__(self, cl, name)
        AzimuthalIntegrationDeviceServer.init_device(self)

    def set_state(self, state) :
        if(state == PyTango.DevState.OFF) :
            if(self.__azimuthalIntegratorTask) :
                self.__azimuthalIntegratorTask = None
                ctControl = _control_ref()
                extOpt = ctControl.externalOperation()
                extOpt.delOp(self.AZIMUTHAL_TASK_NAME)
        elif(state == PyTango.DevState.ON) :
            if not self.__azimuthalIntegratorTask:
                try:
                    ctControl = _control_ref()
                    extOpt = ctControl.externalOperation()
                    self.__azimuthalIntegratorTask = extOpt.addOp(Core.USER_SINK_TASK,
                                                         self.AZIMUTHAL_TASK_NAME,
                                                         self._runLevel)
                    if not self.__pyFAISink:
                        self.__pyFAISink = SinkPyFAI()
                    if self.__jsonConfig:
                        self.__pyFAISink.setJsonConfig(self.__jsonConfig)
                    if self.__extension:
                        self.__pyFAISink.setExtension(self.__extension)
                    if self.__subdir:
                        self.__pyFAISink.setSubdir(self.__subdir)
                    self.__azimuthalIntegratorTask.setSinkTask(self.__pyFAISink)
                except:
                    import traceback
                    traceback.print_exc()
                    return
        PyTango.Device_4Impl.set_state(self, state)

    def setBackgroundImage(self, filepath) :
        if(self.__pyFAISink) :
            self.__pyFAISink.setBackgroundFile(filepath)

    def setFlatfieldImage(self, filepath) :
        if(self.__pyFAISink) :
            self.__pyFAISink.setFlatfieldFile(filepath)


    def setJsonConfig(self, filepath) :
        self.__jsonConfig = filepath
        if(self.__pyFAISink) :
            self.__pyFAISink.setJsonConfig(filepath)

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
        if(self.__pyFAISink) :
            self.__pyFAISink.reset()

    def Reconfig(self, shape) :
        if(self.__pyFAISink) :
            self.__pyFAISink.reconfig(shape)

    def read_Parameters(self, the_att):
        """
        Called  when reading the "Parameters" attribute
        """
        if self.__pyFAISink:
            the_att.set_value(self.__pyFAISink.__repr__())
        else:
            the_att.set_value("No pyFAI Sink processlib active for the moment")


class AzimuthalIntegrationDeviceServerClass(PyTango.DeviceClass) :
        #        Class Properties
    class_property_list = {
        }


    #    Device Properties
    device_property_list = {
        }


    #    Command definitions
    cmd_list = {
        'setBackgroundImage':
        [[PyTango.DevString, "Full path of background image file"],
         [PyTango.DevVoid, ""]],

        'setFlatfieldImage':
        [[PyTango.DevString, "Full path of flatfield image file"],
         [PyTango.DevVoid, ""]],

        'setJsonConfig':
        [[PyTango.DevString, "Path of the JSON configuration file or the configuration itself"],
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
        'Reconfig':
        [[PyTango.DevVarUShortArray, "Input image size [1080,1920]"],
         [PyTango.DevVoid, ""]],
        }


    #    Attribute definitions
    attr_list = {
        'RunLevel':
            [[PyTango.DevLong,
            PyTango.SCALAR,
            PyTango.READ_WRITE]],
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
    return AzimuthalIntegrationDeviceServerClass, AzimuthalIntegrationDeviceServer

#!/usr/bin/env python
# coding: utf8
"""
Tango device server for setting up pyFAI azimuthal integrator in a LImA ProcessLib.

Destination path:
Lima/tango/plugins/AzimuthalIntegration  
"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "31/01/2013"
__status__ = "beta"
__docformat__ = 'restructuredtext'

import os, json
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

import PyTango
import numpy
from Lima import Core
from Utils import getDataFromFile, BasePostProcess
import pyFAI

class PyFAISink(Core.Processlib.SinkTaskBase):
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
        self.unit = unit
        self.polarization = 0
        self.dummy = None
        self.delta_dummy = None
        try:
            self.shapeIn = (camera.getFrameDim.getHeight(), camera.getFrameDim.getWidth())
        except Exception as error:
            logger.error("default on shapeIn %s: %s" % (shapeIn, error))
            self.shapeIn = shapeIn

    def do_2D(self):
        return self.nbpt_azim > 1

    def reset(self):
        """
        this is just to force the integrator to initialize
        """
        print "did a reset"
        self.ai.reset()

    def reconfig(self, shape=(2048, 2048)):
        """
        this is just to force the integrator to initialize with a given input image shape
        """
        self.shapeIn = shape
        self.ai.reset()

        if self.do_2D():
            t = threading.Thread(target=self.ai.integrate2d,
                                 name="init2d",
                                 args=(numpy.zeros(self.shapeIn, dtype=numpy.float32),
                                        self.nbpt_rad, self.nbpt_azim),
                                 kwargs=dict(method="lut", unit=self.unit)
                                 )
            t.start()
        else:
            t = threading.Thread(target=self.ai.integrate2d,
                                 name="init2d",
                                 args=(numpy.zeros(self.shapeIn, dtype=numpy.float32),
                                        self.nbpt_rad),
                                 kwargs=dict(method="lut", unit=self.unit)
                                 )
            t.start()

    def process(self, data) :
        ctControl = _control_ref()
        saving = ctControl.saving()
        sav_parms = saving.getParameters()
        directory = sav_parms.directory
        prefix = sav_parms.prefix
        nextNumber = sav_parms.nextNumber
        indexFormat = sav_parms.indexFormat
        output = os.path.join(directory, prefix + indexFormat % (nextNumber + data.frameNumber))
        try:

            if self.do_2D():
                self.ai.integrate2d(data.buffer, self.nbpt_rad, self.nbpt_azim,
                                    method="lut", unit=self.unit, safe=True,
                                    filename=output + ".azim")
            else:
                self.ai.integrate1d(data.buffer,
                                self.nbpt_rad, method="lut", unit=self.unit, safe=True,
                                filename=output + ".xy")
        except :
                print data.buffer.shape, data.buffer.size
                print self.ai
                print self.ai._lut_integrator
                print self.ai._lut_integrator.size
                raise
        # return rData

    def setDarkcurrentFile(self, imagefile):
        try:
            darkcurrentImage = getDataFromFile(filepath)
        except Exception as error:
            logger.warning("setDarkcurrentFile: Unable to read file %s: %s" % (imagefile, error))
        else:
            self.ai.set_darkcurrent(darkcurrentImage.buffer)

    def setFlatfieldFile(self, imagefile):
        try:
            backGroundImage = getDataFromFile(filepath)
        except Exception as error:
            logger.warning("setFlatfieldFile: Unable to read file %s: %s" % (imagefile, error))
        else:
            self.ai.set_flatfield(backGroundImage.buffer)

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
                mask = getDataFromFile(mask_file).buffer
            except Exception as error:
                logger.error("Unable to load mask file %s, error %s" % (mask_file, error))
            else:
                self.ai.mask = mask
        dark_files = [i.strip() for i in config.get("dark_current", "").split(",")
                      if os.path.isfile(i.strip())]
        if dark_files and config.get("do_dark"):
            d0 = getDataFromFile(dark_files[0]).buffer
            darks = numpy.zeros(d0.shape[0], d0.shape[1], len(dark_files), dtype=numpy.float32)
            for i, f in enumerate(dark_files):
                darks[:, :, i] = getDataFromFile(f).buffer
            self.ai.darkcurrent = darks.mean(axis= -1)

        flat_files = [i.strip() for i in config.get("flat_field", "").split(",")
                      if os.path.isfile(i.strip())]
        if flat_files and config.get("do_flat"):
            d0 = getDataFromFile(flat_files[0]).buffer
            flats = numpy.zeros(d0.shape[0], d0.shape[1], len(flat_files), dtype=numpy.float32)
            for i, f in enumerate(flat_files):
                flats[:, :, i] = getDataFromFile(f).buffer
            self.ai.darkcurrent = flats.mean(axis= -1)

        if config.get("do_2D"):
            self.nbpt_azim = int(config.get("azim_pt"))
        else:
            self.nbpt_azim = 1
        if config.get("rad_pt"):
            self.nbpt_rad = int(config.get("rad_pt"))
        self.unit = pyFAI.units.to_unit(config.get("unit", pyFAI.units.TTH_DEG))

        logger.info(self.ai.__repr__())
        self.reset()
        # For now we do not calculate the LUT as the size of the input image is unknown


class AzimuthalIntegratonDeviceServer(BasePostProcess) :
    AZIMUTHAL_TASK_NAME = 'AzimuthalIntegrationTask'
    Core.DEB_CLASS(Core.DebModApplication, 'AzimuthalIntegration')
    def __init__(self, cl, name):
        self.__azimuthalIntegratorTask = None
        self.__jsonConfig = None
        self.__pyFAISink = None
        self.get_device_properties(self.get_device_class())

        BasePostProcess.__init__(self, cl, name)
        AzimuthalIntegratonDeviceServer.init_device(self)

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
                        self.__pyFAISink = PyFAISink()
                    if self.__jsonConfig :
                        self.__pyFAISink.setJsonConfig(self.__jsonConfig)
                    self.__azimuthalIntegratorTask.setSinkTask(self.__pyFAISink)
                except:
                    import traceback
                    traceback.print_exc()
                    return
        PyTango.Device_4Impl.set_state(self, state)

    @Core.DEB_MEMBER_FUNCT
    def setBackgroundImage(self, filepath) :
        deb.Param('setBackgroundImage filepath=%s' % filepath)
        if(self.__pyFAISink) :
            self.__pyFAISink.setBackgroundFile(filepath)

    @Core.DEB_MEMBER_FUNCT
    def setFlatfieldImage(self, filepath) :
        deb.Param('setFlatfieldImage filepath=%s' % filepath)
        if(self.__pyFAISink) :
            self.__pyFAISink.setFlatfieldFile(filepath)


    @Core.DEB_MEMBER_FUNCT
    def setJsonConfig(self, filepath) :
        deb.Param('setJsonConfig: filepath=%s' % filepath)
        if(self.__pyFAISink) :
            self.__pyFAISink.setJsonConfig(filepath)

    @Core.DEB_MEMBER_FUNCT
    def Reset(self) :
        deb.Param('Reset')
        if(self.__pyFAISink) :
            self.__pyFAISink.reset()

    @Core.DEB_MEMBER_FUNCT
    def Reconfig(self, shape) :
        deb.Param('Reconfig: %s' % shape)
        if(self.__pyFAISink) :
            self.__pyFAISink.reconfig(shape)


class AzimuthalIntegratonDeviceServerClass(PyTango.DeviceClass) :
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
        [[PyTango.DevString, "Full path of background image file"],
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
    return AzimuthalIntegratonDeviceServerClass, AzimuthalIntegratonDeviceServer

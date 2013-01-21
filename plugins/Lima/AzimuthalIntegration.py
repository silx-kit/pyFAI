#!/usr/bin/env python
# coding: utf8
"""
Tango device server for setting up pyFAI azimuthal integrator in a LImA ProcessLib.

Destination path:
Lima/tango/plugins/AzimuthalIntegration  
"""
__author__ = "Jérôme Kieffer"

import os, json
import logging
logger = logging.getLogger("lima.tango.pyfai")
import PyTango

from Lima import Core
from Utils import getDataFromFile, BasePostProcess
import pyFAI

class PyFAILink(Core.Processlib.LinkTask):
    def __init__(self, azimuthalIntgrator=None, shapeIn=(966, 1296), shapeOut=(360, 500), unit="r_mm"):
        """
        @param azimuthalIntgrator: pyFAI.AzimuthalIntegrator instance
        
        """
        Core.Processlib.LinkTask.__init__(self)
        if azimuthalIntgrator is None:
            self.ai = pyFAI.AzimuthalIntegrator()
        else:
            self.ai = azimuthalIntgrator
        self.nbpt_azim, self.nbpt_rad = shapeOut
        self.unit = unit
        # this is just to force the integrator to initialize
        _ = self.ai.integrate2d(numpy.zeros(shapeIn, dtype=numpy.float32),
                            self.nbpt_rad, self.nbpt_azim, method="lut", unit=self.unit,)

    def process(self, data) :
        rData = Core.Processlib.Data()
        rData.frameNumber = data.frameNumber
        rData.buffer = self.ai.integrate2d(data.buffer, self.nbpt_rad, self.nbpt_azim,
                                           method="lut", unit=self.unit, safe=False)[0]
        return rData

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

        self.ai.pixel1 = config.get("pixel1", 1)
        self.ai.pixel2 = config.get("pixel2", 1)
        self.ai.dist = config.get("dist", 1)
        self.ai.poni1 = config.get("poni1", 0)
        self.ai.poni2 = config.get("poni2", 0)
        self.ai.rot1 = config.get("rot1", 0)
        self.ai.rot2 = config.get("rot2", 0)
        self.ai.rot3 = config.get("rot3", 0)

        if self.chi_discontinuity_at_0.isChecked():
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

class AzimuthalIntegratonDeviceServer(BasePostProcess) :
    AZIMUTHAL_TASK_NAME = 'AzimuthalIntegrationTask'
    Core.DEB_CLASS(Core.DebModApplication, 'AzimuthalIntegration')
    def __init__(self, cl, name):
        self.__azimuthalIntegratorTask = None
        self.__jsonConfig = None
        self.get_device_properties(self.get_device_class())

        BasePostProcess.__init__(self, cl, name)
        AzimuthalIntegratorDeviceServer.init_device(self)

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
                  self.__azimuthalIntegratorTask = extOpt.addOp(Core.BACKGROUNDSUBSTRACTION,
                                                       self.AZIMUTHAL_TASK_NAME,
                                                       self._runLevel)
                  self.__azimuthalIntegratorTask.setBackgroundImage(self.__backGroundImage)
                except:
                        import traceback
                        traceback.print_exc()
                        return
        PyTango.Device_4Impl.set_state(self, state)

#    @Core.DEB_MEMBER_FUNCT
    def setBackgroundImage(self, filepath) :
        deb.Param('setBackgroundImage filepath=%s' % filepath)
        if(self.__azimuthalIntegratorTask) :
            self.__azimuthalIntegratorTask.setBackgroundFile(filepath)

#    @Core.DEB_MEMBER_FUNCT
    def setFlatfieldImage(self, filepath) :
        deb.Param('setFlatfieldImage filepath=%s' % filepath)
        if(self.__azimuthalIntegratorTask) :
            self.__azimuthalIntegratorTask.setFlatfieldFile(filepath)


#    @Core.DEB_MEMBER_FUNCT
    def setJsonConfig(self, filepath) :
        deb.Param('setJsonConfig: filepath=%s' % filepath)
        if(self.__azimuthalIntegratorTask) :
            self.__azimuthalIntegratorTask.setJsonConfig(filepath)


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
        self.set_type(name);

_control_ref = None
def set_control_ref(control_class_ref) :
    global _control_ref
    _control_ref = control_class_ref

def get_tango_specific_class_n_device() :
   return AzimuthalIntegratorDeviceServerClass, AzimuthalIntegratorDeviceServer

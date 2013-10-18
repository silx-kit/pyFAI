#!/usr/bin/env python
# coding: utf8
from __future__ import with_statement, print_function
"""
LImA ProcessLib example of pyFAI azimuthal integrator Link and Sink 

"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/10/2013"
__status__ = "beta"
__docformat__ = 'restructuredtext'

import os, json, distutils.util, sys, threading, logging
logger = logging.getLogger("lima.pyfai")
# set loglevel at least at INFO
if logger.getEffectiveLevel() > logging.INFO:
    logger.setLevel(logging.INFO)
import numpy
from Lima import Core
#from Utils import BasePostProcess
import sys
import os, json, distutils.util
from os.path import dirname
try:
    import pyFAI
except ImportError:
    cwd = dirname(dirname(dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.join(cwd, "build", "lib.%s-%i.%i" % (distutils.util.get_platform(), sys.version_info[0], sys.version_info[1])))
    import pyFAI


class StartAcqCallback(Core.SoftCallback):
    """
    Class managing the connection from a 
    Lima.Core.CtControl.prepareAcq() to the configuration of the various tasks
    
    Example of usage:
    cam = Basler.Camera(ip)
    iface = Basler.Interface(cam)
    ctrl = Core.CtControl(iface)
    processLink = LinkPyFAI(worker, writer)
    extMgr = ctrl.externalOperation()
    myOp = self.extMgr.addOp(Core.USER_LINK_TASK, "pyFAILink", 0)
    myOp.setLinkTask(processLink)
    callback = StartAcqCallback(ctrl, processLink)
    myOp.registerCallback(callback)
    acq.setAcqNbFrames(0)
    acq.setAcqExpoTime(1.0)
    ctrl.prepareAcq() #Configuration called here !!!!
    ctrl.startAcq()

    """
    def __init__(self, control, task=None):
        """
        
        @param control: Lima.Core.CtControl instance
        @param task: The task one wants to parametrize at startup. Can be a  Core.Processlib.LinkTask or a Core.Processlib.SinkTask
        """
        Core.SoftCallback.__init__(self)
        self._control = control
        self._task = task

    def prepare(self):
        """
        Called with prepareAcq()
        """

        im = self._control.image()
        imdim = im.getImageDim().getSize()

        x = imdim.getWidth()
        y = imdim.getHeight()
        bin = im.getBin()
        binX = bin.getX()
        binY = bin.getY()

        # number of images ...
        acq = self._control.acquisition()
        nbframe = acq.getAcqNbFrames() #to check.
        expo = acq.getAcqExpoTime()
        #ROI see: https://github.com/esrf-bliss/Lima/blob/master/control/include/CtAcquisition.h
        lima_cfg = {"dimX":x,
                    "dimY":y,
                    "binX":binX,
                    "binY":binY,
                    "number_of_frames": nbframe,
                    "exposure_time":expo}
        if self._task._worker is None:
            #Define a default integrator
            centerX = x // 2
            centerY = y // 2
            ai = pyFAI.AzimuthalIntegrator()
            ai.setFit2D(1000, centerX=centerX, centerY=centerY, pixelX=1, pixelY=1)
            worker = pyFAI.worker.Worker(ai)
            worker.unit = "r_mm"
            worker.method = "lut_ocl_gpu"
            worker.nbpt_azim = 360
            worker.nbpt_rad = 500
            worker.output = "numpy"
            print("Worker updated")
            self._task._worker = worker
        else:
            worker = self._task._worker
        worker.reconfig(shape=(y, x), sync=True)
        if self._task._writer:
            config = self._task._worker.get_config()
            self._task._writer.init(config=config, lima_cfg=lima_cfg)
            self._task._writer.flush(worker.radial, worker.azimuthal)

class LinkPyFAI(Core.Processlib.LinkTask):
    """
    This is a ProcessLib task which is a link: 
    it modifies the image for further processing.
    
    It processes every acquired frame with the pyFAI-worker and can optionally 
    save data as HDF5 or EDF    
    """
    def __init__(self, worker=None, writer=None):
        """
        @param worker: pyFAI.worker.Worker instance
        @param writer: pyFAI.io.Writer instance
        """
        Core.Processlib.LinkTask.__init__(self)
        self._worker = worker
        self._writer = writer

    def process(self, data) :
        """
        Callback function
        
        Called for every frame in a different C++ thread.
        """
        rData = Core.Processlib.Data()
        rData.frameNumber = data.frameNumber
        rData.buffer = self._worker.process(data.buffer)
        if self._writer: #optional HDF5 writer
            self._writer.write(rData.buffer, rData.frameNumber)
        return rData

    def __repr__(self):
        """
        pretty print of myself
        """
        lstout = [ "LinkPyFAI Processlib instance","Worker:",self._worker.__repr__(),"Writer:",self._writer.__repr__()]
        return os.linesep.join(lstout)

class SinkPyFAI(Core.Processlib.SinkTaskBase):
    """
    This is a ProcessLib task which is a sink: 
    it processes the image and saves it.
    
    It processes every acquired frame with the pyFAI-worker.
    If no writer is provided, processed data are lost.  
        
    """
    def __init__(self, worker=None, writer=None):
        Core.Processlib.SinkTaskBase.__init__(self)
        self._worker = worker
        self._writer = writer
        if writer  is None:
            logger.error("Without a writer, SinkPyFAI will just dump all data") 

    def __repr__(self):
        """
        pretty print of myself
        """
        lstout = [ "SinkPyFAI Processlib instance","Worker:",self._worker.__repr__(),"Writer:",self._writer.__repr__()]
        return os.linesep.join(lstout)

    def reset(self):
        """
        this is just to force the integrator to initialize
        """
        self.ai.reset()

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
        Callback function
        
        Called for every frame in a different C++ thread.
        """
        rData = Core.Processlib.Data()
        rData.frameNumber = data.frameNumber
        rData.buffer = self._worker.process(data.buffer)
        if self._writer: #optional HDF5 writer
            self._writer.write(rData.buffer, rData.frameNumber)

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
            kwarg["nbPt_rad"] = self.nbpt_rad
            kwarg["nbPt_azim"] = self.nbpt_azim
            if self.extension:
                kwarg["filename"] += self.extension
            else:
                kwarg["filename"] += ".azim"
        else:
            kwarg["nbPt"] = self.nbpt_rad
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
        self.writer.subdir = path

    def setExtension(self, ext):
        """
        enforce the extension of the processed data file written
        """
        if ext:
            self._writer.extension = ext
        else:
            self._writer.extension = None

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

#FOR SINK
#        ctControl = _control_ref()
#        saving = ctControl.saving()
#        sav_parms = saving.getParameters()
#        if not self.subdir:
#            directory = sav_parms.directory
#        elif self.subdir.startswith("/"):
#            directory = self.subdir
#        else:
#            directory = os.path.join(sav_parms.directory, self.subdir)
#        if not os.path.exists(directory):
#            logger.error("Ouput directory does not exist !!!  %s" % directory)
#            try:
#                os.makedirs(directory)
#            except:  # No luck withthreads
#                pass
#        directory = sav_parms.directory
#        prefix = sav_parms.prefix
#        nextNumber = sav_parms.nextNumber
#        indexFormat = sav_parms.indexFormat
#        if self.output is not None:
#            kwarg["filename"] = os.path.join(directory, prefix + indexFormat % (nextNumber + data.frameNumber))

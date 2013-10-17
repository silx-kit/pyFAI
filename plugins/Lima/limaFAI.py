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
__date__ = "11/10/2013"
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
import pyFAI


class StartAcqCallback(Core.SoftCallback):
    """
    Class managing the connection from a 
    Lima.Core.CtControl.prepareAcq() to the configuration of the various tasks
    
    Example of usage:
    cam = Basler.Camera(ip)
    iface = Basler.Interface(cam)
    ctrl = Core.CtControl(iface)
    processLink = FaiLink(worker, writer)
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

class FaiLink(Core.Processlib.LinkTask):
    def __init__(self, worker=None, writer=None):
        Core.Processlib.LinkTask.__init__(self)
        self._worker = worker
        self._writer = writer
        self._sem = threading.Semaphore()

    def process(self, data) :
        rData = Core.Processlib.Data()
        rData.frameNumber = data.frameNumber
        rData.buffer = self._worker.process(data.buffer)
        if self._writer: #optional HDF5 writer
            self._writer.write(rData.buffer, rData.frameNumber)
        return rData

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

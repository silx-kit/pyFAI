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
from Utils import BasePostProcess
import pyFAI

class FaiLink(Core.Processlib.LinkTask):
    def __init__(self, worker=None):
        Core.Processlib.LinkTask.__init__(self)
        self._worker = worker
        self._sem = threading.Semaphore()

    def process(self, data) :
        if  self._worker is None:
            with self._sem:
                if  self._worker is None:
                    shape = data.buffer.shape
                    centerX = shape[1] // 2
                    centerY = shape[0] // 2
                    ai = pyFAI.AzimuthalIntegrator()
                    ai.setFit2D(1000, centerX=centerX, centerY=centerY, pixelX=1, pixelY=1)
                    worker = pyFAI.worker.Worker(ai)
                    self._worker = worker
                    self._worker.unit = "r_mm"
                    self._worker.nbpt_azim = 500
                    self._worker.nbpt_rad = 360
                    self._worker.reconfig(shape=shape)
                    self.output = "numpy"
        rData = Core.Processlib.Data()
        rData.frameNumber = data.frameNumber
        rData.buffer = self._worker.process(data.buffer)
        return rData

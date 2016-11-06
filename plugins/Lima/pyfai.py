#!/usr/bin/python
#coding: utf-8
#Class defining the link task for azimuthal regrouping in process lib for Lima
#
import logging
import time
import numpy
from Lima import Core, Basler
import pyFAI
import fabio

logger = logging.getLogger("PyFAI.LImA.ProcessLib")

IP = "192.168.5.19"
EXPO = 0.1 #sec

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
            data = fabio.open(imagefile).data
        except Exception as error:
            data = None
            logger.warning("setDarkcurrentFile: Unable to read file %s: %s", imagefile, error)
        else:
            self.ai.set_darkcurrent = data

    def setFlatfieldFile(self, imagefile):
        try:
            data = fabio.open(imagefile).data
        except Exception as error:
            data = None
            logger.warning("setFlatfieldFile: Unable to read file %s: %s", imagefile, error)
        self.ai.set_flatfield(data)

if __name__ == "__main__":

    ai = pyFAI.AzimuthalIntegrator(dist=1, poni1=0.001875, poni2=0.00225, detector="Basler")

    cam = Basler.Camera(IP)
    iface = Basler.Interface(cam)
    ctrl = Core.CtControl(iface)
    extMgr = ctrl.externalOperation()
    myOp = extMgr.addOp(Core.USER_LINK_TASK, "azimGroup", 0)
    myTask = PyFAILink(ai)
    myOp.setLinkTask(myTask)
    acq = ctrl.acquisition()
#    acq.setAcqNbFrames(10)
    acq.setAcqNbFrames(0)
    acq.setAcqExpoTime(EXPO)
    ctrl.prepareAcq()
    ctrl.startAcq()
    print ctrl.getStatus()
    time.sleep(1)
    base_img = ctrl.ReadBaseImage()
    proc_img = ctrl.ReadImage()
    from matplotlib import pyplot
    fig = pyplot.figure()
    subplot1 = fig.add_subplot(1, 2, 1)
    subplot2 = fig.add_subplot(1, 2, 2)
    subplot1.imshow(base_img.buffer, cmap="gray")
    subplot2.imshow(proc_img.buffer, cmap="gray")
    fig.show()
    pyplot.ion()
    while True:
        base_img = ctrl.ReadBaseImage()
        proc_img = ctrl.ReadImage()
        subplot1.imshow(base_img.buffer, cmap="gray")
        subplot2.imshow(proc_img.buffer, cmap="gray")
        time.sleep(EXPO)
        fig.canvas.draw()

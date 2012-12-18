#!/usr/bin/python
#coding: utf8
#Class defining the link task for azimuthal regrouping in process lib for Lima
# 
import time
from Lima import Core, Basler
import pyFAI

IP = "192.168.5.19"
EXPO = 0.1 #sec

class PyFAILink(Core.Processlib.LinkTask):
    def __init__(self, azimuthalIntgrator, shapeIn=(966, 1296), shapeOut=(360, 500), unit="r_mm"):
        Core.Processlib.LinkTask.__init__(self)
        self.ai = azimuthalIntgrator
        self.ai._lut_integrator = self.ai.setup_LUT(shape=shapeIn, nbPt=shapeOut, unit=unit)
        self.nbpt_azim, self.nbpt_rad = shapeOut
        self.unit = unit

    def process(self, data) :
       #print "max src:",data.buffer.min(),data.buffer.max()
       rData = Core.Processlib.Data()
       rData.frameNumber = data.frameNumber
       rData.buffer = self.ai.integrate2d(data.buffer, self.nbpt_rad, self.nbpt_azim, method="lut", unit=self.unit)[0]
       #print "max src:",rData.buffer.min(),rData.buffer.max()
       return rData


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

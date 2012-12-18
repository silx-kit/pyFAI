#!/usr/bin/python
#coding: utf8
#Class defining the link task for azimuthal regrouping in process lib for Lima
# 

from Lima import Core, Basler
import pyFAI

IP = "192.168.5.19"

class MyLink(Core.Processlib.LinkTask):
    def __init__(self, ai=pyFAI.AzimuthalIntegrator(1,), shapeIn=(), shapeOut=(500, 360), unit="r_mm"):
        Core.Processlib.LinkTask.__init__(self)
        self.ai = ai
#        self.ai.

    def process(self, data) :
       #print "max src:",data.buffer.min(),data.buffer.max()
       rData = Core.Processlib.Data()
       rData.frameNumber = data.frameNumber
       rData.buffer = ai.integrate2d(data.buffer)[0]
       #print "max src:",rData.buffer.min(),rData.buffer.max()
       return rData

if __name__ == "__main__":
    cam = Basler.Camera(IP)
    iface = Basler.Interface(cam)
    ctrl = Core.CtControl(iface)
    extMgr = ctrl.externalOperation()
    myOp = extMgr.addOp(Core.USER_LINK_TASK, "myTask", 0)
    myTask = MyLink(10)
    myOp.setLinkTask(myTask)
    acq = ctrl.acquisition()
    acq.setAcqNbFrames(10)
    acq.setAcqExpoTime(0.1)
    ctrl.prepareAcq()
    ctrl.startAcq()
    print ctrl.getStatus()
    base_img = ctrl.ReadBaseImage()
    proc_iimg = ctrl.ReadImage()

#!/usr/bin/python
import time
import signal
import threading
import numpy
import visvis as vv
import pyFAI
import pyopencl
from Lima import Core, Basler

fps = 30
acqt = 1.0 / fps
ip = "192.168.5.19"

cam = Basler.Camera(ip)
iface = Basler.Interface(cam)
ctrl = Core.CtControl(iface)


class MyLink(Core.Processlib.LinkTask):
    def __init__(self, dist, outshape=(400, 360)):
        Core.Processlib.LinkTask.__init__(self)
        self.__dist = dist
        self.__ai = None
        self.__sem = threading.Semaphore()
        self.outshape = outshape
    def process(self, data) :
        if  self.__ai is None:
            with self.__sem:
                if  self.__ai is None:
                    shape = data.buffer.shape
                    centerX = shape[1] // 2
                    centerY = shape[0] // 2
                    ai = pyFAI.AzimuthalIntegrator(detector=pyFAI.detectors.Basler())
                    ai.setFit2D(self.__dist, centerX=centerX, centerY=centerY)
                    ai.integrate2d(numpy.zeros(shape, dtype=numpy.float32), self.outshape[0], self.outshape[1], unit="r_mm", method="lut_ocl")
                    self.__ai = ai
        rData = Core.Processlib.Data()
        rData.frameNumber = data.frameNumber
        rData.buffer = self.__ai.integrate2d(data.buffer, self.outshape[0], self.outshape[1], unit="r_mm", method="lut_ocl")[0]
        return rData

extMgr = ctrl.externalOperation()
myOp = extMgr.addOp(Core.USER_LINK_TASK, "myTask", 0)
myTask = MyLink(10)
myOp.setLinkTask(myTask)
a = ctrl.acquisition()
a.setAcqNbFrames(0)
a.setAcqExpoTime(acqt)
ctrl.prepareAcq()
ctrl.startAcq()

while ctrl.getStatus().ImageCounters.LastImageReady < 1:
    print(ctrl.getStatus())
    time.sleep(0.5)
print(ctrl.getStatus())

raw_img = ctrl.ReadBaseImage().buffer
fai_img = ctrl.ReadImage().buffer
vv.figure()
rawplot = vv.subplot(121)
faiplot = vv.subplot(122)
rawtex = vv.imshow(raw_img, axes=rawplot)
faitex = vv.imshow(fai_img, axes=faiplot)
while 1:
    rawtex.SetData(ctrl.ReadBaseImage().buffer)
    faitex.SetData(ctrl.ReadImage().buffer)
    time.sleep(acqt)
    vv.processEvents()

#!/usr/bin/env python
from picamera.array import PiYUVArray
from picamera import PiCamera
from PIL import Image
from PyQt4 import QtCore
import numpy
import pyFAI
import time

resolution = (640, 480) 
shape = 500, 360
fps = 2
ai = pyFAI.AzimuthalIntegrator(detector="raspberry")
ai.setFit2D(1000,resolution[0]//2, resolution[1]//2)

import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(12,5))
sp1 = fig.add_subplot(1,2,1)
sp2 = fig.add_subplot(1,2,2)
nb = numpy.random.randint(0,255,size=resolution[0]*resolution[1]).reshape((resolution[1], resolution[0]))
i2 = ai.integrate2d(nb, shape[0], shape[1], method="csr", unit="r_mm")[0]
im1 = sp1.imshow(nb, interpolation="nearest", cmap="gray")
im2 = sp2.imshow(i2, interpolation="nearest", cmap="gray")
fig.show()
t0 = time.time()
with PiCamera() as camera:
    camera.resolution = resolution
    camera.framerate = fps
    with PiYUVArray(camera, size=resolution) as raw:
        for f in camera.capture_continuous(raw, format="yuv", use_video_port=True):
            frame = raw.array
            nb = frame[...,0]
            i2 = ai.integrate2d(nb, shape[0], shape[1], method="csr", unit="r_mm")[0]
            im1.set_data(nb)
            im2.set_data(i2)
            t1 = time.time()
            print("fps: %.3f"%(1.0/(t1-t0)))
            t0 = t1
            fig.canvas.draw()
            QtCore.QCoreApplication.processEvents()
            raw.truncate(0)

            

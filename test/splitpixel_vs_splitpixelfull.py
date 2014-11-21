#!/usr/bin/python

import pyFAI
import numpy
from pyFAI import splitPixelFull, splitPixel

ai = pyFAI.AzimuthalIntegrator(detector="Fairchild")
shape = (2048, 2048)
data = numpy.zeros(shape)
data[100, 200] = 1

tth, I = ai.integrate1d(data, 10000, correctSolidAngle=False, method="splitpixel", unit="2th_deg")

res_splitPixelFull = splitPixelFull.fullSplit1D(ai._corner4Da, data, bins=10000)
res_splitPixel = splitPixel.fullSplit1D(ai._corner4Da, data, bins=10000)

for i, ary in enumerate(("tth", "I", "unweight", "weight")):
    print("Error on %s: %s" % (i, abs(res_splitPixelFull[i] - res_splitPixel[i]).max()))



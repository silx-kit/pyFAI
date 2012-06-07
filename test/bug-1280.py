#!/usr/bin/python
import sys, os
import numpy
import fabio
from utilstest import  UtilsTest
pyFAI = UtilsTest.pyFAI
data = fabio.open(UtilsTest.getimage("1788/moke.tif")).data
ai = pyFAI.AzimuthalIntegrator.sload("moke.poni")
ai.xrpd(data, 1000)
tth = ai.twoThetaArray(data.shape)
dtth = ai.delta2Theta(data.shape)
o1 = ai.xrpd(data, 1000)
o2 = ai.xrpd(data, 1000, tthRange=[3.5, 12.5])
o3 = ai.xrpd(data, 1000, chiRange=[10, 80])
o4 = ai.xrpd2(data, 100, 36, tthRange=[3.5, 12.5], chiRange=[10, 80])
from pylab import  *
plot(o1[0], o1[1], "b")
plot(o2[0], o2[1], "r")
plot(o3[0], o3[1], "g")
imshow(o4[0])
show()

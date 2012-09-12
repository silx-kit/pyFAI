#!/usr/bin/python
import os, time
import pyFAI, fabio

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test", "testimages")
spline = os.path.join(root, "halfccd.spline")
poni = os.path.join(root, "LaB6.poni")
res = []
with open(poni, "r") as f:
    for l in f:
        if l.startswith("SplineFile"):
            res.append("SplineFile: %s%s" % (spline, os.linesep))
        else:
            res.append(l)
with open(poni, "w") as f:
    f.writelines(res)
edf = os.path.join(root, "LaB6_0020.edf")

img = fabio.open(edf)
ai = pyFAI.load(poni)
ai.xrpd(img.data, 2048)
tth = ai._ttha.ravel().astype("float32")
dtth = ai._dttha.ravel().astype("float32")
data = img.data.ravel().astype("float32")

import splitBBox
t0 = time.time()
ra, rb, rc, rd = splitBBox.histoBBox1d(data, tth, dtth, bins=2048)
t1 = time.time()
ref_time = t1 - t0
print("ref time: %.3fs" % ref_time)

#import paraSplitBBox
#t0 = time.time()
#a, b, c, d = paraSplitBBox.histoBBox1d(data, tth, dtth, bins=2048)
#t1 = time.time()
#psbb_time = t1 - t0
#print("Parallel Split Bounding Box: %.3fs" % ref_time)
#print abs(ra - a).max(), abs(rb - b).max(), abs(rc - c).max(), abs(rd - d).max()

print "With LUT"
import splitBBoxLUT
#a, b, c, d, ee = splitBBoxLUT.histoBBox1d(data, tth, dtth, bins=2048)
#print "LUT max =", ee.max()
t0 = time.time()
integ = splitBBoxLUT.HistoBBox1d(tth, dtth, bins=2048)
t1 = time.time()
a, b, c, d = integ.integrate(data)
t2 = time.time()
print("LUT creation: %.3fs; integration %.3f" % (t1 - t0, t2 - t1))
print abs(ra - a).max(), abs(rb - b).max(), abs(rc - c).max(), abs(rd - d).max()
t1 = time.time()
a, b, c, d = integ.integrate(data)
t2 = time.time()
print "speed-up:", ref_time / (t2 - t1)
from pylab import *
#plot(ee)
plot(a, b)
plot(ra, rb)
show()
raw_input("Enter")

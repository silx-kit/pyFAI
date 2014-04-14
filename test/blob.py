#!/usr/bin/python
# coding: utf8
import sys
from math import sqrt
import fabio
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
import pyFAI.blob_detection

# data = fabio.open(UtilsTest.options.args[0]).data
# msk = fabio.open(UtilsTest.options.args[1]).data
data = fabio.open("../../testimages/halfccd.edf").data
msk = fabio.open("../../testimages/halfccd_8_mask.tiff").data
bd = pyFAI.blob_detection.BlobDetection(data, mask=msk)

import pylab
pylab.ion()
f=pylab.figure()
ax = f.add_subplot(111)
ax.imshow(bd.raw)
bd._one_octave(True, False, False)
# print("Octave #%i total kp: %i" % (i, bd.keypoints.size))
#for kp  in bd.keypoints:
#    ds = sqrt(kp.scale)
#    ax.annotate("", xy=(kp.x, kp.y), xytext=(kp.x + ds, kp.y + ds),
#                arrowprops=dict(facecolor='blue', shrink=0.05),)
ax.plot(bd.keypoints[:].x, bd.keypoints[:].y, ".g")
n3kp = bd.keypoints

#bd._one_octave(False, False, True)
#ax.plot(bd.keypoints[:].x, bd.keypoints[:].y, ".r")
#n5kp = bd.keypoints
#print(len(n3kp), len(n5kp))
#same = [i for i in n3kp if i not in n5kp]
#print(len(same))
f.show()
raw_input()

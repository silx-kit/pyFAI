#!/usr/bin/python
# coding: utf8
import sys
import fabio
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
import pyFAI.blob_detection

data = fabio.open(UtilsTest.options.args[0]).data
bd = pyFAI.blob_detection.BlobDetection(data)

import pylab
pylab.ion()
f=pylab.figure()
ax = f.add_subplot(111)
ax.imshow(data)
for x, y, dx, dy, sigma in zip(bd._one_octave(False, False)):
    if sigma<1:
        ax.annotate("", xy=(x+dx,y+dy), xytext=(x,y),
                arrowprops=dict(facecolor='blue', shrink=0.05),)
    elif sigma<2:
        ax.annotate("", xy=(x+dx,y+dy), xytext=(x,y),
                arrowprops=dict(facecolor='green', shrink=0.05),)
    elif sigma < 4:
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(facecolor='yellow', shrink=0.05),)
    else:
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(facecolor='red', shrink=0.05),)
f.show()
raw_input()

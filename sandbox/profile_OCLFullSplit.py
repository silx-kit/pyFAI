# -*- coding: utf-8 -*-
"""
Created on Fri Mar 07 09:52:51 2014

@author: ashiotis
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import fabio
from pylab import *
print("#"*50)
pyFAI = sys.modules["pyFAI"]
from pyFAI import OCLFullSplit
# logger = utilstest.getLogger("profile")


ai = pyFAI.load("testimages/halfccd.poni")
data = fabio.open("testimages/halfccd.edf").data

workgroup_size = 256
bins = 1000

pos_in = ai.array_from_unit(data.shape, "corner", unit="2th_deg", scale=False)

pos = pos_in.reshape(pos_in.size / 8, 4, 2)

pos_size = pos.size
# size = data.size
size = pos_size / 8


foo = OCLFullSplit.OCLFullSplit1d(pos, bins)

print(foo.pos0Range)
print(foo.pos1Range)

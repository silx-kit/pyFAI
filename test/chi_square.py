#!/usr/bin/python
#coding: utf-8

#tests if the distribution of Chi2 is centered around 1:
# Needs a large dataset (thousands of images)

import sys
import glob
import pylab
pylab.ion()
import numpy
from math import sqrt
import fabio
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

ai = pyFAI.AzimuthalIntegrator(detector="Pilatus1M")
ai.setFit2D(directDist=2849, centerX=8.900000e+02, centerY=7.600000e+01)
ai.wavelength = 9.919000e-11
images = glob.glob("/mnt/data/BM29/water/daniel/raw/water_029_0*.edf")
images.sort()

I_splitBB = [];sigma_splitBB = [];I_splitFull = [];sigma_splitFull = [];I_nosplit = [];sigma_nosplit = []
for fn in images[:]:
    img = fabio.open(fn).data
    print(fn);
    variance = numpy.maximum(img, 1)
    q, i, s = ai.integrate1d(img, 1040, unit="q_nm^-1", method="numpy", variance=variance)
    I_nosplit.append(i)
    sigma_nosplit.append(s)
    q, i, s = ai.integrate1d(img, 1040, unit="q_nm^-1", method="splitbbox", variance=variance)
    I_splitBB.append(i)
    sigma_splitBB.append(s)
    q, i, s = ai.integrate1d(img, 1040, unit="q_nm^-1", method="splitpixel", variance=variance)
    I_splitFull.append(i)
    sigma_splitFull.append(s)

I_splitBB = numpy.vstack(I_splitBB)
I_splitFull = numpy.vstack(I_splitFull)
I_nosplit = numpy.vstack(I_nosplit)
sigma_nosplit = numpy.vstack(sigma_nosplit)
sigma_splitBB = numpy.vstack(sigma_splitBB)
sigma_splitFull = numpy.vstack(sigma_splitFull)
Chi2_splitBB = [];Chi2_splitFull = []; Chi2_nosplit = []
Iavg_splitFull = I_splitFull.mean(axis=0)
Iavg_splitBB = I_splitBB.mean(axis=0)
Iavg_nosplit = I_nosplit.mean(axis=0)

for i in range(I_splitBB.shape[0]):
    Chi2_splitBB.append((((I_splitBB[i] - Iavg_splitBB) / sigma_splitBB[i]) ** 2).mean())
    Chi2_splitFull.append((((I_splitFull[i] - Iavg_splitFull) / sigma_splitFull[i]) ** 2).mean())
    Chi2_nosplit.append((((I_nosplit[i] - Iavg_nosplit) / sigma_nosplit[i]) ** 2).mean())
pylab.hist(Chi2_splitBB, 50, label="splitBB")
pylab.hist(Chi2_splitFull, 50, label="splitFull")
pylab.hist(Chi2_nosplit, 50, label="no_split")
pylab.xlabel("$\chi^2$")
pylab.ylabel("count")
pylab.legend()
pylab.show()

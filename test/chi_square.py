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

I_split = [];sigma_split = [];I_nosplit = [];sigma_nosplit = []
for fn in images:
    img = fabio.open(fn).data
    print(fn);
    variance = numpy.maximum(img, 1)
    q, i, s = ai.integrate1d(img, 1040, unit="q_nm^-1", method="numpy", variance=variance)
    I_nosplit.append(i)
    sigma_nosplit.append(s)
    q, i, s = ai.integrate1d(img, 1040, unit="q_nm^-1", method="lut", variance=variance)
    I_split.append(i)
    sigma_split.append(s)
I_split = numpy.vstack(I_split)
I_nosplit = numpy.vstack(I_nosplit)
sigma_nosplit = numpy.vstack(sigma_nosplit)
sigma_split = numpy.vstack(sigma_split)
Chi2_split = []; Chi2_nosplit = []
Iavg_split = I_split.mean(axis=0)
Iavg_nosplit = I_nosplit.mean(axis=0)
for i in range(len(images)):
    Chi2_split.append((((I_split[i] - Iavg_split) / sigma_split[i]) ** 2).mean())
    Chi2_nosplit.append((((I_nosplit[i] - Iavg_nosplit) / sigma_nosplit[i]) ** 2).mean())
pylab.hist(Chi2_split, 50, label="split")
pylab.hist(Chi2_nosplit, 50, label="no_split")
pylab.legend()
pylab.show()

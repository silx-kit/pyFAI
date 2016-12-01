#!/usr/bin/env python
# coding: utf-8
# author: Jérôme Kieffer
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# tests if the distribution of Chi2 is centered around 1:
# Needs a large dataset (thousands of images)

import os
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
from lxml import etree

ai = pyFAI.AzimuthalIntegrator(detector="Pilatus1M")
images = glob.glob("/data/bm29/inhouse/opd29/20140430/raw/water_008_*.edf")
images.sort()
img = images[0]
xml = etree.parse(os.path.splitext(img)[0] + ".xml")
wl = float(xml.xpath("//wavelength")[0].getchildren()[0].text)
centerX = float(xml.xpath("//beamCenter_1")[0].getchildren()[0].text)
centerY = float(xml.xpath("//beamCenter_2")[0].getchildren()[0].text)
directDist = float(xml.xpath("//detectorDistance")[0].getchildren()[0].text) * 1000.0
msk = xml.xpath("//maskFile")[0].getchildren()[0].getchildren()[0].text
msk = numpy.logical_or(fabio.open(msk).data, ai.detector.mask)
ai.setFit2D(directDist=directDist, centerX=centerX, centerY=centerY)
ai.wavelength = wl

I_splitBB = [];sigma_splitBB = [];I_splitFull = [];sigma_splitFull = [];I_nosplit = [];sigma_nosplit = []
for fn in images[:10]:
    img = fabio.open(fn).data
    xml = etree.parse(os.path.splitext(fn)[0] + ".xml")
    monitor = float(xml.xpath("//beamStopDiode")[0].getchildren()[0].text)
    print(fn, monitor);
    variance = numpy.maximum(img, 1)
    q, i, s = ai.integrate1d(img, 1040, unit="q_nm^-1", method="numpy", variance=variance, mask=msk, normalization_factor=monitor)
    I_nosplit.append(i)
    sigma_nosplit.append(s)
    q, i, s = ai.integrate1d(img, 1040, unit="q_nm^-1", method="splitbbox", variance=variance, mask=msk, normalization_factor=monitor)
    I_splitBB.append(i)
    sigma_splitBB.append(s)
    q, i, s = ai.integrate1d(img, 1040, unit="q_nm^-1", method="splitpixel", variance=variance, mask=msk, normalization_factor=monitor)
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

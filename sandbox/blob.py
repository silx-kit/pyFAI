# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import print_function
import sys, scipy
import matplotlib
matplotlib.use('Qt4Agg')
import pylab
from math import sqrt
import fabio, numpy
import logging
from pyFAI.test.utilstest import UtilsTest
logger = logging.getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.blob_detection import BlobDetection
from pyFAI.detectors import detector_factory
from pyFAI.third_party import six


def somme(im):
    im[1:-1, 1:-1] += im[:-2, 1:-1] + im[2:, 1:-1] + im[1:-1, :-2] + im[1:-1, 2:]  # + im[:-2, :-2] + im[2:, 2:] + im[2:, :-2] + im[:-2, 2:]
    return im


def image_test():
    "Creating a test image containing several gaussian of several sizes"
    img = numpy.zeros((128 * 4, 128 * 4))
    a = numpy.linspace(0.5, 8, 16)
    xc = [64, 64, 64, 64, 192, 192, 192, 192, 320, 320, 320, 320, 448, 448, 448, 448]
    yc = [64, 192, 320, 448, 64, 192, 320, 448, 64, 192, 320, 448, 64, 192, 320, 448]
    cpt = 0
    for sigma in a:
        img = make_gaussian(img, sigma, xc[cpt], yc[cpt])
        cpt = cpt + 1
    img = add_noise(img, 0.1)
    return img


def image_test_rings():
    "Creating a test image containing gaussian spots on several rings"
    rings = 10
    mod = 50
    detector = detector_factory("Titan")
    sigma = detector.pixel1 * 4
    shape = detector.max_shape
    ai = AzimuthalIntegrator(detector=detector)
    ai.setFit2D(1000, 1000, 1000)
    r = ai.rArray(shape)
    r_max = r.max()
    chi = ai.chiArray(shape)
    img = numpy.zeros(shape)
    modulation = (1 + numpy.sin(5 * r + chi * mod))
    for radius in numpy.linspace(0, r_max, rings):
        img += numpy.exp(-(r - radius) ** 2 / (2 * (sigma * sigma)))

    img *= modulation
    img = add_noise(img, 0.0)
    return img


def add_noise(img, rate):
    noise = numpy.random.random(img.shape) * rate
    return img + noise


def make_gaussian(im, sigma, xc, yc):
    "Creating 2D gaussian to be put in a test image"
    e = 1
    angle = 0
    sx = sigma * (1 + e)
    sy = sigma * (1 - e)
    size = int(8 * sigma + 1)
    if size % 2 == 0 :
        size += 1
    x = numpy.arange(0, size, 1, float)
    y = x[:, numpy.newaxis]
#     x = x * 2
    x0 = y0 = size // 2
    gausx = numpy.exp(-4 * numpy.log(2) * (x - x0) ** 2 / sx ** 2)
    gausy = numpy.exp(-4 * numpy.log(2) * (y - y0) ** 2 / sy ** 2)
    gaus = 0.01 + gausx * gausy
    im[xc - size / 2:xc + size / 2 + 1, yc - size / 2:yc + size / 2 + 1] = scipy.ndimage.rotate(gaus, angle, reshape=False)
    return im


if len(UtilsTest.options.args) > 0:
    data = fabio.open(UtilsTest.options.args[0]).data
    if len(UtilsTest.options.args) > 1:
        msk = fabio.open(UtilsTest.options.args[1]).data
    else:
        msk = None
else:
    data = image_test_rings()
    msk = None


bd = BlobDetection(data, mask=msk)  # , cur_sigma=0.25, init_sigma=numpy.sqrt(2)/2, dest_sigma=numpy.sqrt(2), scale_per_octave=2)

pylab.ion()
f = pylab.figure(1)
ax = f.add_subplot(111)
ax.imshow(numpy.log1p(data), interpolation='nearest')

for i in range(5):
    print('Octave #%i' % i)
    bd._one_octave(shrink=True, refine=True, n_5=False)
    print("Octave #%i Total kp: %i" % (i, bd.keypoints.size))

# bd._one_octave(False, True ,False)

print('Final size of keypoints : %i' % bd.keypoints.size)

i = 0
# for kp  in bd.keypoints:
#     ds = kp.sigma
#     ax.annotate("", xy=(kp.x, kp.y), xytext=(kp.x+ds, kp.y+ds),
#                 arrowprops=dict(facecolor='blue', shrink=0.05),)
sigma = bd.keypoints.sigma
for i, c in enumerate("ygrcmykw"):
#    j = 2 ** i
    m = numpy.logical_and(sigma >= i, sigma < (i + 1))
    ax.plot(bd.keypoints[m].x, bd.keypoints[m].y, "o" + c, label=str(i))
ax.legend()

if sigma.size > 0:
    h = pylab.figure(2)
    x, y, o = pylab.hist(sigma, bins=100)
    h.show()

    index = numpy.where(x == x.max())
    kp = bd.keypoints[bd.keypoints.sigma > y[index]]
else : kp = bd.keypoints

# pylab.figure()
# pylab.imshow(numpy.log1p(data), interpolation = 'nearest')
# pylab.plot(kp.x,kp.y,'og')

f.show()
six.moves.input()

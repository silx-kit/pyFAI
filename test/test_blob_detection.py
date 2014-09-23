#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: PyFAI: https://github.com/kif/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France

"test suite for blob detection cython accelerated code"

__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme Kieffer"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/05/2014"

import sys
import unittest
import numpy
from utilstest import getLogger  # UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI.detectors import detector_factory
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.blob_detection import BlobDetection, local_max
from pyFAI import _blob

from pyFAI import _blob

def image_test_rings():
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
    return img * modulation

class TestBlobDetection(unittest.TestCase):
    img = None
    
    def setUp(self):
        if self.img is None:
            self.img = image_test_rings()
        
    def test_local_max(self):
        bd = BlobDetection(self.img)
        bd._one_octave(shrink=False, refine=False, n_5=False)
        self.assert_(numpy.alltrue(_blob.local_max(bd.dogs, bd.cur_mask, False) == \
                                         local_max(bd.dogs, bd.cur_mask, False)), "max test, 3x3x3")
        self.assert_(numpy.alltrue(_blob.local_max(bd.dogs, bd.cur_mask, True) == \
                                         local_max(bd.dogs, bd.cur_mask, True)), "max test, 3x5x5")

def test_suite_all_blob_detection():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestBlobDetection("test_local_max"))
#    testSuite.addTest(TestConvolution("test_vertical_convolution"))
#    testSuite.addTest(TestConvolution("test_gaussian"))
#    testSuite.addTest(TestConvolution("test_gaussian_filter"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_blob_detection()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)

        

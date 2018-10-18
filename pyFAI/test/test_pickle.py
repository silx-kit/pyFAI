#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

from __future__ import absolute_import, division, print_function

"""Test suite for pickled objects"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/10/2018"


import numpy
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import detector_factory
from pickle import dumps, loads
import unittest
import logging
logger = logging.getLogger(__name__)


class TestPickle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestPickle, cls).setUpClass()
        cls.ai = AzimuthalIntegrator(1.0, detector="Pilatus100k")
        cls.ai.wavelength = 1e-10
        cls.npt = 100
        cls.data = numpy.random.random(cls.ai.detector.shape)

    @classmethod
    def tearDownClass(cls):
        super(TestPickle, cls).tearDownClass()
        cls.data = cls.ai = cls.npt = None

    def test_Detector_pickle(self):
        det = self.ai.detector  # type: Detector
        dets = dumps(det)
        self.assert_(dets, "pickle works")
        rest = loads(dets)
        self.assert_(rest, "unpickle works")
        self.assertEqual(rest.shape, self.ai.detector.MAX_SHAPE)

        # test the binning
        mar = detector_factory("RayonixMx225")
        mar.guess_binning((2048, 2048))
        self.assertEqual(mar.binning, (3, 3), "binning OK")
        mars = dumps(mar)
        marr = loads(mars)
        self.assertEqual(mar.binning, marr.binning, "restored binning OK")

    def test_AzimuthalIntegrator_pickle(self):
        spectra = self.ai.integrate1d(self.data, self.npt)  # force lut generation
        ais = dumps(self.ai)
        newai = loads(ais)  # type: AzimuthalIntegrator
        self.assertEqual(newai._cached_array.keys(), self.ai._cached_array.keys())
        for key in self.ai._cached_array.keys():
            if isinstance(self.ai._cached_array[key], numpy.ndarray):
                self.assertEqual(abs(newai._cached_array[key] - self.ai._cached_array[key]).max(), 0,
                                 "key %s is the same" % key)
            else:
                self.assertEqual(newai._cached_array[key], self.ai._cached_array[key],
                                 "key %s is the same: %s %s" %
                                 (key, newai._cached_array[key], self.ai._cached_array[key]))
        for first, second in zip(newai.integrate1d(self.data, self.npt), spectra):
            self.assertEqual(abs(first - second).max(), 0, "Spectra are the same")

    def test_Calibrant(self):
        from pyFAI import calibrant
        calibrant = calibrant.CalibrantFactory()('AgBh')
        assert dumps(calibrant)
        assert loads(dumps(calibrant))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestPickle))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

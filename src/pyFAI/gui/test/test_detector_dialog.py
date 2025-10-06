#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite for detector dialog"""

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/10/2025"

import unittest
import logging

from silx.gui.utils import testutils
from pyFAI.test.utilstest import UtilsTest
from ..dialog.DetectorSelectorDialog import DetectorSelectorDrop
from pyFAI import detectors

logger = logging.getLogger(__name__)


class TestDetectorDialog(testutils.TestCaseQt):

    def test_detector(self):
        detector = detectors.Detector()
        widget = DetectorSelectorDrop()
        widget.setDetector(detector)
        newDetector = widget.detector()
        self.assertEqual(detector, newDetector)

    def test_custom_detector(self):
        detector = detectors.Detector(pixel1=1, pixel2=2, max_shape=(100, 100))
        widget = DetectorSelectorDrop()
        widget.setDetector(detector)
        newDetector = widget.detector()
        self.assertEqual(detector, newDetector)

    def test_detector_splinefile(self):
        splinefile = UtilsTest.getimage("frelon.spline")
        detector = detectors.Detector()
        detector.splinefile = splinefile
        widget = DetectorSelectorDrop()
        widget.setDetector(detector)
        newDetector = widget.detector()
        self.assertEqual(detector, newDetector)

    def test_eiger(self):
        detector = detectors.Eiger1M()
        widget = DetectorSelectorDrop()
        widget.setDetector(detector)
        newDetector = widget.detector()
        self.assertEqual(detector, newDetector)

    def test_frelon_splinefile(self):
        splinefile = UtilsTest.getimage("frelon.spline")
        detector = detectors.FReLoN(splinefile=splinefile)
        widget = DetectorSelectorDrop()
        widget.setDetector(detector)
        newDetector = widget.detector()
        self.assertEqual(detector, newDetector)

    def test_nexus(self):
        wos = UtilsTest.getimage("WOS.h5")
        detector = detectors.NexusDetector(wos)
        widget = DetectorSelectorDrop()
        widget.setDetector(detector)
        newDetector = widget.detector()
        self.assertEqual(detector, newDetector)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestDetectorDialog))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

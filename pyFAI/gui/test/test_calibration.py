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

"""Test suite for the calibration GUI"""

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/10/2018"

import unittest
import logging
import numpy

from silx.gui import qt
try:
    from silx.gui.utils import testutils
except ImportError:
    # silx 0.8 and earlly
    from silx.gui.test import utils as testutils

import pyFAI.resources
import pyFAI.calibrant
import pyFAI.detectors
from pyFAI.gui.calibration.CalibrationWindow import CalibrationWindow
from pyFAI.gui.calibration.CalibrationContext import CalibrationContext

_logger = logging.getLogger(__name__)


class TestCalibration(testutils.TestCaseQt):

    @classmethod
    def setUpClass(cls):
        super(TestCalibration, cls).setUpClass()
        pyFAI.resources.silx_integration()

    def setUp(self):
        # FIXME: It would be good to remove this singleton
        CalibrationContext._instance = None
        super(TestCalibration, self).setUp()

    def create_context(self):
        settings = qt.QSettings()
        context = CalibrationContext(settings)
        return context

    def display_each_tasks(self, window):
        for _ in range(4):
            window.nextTask()
            self.qWait(200)

    def test_empty_data(self):
        context = self.create_context()
        window = CalibrationWindow(context)
        window.setVisible(True)
        self.qWaitForWindowExposed(window)
        self.display_each_tasks(window)

    def test_with_set_data(self):
        context = self.create_context()
        # set data before launching the application
        experimentSettings = context.getCalibrationModel().experimentSettingsModel()
        experimentSettings.image().setValue(numpy.array([[10, 11], [12, 13]]))
        experimentSettings.calibrantModel().setCalibrant(pyFAI.calibrant.get_calibrant("LaB6"))
        experimentSettings.detectorModel().setDetector(pyFAI.detectors.FReLoN())
        experimentSettings.wavelength().setValue(60)
        window = CalibrationWindow(context)
        window.setVisible(True)
        self.qWaitForWindowExposed(window)
        self.display_each_tasks(window)

    def test_then_set_data(self):
        context = self.create_context()
        window = CalibrationWindow(context)
        window.setVisible(True)
        self.qWaitForWindowExposed(window)
        # set data while the application is working
        experimentSettings = context.getCalibrationModel().experimentSettingsModel()
        self.qWait(100)
        experimentSettings.image().setValue(numpy.array([[10, 11], [12, 13]]))
        self.qWait(100)
        experimentSettings.calibrantModel().setCalibrant(pyFAI.calibrant.get_calibrant("LaB6"))
        self.qWait(100)
        experimentSettings.detectorModel().setDetector(pyFAI.detectors.FReLoN())
        self.qWait(100)
        experimentSettings.wavelength().setValue(60)
        self.display_each_tasks(window)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestCalibration))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

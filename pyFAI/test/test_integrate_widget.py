#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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
from unicodedata import decimal

__doc__ = "test suite for worker"
__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/06/2016"

import os
import sys
import unittest
import numpy
import fabio
from .utilstest import getLogger
from .. import units
from ..worker import Worker
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..gui_utils import has_Qt
if has_Qt:
    from ..integrate_widget import AIWidget
from .utilstest import UtilsTest

logger = getLogger(__file__)


class AIWidgetMocked():

    def __init__(self, result=None):
        pass


class TestAIWidget(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ..gui_utils import QtGui
        if has_Qt:
            cls.app = QtGui.QApplication([])

    def setUp(self):
        if not has_Qt:
            self.skipTest("Qt is not available")

    @classmethod
    def tearDownClass(cls):
        cls.app = None

    def test_process_no_data(self):
        widget = AIWidget(json_file=None)
        widget.set_ponifile(UtilsTest.getimage("1893/Pilatus1M.poni"))
        widget.nbpt_rad.setText("2")
        result = widget.proceed()
        self.assertIsNone(result)

    def test_process_numpy_1d(self):
        ponifile = UtilsTest.getimage("1893/Pilatus1M.poni")
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = [[23.5, 9.9]]

        widget = AIWidget(json_file=None)
        widget.set_ponifile(ponifile)
        widget.do_2D.setChecked(False)
        widget.nbpt_rad.setText("2")
        widget.set_input_data(numpy.array([data]), "foo")
        result = widget.proceed()
        numpy.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_process_numpy_2d(self):
        ponifile = UtilsTest.getimage("1893/Pilatus1M.poni")
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = [[[7.5, 5.3], [50.1, 12.6]]]

        widget = AIWidget(json_file=None)
        widget.set_ponifile(ponifile)
        widget.do_2D.setChecked(True)
        widget.nbpt_azim.setText("2")
        widget.nbpt_rad.setText("2")
        widget.set_input_data(numpy.array([data]), "foo")
        result = widget.proceed()
        numpy.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_process_array_1d(self):
        ponifile = UtilsTest.getimage("1893/Pilatus1M.poni")
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = [[[1.9, 1.9], [23.5, 9.9]]]

        widget = AIWidget(json_file=None)
        widget.set_ponifile(ponifile)
        widget.do_2D.setChecked(False)
        widget.nbpt_rad.setText("2")
        widget.set_input_data([data], "foo")
        result = widget.proceed()
        numpy.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_process_array_2d(self):
        ponifile = UtilsTest.getimage("1893/Pilatus1M.poni")
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = [[[[7.5, 5.3], [50.1, 12.6]], [2.0, 2.0], [-124.5, -124.2]]]

        widget = AIWidget(json_file=None)
        widget.set_ponifile(ponifile)
        widget.do_2D.setChecked(True)
        widget.nbpt_azim.setText("2")
        widget.nbpt_rad.setText("2")
        widget.set_input_data([data], "foo")
        result = widget.proceed()
        # simplify representation
        self.assertEqual(len(result), len(expected))
        self.assertEqual(len(result[0]), len(expected[0]))
        for i in range(len(result[0])):
            numpy.testing.assert_array_almost_equal(result[0][i], expected[0][i], decimal=1)


if sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
    # On linux and no DISPLAY available (e.g., ssh without -X)
    logger.warning('pyFAI.integrate_widget tests disabled (DISPLAY env. variable not set)')

    class SkipGUITest(unittest.TestCase):
        def runTest(self):
            self.skipTest(
                'pyFAI.integrate_widget tests disabled (DISPLAY env. variable not set)')

    def suite():
        suite = unittest.TestSuite()
        suite.addTest(SkipGUITest())
        return suite
else:
    def suite():
        testsuite = unittest.TestSuite()
        test_names = unittest.getTestCaseNames(TestAIWidget, "test")
        for test in test_names:
            testsuite.addTest(TestAIWidget(test))
        return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

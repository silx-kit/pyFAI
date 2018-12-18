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

"""Test suite for worker"""

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/12/2018"

import os
import sys
import unittest
import numpy
import logging

from silx.gui import qt
from ...gui.IntegrationDialog import IntegrationDialog
from ...gui.widgets.WorkerConfigurator import WorkerConfigurator
from pyFAI.test.utilstest import UtilsTest

logger = logging.getLogger(__name__)


class TestIntegrationDialog(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = None
        if sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
            # On linux and no DISPLAY available (e.g., ssh without -X)
            logger.warning('pyFAI.integrate_widget tests disabled (DISPLAY env. variable not set)')
            cls.app = None
        elif qt is not None:
            cls.app = qt.QApplication([])

    def setUp(self):
        if qt is None:
            self.skipTest("Qt is not available")
        if self.__class__.app is None:
            self.skipTest("DISPLAY env. is not set")

    @classmethod
    def tearDownClass(cls):
        cls.app = None

    def test_process_no_data(self):
        widget = IntegrationDialog(json_file=None)
        dico = {"poni": UtilsTest.getimage("Pilatus1M.poni"),
                "nbpt_rad": 2}
        widget.set_config(dico)
        result = widget.proceed()
        self.assertIsNone(result)

    def test_process_numpy_1d(self):
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = [[23.5, 9.9]]

        widget = IntegrationDialog(json_file=None)
        dico = {"poni": UtilsTest.getimage("Pilatus1M.poni"),
                "do_2D": False,
                "nbpt_rad": 2,
                "method": "splitbbox"}
        widget.set_config(dico)
        widget.set_input_data(numpy.array([data]))
        result = widget.proceed()
        numpy.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_process_numpy_2d(self):
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = [[[5.6, 4.5], [41.8, 9.3]]]
        widget = IntegrationDialog(json_file=None)
        dico = {"poni": UtilsTest.getimage("Pilatus1M.poni"),
                "do_2D": True,
                "nbpt_azim": 2,
                "nbpt_rad": 2,
                "method": "splitbbox"}
        widget.set_config(dico)
        widget.set_input_data(numpy.array([data]))
        result = widget.proceed()
        numpy.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_process_array_1d(self):
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = [[[1.9, 1.9], [23.5, 9.9]]]
        widget = IntegrationDialog(json_file=None)
        dico = {"poni": UtilsTest.getimage("Pilatus1M.poni"),
                "do_2D": False,
                "nbpt_rad": 2}
        widget.set_config(dico)
        widget.set_input_data([data])
        result = widget.proceed()
        numpy.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_process_array_2d(self):
        data = numpy.array([[0, 0], [0, 100], [0, 0]])
        expected = [[[[5.6, 4.5], [41.8, 9.3]], [2.0, 2.0], [-124.5, -124.2]]]
        widget = IntegrationDialog(json_file=None)
        dico = {"poni": UtilsTest.getimage("Pilatus1M.poni"),
                "do_2D": True,
                "nbpt_azim": 2,
                "nbpt_rad": 2,
                "method": "splitbbox"}
        widget.set_config(dico)
        widget.set_input_data([data])
        result = widget.proceed()
        # simplify representation
        self.assertEqual(len(result), len(expected))
        self.assertEqual(len(result[0]), len(expected[0]))
        for i in range(len(result[0])):
            numpy.testing.assert_array_almost_equal(result[0][i], expected[0][i], decimal=1)

    def test_config_flatdark_v1(self):
        dico = {"dark_current": "a,b,c",
                "flat_field": "a,b,d"}
        widget = WorkerConfigurator()
        widget.setConfig(dico)
        dico = widget.getConfig()
        self.assertEqual(dico["dark_current"], ["a", "b", "c"])
        self.assertEqual(dico["flat_field"], ["a", "b", "d"])

    def test_config_flatdark_v2(self):
        dico = {"dark_current": ["a", "b", "c"],
                "flat_field": ["a", "b", "d"]}
        widget = WorkerConfigurator()
        widget.setConfig(dico)
        dico = widget.getConfig()
        self.assertEqual(dico["dark_current"], ["a", "b", "c"])
        self.assertEqual(dico["flat_field"], ["a", "b", "d"])


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestIntegrationDialog))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

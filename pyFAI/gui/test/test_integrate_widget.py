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
__date__ = "16/05/2019"

import unittest
import logging

from silx.gui import qt
from ...gui.widgets.WorkerConfigurator import WorkerConfigurator
from silx.gui.utils import testutils
from pyFAI.test.utilstest import UtilsTest
from pyFAI.io import integration_config


logger = logging.getLogger(__name__)


class TestIntegrationDialog(testutils.TestCaseQt):

    @classmethod
    def setUpClass(cls):
        super(TestIntegrationDialog, cls).setUpClass()
        config = {"poni": UtilsTest.getimage("Pilatus1M.poni")}
        integration_config.normalize(config, inplace=True)
        cls.base_config = config

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

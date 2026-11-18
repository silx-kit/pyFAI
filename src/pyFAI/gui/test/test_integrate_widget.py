#!/usr/bin/env python
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

"""Test suite for worker"""

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/11/2026"

import logging
import unittest

from silx.gui import qt

from ..utils.testutils import TestCaseQt

from pyFAI.io import integration_config
from pyFAI.test.utilstest import UtilsTest

from ...gui.widgets.WorkerConfigurator import WorkerConfigurator
from ...worker import WorkerFiber

logger = logging.getLogger(__name__)


class TestIntegrationDialog(TestCaseQt):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
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

    def test_config_correct_absorption(self):
        "#2953: tristate checkbox: None=automatic (partially checked), True, False"
        states = {None: qt.Qt.CheckState.PartiallyChecked,
                  True: qt.Qt.CheckState.Checked,
                  False: qt.Qt.CheckState.Unchecked}
        widget = WorkerConfigurator()
        self.assertEqual(widget.do_absorption.checkState(), states[None], "automatic by default")
        self.assertIsNone(widget.getConfig()["correct_absorption"])
        for value, state in states.items():
            widget.setConfig({"correct_absorption": value})
            self.assertEqual(widget.do_absorption.checkState(), state)
            self.assertIs(widget.getConfig()["correct_absorption"], value)

    def test_absorption_both_tabs(self):
        "#2953: the absorption correction is available for azimuthal and fiber integration"
        widget = WorkerConfigurator()
        for index in (1, 0):
            widget.tabWidget.setCurrentIndex(index)
            self.assertTrue(widget.do_absorption.isEnabled(), index)

    def test_config_correct_absorption_fiber(self):
        "#2953: the checkbox is common to both tabs: its state goes to the fiber configuration"
        widget = WorkerConfigurator()
        config = WorkerFiber().get_worker_config()
        for value in (None, True, False):
            config.correct_absorption = value
            widget.setConfig(config.as_dict())
            self.assertEqual(widget.tabWidget.currentIndex(), 1)
            self.assertIs(widget.getConfig()["correct_absorption"], value)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestIntegrationDialog))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

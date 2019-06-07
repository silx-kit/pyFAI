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
__date__ = "16/05/2019"

import unittest
import logging
import numpy

from silx.gui.utils import testutils
from ..model.PeakModel import PeakModel
from ..model.ListModel import ListModel
from ..model.DataModel import DataModel


_logger = logging.getLogger(__name__)


class TestPeakModelization(testutils.TestCaseQt):

    def test_merge(self):
        data1 = numpy.array([[2, 3], [3, 3]])
        data2 = numpy.array([[4, 4], [3, 3]])
        peak = PeakModel()
        peak.setCoords(data1)
        peak.mergeCoords(data2)
        self.assertEqual(len(peak.coords()), 3)

    def test_distanceTo(self):
        data1 = numpy.array([[2, 3], [3, 3]])
        peak = PeakModel()
        peak.setCoords(data1)
        result = peak.distanceTo((2, 3))
        self.assertEqual(result, 0)


class TestListModel(testutils.TestCaseQt):

    def test_add(self):
        listModel = ListModel()
        listener = testutils.SignalListener()
        listModel.changed[object].connect(listener.partial("changed"))
        listModel.structureChanged.connect(listener.partial("structureChanged"))
        listModel.contentChanged.connect(listener.partial("contentChanged"))
        data = DataModel()
        listModel.append(data)
        self.assertEqual(len(listModel), 1)
        self.assertEqual(len([args for args in listener.arguments() if args[0] == "structureChanged"]), 1)
        self.assertEqual(len([args for args in listener.arguments() if args[0] == "contentChanged"]), 0)
        self.assertEqual(len([args for args in listener.arguments() if args[0] == "changed"]), 1)
        events = [args for args in listener.arguments() if args[0] == "changed"][0][1]
        self.assertTrue(events.hasStructuralEvents())
        self.assertTrue(events.hasOnlyStructuralEvents())
        self.assertFalse(events.hasUpdateEvents())
        self.assertFalse(events.hasOnlyUpdateEvents())

    def test_remove(self):
        listModel = ListModel()
        data = DataModel()
        listModel.append(DataModel())
        listModel.append(data)
        listModel.append(DataModel())
        listener = testutils.SignalListener()
        listModel.changed[object].connect(listener.partial("changed"))
        listModel.structureChanged.connect(listener.partial("structureChanged"))
        listModel.contentChanged.connect(listener.partial("contentChanged"))
        listModel.remove(data)
        self.assertEqual(len(listModel), 2)
        self.assertEqual(len([args for args in listener.arguments() if args[0] == "structureChanged"]), 1)
        self.assertEqual(len([args for args in listener.arguments() if args[0] == "contentChanged"]), 0)
        self.assertEqual(len([args for args in listener.arguments() if args[0] == "changed"]), 1)
        events = [args for args in listener.arguments() if args[0] == "changed"][0][1]
        self.assertTrue(events.hasStructuralEvents())
        self.assertTrue(events.hasOnlyStructuralEvents())
        self.assertFalse(events.hasUpdateEvents())
        self.assertFalse(events.hasOnlyUpdateEvents())

    def test_change(self):
        listModel = ListModel()
        data = DataModel()
        listModel.append(data)
        listener = testutils.SignalListener()
        listModel.changed[object].connect(listener.partial("changed"))
        listModel.structureChanged.connect(listener.partial("structureChanged"))
        listModel.contentChanged.connect(listener.partial("contentChanged"))
        data.setValue(666)
        self.assertEqual(len(listModel), 1)
        self.assertEqual(len([args for args in listener.arguments() if args[0] == "structureChanged"]), 0)
        self.assertEqual(len([args for args in listener.arguments() if args[0] == "contentChanged"]), 1)
        self.assertEqual(len([args for args in listener.arguments() if args[0] == "changed"]), 1)
        events = [args for args in listener.arguments() if args[0] == "changed"][0][1]
        self.assertFalse(events.hasStructuralEvents())
        self.assertFalse(events.hasOnlyStructuralEvents())
        self.assertTrue(events.hasUpdateEvents())
        self.assertTrue(events.hasOnlyUpdateEvents())


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestPeakModelization))
    testsuite.addTest(loader(TestListModel))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

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

"""Test QuantityEdit widget"""

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/10/2020"

import unittest
import logging

from silx.gui import qt
from silx.gui.utils import testutils
from ..QuantityEdit import QuantityEdit
from pyFAI.gui.utils import units
from pyFAI.gui.model.DataModel import DataModel

logger = logging.getLogger(__name__)


class TestQuantityEdit(testutils.TestCaseQt):

    def test_same_distance(self):
        w = QuantityEdit()
        model = DataModel(w)
        model.setValue(100)
        w.setModelUnit(units.Unit.MILLIMETER)
        w.setDisplayedUnit(units.Unit.MILLIMETER)
        w.setModel(model)
        self.assertEqual(w.text(), "100")

    def test_different_distance(self):
        w = QuantityEdit()
        model = DataModel(w)
        model.setValue(1000)
        w.setModelUnit(units.Unit.MILLIMETER)
        w.setDisplayedUnit(units.Unit.METER)
        w.setModel(model)
        self.assertEqual(w.text(), "1.0")

    def test_same_distance_with_model(self):
        w = QuantityEdit()
        model = DataModel(w)
        model.setValue(100)
        lengthUnit = DataModel()
        lengthUnit.setValue(units.Unit.MILLIMETER)
        w.setModelUnit(units.Unit.MILLIMETER)
        w.setDisplayedUnitModel(lengthUnit)
        w.setModel(model)
        self.assertEqual(w.text(), "100")

    def test_different_distance_with_model(self):
        w = QuantityEdit()
        model = DataModel(w)
        model.setValue(1000)
        lengthUnit = DataModel()
        lengthUnit.setValue(units.Unit.METER)
        w.setModelUnit(units.Unit.MILLIMETER)
        w.setDisplayedUnitModel(lengthUnit)
        w.setModel(model)
        self.assertEqual(w.text(), "1.0")

    def test_update_unit_model(self):
        w = QuantityEdit()
        model = DataModel(w)
        model.setValue(1000)
        lengthUnit = DataModel()
        lengthUnit.setValue(units.Unit.METER)
        w.setModelUnit(units.Unit.MILLIMETER)
        w.setDisplayedUnitModel(lengthUnit)
        w.setModel(model)
        self.assertEqual(w.text(), "1.0")
        lengthUnit.setValue(units.Unit.MILLIMETER)
        self.assertEqual(w.text(), "1000")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestQuantityEdit))
    return testsuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

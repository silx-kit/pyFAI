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

"""Test suite to scripts"""

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/10/2018"

import unittest
import logging

from silx.gui import qt
from .. import validators


_logger = logging.getLogger(__name__)


def typing(validator, text, pos, keys):
    """Simulate typing text from a cursor position using a validator"""

    for key in keys:
        text = text[:pos] + key + text[pos:]
        pos = pos + 1
        state, nextText, nextPos = validator.validate(text, pos)
        if state != qt.QValidator.Invalid:
            text, pos = nextText, nextPos
    return state, text, pos


class TestIntValidator(unittest.TestCase):

    def testValid(self):
        validator = validators.IntegerAndEmptyValidator()
        state, text, pos = validator.validate("10", 0)
        self.assertEqual(state, qt.QValidator.Acceptable)
        self.assertEqual(text, "10")
        self.assertEqual(pos, 0)
        self.assertEqual(validator.toValue(text), (10, True))

    def testInvalid(self):
        validator = validators.IntegerAndEmptyValidator()
        state, _, _ = validator.validate("1.0", 0)
        self.assertEqual(state, qt.QValidator.Invalid)

    def testAcceptableEmpty(self):
        validator = validators.IntegerAndEmptyValidator()
        state, text, pos = validator.validate("", 0)
        self.assertEqual(state, qt.QValidator.Acceptable)
        self.assertEqual(text, "")
        self.assertEqual(pos, 0)
        self.assertEqual(validator.toValue(text), (None, True))


class TestDoubleValidator(unittest.TestCase):

    def testValid(self):
        validator = validators.DoubleAndEmptyValidator()
        state, text, pos = validator.validate("1.2", 0)
        self.assertEqual(state, qt.QValidator.Acceptable)
        self.assertEqual(text, "1.2")
        self.assertEqual(pos, 0)
        self.assertEqual(validator.toValue(text), (1.2, True))

    def testRejectedEmpty(self):
        validator = validators.DoubleValidator()
        state, text, pos = validator.validate("", 0)
        self.assertNotEqual(state, qt.QValidator.Acceptable)
        self.assertEqual(text, "")
        self.assertEqual(pos, 0)

    def testAcceptableEmpty(self):
        validator = validators.DoubleAndEmptyValidator()
        state, text, pos = validator.validate("", 0)
        self.assertEqual(state, qt.QValidator.Acceptable)
        self.assertEqual(text, "")
        self.assertEqual(pos, 0)
        self.assertEqual(validator.toValue(text), (None, True))

    def testTypingFromEmpty(self):
        validator = validators.DoubleValidator()
        state, text, pos = typing(validator, "", 0, ["1", ".", "2", "3"])
        self.assertEqual(state, qt.QValidator.Acceptable)
        self.assertEqual(text, "1.23")
        self.assertEqual(pos, 4)

    def testTypingDotOverDot(self):
        validator = validators.DoubleValidator()
        state, text, pos = typing(validator, "10.23", 2, ["."])
        self.assertEqual(state, qt.QValidator.Acceptable)
        self.assertEqual(text, "10.23")
        self.assertEqual(pos, 3)

    def testTypingDotAfterDot(self):
        validator = validators.DoubleValidator()
        state, text, pos = typing(validator, "10.23", 4, ["."])
        self.assertEqual(state, qt.QValidator.Acceptable)
        self.assertEqual(text, "102.3")
        self.assertEqual(pos, 4)

    def testTypingDotBeforeDot(self):
        validator = validators.DoubleValidator()
        state, text, pos = typing(validator, "10.23", 1, ["."])
        self.assertEqual(state, qt.QValidator.Acceptable)
        self.assertEqual(text, "1.023")
        self.assertEqual(pos, 2)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestIntValidator))
    testsuite.addTest(loader(TestDoubleValidator))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
#
# ###########################################################################*/
__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "24/10/2017"


import logging
import os
import sys
import unittest


_logger = logging.getLogger(__name__)


def suite():

    test_suite = unittest.TestSuite()

    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
        # On Linux and no DISPLAY available (e.g., ssh without -X)
        _logger.warning('pyfai.gui tests disabled (DISPLAY env. variable not set)')

        class SkipGUITest(unittest.TestCase):
            def runTest(self):
                self.skipTest(
                    'pyfai.gui tests disabled (DISPLAY env. variable not set)')

        test_suite.addTest(SkipGUITest())
        return test_suite

    elif os.environ.get('WITH_QT_TEST', 'True') == 'False':
        # Explicitly disabled tests
        _logger.warning(
            "pyfai.gui tests disabled (env. variable WITH_QT_TEST=False)")

        class SkipGUITest(unittest.TestCase):
            def runTest(self):
                self.skipTest(
                    "pyfai.gui tests disabled (env. variable WITH_QT_TEST=False)")

        test_suite.addTest(SkipGUITest())
        return test_suite

    # Import here to avoid loading QT if tests are disabled

    from ..dialog import test as test_dialog

    test_suite.addTest(test_dialog.suite())
    return test_suite

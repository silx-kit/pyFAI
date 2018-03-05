#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2016-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test module for pyFAI GUI"""

from __future__ import absolute_import, division, print_function

__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/03/2018"

import sys
import os
import unittest
import logging
from pyFAI.test.utilstest import UtilsTest


_logger = logging.getLogger(__name__)


def suite():

    test_suite = unittest.TestSuite()

    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
        # On Linux and no DISPLAY available (e.g., ssh without -X)
        _logger.warning('pyFAI.gui tests disabled (DISPLAY env. variable not set)')

        class SkipGUITest(unittest.TestCase):
            def runTest(self):
                self.skipTest(
                    'pyFAI.gui tests disabled (DISPLAY env. variable not set)')

        test_suite.addTest(SkipGUITest())
        return test_suite

    from . import test_integrate_widget
    from . import test_scripts
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_integrate_widget.suite())
    test_suite.addTest(test_scripts.suite())
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

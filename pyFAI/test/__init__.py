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

"""Test module for pyFAI"""

from __future__ import absolute_import, division, print_function

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/05/2019"

import sys
import os
import unittest

from . import utilstest
from pyFAI.test.utilstest import test_options


# Issue https://github.com/silx-kit/fabio/pull/291
# Relative to fabio 0.8
import fabio
if fabio.hdf5image.Hdf5Image.close.__module__ != "fabio.hdf5image":
    def close(self):
        if self.hdf5 is not None:
            self.hdf5.close()
            self.hdf5 = None
            self.dataset = None
    fabio.hdf5image.Hdf5Image.close = close


def suite():
    # Importing locally to prevent premature initialization of UtilsTest
    # and preventing the test skipping
    from . import test_all
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_all.suite())
    return test_suite


def run_tests():
    """Run test complete test_suite"""
    test_options.configure()
    runner = unittest.TextTestRunner()
    if not runner.run(suite()).wasSuccessful():
        print("Test suite failed")
        return 1
    else:
        print("Test suite succeeded")
        return 0

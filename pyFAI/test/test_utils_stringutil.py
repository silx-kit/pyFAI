# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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


from __future__ import absolute_import, print_function, division

"""Test module for utils.string module"""

__author__ = "valentin.valls@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/01/2018"
__status__ = "development"
__docformat__ = 'restructuredtext'

import unittest
import logging
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from ..utils import stringutil


class TestUtilsString(unittest.TestCase):

    def test_default_behaviour_nothing(self):
        self.assertEquals(stringutil.safe_format("aaaa", {}), "aaaa")

    def test_default_behaviour_list(self):
        self.assertEquals(stringutil.safe_format("aaaa{0}{1}", (10, "aaaa")), "aaaa10aaaa")

    def test_default_behaviour_dict(self):
        self.assertEquals(stringutil.safe_format("aaaa{a}{b}", {"a": 10, "b": "aaaa"}), "aaaa10aaaa")

    def test_default_behaviour_object(self):
        args = {"a": (10, 1), "b": TestUtilsString}
        expected = "aaaa10TestUtilsString"
        self.assertEquals(stringutil.safe_format("aaaa{a[0]}{b.__name__}", args), expected)

    def test_missing_index(self):
        self.assertEquals(stringutil.safe_format("aaaa{0}{1}{2}", (10, "aaaa")), "aaaa10aaaa{2}")

    def test_missing_key(self):
        self.assertEquals(stringutil.safe_format("aaaa{a}{b}{c}", {"a": 10, "b": "aaaa"}), "aaaa10aaaa{c}")

    def test_missing_object(self):
        expected = "aaaa{a[0]}{b.__name__}"
        self.assertEquals(stringutil.safe_format("aaaa{a[0]}{b.__name__}", {}), expected)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestUtilsString))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()

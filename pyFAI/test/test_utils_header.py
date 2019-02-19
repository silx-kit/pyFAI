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


from __future__ import division, print_function, absolute_import

"""Test suite for math utilities library"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/02/2019"

import unittest
import numpy
import logging
import fabio

logger = logging.getLogger(__name__)

from .utilstest import UtilsTest
from ..utils import header_utils


class TestMonitorName(unittest.TestCase):

    def setUp(self):
        header = {
            "mon1": "100",
            "bad": "foo",
            "counter_pos": "12 13 14 foo",
            "counter_mne": "mon2 mon3 mon4 mon5",
            "bad_size_pos": "foo foo foo",
            "bad_size_mne": "mon2 mon3 mon4 mon5",
            "mne_not_exists_pos": "12 13 14 foo",
            "pos_not_exists_mne": "mon2 mon3 mon4 mon5",
        }
        self.image = fabio.numpyimage.numpyimage(numpy.array([]), header)

    def test_monitor(self):
        result = header_utils._get_monitor_value_from_edf(self.image, "mon1")
        self.assertEquals(100, result)

    def test_monitor_in_counter(self):
        result = header_utils._get_monitor_value_from_edf(self.image, "counter/mon3")
        self.assertEquals(13, result)

    def test_bad_monitor(self):
        self.assertRaises(header_utils.MonitorNotFound, header_utils._get_monitor_value_from_edf, self.image, "bad")

    def test_bad_monitor_in_counter(self):
        self.assertRaises(header_utils.MonitorNotFound, header_utils._get_monitor_value_from_edf, self.image, "counter/mon5")

    def test_bad_counter_syntax(self):
        self.assertRaises(header_utils.MonitorNotFound, header_utils._get_monitor_value_from_edf, self.image, "counter/mon5/1")

    def test_missing_monitor(self):
        self.assertRaises(header_utils.MonitorNotFound, header_utils._get_monitor_value_from_edf, self.image, "not_exists")

    def test_missing_counter(self):
        self.assertRaises(header_utils.MonitorNotFound, header_utils._get_monitor_value_from_edf, self.image, "not_exists/mon")

    def test_missing_counter_monitor(self):
        self.assertRaises(header_utils.MonitorNotFound, header_utils._get_monitor_value_from_edf, self.image, "counter/not_exists")

    def test_missing_counter_mne(self):
        self.assertRaises(header_utils.MonitorNotFound, header_utils._get_monitor_value_from_edf, self.image, "mne_not_exists/mon")

    def test_missing_counter_pos(self):
        self.assertRaises(header_utils.MonitorNotFound, header_utils._get_monitor_value_from_edf, self.image, "pos_not_exists/mon")

    def test_missing_counter_pos_element(self):
        self.assertRaises(header_utils.MonitorNotFound, header_utils._get_monitor_value_from_edf, self.image, "bad_size/mon")

    def test_edf_file_motor(self):
        image = fabio.open(UtilsTest.getimage("Pilatus1M.edf"))
        result = header_utils._get_monitor_value_from_edf(image, "motor/lx")
        self.assertEqual(result, -0.2)

    def test_edf_file_key(self):
        image = fabio.open(UtilsTest.getimage("Pilatus1M.edf"))
        result = header_utils._get_monitor_value_from_edf(image, "scan_no")
        self.assertEqual(result, 19)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestMonitorName))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()

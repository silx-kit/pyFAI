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
import shutil
import os
import fabio
import h5py

logger = logging.getLogger(__name__)

from .utilstest import UtilsTest
from ..utils import header_utils


class TestEdfMonitor(unittest.TestCase):

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


class TestHdf5Monitor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestHdf5Monitor, cls).setUpClass()
        cls.tempDir = os.path.join(UtilsTest.tempdir, cls.__name__)
        os.makedirs(cls.tempDir)
        cls.file = os.path.join(cls.tempDir, "file.h5")

        h5 = h5py.File(cls.file)
        data = numpy.array([1.0, 1.2, 1.3, 1.4]) + numpy.array([0, 0, 0]).reshape(-1, 1)
        h5["images"] = data.reshape(-1, 2, 2)
        h5["header/bar/vector"] = numpy.array([1.0, 1.2, 1.3])
        h5["header/bar/const"] = 1.5
        h5["header/bar/bad_type"] = numpy.array([1.0, 1.2j, 1.3]).reshape(1, 3, 1)
        h5["header/bar/bad_shape"] = numpy.array([1.0, 1.2, 1.3]).reshape(1, 3, 1)
        h5["header/bar/bad_size"] = numpy.array([1.0, 1.2])
        h5.close()

    @classmethod
    def tearDownClass(cls):
        super(TestHdf5Monitor, cls).tearDownClass()
        shutil.rmtree(cls.tempDir)
        cls.tempDir = None

    def test_vector_monitor(self):
        pass

    def test_const_monitor(self):
        monitor_key = "/header/bar/const"
        with fabio.open(self.file + "::/images") as image:
            for iframe in range(image.nframes):
                frame = image.getframe(iframe)
                result = header_utils.get_monitor_value(frame, monitor_key)
                self.assertEquals(1.5, result)

    def test_missing_monitor(self):
        monitor_key = "/header/bar/vector"
        expected_values = [1.0, 1.2, 1.3]
        with fabio.open(self.file + "::/images") as image:
            for iframe in range(image.nframes):
                frame = image.getframe(iframe)
                result = header_utils.get_monitor_value(frame, monitor_key)
                self.assertAlmostEqual(result, expected_values[iframe])

    def test_bad_type_monitor(self):
        monitor_key = "/header/bar/bad_type"
        with fabio.open(self.file + "::/images") as image:
            frame = image.getframe(0)
            with self.assertRaises(header_utils.MonitorNotFound):
                header_utils.get_monitor_value(frame, monitor_key)

    def test_bad_shape_monitor(self):
        monitor_key = "/header/bar/bad_shape"
        with fabio.open(self.file + "::/images") as image:
            frame = image.getframe(0)
            with self.assertRaises(header_utils.MonitorNotFound):
                header_utils.get_monitor_value(frame, monitor_key)

    def test_bad_size_monitor(self):
        monitor_key = "/header/bar/bad_size"
        expected_values = [1.0, 1.2, header_utils.MonitorNotFound]
        with fabio.open(self.file + "::/images") as image:
            for iframe in range(image.nframes):
                frame = image.getframe(iframe)
                expected_value = expected_values[iframe]
                if isinstance(expected_value, type(Exception)) and issubclass(expected_value, Exception):
                    with self.assertRaises(expected_value):
                        header_utils.get_monitor_value(frame, monitor_key)
                else:
                    result = header_utils.get_monitor_value(frame, monitor_key)
                    self.assertAlmostEqual(result, expected_value)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestEdfMonitor))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()

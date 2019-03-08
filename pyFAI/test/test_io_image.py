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

"Test suite for worker"

from __future__ import absolute_import, division, print_function

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/03/2019"


import unittest
import logging
import os.path
import shutil
import numpy
import h5py

from silx.io.url import DataUrl

from ..io import image as image_mdl
from . import utilstest


logger = logging.getLogger(__name__)


class TestReadImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = os.path.join(utilstest.test_options.tempdir, cls.__name__)
        os.makedirs(cls.directory)

        cls.a = os.path.join(cls.directory, "a.npy")
        cls.b = os.path.join(cls.directory, "b.h5")

        cls.shape = (2, 2)
        ones = numpy.ones(shape=cls.shape)
        numpy.save(cls.a, ones)

        with h5py.File(cls.b) as h5:
            h5["/image"] = ones
            h5["/number"] = 10
            h5["/group/foo"] = 10

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory)

    def test_image_path(self):
        abs_a = os.path.abspath(self.a)
        image = image_mdl.read_image_data(abs_a)
        self.assertIsNotNone(image)

    def test_silx_url(self):
        abs_a = os.path.abspath(self.a)
        url = DataUrl(abs_a).path()
        image = image_mdl.read_image_data(url)
        self.assertIsNotNone(image)

    def test_not_existing_file(self):
        with self.assertRaises(Exception):
            image_mdl.read_image_data("fooobar.not.existing")

    def test_fabio_hdf5_file(self):
        abs_b = os.path.abspath(self.b)
        image = image_mdl.read_image_data(abs_b + "::/image")
        self.assertIsNotNone(image)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestReadImage))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

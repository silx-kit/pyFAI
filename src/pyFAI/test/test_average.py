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

"""test suite for average library
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "21/05/2024"

import unittest
import numpy
import os
import ast
import logging
import fabio
from .utilstest import UtilsTest
from .. import average

logger = logging.getLogger(__name__)

# TODO add tests from
# - boundingBox
# - removeSaturatedPixel


class TestAverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls)->None:
        super(TestAverage, cls).setUpClass()
        cls.rng = UtilsTest.get_rng()
        cls.unbinned = cls.rng.random((64, 32))
        cls.dark = cls.unbinned.astype("float32")
        cls.flat = 1 + cls.rng.random((64, 32))
        cls.raw = cls.flat + cls.dark
        cls.tmp_file = os.path.join(UtilsTest.tempdir, "testUtils_average.edf")

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.dark = self.flat = self.raw = self.tmp_file = None

    def test_average_dark(self):
        """
        Some testing for dark averaging
        """
        one = average.average_dark([self.dark])
        self.assertEqual(abs(self.dark - one).max(), 0, "data are the same")

        two = average.average_dark([self.dark, self.dark])
        self.assertEqual(abs(self.dark - two).max(), 0, "data are the same: mean test")

        three = average.average_dark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark)], "median")
        self.assertEqual(abs(self.dark - three).max(), 0, "data are the same: median test")

        four = average.average_dark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark)], "min")
        self.assertEqual(abs(numpy.zeros_like(self.dark) - four).max(), 0, "data are the same: min test")

        five = average.average_dark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark)], "max")
        self.assertEqual(abs(numpy.ones_like(self.dark) - five).max(), 0, "data are the same: max test")

        six = average.average_dark([numpy.ones_like(self.dark), self.dark, numpy.zeros_like(self.dark), self.dark, self.dark], "median", .001)
        self.assertTrue(abs(self.dark - six).max() < 1e-4, "data are the same: test threshold")

    def test_quantile(self):
        shape = (100, 100)
        dtype = numpy.float32
        image1 = self.rng.random(shape).astype(dtype)
        image2 = self.rng.random(shape).astype(dtype)
        image3 = numpy.zeros(shape, dtype=dtype)
        image4 = numpy.ones(shape, dtype=dtype)
        image5 = self.rng.random(shape).astype(dtype)
        expected = (image1 + image2 + image5) / 3.0
        result = average.average_images(
            [image1, image2, image3, image4, image5],
            quantiles=(0.2, 0.8),
            threshold=0,
            filter_="quantiles",
            fformat=None)

        self.assertTrue(numpy.allclose(result, expected),
                        "average with quantiles gives bad results")

    def test_output_file(self):
        if fabio.hexversion < 262147:
            self.skipTest("The version of the FabIO library is too old: %s, please upgrade to 0.4+. Skipping test for now" % fabio.version)
        file_name = average.average_images([self.raw], darks=[self.dark], flats=[self.flat], threshold=0, output=self.tmp_file)
        with fabio.open(file_name) as fimg:
            result = fimg.data
        expected = numpy.ones_like(self.dark)
        self.assertTrue(abs(expected - result).mean() < 1e-2, "average_images")

    def test_min_filter(self):
        algorith = average.MinAveraging()
        algorith.init()
        min_array = numpy.array([[1]])
        max_array = numpy.array([[500]])
        algorith.add_image(min_array)
        algorith.add_image(max_array)
        result = algorith.get_result()
        numpy.testing.assert_array_almost_equal(result, min_array, decimal=3)

    def test_max_filter(self):
        algorith = average.MaxAveraging()
        algorith.init()
        min_array = numpy.array([[1]])
        max_array = numpy.array([[500]])
        algorith.add_image(min_array)
        algorith.add_image(max_array)
        result = algorith.get_result()
        numpy.testing.assert_array_almost_equal(result, max_array, decimal=3)

    def test_sum_filter(self):
        algorith = average.SumAveraging()
        algorith.init()
        array1 = numpy.array([[1, -20, 500]])
        array2 = numpy.array([[500, 1, -20]])
        algorith.add_image(array1)
        algorith.add_image(array2)
        result = algorith.get_result()
        numpy.testing.assert_array_almost_equal(result, (array1 + array2), decimal=3)

    def test_mean_filter(self):
        algorith = average.MeanAveraging()
        algorith.init()
        array1 = numpy.array([[1, -20, 500]])
        array2 = numpy.array([[500, 1, -20]])
        algorith.add_image(array1)
        algorith.add_image(array2)
        result = algorith.get_result()
        numpy.testing.assert_array_almost_equal(result, (array1 + array2) * 0.5, decimal=3)

    def test_average_monitor(self):
        data1 = numpy.array([[1.0, 3.0], [3.0, 4.0]])
        data2 = numpy.array([[2.0, 2.0], [1.0, 4.0]])
        data3 = numpy.array([[3.0, 1.0], [2.0, 4.0]])
        mon1, mon2, mon3 = 0.1, 1.0, 3.1
        image1 = fabio.numpyimage.numpyimage(data1)
        image1.header["mon"] = str(mon1)
        image2 = fabio.numpyimage.numpyimage(data2)
        image2.header["mon"] = str(mon2)
        image3 = fabio.numpyimage.numpyimage(data3)
        image3.header["mon"] = str(mon3)
        image_ignored = fabio.numpyimage.numpyimage(data3)

        expected_result = data1 / mon1 + data2 / mon2 + data3 / mon3
        filename = average.average_images([image1, image2, image3, image_ignored], threshold=0, filter_="sum", monitor_key="mon", output=self.tmp_file)
        with fabio.open(filename) as fimg:
            result = fimg.data
        numpy.testing.assert_array_almost_equal(result, expected_result, decimal=3)

    def test_writed_properties(self):
        writer = average.MultiFilesAverageWriter("foo", "edf", dry_run=True)
        algorithm = average.AverageDarkFilter(filter_name="quantiles", cut_off=None, quantiles=(0.2, 0.8))
        image1 = self.rng.random((1, 1)).astype(numpy.float32)

        averager = average.Average()
        averager.set_writer(writer)
        averager.set_images([image1])
        averager.add_algorithm(algorithm)
        averager.process()

        fabio_image = writer.get_fabio_image(algorithm)
        header = fabio_image.header
        self.assertEqual(ast.literal_eval(header["cutoff"]), None)
        self.assertEqual(ast.literal_eval(header["quantiles"]), (0.2, 0.8))


class TestAverageAlgorithmFactory(unittest.TestCase):

    def test_max(self):
        alrorithm = average.create_algorithm("max")
        self.assertTrue(isinstance(alrorithm, average.MaxAveraging))

    def test_sum(self):
        alrorithm = average.create_algorithm("sum")
        self.assertTrue(isinstance(alrorithm, average.SumAveraging))

    def test_median(self):
        alrorithm = average.create_algorithm("median")
        self.assertTrue(isinstance(alrorithm, average.AverageDarkFilter))
        self.assertEqual(alrorithm.name, "median")

    def test_sum_cutoff(self):
        alrorithm = average.create_algorithm("sum", cut_off=0.5)
        self.assertTrue(isinstance(alrorithm, average.AverageDarkFilter))
        self.assertEqual(alrorithm.name, "sum")

    def test_quantiles_no_params(self):
        self.assertRaises(average.AlgorithmCreationError, average.create_algorithm, "quantiles")

    def test_unknown_filter(self):
        self.assertRaises(average.AlgorithmCreationError, average.create_algorithm, "not_existing")


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestAverage))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()

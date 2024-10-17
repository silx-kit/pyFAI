#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2024-2024 Australian Synchrotron
#                  2024-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Emily Massahud
#                            Jerome Kieffer
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

"""Test suite for RingExtraction class."""

__authors__ = ["Emily Massahud", "Jérôme Kieffer"]
__contact__ = "Jérôme.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/10/2024"

import sys
import unittest
from unittest import mock

import numpy

from .. import ring_extraction
from ..ring_extraction import RingExtraction


class RingExtractionTestBase(unittest.TestCase):
    def setUp(self):
        self.single_geometry = mock.MagicMock()
        self.massif = mock.MagicMock()
        self.ring_extraction = RingExtraction(self.single_geometry, self.massif)
        two_theta_values = numpy.array([1, 2, 3, 4, 5])
        self.ring_extraction.two_theta_values = two_theta_values
        self.ones_array = numpy.ones((5, 5))
        self.ring_extraction.two_theta_array = self.ones_array
        self.expected_two_theta_dictionary = {
            "min": 0.75,
            "max": 1.25,
        }

        self.numpy_patcher = mock.patch(ring_extraction.__name__ + ".numpy", autospec=True)
        self.mock_numpy = self.numpy_patcher.start()

    def tearDown(self):
        mock.patch.stopall()


class TestCalibAttributes(RingExtractionTestBase):
    def test_calib_attributes(self):
        ring_extraction_attributes = [
            "single_geometry",
            "image",
            "detector",
            "calibrant",
            "massif",
            "two_theta_array",
            "two_theta_values",
        ]

        for attribute_name in ring_extraction_attributes:
            with self.subTest(attribute_name=attribute_name):
                self.assertTrue(
                    hasattr(self.ring_extraction, attribute_name),
                    f"{attribute_name} is missing",
                )


class TestExtractControlPoints(RingExtractionTestBase):
    def setUp(self):
        super().setUp()
        self.mock_control_points = mock.patch((ring_extraction.__name__ + ".ControlPoints")).start()
        self.mock_extract_peaks_in_one_ring = mock.patch.object(
            self.ring_extraction, "extract_list_of_peaks_in_one_ring"
        ).start()

    def tearDown(self):
        RingExtractionTestBase.tearDown(self)
        self.mock_control_points = self.mock_extract_peaks_in_one_ring = None

    def test_extract_control_points(self):
        # Act
        self.ring_extraction.extract_control_points(max_number_of_rings=3)

        # Assert
        self.mock_control_points.assert_called_once_with(calibrant=self.ring_extraction.calibrant)

        self.assertEqual(self.mock_extract_peaks_in_one_ring.call_count, 3)

        # work arround for this issue https://github.com/python/cpython/issues/84147 and
        # #2189 related to threads being used and the function callability checked.
        self.mock_extract_peaks_in_one_ring.assert_called()
        # self.mock_extract_peaks_in_one_ring.assert_has_calls(
        #     [mock.call(0, 1), mock.call(1, 1), mock.call(2, 1)],
        #     any_order=False)


class TestExtractOneRing(RingExtractionTestBase):
    def setUp(self):
        super().setUp()
        self.ring_extraction.detector.mask = "detector_mask_string"

        self.mock_calculate_num_of_points_to_keep = mock.patch.object(
            RingExtraction, "_calculate_num_of_points_to_keep", return_value=5
        ).start()
        self.mock_remove_low_intensity_pixels = mock.patch.object(
            RingExtraction,
            "_remove_low_intensity_pixels_from_mask",
            return_value=(numpy.ones((5, 5)), 0),
        ).start()
        self.mock_create_mask_around_ring = mock.patch.object(
            RingExtraction, "_create_mask_around_ring", return_value=self.ones_array
        ).start()
        self.mock_marching_squares = mock.patch(
            ring_extraction.__name__ + ".marchingsquares.MarchingSquaresMergeImpl"
        ).start()
        mock_marching_squares_instance = self.mock_marching_squares.return_value

        self.pixels_at_two_theta_level = numpy.array(([1, 1], [2, 2], [3, 3]))
        mock_marching_squares_instance.find_pixels.return_value = self.pixels_at_two_theta_level

        self.mock_calculate_min_distance_between_control_points = mock.patch.object(
            RingExtraction,
            "_calculate_min_distance_between_control_points",
            return_value=0.1,
        ).start()

    def test_extract_list_of_peaks_in_one_ring(
        self,
    ):
        # Act
        self.ring_extraction.extract_list_of_peaks_in_one_ring(ring_index=1)

        # Assert
        self.mock_marching_squares.assert_called_once()
        self.assertTrue(
            numpy.array_equal(self.mock_marching_squares.call_args_list[0][0][0], self.ones_array)
        )
        self.assertEqual(
            self.mock_marching_squares.call_args_list[0][1]["mask"],
            "detector_mask_string",
        )
        self.assertTrue(self.mock_marching_squares.call_args_list[0][1]["use_minmax_cache"], True)

        self.mock_create_mask_around_ring.assert_called_once_with(1)
        self.mock_remove_low_intensity_pixels.assert_called_once()
        self.assertTrue(
            numpy.array_equal(
                self.mock_remove_low_intensity_pixels.call_args_list[0][0][0],
                self.ones_array,
            )
        )
        self.mock_calculate_num_of_points_to_keep.assert_called_once_with(
            self.pixels_at_two_theta_level, 1
        )

        self.mock_calculate_min_distance_between_control_points.assert_called_once_with(
            [[1, 1], [2, 2], [3, 3]], 1
        )

        self.ring_extraction.massif.peaks_from_area.assert_called_once()
        self.assertTrue(
            numpy.array_equal(
                self.ring_extraction.massif.peaks_from_area.call_args_list[0][0][0],
                self.ones_array,
            )
        )
        self.assertEqual(
            self.ring_extraction.massif.peaks_from_area.call_args_list[0][1]["keep"],
            5,
        )
        self.assertEqual(
            self.ring_extraction.massif.peaks_from_area.call_args_list[0][1]["dmin"],
            0.1,
        )
        self.assertEqual(
            self.ring_extraction.massif.peaks_from_area.call_args_list[0][1]["seed"],
            set(((1, 1), (2, 2), (3, 3))),
        )
        self.assertEqual(
            self.ring_extraction.massif.peaks_from_area.call_args_list[0][1]["ring"],
            1,
        )


class TestGetUniqueTwoThetaValuesInImage(RingExtractionTestBase):
    def test_get_unique_two_theta_values_in_image(self):
        two_theta_values = numpy.array([0.5, 1, 1.5, 2, 2.5, 1])
        self.ring_extraction.calibrant = mock.MagicMock()

        self.mock_numpy.array.return_value = two_theta_values

        # Act
        self.ring_extraction._get_unique_two_theta_values_in_image()

        # Assert
        self.ring_extraction.calibrant.get_2th.assert_called_once_with()
        self.mock_numpy.unique.assert_called_once_with(two_theta_values)


class TestCreateMaskAroundRing(RingExtractionTestBase):
    def setUp(self):
        super().setUp()
        zeros_array = numpy.zeros((3, 3))
        self.ring_extraction.two_theta_array = zeros_array
        self.mock_get_two_theta_min_max = mock.patch.object(
            RingExtraction,
            "_get_two_theta_min_max",
            return_value=self.expected_two_theta_dictionary,
        ).start()

    def test_create_mask_around_ring(self):
        # Act
        self.ring_extraction._create_mask_around_ring(ring_index=0)

        # Assert
        self.mock_get_two_theta_min_max.assert_called_once_with(0)
        self.mock_numpy.logical_and.assert_called_once()
        first_call_logical_and = (
            numpy.zeros((3, 3), dtype=bool),
            numpy.ones((3, 3), dtype=bool),
        )
        self.assertTrue(
            numpy.array_equal(
                self.mock_numpy.logical_and.call_args_list[0][0], first_call_logical_and
            )
        )


class TestGetTwoThetaMinMax(RingExtractionTestBase):
    def test_get_two_theta_min_max(self):
        # Act
        two_theta_dictionary = self.ring_extraction._get_two_theta_min_max(0)

        # Assert
        self.assertEqual(
            set(two_theta_dictionary.keys()),
            set(self.expected_two_theta_dictionary.keys()),
        )


class TestRemoveLowIntensityPixelsFromMask(RingExtractionTestBase):
    def setUp(self):
        super().setUp()
        self.mock_numpy.logical_and.side_effect = [
            numpy.ones((3, 3), dtype=bool),
            numpy.ones((3, 3), dtype=bool),
            numpy.ones((32, 32), dtype=bool),
        ]
        self.mock_calculate_mean_and_std_of_intensities_in_mask = mock.patch.object(
            RingExtraction,
            "_calculate_mean_and_std_of_intensities_in_mask",
            return_value=(0.9, 0.1),
        ).start()
        self.ring_extraction.image = numpy.ones((3, 3))
        self.mock_mask = numpy.ones((3, 3), dtype=bool)

    def test_remove_low_intensity_pixels_from_mask(self):
        # Act
        (
            final_mask,
            upper_limit,
        ) = self.ring_extraction._remove_low_intensity_pixels_from_mask(self.mock_mask)

        # Assert
        self.assertIsInstance(final_mask, numpy.ndarray)
        self.assertEqual(upper_limit, 0.9)
        self.mock_calculate_mean_and_std_of_intensities_in_mask.assert_called_once_with(
            self.mock_mask
        )

        expected_first_call = (numpy.zeros((3, 3)), numpy.ones((3, 3)))
        expected_second_call = (numpy.ones((3, 3)), numpy.ones((3, 3)))
        actual_calls = self.mock_numpy.logical_and.call_args_list
        self.assertEqual(len(actual_calls), 2, "numpy.logical_and was not called twice as expected")
        self.assertTrue(numpy.array_equal(actual_calls[0][0][0], expected_first_call[0]))
        self.assertTrue(numpy.array_equal(actual_calls[0][0][1], expected_first_call[1]))
        self.assertTrue(numpy.array_equal(actual_calls[1][0][0], expected_second_call[0]))
        self.assertTrue(numpy.array_equal(actual_calls[1][0][1], expected_second_call[1]))

    def test_remove_low_intensity_pixels_from_mask_enough_points(self):
        # Act
        self.ring_extraction._remove_low_intensity_pixels_from_mask(self.mock_mask)

        # Assert
        self.mock_calculate_mean_and_std_of_intensities_in_mask.assert_called_once_with(
            self.mock_mask
        )

        expected_call_args = (numpy.zeros((3, 3)), numpy.ones((3, 3)))
        actual_call_args = self.mock_numpy.logical_and.call_args_list[0][0]
        self.assertTrue(numpy.array_equal(actual_call_args[0], expected_call_args[0]))
        self.assertTrue(numpy.array_equal(actual_call_args[1], expected_call_args[1]))


class TestCalcMeanStdOfIntensitiesInMask(RingExtractionTestBase):
    def test_calculate_mean_and_std_of_intensities_in_mask(self):
        mock_mask = mock.MagicMock()
        mock_flattened_mask = numpy.ones(9, dtype=bool)
        mock_mask.flatten.return_value = mock_flattened_mask

        mock_image = mock.MagicMock()
        mock_flattened_image = mock.MagicMock()
        mock_image.flatten.return_value = mock_flattened_image
        mock_pixel_intensities_in_mask = mock_flattened_image[numpy.where(mock_flattened_mask)]
        mean = 1
        std = 0.1
        mock_pixel_intensities_in_mask.mean.return_value = mean
        mock_pixel_intensities_in_mask.std.return_value = std

        self.ring_extraction.image = mock_image

        # Act
        result = self.ring_extraction._calculate_mean_and_std_of_intensities_in_mask(mock_mask)

        # Assert
        self.assertEqual(result, (mean, std))
        mock_mask.flatten.assert_called_once_with()
        mock_image.flatten.assert_called_once_with()
        self.mock_numpy.where.assert_called_once_with(mock_flattened_mask)


class TestCalcPointsToKeep(RingExtractionTestBase):
    def test_calculate_num_of_points_to_keep(self):
        self.ring_extraction.image = numpy.zeros((10, 10))
        pixel_list_at_two_theta_level = numpy.array([[2, 2], [3, 3], [4, 4]])
        sample_array = numpy.array([[1, 2], [3, 4]])

        self.mock_numpy.unique.return_value = numpy.array([1, 2, 3, 4])
        self.mock_numpy.rad2deg.return_value = sample_array
        # Act
        points_to_keep = self.ring_extraction._calculate_num_of_points_to_keep(
            pixel_list_at_two_theta_level, 1
        )

        # Assert
        self.single_geometry.geometry_refinement.chiArray.assert_called_once_with((10, 10))
        self.mock_numpy.unique.assert_called_once_with(sample_array)
        self.assertEqual(points_to_keep, 4)


class TestCalcMinDistBetweenControlPoints(RingExtractionTestBase):
    def setUp(self):
        super().setUp()
        self.numpy_patcher.stop()
        self.beam_centre_coords = numpy.array((1.5, 2.1))
        self.mock_get_beam_centre_coords = mock.patch.object(
            self.ring_extraction,
            "_get_beam_centre_coords",
            return_value=self.beam_centre_coords,
        ).start()

    def test_calculate_min_distance_between_control_points(self):
        # Arrange
        pixel_list_at_two_theta_level = [[5, 5]]

        # Act
        min_dist_between_control_points = (
            self.ring_extraction._calculate_min_distance_between_control_points(
                pixel_list_at_two_theta_level, 1
            )
        )

        # Assert
        self.mock_get_beam_centre_coords.assert_called_once_with()
        self.assertAlmostEqual(min_dist_between_control_points, 0.0793309)


class TestGetBeamCentreCoords(RingExtractionTestBase):
    def setUp(self):
        super().setUp()
        self.numpy_patcher.stop()
        self.mock_ai = self.single_geometry.get_ai.return_value
        self.mock_ai.getFit2D.return_value = {"centerX": 1, "centerY": 2}

    def test_get_beam_centre_coords(self):
        # Act
        beam_centre = self.ring_extraction._get_beam_centre_coords()

        # Assert
        self.single_geometry.get_ai().getFit2D.assert_called_once_with()
        self.assertTrue(numpy.array_equal(beam_centre, numpy.array((2, 1))))


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestCalibAttributes))
    testsuite.addTest(loader(TestExtractControlPoints))
    testsuite.addTest(loader(TestExtractOneRing))
    testsuite.addTest(loader(TestGetUniqueTwoThetaValuesInImage))
    testsuite.addTest(loader(TestCreateMaskAroundRing))
    testsuite.addTest(loader(TestGetTwoThetaMinMax))
    testsuite.addTest(loader(TestRemoveLowIntensityPixelsFromMask))
    testsuite.addTest(loader(TestCalcMeanStdOfIntensitiesInMask))
    testsuite.addTest(loader(TestCalcPointsToKeep))
    testsuite.addTest(loader(TestCalcMinDistBetweenControlPoints))
    testsuite.addTest(loader(TestGetBeamCentreCoords))
    return testsuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

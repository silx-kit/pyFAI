# -*- coding: utf-8 -*-
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

"""
Module used for extracting control points in a calibrant image to be used for geometry refinement.
"""

from __future__ import annotations

__authors__ = ["Emily Massahud", "Jérôme Kieffer"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/02/2024"
__status__ = "development"


import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy
from silx.image import marchingsquares

from .control_points import ControlPoints
from .goniometer import SingleGeometry
from .massif import Massif

logger = logging.getLogger(__name__)


class RingExtraction:
    """Class to perform extraction of control points from a calibration image."""

    def __init__(self, single_geometry: SingleGeometry, massif: Optional[Massif] = None):
        """
        Parameters
        ----------
        single_geometry : SingleGeometry
            instance of pyFAI class
        massif : Optional[Massif]
            instance of Massif class, which defines an area around a peak; it is used to find
            neighboring peaks, by default None
        """

        self.single_geometry = single_geometry
        self.image = self.single_geometry.image
        self.detector = self.single_geometry.detector
        self.calibrant = self.single_geometry.calibrant

        if massif:
            self.massif = massif
        else:
            if self.detector:
                mask = self.detector.dynamic_mask(self.image)
            else:
                mask = None
            self.massif = Massif(self.image, mask)

        self.two_theta_array = self.single_geometry.geometry_refinement.twoThetaArray()
        self.two_theta_values = self._get_unique_two_theta_values_in_image()

    def extract_control_points(
        self, max_number_of_rings: Optional[int] = None, points_per_degree: float = 1
    ) -> ControlPoints:
        """
        Primary method of RingExtraction class. Runs extract_control_points_in_one_ring for all
        rings in image, or for max_number_of_rings, if not None.

        Parameters
        ----------
        max_number_of_rings : Optional[int]
            Maximum number of diffraction rings to extract points from, by default None
        points_per_degree : float, optional
            number of control points per azimuthal degree (increase for better precision), by
            default 1.0

        Returns
        -------
        ControlPoints
            Instance of the pyFAI class with extracted peaks
        """

        control_points = ControlPoints(calibrant=self.calibrant)

        if max_number_of_rings is None:
            max_number_of_rings = self.two_theta_values.size

        tasks = {}
        with ThreadPoolExecutor() as executor:
            for ring_index in range(min(max_number_of_rings, self.two_theta_values.size)):
                future = executor.submit(
                    self.extract_list_of_peaks_in_one_ring, ring_index, points_per_degree
                )
                tasks[future] = ring_index

            for future in as_completed(tasks):
                list_of_peaks_in_ring = future.result()
                ring_index = tasks[future]
                if list_of_peaks_in_ring:
                    control_points.append(list_of_peaks_in_ring, ring_index)

        return control_points

    def extract_list_of_peaks_in_one_ring(
        self, ring_index: int, points_per_degree: float = 1.0
    ) -> Optional[list[tuple[float, float]]]:
        """
        Using massif.peaks_from_area, get all pixel coordinates inside a mask of pixels around a
        diffraction ring above a certain intensity, provided the desired number of points to keep,
        and the minimum distance between peaks.

        Parameters
        ----------
        ring_index : int
            ring number
        points_per_degree : float, optional
            number of control points per azimuthal degree (increase for better precision), by
            default 1.0

        Returns
        -------
        Optional[list[tuple[float, float]]]
            peaks at given ring index
        """
        marching_squares_algorithm = marchingsquares.MarchingSquaresMergeImpl(
            self.two_theta_array, mask=self.detector.mask, use_minmax_cache=True
        )

        initial_mask = self._create_mask_around_ring(ring_index)

        mask_size = initial_mask.sum(dtype=int)
        if mask_size > 0:
            final_mask, upper_limit = self._remove_low_intensity_pixels_from_mask(initial_mask)

            pixels_at_two_theta_level = marching_squares_algorithm.find_pixels(
                self.two_theta_values[ring_index]
            )
            seeds = set((i[0], i[1]) for i in pixels_at_two_theta_level if final_mask[i[0], i[1]])

            num_points_to_keep = self._calculate_num_of_points_to_keep(
                pixels_at_two_theta_level, points_per_degree
            )
            if num_points_to_keep > 0:
                min_distance_between_control_points = (
                    self._calculate_min_distance_between_control_points(
                        pixels_at_two_theta_level.tolist(), points_per_degree
                    )
                )

                logger.info(
                    "Extracting datapoints for ring %s (2theta = %.2f deg); searching for %i pts"
                    " out of %i with I>%.1f, dmin=%.1f",
                    ring_index,
                    numpy.degrees(self.two_theta_values[ring_index]),
                    num_points_to_keep,
                    final_mask.sum(dtype=int),
                    upper_limit,
                    min_distance_between_control_points,
                )

                return self.massif.peaks_from_area(
                    final_mask,
                    keep=num_points_to_keep,
                    dmin=min_distance_between_control_points,
                    seed=seeds,
                    ring=ring_index,
                )
            return None
        return None

    def _get_unique_two_theta_values_in_image(self) -> numpy.ndarray:
        """
        Calculates all two theta values covered by the image with the current detector and geometry

        Returns
        -------
        numpy.ndarray
            array containing all two theta values for calibrant at present wavelength
        """
        two_theta_values = numpy.unique(
            numpy.array([i for i in self.calibrant.get_2th() if i is not None])
        )
        largest_two_theta_in_image = self.two_theta_array.max()
        two_theta_values_in_image = numpy.array(
            [two_theta for two_theta in two_theta_values if two_theta <= largest_two_theta_in_image]
        )
        return two_theta_values_in_image

    def _create_mask_around_ring(self, ring_index: int) -> numpy.ndarray:
        """
        Creates a mask of valid pixels around each ring, of thickness equal to 1/2 the distance
        between the centre of two adjacent rings.


        Parameters
        ----------
        ring_index : int
            Ring number

        Returns
        -------
        numpy.ndarray
            Mask of valid pixels around each ring
        """
        two_theta_min_max = self._get_two_theta_min_max(ring_index)
        two_theta_min, two_theta_max = (two_theta_min_max["min"], two_theta_min_max["max"])

        initial_mask = numpy.logical_and(
            self.two_theta_array >= two_theta_min, self.two_theta_array < two_theta_max
        )
        if self.detector.mask is not None:
            detector_mask_bool = self.detector.mask.astype(bool)
            initial_mask &= ~detector_mask_bool

        return initial_mask

    def _get_two_theta_min_max(self, ring_index: int) -> dict[str, float]:
        """
        Calculates the distance between 2 consecutive rings (in radians), and returns an
        interval equal to +- 1/4 of this distance centred around the ring position, namely
        a minimum and maximum values for the two theta of the ring.

        Parameters
        ----------
        ring_index : int
            ring number
        Returns
        -------
        dict[str, float]
            dictionary containing minimum and maximum values for two theta
        """
        if ring_index == 0:
            delta_two_theta = (self.two_theta_values[1] - self.two_theta_values[0]) / 4
        else:
            delta_two_theta = (
                self.two_theta_values[ring_index] - self.two_theta_values[ring_index - 1]
            ) / 4

        two_theta_min = self.two_theta_values[ring_index] - delta_two_theta
        two_theta_max = self.two_theta_values[ring_index] + delta_two_theta

        return {"min": two_theta_min, "max": two_theta_max}

    def _remove_low_intensity_pixels_from_mask(
        self, mask: numpy.ndarray
    ) -> tuple[numpy.ndarray, float]:
        """
        Creates a final mask of valid pixels to be used in peak extraction, by removing low
        intensity pixels from the initial mask

        Parameters
        ----------
        mask : numpy.ndarray
            mask of valid pixels

        Returns
        -------
        tuple[numpy.ndarray, float]
            final mask of valid pixels, upper limit of intensity to mask
        """
        mean, std = self._calculate_mean_and_std_of_intensities_in_mask(mask)
        upper_limit = mean + std
        high_pixel_intensity_coords = self.image > upper_limit
        final_mask = numpy.logical_and(high_pixel_intensity_coords, mask)
        final_mask_size = final_mask.sum(dtype=int)
        minimum_mask_size = 1000  # copied this from original method, don't know why this number
        if final_mask_size < minimum_mask_size:
            upper_limit = mean
            final_mask = numpy.logical_and(self.image > upper_limit, mask)
            final_mask_size = final_mask.sum()
        return final_mask, upper_limit

    def _calculate_mean_and_std_of_intensities_in_mask(
        self, mask: numpy.ndarray
    ) -> tuple[float, float]:
        """
        Calculates mean and standard deviation of pixel intensities of image which are emcompassed
        by mask.

        Parameters
        ----------
        mask : numpy.ndarray
            mask of valid pixels

        Returns
        -------
        tuple[float, float]
            mean and standard deviation of pixel intensities
        """
        flattened_mask = mask.flatten()
        flattened_image = self.image.flatten()
        pixel_intensities_in_mask = flattened_image[numpy.where(flattened_mask)]
        mean = pixel_intensities_in_mask.mean()
        std = pixel_intensities_in_mask.std()

        return mean, std

    def _calculate_num_of_points_to_keep(
        self, pixels_at_two_theta_level: numpy.ndarray, points_per_degree: float
    ) -> int:
        """
        Calculate number of azimuthal degrees in ring, then multiply by self.points_per_degree

        Parameters
        ----------
        pixels_at_two_theta_level : numpy.ndarray
            Array of pixels in the image located in the ring at the current two theta value
        points_per_degree : float
            number of control points per azimuthal degree (increase for better precision)

        Returns
        -------
        int
            Number of points to keep as control points
        """
        image_shape = self.image.shape
        azimuthal_angles_array = self.single_geometry.geometry_refinement.chiArray(image_shape)
        azimuthal_degrees_array_in_ring = azimuthal_angles_array[
            pixels_at_two_theta_level[:, 0].clip(0, image_shape[0]),
            pixels_at_two_theta_level[:, 1].clip(0, image_shape[1]),
        ]
        number_of_azimuthal_degrees = numpy.unique(
            numpy.rad2deg(azimuthal_degrees_array_in_ring).round()
        ).size
        return int(number_of_azimuthal_degrees * points_per_degree)

    def _calculate_min_distance_between_control_points(
        self, pixel_list_at_two_theta_level: list[list[int]], points_per_degree: float
    ) -> float:
        """
        Get a random value from the ring and subtract from beam_centre_coords to calculate ring
        radius; grabbing any value from pixel_list_at_two_theta_level will lead to the same radius.
        then calculate the distance between 2 points separated by 1 degree in ring circumference:
            ring_radius * 0.0174533 (equals a degree in radians)
        only good for shapes which are close to circular.

        Parameters
        ----------
        pixel_list_at_two_theta_level : list[list[int]]
            List of pixels in the image located in the ring at the current two theta value

        Returns
        -------
        float
            Minimum accepted distance between 2 control points
        """

        # TODO implementation for elliptical rings needs semi-minor axis to ensure the min value for
        # arc, and ellipse centre, which differs from beam centre in elliptical projections
        beam_centre_coords = self._get_beam_centre_coords()
        random_point_in_ring = numpy.array(random.sample(pixel_list_at_two_theta_level, 1)[0])
        ring_radius_px = numpy.sqrt(sum(x**2 for x in (random_point_in_ring - beam_centre_coords)))
        one_degree = 1
        ring_dist_in_one_deg = ring_radius_px * numpy.radians(one_degree)
        return ring_dist_in_one_deg / points_per_degree

    def _get_beam_centre_coords(self) -> numpy.ndarray:
        """
        Use AzimuthalIntegrator's method `getFit2D` to get the pixel coordinates of the beam centre

        Returns
        -------
        numpy.ndarray
            beam centre coordinates [y,x], to match pyFAI order.
        """
        fit_2d = self.single_geometry.get_ai().getFit2D()
        beam_centre_x = fit_2d["centerX"]
        beam_centre_y = fit_2d["centerY"]
        return numpy.array([beam_centre_y, beam_centre_x])

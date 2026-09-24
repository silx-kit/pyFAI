#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2026-2026 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suite validating the detector orientation against the position of
control points in the laboratory frame.

The detector `orientation` is only a re-indexing of the pixels: it changes which
array index maps onto a given physical position, it does not move the detector.
Consequently, mirroring the coordinates of a control point and selecting the
matching orientation must give back **exactly** the same position in space.

The reference dataset is synthetic: ~2800 control points spread over 16 LaB6
rings, for the geometry stored in the header of `orientation/orientation_3.edf`.
Reference control points are also provided for the three other orientations,
picked independently (hence a slightly different number of points), which allows
a cross-check that does not depend on the mirroring rule used here.
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/09/2026"

import logging
import unittest
from typing import ClassVar

import numpy
from scipy.spatial import cKDTree

from .. import geometry
from ..calibrant import get_calibrant
from ..containers import FixedParameters
from ..control_points import ControlPoints
from ..detectors import detector_factory
from ..geometry.imaged11 import (
    ORIENTATION_TO_FLIP_MATRIX,
    convert_from_ImageD11,
    convert_to_ImageD11,
)
from ..geometry.utils import (
    CORNER_OF_THE_DETECTOR,
    FLIPPED_AXES,
    convert_orientation,
    detector_corner,
)
from ..geometryRefinement import GeometryRefinement
from .utilstest import UtilsTest

logger = logging.getLogger(__name__)


class TestOrientationPositions(unittest.TestCase):
    """Position in space of control points across the 4 detector orientations"""

    DETECTOR = "Eiger2_1M"
    WAVELENGTH = 1e-10
    # geometry used to generate the synthetic dataset, see the EDF header
    GEOMETRY: ClassVar[dict] = {"dist": 0.04, "poni1": 0.05, "poni2": 0.06,
                                "rot1": 0.07, "rot2": 0.08, "rot3": 0.0}
    FLIPPED_AXES: ClassVar[dict] = FLIPPED_AXES

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.shape = detector_factory(cls.DETECTOR).shape
        points = numpy.array(cls.load_control_points(3))
        cls.d1 = points[:, 0]
        cls.d2 = points[:, 1]
        cls.ring = points[:, 2].astype(int)

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        cls.d1 = cls.d2 = cls.ring = None

    @staticmethod
    def load_control_points(orientation):
        """Read the reference control points for one orientation

        :param orientation: 1, 2, 3 or 4
        :return: list of [dim1, dim2, ring index]
        """
        filename = UtilsTest.getimage(f"orientation/orientation_{orientation}.npt")
        return ControlPoints(filename).getList()

    def positions_from_file(self, orientation):
        """Position in space of the control points picked for one orientation

        :param orientation: 1, 2, 3 or 4
        :return: (n, 3) array of positions in meter
        """
        points = numpy.array(self.load_control_points(orientation))
        ai = self.build_geometry(orientation)
        return numpy.array(ai.calc_pos_zyx(d1=points[:, 0], d2=points[:, 1])).T

    def build_geometry(self, orientation):
        """Reference geometry, with the detector set to the given orientation

        :param orientation: 1, 2, 3 or 4
        :return: Geometry instance
        """
        detector = detector_factory(self.DETECTOR, {"orientation": orientation})
        return geometry.Geometry(detector=detector, wavelength=self.WAVELENGTH,
                                 **self.GEOMETRY)

    def mirror(self, d1, d2, orientation, center=True):
        """Mirror pixel coordinates to address the same physical position once
        the detector is set to `orientation`.

        Nota: the offset differs with the convention, and both are exact:
        a pixel *index* n has its center at n+0.5, so indexes mirror as
        `shape - 1 - n`, while corner coordinates span [0, shape] and mirror
        as `shape - n`.

        :param d1: coordinates along the slow dimension
        :param d2: coordinates along the fast dimension
        :param orientation: 1, 2, 3 or 4
        :param center: True for pixel centers, False for pixel corners
        :return: 2-tuple of mirrored coordinates
        """
        offset = 1 if center else 0
        flip1, flip2 = self.FLIPPED_AXES[orientation]
        if flip1:
            d1 = self.shape[0] - offset - d1
        if flip2:
            d2 = self.shape[1] - offset - d2
        return d1, d2

    def test_position_is_orientation_invariant(self):
        """The very point of this suite: mirroring the control points and
        selecting the matching orientation gives back the same position in
        space. Would fail if the orientation were applied twice."""
        reference = numpy.array(self.build_geometry(3).calc_pos_zyx(d1=self.d1, d2=self.d2))
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                d1, d2 = self.mirror(self.d1, self.d2, orientation)
                got = numpy.array(self.build_geometry(orientation).calc_pos_zyx(d1=d1, d2=d2))
                delta = abs(got - reference).max()
                self.assertLess(delta, 1e-9,
                                f"orientation {orientation}: position differs by {delta} m")

    def test_detector_position_is_orientation_invariant(self):
        """Same check one layer below, on the detector alone, for both the pixel
        center and the pixel corner conventions."""
        for center in (True, False):
            detector = detector_factory(self.DETECTOR, {"orientation": 3})
            reference = detector.calc_cartesian_positions(self.d1, self.d2, center=center)
            for orientation in (1, 2, 4):
                with self.subTest(orientation=orientation, center=center):
                    d1, d2 = self.mirror(self.d1, self.d2, orientation, center=center)
                    detector = detector_factory(self.DETECTOR, {"orientation": orientation})
                    got = detector.calc_cartesian_positions(d1, d2, center=center)
                    delta = max(abs(a - b).max() for a, b in zip(got[:2], reference[:2]))
                    self.assertLess(delta, 1e-9,
                                    f"orientation {orientation}, center={center}: "
                                    f"position differs by {delta} m")

    def test_cython_matches_python(self):
        """The Cython and the pure numpy implementations agree, whatever the
        orientation."""
        for orientation in (1, 2, 3, 4):
            with self.subTest(orientation=orientation):
                ai = self.build_geometry(orientation)
                cython = numpy.array(ai.calc_pos_zyx(d1=self.d1, d2=self.d2, use_cython=True))
                python = numpy.array(ai.calc_pos_zyx(d1=self.d1, d2=self.d2, use_cython=False))
                delta = abs(cython - python).max()
                self.assertLess(delta, 1e-9, f"orientation {orientation}: "
                                             f"cython and python differ by {delta} m")

    def test_reference_files_describe_the_same_points(self):
        """Strongest cross-check, and it does not use `mirror()` at all: the
        control points picked independently for each orientation must describe
        the same cloud of physical points once each is combined with its own
        orientation.

        The files do not hold exactly the same peaks (they were picked
        separately, hence a slightly different count), so only the bulk of the
        cloud can be compared: unmatched points fall onto a neighbour of the
        same ring, a couple of pixels away. Hence the median."""
        reference = self.positions_from_file(3)
        tree = cKDTree(reference)
        pixel = detector_factory(self.DETECTOR).pixel1
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                distance, _ = tree.query(self.positions_from_file(orientation))
                median = numpy.median(distance) / pixel
                self.assertLess(median, 0.1,
                                f"orientation {orientation}: median distance to the "
                                f"orientation 3 cloud is {median} pixel")

    def test_derived_angles_are_orientation_invariant(self):
        """Since a mirrored control point is the very same point in space, every
        derived quantity must match, `chi` included. This is the user visible
        symptom of applying the orientation twice."""
        ai3 = self.build_geometry(3)
        reference = {"tth": ai3.tth(self.d1, self.d2),
                     "chi": ai3.chi(self.d1, self.d2),
                     "q": ai3.qFunction(self.d1, self.d2)}
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                d1, d2 = self.mirror(self.d1, self.d2, orientation)
                ai = self.build_geometry(orientation)
                for name, expected in (("tth", ai.tth(d1, d2)),
                                       ("chi", ai.chi(d1, d2)),
                                       ("q", ai.qFunction(d1, d2))):
                    delta = abs(expected - reference[name]).max()
                    self.assertLess(delta, 1e-9, f"orientation {orientation}: "
                                                 f"{name} differs by {delta}")

    def test_reference_points_lie_on_rings(self):
        """Cross-check independent of the mirroring rule: the control points
        picked for each orientation land on the expected LaB6 rings."""
        calibrant = get_calibrant("LaB6")
        calibrant.wavelength = self.WAVELENGTH
        expected = numpy.array(calibrant.get_2th())
        for orientation in (1, 2, 3, 4):
            with self.subTest(orientation=orientation):
                points = numpy.array(self.load_control_points(orientation))
                ai = self.build_geometry(orientation)
                tth = ai.tth(points[:, 0], points[:, 1])
                residual = numpy.rad2deg(abs(tth - expected[points[:, 2].astype(int)]))
                self.assertLess(numpy.median(residual), 0.01,
                                f"orientation {orientation}: median residual "
                                f"{numpy.median(residual)} deg")
                self.assertLess(residual.max(), 0.1,
                                f"orientation {orientation}: max residual "
                                f"{residual.max()} deg")


class TestOrientationRefinement(unittest.TestCase):
    """Refining the geometry from the control points must give back the
    reference parameters, whatever the orientation the detector is declared in,
    provided the control points are the ones picked for that orientation.

    `rot3` and the wavelength are fixed during the refinement: the reference
    setup is invariant under `rot3`, which is therefore not constrained by the
    data and is not optimised.
    """

    DETECTOR = TestOrientationPositions.DETECTOR
    WAVELENGTH = TestOrientationPositions.WAVELENGTH
    REFERENCE: ClassVar[dict] = dict(TestOrientationPositions.GEOMETRY)
    # the refinement lands within these of the reference, while a wrong
    # orientation is off by ~2e-2 m and ~1.6e-1 rad, i.e. orders of magnitude more
    TOLERANCE: ClassVar[dict] = {"dist": 1e-4, "poni1": 1e-4, "poni2": 1e-4,
                                 "rot1": 1e-3, "rot2": 1e-3}

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.detector = detector_factory(cls.DETECTOR)
        cls.calibrant = get_calibrant("LaB6")
        cls.calibrant.wavelength = cls.WAVELENGTH
        # refinements are the expensive part, run them once
        cls.matched = {o: cls.refine(cls.control_points(o), o) for o in (1, 2, 3, 4)}
        points3 = cls.control_points(3)
        cls.declared = {o: cls.refine(points3, o) for o in (1, 2, 3, 4)}

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        cls.matched = cls.declared = cls.calibrant = None

    @staticmethod
    def control_points(orientation):
        return numpy.array(TestOrientationPositions.load_control_points(orientation))

    @classmethod
    def refine(cls, points, orientation):
        """Refine the geometry, keeping `rot3` and the wavelength fixed

        :return: dict of refined parameters
        """
        detector = detector_factory(cls.DETECTOR, {"orientation": orientation})
        refiner = GeometryRefinement(data=points, calibrant=cls.calibrant,
                                     detector=detector, wavelength=cls.WAVELENGTH,
                                     dist=0.05,
                                     poni1=detector.shape[0] * detector.pixel1 / 2,
                                     poni2=detector.shape[1] * detector.pixel2 / 2,
                                     rot1=0, rot2=0, rot3=0)
        refiner.refine3(1000000, fix=FixedParameters(["wavelength", "rot3"]))
        return {key: getattr(refiner, key)
                for key in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")}

    def test_refinement_is_orientation_invariant(self):
        """Each file refined with its own orientation gives back the reference:
        flipping the image and declaring the matching orientation cancel out."""
        for orientation, found in self.matched.items():
            with self.subTest(orientation=orientation):
                for key, tolerance in self.TOLERANCE.items():
                    self.assertAlmostEqual(found[key], self.REFERENCE[key], delta=tolerance,
                                           msg=f"orientation {orientation}: {key} is "
                                               f"{found[key]} instead of {self.REFERENCE[key]}")

    def test_refinement_differs_with_another_orientation(self):
        """Refining the *same* data while declaring another orientation converges
        just as well but towards different parameters."""
        reference = self.declared[3]
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                found = self.declared[orientation]
                flip1, flip2 = FLIPPED_AXES[orientation]
                if flip1:
                    self.assertGreater(abs(found["poni1"] - reference["poni1"]), 1e-3,
                                       "poni1 should be mirrored")
                    self.assertLess(found["rot2"] * reference["rot2"], 0, "rot2 should change sign")
                if flip2:
                    self.assertGreater(abs(found["poni2"] - reference["poni2"]), 1e-3,
                                       "poni2 should be mirrored")
                    self.assertLess(found["rot1"] * reference["rot1"], 0, "rot1 should change sign")

    def test_convert_orientation_matches_refinement(self):
        """`convert_orientation` reproduces what the refinement finds when the
        same data is declared in another orientation."""
        reference = self.declared[3]
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                expected = convert_orientation(reference, self.detector, 3, orientation)
                found = self.declared[orientation]
                for key, tolerance in self.TOLERANCE.items():
                    self.assertAlmostEqual(found[key], expected[key], delta=tolerance,
                                           msg=f"orientation {orientation}: {key} is "
                                               f"{found[key]}, converted gives {expected[key]}")

    def test_convert_orientation_round_trip(self):
        """Converting back and forth restores the parameters."""
        reference = self.declared[3]
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                there = convert_orientation(reference, self.detector, 3, orientation)
                back = convert_orientation(there, self.detector, orientation, 3)
                for key in self.TOLERANCE:
                    self.assertAlmostEqual(back[key], reference[key], places=12,
                                           msg=f"{key} not restored by the round trip")


class TestImageD11Interoperability(unittest.TestCase):
    """pyFAI and ImageD11 must place the pixels at the very same position in the
    laboratory frame, for every detector orientation.

    The two frames are related by the change of axes documented in
    `doc/source/geometry_conversion.rst`:
    pyFAI (beam, slow, fast) == ImageD11 (x, z, -y). The sign on the horizontal
    transverse axis is physical: it points towards the center of the storage
    ring for pyFAI, away from it for ImageD11.
    """

    DETECTOR = TestOrientationPositions.DETECTOR
    WAVELENGTH = TestOrientationPositions.WAVELENGTH
    # a geometry with all three rotations, rot3 included, so that every term of
    # the conversion is exercised
    GEOMETRY: ClassVar[dict] = {"dist": 0.04, "poni1": 0.05, "poni2": 0.06,
                                "rot1": 0.07, "rot2": 0.08, "rot3": 0.23}
    # half a pixel is 3.75e-5 m for this detector, aim one order of magnitude below
    TOLERANCE = 1e-5

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            from ImageD11 import transform
        except ImportError:
            raise unittest.SkipTest("ImageD11 is not installed") from None
        cls.transform = transform
        detector = detector_factory(cls.DETECTOR)
        slow, fast = numpy.meshgrid(numpy.linspace(0, detector.shape[0] - 1, 9),
                                    numpy.linspace(0, detector.shape[1] - 1, 11),
                                    indexing="ij")
        cls.slow = slow.ravel()
        cls.fast = fast.ravel()

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        cls.transform = cls.slow = cls.fast = None

    def build_geometry(self, orientation):
        detector = detector_factory(self.DETECTOR, {"orientation": orientation})
        return geometry.Geometry(detector=detector, wavelength=self.WAVELENGTH,
                                 **self.GEOMETRY)

    def imaged11_positions(self, parameters):
        """Position of the pixels according to ImageD11, in meter, reordered
        the pyFAI way: (beam, slow, fast)

        :param parameters: ImageD11Geometry, in meter (distance_unit="m")
        """
        # compute_xyz_lab ignores the extra keys (shape, spline, wavelength)
        lab = self.transform.compute_xyz_lab([self.slow, self.fast],
                                             **parameters._asdict())
        # (x, y, z) of ImageD11 -> (beam, slow, fast) of pyFAI
        return numpy.array([lab[0], lab[2], -lab[1]])

    def test_same_position_in_space(self):
        """The core requirement: same xyz, better than half a pixel."""
        for orientation in (1, 2, 3, 4):
            with self.subTest(orientation=orientation):
                ai = self.build_geometry(orientation)
                expected = numpy.array(ai.calc_pos_zyx(d1=self.slow, d2=self.fast))
                parameters = convert_to_ImageD11(ai, distance_unit="m")
                delta = abs(self.imaged11_positions(parameters) - expected).max()
                self.assertLess(delta, self.TOLERANCE,
                                f"orientation {orientation}: positions differ by {delta} m")

    def test_orientation_maps_to_the_flip_matrix(self):
        """The o-matrix of ImageD11 carries the orientation of pyFAI, composed
        with the `o22 = -1` which the two axis conventions already require."""
        for orientation in (1, 2, 3, 4):
            with self.subTest(orientation=orientation):
                parameters = convert_to_ImageD11(self.build_geometry(orientation))
                o11, o22 = ORIENTATION_TO_FLIP_MATRIX[orientation]
                self.assertEqual(parameters.o11, o11, "o11")
                self.assertEqual(parameters.o22, o22, "o22")
                self.assertEqual((parameters.o12, parameters.o21), (0, 0), "no transposition")
                # each mirrored axis flips its own sign with respect to orientation 3
                flip_slow, flip_fast = FLIPPED_AXES[orientation]
                reference = ORIENTATION_TO_FLIP_MATRIX[3]
                self.assertEqual(o11, -reference[0] if flip_slow else reference[0])
                self.assertEqual(o22, -reference[1] if flip_fast else reference[1])

    def test_round_trip(self):
        """Converting to ImageD11 and back restores the pyFAI geometry."""
        for orientation in (1, 2, 3, 4):
            with self.subTest(orientation=orientation):
                ai = self.build_geometry(orientation)
                back = convert_from_ImageD11(convert_to_ImageD11(ai, distance_unit="m"))
                for key, expected in self.GEOMETRY.items():
                    self.assertAlmostEqual(getattr(back, key), expected, places=9,
                                           msg=f"orientation {orientation}: {key}")
                self.assertEqual(back.detector.orientation, orientation, "orientation restored")

    def test_tth_matches(self):
        """Scattering angles agree as well, which is what ImageD11 users
        actually consume."""
        for orientation in (1, 2, 3, 4):
            with self.subTest(orientation=orientation):
                ai = self.build_geometry(orientation)
                positions = self.imaged11_positions(convert_to_ImageD11(ai, distance_unit="m"))
                radius = numpy.sqrt(positions[1] ** 2 + positions[2] ** 2)
                tth = numpy.arctan2(radius, positions[0])
                delta = abs(numpy.rad2deg(tth - ai.tth(self.slow, self.fast))).max()
                self.assertLess(delta, 1e-3, f"orientation {orientation}: "
                                             f"2theta differs by {delta} deg")


class TestIrregularPixels(unittest.TestCase):
    """`poni1`/`poni2` are distances from the corner of pixel (0, 0), so
    mirroring them needs the real corners of the detector.

    Those corners are not at 0 and `shape * pixel_size`: modular detectors have
    wider pixels along the module borders, which pushes the far corner several
    millimetres beyond the naive product, and a spline-corrected detector has
    the corner of its pixel (0, 0) displaced away from the origin.
    """

    REGULAR = "Eiger2_1M"
    SPLINE = "Frelon"
    # module borders make these reach beyond shape * pixel_size, by the amount
    # given here along the slow and the fast dimension, in meter
    IRREGULAR: ClassVar[dict] = {"ImXPadS140": (3.90e-4, 2.34e-3),
                                 "ImXPadS70": (0.0, 2.34e-3),
                                 "Xpad_flat": (2.499e-2, 2.34e-3)}
    # get_pixel_corners stores float32, i.e. ~0.1 µm over a 0.1 m detector
    PLACES = 7
    FLIPS = ((False, False), (True, False), (False, True), (True, True))

    @staticmethod
    def build_detector(name, orientation):
        """Detector with its real pixel grid loaded.

        `calc_cartesian_positions` falls back on a uniform grid built from
        `pixel1`/`pixel2` as long as `_pixel_corners` has not been populated,
        which would hide the very irregularity under test here.

        :param name: name of the detector
        :param orientation: 1, 2, 3 or 4
        :return: Detector instance
        """
        detector = detector_factory(name, {"orientation": orientation})
        detector.get_pixel_corners()
        return detector

    @classmethod
    def build_spline_detector(cls):
        """FReLoN with its distortion spline, the reference irregular grid

        :return: Detector instance
        """
        return detector_factory(cls.SPLINE,
                                {"splineFile": UtilsTest.getimage("frelon.spline")})

    def test_regular_detector_matches_the_naive_product(self):
        """On a detector with identical pixels the corners are at 0 and
        `shape * pixel_size`, which is what the naive rule assumed."""
        detector = detector_factory(self.REGULAR)
        length1 = detector.shape[0] * detector.pixel1
        length2 = detector.shape[1] * detector.pixel2
        expected = {(False, False): (0.0, 0.0),
                    (True, False): (length1, 0.0),
                    (False, True): (0.0, length2),
                    (True, True): (length1, length2)}
        for flips, (slow, fast) in expected.items():
            with self.subTest(flips=flips):
                got = detector_corner(detector, *flips)
                self.assertAlmostEqual(got[0], slow, self.PLACES, "slow")
                self.assertAlmostEqual(got[1], fast, self.PLACES, "fast")

    def test_irregular_detector_reaches_beyond_the_naive_product(self):
        """The regression: `shape * pixel_size` under-estimates the size of a
        modular detector, by 25 mm on Xpad_flat."""
        for name, (excess1, excess2) in self.IRREGULAR.items():
            with self.subTest(detector=name):
                detector = detector_factory(name)
                far = detector_corner(detector, True, True)
                self.assertAlmostEqual(far[0] - detector.shape[0] * detector.pixel1,
                                       excess1, self.PLACES, "far corner, slow")
                self.assertAlmostEqual(far[1] - detector.shape[1] * detector.pixel2,
                                       excess2, self.PLACES, "far corner, fast")

    def test_corner_is_the_vertex_at_the_matching_index(self):
        """The pixel and the vertex both have to follow the mirrored axes, so
        the result must be the position of the corner of the detector in index
        space: (0, 0), (n1, 0), (0, n2) or (n1, n2)."""
        detectors = [detector_factory(name)
                     for name in [self.REGULAR, *self.IRREGULAR]]
        detectors.append(self.build_spline_detector())
        for detector in detectors:
            shape = detector.shape
            for flips in self.FLIPS:
                with self.subTest(detector=detector.name, flips=flips):
                    index1 = numpy.array([shape[0] if flips[0] else 0], dtype=numpy.float64)
                    index2 = numpy.array([shape[1] if flips[1] else 0], dtype=numpy.float64)
                    slow, fast, _ = detector.calc_cartesian_positions(index1, index2,
                                                                      center=False)
                    got = detector_corner(detector, *flips)
                    # the shape of the result varies with the detector, ravel
                    # rather than convert: numpy >= 2.3 rejects float() on an
                    # array which holds a single value but is not 0-dimensional
                    self.assertAlmostEqual(got[0], numpy.ravel(slow)[0], self.PLACES, "slow")
                    self.assertAlmostEqual(got[1], numpy.ravel(fast)[0], self.PLACES, "fast")

    def test_spline_corner_is_not_an_extremum(self):
        """Why a specific vertex is needed rather than a min or a max: the
        pixels of a spline-corrected detector are distorted quadrilaterals, so
        the vertex bounding the sensor is not the extreme one of its pixel."""
        detector = self.build_spline_detector()
        corners = detector.get_pixel_corners()
        deltas = []
        for flips in self.FLIPS:
            index1, index2, vertex = CORNER_OF_THE_DETECTOR[flips]
            pixel = corners[index1, index2]
            for dim, flipped in enumerate(flips, start=1):
                extremum = pixel[:, dim].max() if flipped else pixel[:, dim].min()
                deltas.append(abs(float(pixel[vertex, dim] - extremum)))
        self.assertGreater(max(deltas), 1e-6,
                           "the spline does not displace the vertices, "
                           "this detector no longer exercises the case")

    def test_conversion_is_an_involution(self):
        """Converting to another orientation and back restores the geometry."""
        parameters = {"dist": 0.04, "poni1": 0.011, "poni2": 0.013,
                      "rot1": 0.07, "rot2": 0.08, "rot3": 0.0}
        for name in [self.REGULAR, *self.IRREGULAR]:
            detector = detector_factory(name)
            for orientation in (1, 2, 4):
                with self.subTest(detector=name, orientation=orientation):
                    there = convert_orientation(parameters, detector, 3, orientation)
                    back = convert_orientation(there, detector, orientation, 3)
                    for key, expected in parameters.items():
                        self.assertAlmostEqual(back[key], expected, 12, key)

    def test_conversion_mirrors_the_positions(self):
        """The physical content of the conversion: expressing the same data in
        a mirrored orientation negates the laboratory coordinate of the
        mirrored axis and leaves the other two alone."""
        parameters = {"dist": 0.04, "poni1": 0.011, "poni2": 0.013,
                      "rot1": 0.07, "rot2": 0.08, "rot3": 0.0}
        for detector_name in [self.REGULAR, "ImXPadS70"]:
            reference = self.build_detector(detector_name, 3)
            beam, slow, fast = geometry.Geometry(detector=reference, wavelength=1e-10,
                                                **parameters).calc_pos_zyx(corners=False)
            for orientation in (1, 2, 4):
                with self.subTest(detector=detector_name, orientation=orientation):
                    flip1, flip2 = FLIPPED_AXES[orientation]
                    converted = convert_orientation(parameters, reference, 3, orientation)
                    detector = self.build_detector(detector_name, orientation)
                    got = geometry.Geometry(detector=detector, wavelength=1e-10,
                                            **converted).calc_pos_zyx(corners=False)
                    axes = [(beam, "beam"),
                            (-slow if flip1 else slow, "slow"),
                            (-fast if flip2 else fast, "fast")]
                    for axis, (expected, label) in enumerate(axes):
                        delta = abs(got[axis] - expected).max()
                        self.assertLess(delta, 1e-7, f"{label} differs by {delta} m")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestOrientationPositions))
    testsuite.addTest(loader(TestOrientationRefinement))
    testsuite.addTest(loader(TestImageD11Interoperability))
    testsuite.addTest(loader(TestIrregularPixels))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

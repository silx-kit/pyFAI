#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2024 European Synchrotron Radiation Facility, Grenoble, France
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


"""test suite for Fiber integrator class"""

__author__ = "Edgar Gutiérrez Fernández"
__contact__ = "edgar.gutierrez-fernandez@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "23/10/2024"

import unittest
import numpy
import logging
from ..calibrant import get_calibrant
logger = logging.getLogger(__name__)
from ..integrator.fiber import FiberIntegrator
from ..detectors import detector_factory
from ..units import get_unit_fiber


class TestFiberIntegrator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dist = 0.1
        poni1 = 0.02
        poni2 = 0.02
        detector = detector_factory("Eiger2_4M")
        wavelength = 1e-10
        calibrant = get_calibrant("LaB6")

        cls.fi = FiberIntegrator(dist=dist,
                                 poni1=poni1,
                                 poni2=poni2,
                                 wavelength=wavelength,
                                 detector=detector,
                             )
        cls.data = calibrant.fake_calibration_image(ai=cls.fi)

    def test_integrate2d_no_parameters(self):
        res2d_ref = self.fi.integrate2d_grazing_incidence(data=self.data)
        res2d = self.fi.integrate2d(data=self.data)

        assert numpy.array_equal(res2d_ref.radial, res2d.radial)
        assert numpy.array_equal(res2d_ref.azimuthal, res2d.azimuthal)
        assert numpy.array_equal(res2d_ref.intensity, res2d.intensity)

    def test_integrate2d_deprecated_parameters(self):
        res2d_ref = self.fi.integrate2d_grazing_incidence(data=self.data, npt_ip=500, npt_oop=500)
        res2d_deprecated = self.fi.integrate2d(data=self.data, npt_horizontal=500, npt_vertical=500)

        assert numpy.array_equal(res2d_ref.radial, res2d_deprecated.radial)
        assert numpy.array_equal(res2d_ref.azimuthal, res2d_deprecated.azimuthal)
        assert numpy.array_equal(res2d_ref.intensity, res2d_deprecated.intensity)

    def test_integrate2d_explicit_units(self):
        res2d_ref = self.fi.integrate2d(data=self.data)

        res2d_string_units = self.fi.integrate2d(data=self.data, unit_ip="qip_nm^-1", unit_oop="qoop_nm^-1")
        assert numpy.array_equal(res2d_ref.radial, res2d_string_units.radial)
        assert numpy.array_equal(res2d_ref.azimuthal, res2d_string_units.azimuthal)
        assert numpy.array_equal(res2d_ref.intensity, res2d_string_units.intensity)

        unit_qip = get_unit_fiber(name="qip_nm^-1")
        unit_qoop = get_unit_fiber(name="qoop_nm^-1")
        res2d_fiber_units = self.fi.integrate2d(data=self.data, unit_ip=unit_qip, unit_oop=unit_qoop)
        assert numpy.array_equal(res2d_ref.radial, res2d_fiber_units.radial)
        assert numpy.array_equal(res2d_ref.azimuthal, res2d_fiber_units.azimuthal)
        assert numpy.array_equal(res2d_ref.intensity, res2d_fiber_units.intensity)

    def test_integrate2d_preferences(self):
        incident_angle = 0.2
        tilt_angle = 1.0
        sample_orientation = 3

        res2d_ref = self.fi.integrate2d(data=self.data, incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)

        incident_angle_2 = 0.77
        tilt_angle_2 = -1.0
        sample_orientation_2 = 2

        unit_qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle_2, tilt_angle=tilt_angle_2, sample_orientation=sample_orientation_2)
        unit_qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle_2, tilt_angle=tilt_angle_2, sample_orientation=sample_orientation_2)
        res2d_explicit_angles = self.fi.integrate2d(data=self.data, unit_ip=unit_qip, unit_oop=unit_qoop,
                                                     incident_angle=incident_angle,
                                                     tilt_angle=tilt_angle,
                                                     sample_orientation=sample_orientation)

        assert numpy.array_equal(res2d_ref.radial, res2d_explicit_angles.radial)
        assert numpy.array_equal(res2d_ref.azimuthal, res2d_explicit_angles.azimuthal)
        assert numpy.array_equal(res2d_ref.intensity, res2d_explicit_angles.intensity)

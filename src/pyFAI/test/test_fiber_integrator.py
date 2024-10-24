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

    def test_integrate2d_equivalences(self):
        res2d_gi = self.fi.integrate2d_grazing_incidence(data=self.data)
        res2d_fiber = self.fi.integrate2d_fiber(data=self.data)

        self.assertEqual(abs(res2d_gi.radial - res2d_fiber.radial).max(), 0)
        self.assertEqual(abs(res2d_gi.azimuthal - res2d_fiber.azimuthal).max(), 0)
        self.assertEqual(abs(res2d_gi.intensity - res2d_fiber.intensity).max(), 0)

    def test_integrate2d_equivalences_parameters(self):
        incident_angle = 0.2
        tilt_angle = 1.0
        sample_orientation = 3

        res2d_gi = self.fi.integrate2d_grazing_incidence(data=self.data, incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)

        qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)
        qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)

        res2d_fiber = self.fi.integrate2d_fiber(data=self.data, unit_ip=qip, unit_oop=qoop)

        self.assertEqual(abs(res2d_gi.radial - res2d_fiber.radial).max(), 0)
        self.assertEqual(abs(res2d_gi.azimuthal - res2d_fiber.azimuthal).max(), 0)
        self.assertEqual(abs(res2d_gi.intensity - res2d_fiber.intensity).max(), 0)

    def test_integrate2d_deprecated_parameters(self):
        npt_ip = 500
        npt_oop = 500
        ip_range = [0,5]
        oop_range = [0,20]

        incident_angle = 0.2
        tilt_angle = 1.0
        sample_orientation = 3

        qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)
        qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)

        res2d_ref = self.fi.integrate2d_grazing_incidence(data=self.data,
                                                          npt_ip=npt_ip, npt_oop=npt_oop,
                                                          unit_ip=qip, unit_oop=qoop,
                                                          ip_range=ip_range, oop_range=oop_range)
        res2d_deprecated = self.fi.integrate2d_grazing_incidence(data=self.data,
                                                                 npt_horizontal=npt_ip, npt_vertical=npt_oop,
                                                                 horizontal_unit=qip, vertical_unit=qoop,
                                                                 horizontal_unit_range=ip_range, vertical_unit_range=oop_range,
                                                                 )

        self.assertEqual(abs(res2d_ref.radial - res2d_deprecated.radial).max(), 0)
        self.assertEqual(abs(res2d_ref.azimuthal - res2d_deprecated.azimuthal).max(), 0)
        self.assertEqual(abs(res2d_ref.intensity - res2d_deprecated.intensity).max(), 0)

    def test_integrate2d_explicit_units(self):
        res2d_ref = self.fi.integrate2d_grazing_incidence(data=self.data)

        res2d_string_units = self.fi.integrate2d_grazing_incidence(data=self.data, unit_ip="qip_nm^-1", unit_oop="qoop_nm^-1")

        self.assertEqual(abs(res2d_ref.radial - res2d_string_units.radial).max(), 0)
        self.assertEqual(abs(res2d_ref.azimuthal - res2d_string_units.azimuthal).max(), 0)
        self.assertEqual(abs(res2d_ref.intensity - res2d_string_units.intensity).max(), 0)

        unit_qip = get_unit_fiber(name="qip_nm^-1")
        unit_qoop = get_unit_fiber(name="qoop_nm^-1")
        res2d_fiber_units = self.fi.integrate2d_grazing_incidence(data=self.data, unit_ip=unit_qip, unit_oop=unit_qoop)

        self.assertEqual(abs(res2d_ref.radial - res2d_fiber_units.radial).max(), 0)
        self.assertEqual(abs(res2d_ref.azimuthal - res2d_fiber_units.azimuthal).max(), 0)
        self.assertEqual(abs(res2d_ref.intensity - res2d_fiber_units.intensity).max(), 0)

    def test_integrate2d_priority(self):
        incident_angle = 0.2
        tilt_angle = 1.0
        sample_orientation = 3

        res2d_ref = self.fi.integrate2d_grazing_incidence(data=self.data, incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)

        incident_angle_2 = 0.77
        tilt_angle_2 = -1.0
        sample_orientation_2 = 2

        unit_qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle_2, tilt_angle=tilt_angle_2, sample_orientation=sample_orientation_2)
        unit_qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle_2, tilt_angle=tilt_angle_2, sample_orientation=sample_orientation_2)
        res2d_explicit_angles = self.fi.integrate2d_grazing_incidence(data=self.data, unit_ip=unit_qip, unit_oop=unit_qoop,
                                                     incident_angle=incident_angle,
                                                     tilt_angle=tilt_angle,
                                                     sample_orientation=sample_orientation)

        self.assertEqual(abs(res2d_ref.radial - res2d_explicit_angles.radial).max(), 0)
        self.assertEqual(abs(res2d_ref.azimuthal - res2d_explicit_angles.azimuthal).max(), 0)
        self.assertEqual(abs(res2d_ref.intensity - res2d_explicit_angles.intensity).max(), 0)

    def test_integrate1d_equivalences(self):
        npt_ip = 500
        res1d_gi = self.fi.integrate1d_grazing_incidence(data=self.data, npt_ip=npt_ip)
        res1d_fiber = self.fi.integrate1d_fiber(data=self.data, npt_ip=npt_ip)

        self.assertEqual(abs(res1d_gi.radial - res1d_fiber.radial).max(), 0)
        self.assertEqual(abs(res1d_gi.intensity - res1d_fiber.intensity).max(), 0)

    def test_integrate1d_equivalences_parameters(self):
        npt_ip = 500
        npt_oop = 500
        ip_range = [0,5]
        oop_range = [0,20]

        incident_angle = 0.2
        tilt_angle = 1.0
        sample_orientation = 3

        res1d_gi = self.fi.integrate1d_grazing_incidence(data=self.data,
                                                         npt_ip=npt_ip, ip_range=ip_range,
                                                         npt_oop=npt_oop, oop_range=oop_range,
                                                         incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation,
                                                         )

        qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)
        qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)

        res1d_fiber = self.fi.integrate1d_fiber(data=self.data,
                                                npt_ip=npt_ip, ip_range=ip_range,
                                                npt_oop=npt_oop, oop_range=oop_range,
                                                unit_ip=qip, unit_oop=qoop,
                                                )

        self.assertEqual(abs(res1d_gi.radial - res1d_fiber.radial).max(), 0)
        self.assertEqual(abs(res1d_gi.intensity - res1d_fiber.intensity).max(), 0)

    def test_integrate1d_explicit_units(self):
        npt_ip = 500
        incident_angle = 0.2
        tilt_angle = 1.0
        sample_orientation = 3

        res1d_ref = self.fi.integrate_grazing_incidence(data=self.data, npt_ip=npt_ip,
                                                        incident_angle=incident_angle,
                                                        tilt_angle=tilt_angle,
                                                        sample_orientation=sample_orientation,
                                                        )

        res1d_string_units = self.fi.integrate_grazing_incidence(data=self.data, npt_ip=npt_ip,
                                                                 unit_ip="qip_nm^-1", unit_oop="qoop_nm^-1",
                                                                 incident_angle=incident_angle,
                                                                 tilt_angle=tilt_angle,
                                                                 sample_orientation=sample_orientation,
        )

        self.assertEqual(abs(res1d_ref.radial - res1d_string_units.radial).max(), 0)
        self.assertEqual(abs(res1d_ref.intensity - res1d_string_units.intensity).max(), 0)

        qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)
        qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)

        res1d_fiber_units = self.fi.integrate_grazing_incidence(data=self.data, npt_ip=npt_ip,
                                                                unit_ip=qip, unit_oop=qoop,
        )

        self.assertEqual(abs(res1d_ref.radial - res1d_fiber_units.radial).max(), 0)
        self.assertEqual(abs(res1d_ref.intensity - res1d_fiber_units.intensity).max(), 0)

    def test_integrate1d_priority(self):
        npt_ip = 500
        incident_angle = 0.2
        tilt_angle = 1.0
        sample_orientation = 3
        res1d_ref = self.fi.integrate_grazing_incidence(data=self.data, npt_ip=npt_ip,
                                                        incident_angle=incident_angle,
                                                        tilt_angle=tilt_angle,
                                                        sample_orientation=sample_orientation,
                                                        )

        incident_angle_2 = 0.77
        tilt_angle_2 = -1.0
        sample_orientation_2 = 2

        qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle_2, tilt_angle=tilt_angle_2, sample_orientation=sample_orientation_2)
        qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle_2, tilt_angle=tilt_angle_2, sample_orientation=sample_orientation_2)

        res1d = self.fi.integrate_grazing_incidence(data=self.data, npt_ip=npt_ip,
                                                    unit_ip=qip, unit_oop=qoop,
                                                    incident_angle=incident_angle,
                                                    tilt_angle=tilt_angle,
                                                    sample_orientation=sample_orientation,
                                                    )

        self.assertEqual(abs(res1d_ref.radial - res1d.radial).max(), 0)
        self.assertEqual(abs(res1d_ref.intensity - res1d.intensity).max(), 0)

        res1d_wrong = self.fi.integrate_grazing_incidence(data=self.data, npt_ip=npt_ip,
                                                    unit_ip=qip, unit_oop=qoop,
                                                    )

        self.assertFalse(abs(res1d_ref.radial - res1d_wrong.radial).max(), 0)
        self.assertFalse(abs(res1d_ref.intensity - res1d_wrong.intensity).max(), 0)

    def test_sample_orientation_equivalence(self):
        incident_angle = 0.0
        tilt_angle =0.0
        npt_ip = 100
        npt_oop = 100

        threshold = 1.0

        sample_orientation = 1
        oop_range_1 = [-5,5]
        ip_range_1 = [0,20]
        res_so_1 = self.fi.integrate_grazing_incidence(data=self.data, npt_ip=npt_ip, npt_oop=npt_oop,
                                                       ip_range=ip_range_1, oop_range=oop_range_1,
                                                       incident_angle=incident_angle, tilt_angle=tilt_angle,
                                                       sample_orientation=sample_orientation,
                                                       vertical_integration=False,
        )

        sample_orientation = 2
        ip_range_2 = [-5,5]
        oop_range_2 = [0,20]
        res_so_2 = self.fi.integrate_grazing_incidence(data=self.data, npt_ip=npt_ip, npt_oop=npt_oop,
                                                       ip_range=ip_range_2, oop_range=oop_range_2,
                                                       incident_angle=incident_angle, tilt_angle=tilt_angle,
                                                       sample_orientation=sample_orientation,
                                                       vertical_integration=True,
        )

        self.assertLess((abs(res_so_1.intensity) - abs(res_so_2.intensity)).max(), threshold)

        sample_orientation = 3
        oop_range_3 = [-5,5]
        ip_range_3 = [-20,0]
        res_so_3 = self.fi.integrate_grazing_incidence(data=self.data, npt_ip=npt_ip, npt_oop=npt_oop,
                                                       ip_range=ip_range_3, oop_range=oop_range_3,
                                                       incident_angle=incident_angle, tilt_angle=tilt_angle,
                                                       sample_orientation=sample_orientation,
                                                       vertical_integration=False,
        )

        self.assertLess((abs(res_so_3.intensity) - abs(res_so_2.intensity)).max(), threshold)

        sample_orientation = 4
        oop_range_4 = [-20,5]
        ip_range_4 = [-5,5]
        res_so_4 = self.fi.integrate_grazing_incidence(data=self.data, npt_ip=npt_ip, npt_oop=npt_oop,
                                                       ip_range=ip_range_4, oop_range=oop_range_4,
                                                       incident_angle=incident_angle, tilt_angle=tilt_angle,
                                                       sample_orientation=sample_orientation,
                                                       vertical_integration=True,
        )

        self.assertLess((abs(res_so_4.intensity) - abs(res_so_3.intensity)).max(), threshold)

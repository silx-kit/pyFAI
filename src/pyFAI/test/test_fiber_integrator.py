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
__date__ = "16/01/2025"

import unittest
import logging
import numpy
from ..calibrant import get_calibrant
logger = logging.getLogger(__name__)
from ..integrator.fiber import FiberIntegrator
from ..integrator.azimuthal import AzimuthalIntegrator
from ..detectors import detector_factory
from ..units import get_unit_fiber
from ..units import parse_fiber_unit
from ..units import UnitFiber
from ..test.utilstest import UtilsTest
from .. import load

class TestFiberIntegrator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dist = 0.1
        poni1 = 0.02
        poni2 = 0.02
        detector = detector_factory("Pilatus100k")
        wavelength = 1e-10
        cls.calibrant = get_calibrant("LaB6")

        cls.fi = FiberIntegrator(dist=dist,
                                 poni1=poni1,
                                 poni2=poni2,
                                 wavelength=wavelength,
                                 detector=detector,
                             )
        cls.data = cls.calibrant.fake_calibration_image(ai=cls.fi)
        cls.poni_p1m = UtilsTest.getimage("Pilatus1M.poni")

    def test_instantiation(self):
        p1m = detector_factory("Pilatus1M")
        _ai = AzimuthalIntegrator(detector=p1m)
        p1m_data = self.calibrant.fake_calibration_image(ai=_ai)

        fi_load = load(filename=self.poni_p1m, type_="pyFAI.integrator.fiber.FiberIntegrator")
        res2d_load = fi_load.integrate2d_fiber(data=p1m_data)

        fi_direct = FiberIntegrator()
        fi_direct.load(self.poni_p1m)
        res2d_direct = fi_direct.integrate2d_fiber(data=p1m_data)

        self.assertEqual(abs(res2d_load.radial - res2d_direct.radial).max(), 0)
        self.assertEqual(abs(res2d_load.azimuthal - res2d_direct.azimuthal).max(), 0)
        self.assertEqual(abs(res2d_load.intensity - res2d_direct.intensity).max(), 0)

        ai = AzimuthalIntegrator()
        ai.load(self.poni_p1m)
        fi_from_ai = ai.promote(type_="pyFAI.integrator.fiber.FiberIntegrator")
        res2d_from_ai = fi_from_ai.integrate2d_fiber(data=p1m_data)

        self.assertEqual(abs(res2d_load.radial - res2d_from_ai.radial).max(), 0)
        self.assertEqual(abs(res2d_load.azimuthal - res2d_from_ai.azimuthal).max(), 0)
        self.assertEqual(abs(res2d_load.intensity - res2d_from_ai.intensity).max(), 0)


    def test_parse_units(self):
        gi_parameters_default = {"incident_angle" : 0.0,
                                 "tilt_angle" : 0.0,
                                 "sample_orientation" : 1,
        }
        gi_parameters_1 = {"incident_angle" : 0.2,
                           "tilt_angle" : 0.4,
                           "sample_orientation" : 2,
        }

        qip_str_1 = parse_fiber_unit(unit='qip_nm^-1')
        qoop_str_1 = parse_fiber_unit(unit='qoop_nm^-1')
        qip_str_2 = parse_fiber_unit(unit='qip_nm^-1', **gi_parameters_1)
        qoop_str_2 = parse_fiber_unit(unit='qoop_nm^-1', **gi_parameters_1)

        qip_unit_1 = parse_fiber_unit(unit=get_unit_fiber(name='qip_A^-1'))
        qoop_unit_1 = parse_fiber_unit(unit=get_unit_fiber(name='qoop_A^-1'))
        qip_unit_2 = parse_fiber_unit(unit=get_unit_fiber(name='qip_A^-1'), **gi_parameters_1)
        qoop_unit_2 = parse_fiber_unit(unit=get_unit_fiber(name='qoop_A^-1'), **gi_parameters_1)

        self.assertIsInstance(qip_str_1, UnitFiber)
        self.assertIsInstance(qoop_str_1, UnitFiber)
        for k,v in gi_parameters_default.items():
            self.assertEqual(getattr(qip_str_1, k), v)
            self.assertEqual(getattr(qoop_str_1, k), v)

        self.assertIsInstance(qip_str_2, UnitFiber)
        self.assertIsInstance(qoop_str_2, UnitFiber)
        for k,v in gi_parameters_1.items():
            self.assertEqual(getattr(qip_str_2, k), v)
            self.assertEqual(getattr(qoop_str_2, k), v)

        self.assertIsInstance(qip_unit_1, UnitFiber)
        self.assertIsInstance(qoop_unit_1, UnitFiber)
        for k,v in gi_parameters_default.items():
            self.assertEqual(getattr(qip_unit_1, k), v)
            self.assertEqual(getattr(qoop_unit_1, k), v)

        self.assertIsInstance(qip_unit_2, UnitFiber)
        self.assertIsInstance(qoop_unit_2, UnitFiber)
        for k,v in gi_parameters_1.items():
            self.assertEqual(getattr(qip_unit_2, k), v)
            self.assertEqual(getattr(qoop_unit_2, k), v)

    def test_parse_wrong_units(self):
        correct = parse_fiber_unit(unit='qip_nm^-1')
        def wrong():
            _ = parse_fiber_unit(unit='q_nm^-1')

        self.assertRaises(Exception, wrong)

    def test_unique_units(self):
        gi_parameters_1 = {"incident_angle" : 0.2,
                           "tilt_angle" : 0.4,
                           "sample_orientation" : 2,
        }

        gi_parameters_2 = {"incident_angle" : 0.6,
                           "tilt_angle" : 0.9,
                           "sample_orientation" : 3,
        }

        gi_parameters_default = {"incident_angle" : 0.0,
                                 "tilt_angle" : 0.0,
                                 "sample_orientation" : 1,
        }

        qip_1 = get_unit_fiber(name='qip_nm^-1', **gi_parameters_1)
        qoop_1 = get_unit_fiber(name='qoop_nm^-1', **gi_parameters_1)

        qip_2 = get_unit_fiber(name='qip_nm^-1')
        qoop_2 = get_unit_fiber(name='qoop_nm^-1')
        qip_2.set_incident_angle(gi_parameters_2['incident_angle'])
        qoop_2.set_incident_angle(gi_parameters_2['incident_angle'])
        qip_2.set_tilt_angle(gi_parameters_2['tilt_angle'])
        qoop_2.set_tilt_angle(gi_parameters_2['tilt_angle'])
        qip_2.set_sample_orientation(gi_parameters_2['sample_orientation'])
        qoop_2.set_sample_orientation(gi_parameters_2['sample_orientation'])

        for k,v in gi_parameters_1.items():
            self.assertEqual(getattr(qip_1, k), v)
            self.assertEqual(getattr(qoop_1, k), v)

        for k,v in gi_parameters_2.items():
            self.assertEqual(getattr(qip_2, k), v)
            self.assertEqual(getattr(qoop_2, k), v)

        qip_3 = get_unit_fiber(name='qip_nm^-1')
        qoop_3 = get_unit_fiber(name='qip_nm^-1')

        for k,v in gi_parameters_default.items():
            self.assertEqual(getattr(qip_3, k), v)
            self.assertEqual(getattr(qoop_3, k), v)

    def test_integrate2d_default(self):
        res2d_ref = self.fi.integrate2d_grazing_incidence(data=self.data)
        res2d_parameters = self.fi.integrate2d_grazing_incidence(data=self.data,
                                                                 npt_ip=1000, npt_oop=1000,
                                                                 unit_ip=get_unit_fiber(name="qip_nm^-1", incident_angle=0.0, tilt_angle=0.0, sample_orientation=1),
                                                                 unit_oop=get_unit_fiber(name="qoop_nm^-1", incident_angle=0.0, tilt_angle=0.0, sample_orientation=1),
                                                                 ip_range=None, oop_range=None,
                                                                 )

        self.assertEqual(abs(res2d_ref.radial - res2d_parameters.radial).max(), 0)
        self.assertEqual(abs(res2d_ref.azimuthal - res2d_parameters.azimuthal).max(), 0)
        self.assertEqual(abs(res2d_ref.intensity - res2d_parameters.intensity).max(), 0)

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

    def test_integrate1d_vertical_runtimeerror(self):
        def res1d_ref():
            wrong = self.fi.integrate1d_grazing_incidence(data=self.data, vertical_integration=True)

        self.assertRaises(RuntimeError, res1d_ref)
        correct = self.fi.integrate1d_grazing_incidence(data=self.data, npt_oop=100, vertical_integration=True)

    def test_integrate1d_horizontal_runtimeerror(self):
        def res1d_ref():
            wrong = self.fi.integrate1d_grazing_incidence(data=self.data, vertical_integration=False)

        self.assertRaises(RuntimeError, res1d_ref)
        correct = self.fi.integrate1d_grazing_incidence(data=self.data, npt_ip=100, vertical_integration=False)

    def test_integrate1d_defaults(self):
        res1d_vertical_ref = self.fi.integrate1d_grazing_incidence(data=self.data, npt_oop=100, vertical_integration=True)
        res1d_vertical = self.fi.integrate1d_grazing_incidence(data=self.data,
                                                               npt_oop=100, npt_ip=500,
                                                               vertical_integration=True)

        self.assertTrue(numpy.allclose(res1d_vertical_ref.radial, res1d_vertical.radial))
        self.assertTrue(numpy.allclose(res1d_vertical_ref.intensity, res1d_vertical.intensity))

        res1d_horizontal_ref = self.fi.integrate1d_grazing_incidence(data=self.data, npt_ip=100, vertical_integration=False)
        res1d_horizontal = self.fi.integrate1d_grazing_incidence(data=self.data,
                                                               npt_oop=500, npt_ip=100,
                                                               vertical_integration=False)

        self.assertTrue(numpy.allclose(res1d_horizontal_ref.radial, res1d_horizontal.radial))
        self.assertTrue(numpy.allclose(res1d_horizontal_ref.intensity, res1d_horizontal.intensity))

    def test_integrate1d_equivalences(self):
        npt_ip = 200
        npt_oop = 100
        res1d_gi = self.fi.integrate1d_grazing_incidence(data=self.data, npt_oop=npt_oop, npt_ip=npt_ip)
        res1d_fiber = self.fi.integrate1d_fiber(data=self.data, npt_oop=npt_oop, npt_ip=npt_ip)

        self.assertEqual(abs(res1d_gi.radial - res1d_fiber.radial).max(), 0)
        self.assertEqual(abs(res1d_gi.intensity - res1d_fiber.intensity).max(), 0)

    def test_integrate1d_equivalences_parameters(self):
        npt_ip = 200
        npt_oop = 100
        ip_range = [0,5]
        oop_range = [0,20]

        incident_angle = 0.2
        tilt_angle = 1.0
        sample_orientation = 3

        res1d_gi = self.fi.integrate1d_grazing_incidence(data=self.data,
                                                         npt_oop=npt_oop, npt_ip=npt_ip,
                                                         ip_range=ip_range, oop_range=oop_range,
                                                         incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation,
                                                         )

        qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)
        qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)

        res1d_fiber = self.fi.integrate1d_fiber(data=self.data,
                                                npt_oop=npt_oop, npt_ip=npt_ip,
                                                ip_range=ip_range, oop_range=oop_range,
                                                unit_ip=qip, unit_oop=qoop,
                                                )

        self.assertEqual(abs(res1d_gi.radial - res1d_fiber.radial).max(), 0)
        self.assertEqual(abs(res1d_gi.intensity - res1d_fiber.intensity).max(), 0)

    def test_integrate1d_explicit_units(self):
        npt_ip = 200
        npt_oop = 100
        incident_angle = 0.2
        tilt_angle = 1.0
        sample_orientation = 3

        res1d_ref = self.fi.integrate_grazing_incidence(data=self.data,
                                                        npt_oop=npt_oop, npt_ip=npt_ip,
                                                        incident_angle=incident_angle,
                                                        tilt_angle=tilt_angle,
                                                        sample_orientation=sample_orientation,
                                                        )

        res1d_string_units = self.fi.integrate_grazing_incidence(data=self.data,
                                                                 npt_oop=npt_oop, npt_ip=npt_ip,
                                                                 unit_ip="qip_nm^-1", unit_oop="qoop_nm^-1",
                                                                 incident_angle=incident_angle,
                                                                 tilt_angle=tilt_angle,
                                                                 sample_orientation=sample_orientation,
        )

        self.assertEqual(abs(res1d_ref.radial - res1d_string_units.radial).max(), 0)
        self.assertEqual(abs(res1d_ref.intensity - res1d_string_units.intensity).max(), 0)

        qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)
        qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)

        res1d_fiber_units = self.fi.integrate_grazing_incidence(data=self.data,
                                                                npt_oop=npt_oop, npt_ip=npt_ip,
                                                                unit_ip=qip, unit_oop=qoop,
        )

        self.assertEqual(abs(res1d_ref.radial - res1d_fiber_units.radial).max(), 0)
        self.assertEqual(abs(res1d_ref.intensity - res1d_fiber_units.intensity).max(), 0)

    def test_integrate1d_priority(self):
        npt_ip = 200
        npt_oop = 100
        incident_angle = 0.2
        tilt_angle = 1.0
        sample_orientation = 3
        res1d_ref = self.fi.integrate_grazing_incidence(data=self.data,
                                                        npt_oop=npt_oop, npt_ip=npt_ip,
                                                        incident_angle=incident_angle,
                                                        tilt_angle=tilt_angle,
                                                        sample_orientation=sample_orientation,
                                                        )

        incident_angle_2 = 0.77
        tilt_angle_2 = -1.0
        sample_orientation_2 = 2

        qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle_2, tilt_angle=tilt_angle_2, sample_orientation=sample_orientation_2)
        qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle_2, tilt_angle=tilt_angle_2, sample_orientation=sample_orientation_2)

        res1d = self.fi.integrate_grazing_incidence(data=self.data,
                                                    npt_oop=npt_oop, npt_ip=npt_ip,
                                                    unit_ip=qip, unit_oop=qoop,
                                                    incident_angle=incident_angle,
                                                    tilt_angle=tilt_angle,
                                                    sample_orientation=sample_orientation,
                                                    )

        self.assertEqual(abs(res1d_ref.radial - res1d.radial).max(), 0)
        self.assertEqual(abs(res1d_ref.intensity - res1d.intensity).max(), 0)

        res1d_wrong = self.fi.integrate_grazing_incidence(data=self.data,
                                                          npt_oop=npt_oop, npt_ip=npt_ip,
                                                          unit_ip=qip, unit_oop=qoop,
                                                    )

        self.assertFalse(abs(res1d_ref.radial - res1d_wrong.radial).max(), 0)
        self.assertFalse(abs(res1d_ref.intensity - res1d_wrong.intensity).max(), 0)

    def test_integrate1d_vertical_deprecated_units(self):
        npt_ip = 200
        npt_oop = 100
        ip_range = [-3,3]
        oop_range = [0,5]
        incident_angle = 0.2
        tilt_angle = 1.0

        qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle)
        qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle)

        res1d_ref = self.fi.integrate_grazing_incidence(data=self.data,
                                                        npt_oop=npt_oop, unit_oop=qoop, oop_range=oop_range,
                                                        npt_ip=npt_ip, unit_ip=qip, ip_range=ip_range,
                                                        incident_angle=incident_angle, tilt_angle=tilt_angle,
                                                        vertical_integration=True,
                                                        )

        res1d_deprecated = self.fi.integrate1d_grazing_incidence(data=self.data,
                                                                 npt_output=npt_oop, output_unit=qoop, output_unit_range=oop_range,
                                                                 npt_integrated=npt_ip, integrated_unit=qip, integrated_unit_range=ip_range,
                                                                 incident_angle=incident_angle, tilt_angle=tilt_angle,
                                                                 )

        self.assertFalse(abs(res1d_ref.radial - res1d_deprecated.radial).max(), 0)
        self.assertFalse(abs(res1d_ref.intensity - res1d_deprecated.intensity).max(), 0)

    def test_integrate1d_horizontal_deprecated_units(self):
        npt_ip = 100
        npt_oop = 100
        ip_range = [-3,3]
        oop_range = [0,5]
        incident_angle = 0.2
        tilt_angle = 1.0

        qip = get_unit_fiber(name="qip_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle)
        qoop = get_unit_fiber(name="qoop_nm^-1", incident_angle=incident_angle, tilt_angle=tilt_angle)

        res1d_ref = self.fi.integrate_grazing_incidence(data=self.data,
                                                        npt_oop=npt_oop, unit_oop=qoop, oop_range=oop_range,
                                                        npt_ip=npt_ip, unit_ip=qip, ip_range=ip_range,
                                                        incident_angle=incident_angle, tilt_angle=tilt_angle,
                                                        vertical_integration=False,
                                                        )

        res1d_deprecated = self.fi.integrate1d_grazing_incidence(data=self.data,
                                                                 npt_output=npt_ip, output_unit=qip, output_unit_range=ip_range,
                                                                 npt_integrated=npt_oop, integrated_unit=qoop, integrated_unit_range=oop_range,
                                                                 incident_angle=incident_angle, tilt_angle=tilt_angle,
                                                                 )

        self.assertFalse(abs(res1d_ref.radial - res1d_deprecated.radial).max(), 0)
        self.assertFalse(abs(res1d_ref.intensity - res1d_deprecated.intensity).max(), 0)

    def test_eight_orientations(self):
        npt_ip = 200
        npt_oop = 200
        ranges = {1 : {"ip" : [2,20], "oop" : [-11,-10], "vertical_integration" : False},
                  2 : {"ip" : [-20,-2], "oop" : [-11,-10], "vertical_integration" : False},
                  3 : {"ip" : [-20,-2], "oop" : [10,11], "vertical_integration" : False},
                  4 : {"ip" : [2,20], "oop" : [10,11], "vertical_integration" : False},
                  5 : {"ip" : [10,11], "oop" : [-20,-2], "vertical_integration" : True},
                  6 : {"ip" : [10,11], "oop" : [2,20], "vertical_integration" : True},
                  7 : {"ip" : [-11,-10], "oop" : [2,20], "vertical_integration" : True},
                  8 : {"ip" : [-11,-10], "oop" : [-20,-2], "vertical_integration" : True},
        }
        result_ref = self.fi.integrate1d_grazing_incidence(data=self.data, npt_ip=npt_ip, npt_oop=npt_oop,
                                                              ip_range=ranges[1]["ip"], oop_range=ranges[1]["oop"],
                                                              vertical_integration=ranges[1]["vertical_integration"], sample_orientation=int(1),
                                                              method=("bbox", "csr", "cython"),
        )

        for so in ranges:
            if so == 1:
                continue
            result = self.fi.integrate1d_grazing_incidence(data=self.data, npt_ip=npt_ip, npt_oop=npt_oop,
                                                              ip_range=ranges[so]["ip"], oop_range=ranges[so]["oop"],
                                                              vertical_integration=ranges[so]["vertical_integration"], sample_orientation=int(so),
                                                              method=("bbox", "csr", "cython"),
            )
            if result.radial.max() < 0:
                intensity = numpy.flip(result.intensity)
            else:
                intensity = result.intensity
            diff = numpy.abs(intensity - result_ref.intensity)
            self.assertLessEqual(diff.max(), 6e-2)

    def test_eight_orientations_fail(self):
        npt_ip = 200
        npt_oop = 200
        ranges = {1 : {"ip" : [2,20], "oop" : [-11,-10], "vertical_integration" : False},
                  2 : {"ip" : [-20,-2], "oop" : [-11,-10], "vertical_integration" : True},
                  3 : {"ip" : [-20,-2], "oop" : [10,11], "vertical_integration" : True},
                  4 : {"ip" : [2,20], "oop" : [10,11], "vertical_integration" : True},
                  5 : {"ip" : [10,11], "oop" : [-20,-2], "vertical_integration" : False},
                  6 : {"ip" : [10,11], "oop" : [2,20], "vertical_integration" : False},
                  7 : {"ip" : [-11,-10], "oop" : [2,20], "vertical_integration" : False},
                  8 : {"ip" : [-11,-10], "oop" : [-20,-2], "vertical_integration" : False},
        }

        result_ref = self.fi.integrate1d_grazing_incidence(data=self.data, npt_ip=npt_ip, npt_oop=npt_oop,
                                                              ip_range=ranges[1]["ip"], oop_range=ranges[1]["oop"],
                                                              vertical_integration=ranges[1]["vertical_integration"], sample_orientation=int(1),
                                                              method=("bbox", "csr", "cython"),
        )

        for so in ranges:
            if so == 1:
                continue
            result = self.fi.integrate1d_grazing_incidence(data=self.data, npt_ip=npt_ip, npt_oop=npt_oop,
                                                              ip_range=ranges[so]["ip"], oop_range=ranges[so]["oop"],
                                                              vertical_integration=ranges[so]["vertical_integration"], sample_orientation=int(so),
                                                              method=("bbox", "csr", "cython"),
            )
            if result.radial.max() < 0:
                intensity = numpy.flip(result.intensity)
            else:
                intensity = result.intensity
            diff = numpy.abs(intensity - result_ref.intensity)
            self.assertGreater(diff.max(), 6e-2)

def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestFiberIntegrator))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

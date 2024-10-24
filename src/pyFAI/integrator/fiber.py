#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

__author__ = "Edgar GUTIERREZ FERNANDEZ "
__contact__ = "edgar.gutierrez-fernandez@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/10/2024"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger(__name__)
import numpy
from .azimuthal import AzimuthalIntegrator
from ..containers import Integrate1dResult
from ..method_registry import IntegrationMethod
from ..io import save_integrate_result
from .. import units

class FiberIntegrator(AzimuthalIntegrator):

    @property
    def incident_angle(self):
        return self._cache_parameters.get('incident_angle', 0.0)

    @property
    def tilt_angle(self):
        return self._cache_parameters.get('tilt_angle', 0.0)

    @property
    def sample_orientation(self):
        return self._cache_parameters.get('sample_orientation', 1)

    def parse_units(self, unit_ip, unit_oop, incident_angle=None, tilt_angle=None, sample_orientation=None):
        if unit_ip is None:
            unit_ip = units.get_unit_fiber("qip_nm^-1")
        else:
            unit_ip = units.to_unit(unit_ip)

        if unit_oop is None:
            unit_oop = units.get_unit_fiber("qoop_nm^-1")
        else:
            unit_oop = units.to_unit(unit_oop)

        if incident_angle is None:
            if isinstance(unit_ip, units.UnitFiber):
                incident_angle = unit_ip.incident_angle
            elif isinstance(unit_oop, units.UnitFiber):
                incident_angle = unit_oop.incident_angle
            else:
                incident_angle = 0.0

        if tilt_angle is None:
            if isinstance(unit_ip, units.UnitFiber):
                tilt_angle = unit_ip.tilt_angle
            elif isinstance(unit_oop, units.UnitFiber):
                tilt_angle = unit_oop.tilt_angle
            else:
                tilt_angle = 0.0

        if sample_orientation is None:
            if isinstance(unit_ip, units.UnitFiber):
                sample_orientation = unit_ip.sample_orientation
            elif isinstance(unit_oop, units.UnitFiber):
                sample_orientation = unit_oop.sample_orientation
            else:
                sample_orientation = 1

        unit_ip = units.to_unit(unit_ip)
        unit_ip.set_incident_angle(incident_angle)
        unit_ip.set_tilt_angle(tilt_angle)
        unit_ip.set_sample_orientation(sample_orientation)

        unit_oop = units.to_unit(unit_oop)
        unit_oop.set_incident_angle(incident_angle)
        unit_oop.set_tilt_angle(tilt_angle)
        unit_oop.set_sample_orientation(sample_orientation)

        return unit_ip, unit_oop

    def reset_integrator(self, incident_angle, tilt_angle, sample_orientation):
        reset = False
        if incident_angle != self.incident_angle:
            logger.info(f"Incident angle set to {incident_angle}. AzimuthalIntegrator will be reset.")
            reset = True
        if tilt_angle != self.tilt_angle:
            logger.info(f"Tilt angle set to {tilt_angle}. AzimuthalIntegrator will be reset.")
            reset = True
        if sample_orientation != self.sample_orientation:
            logger.info(f"Sample orientation set to {sample_orientation}. AzimuthalIntegrator will be reset.")
            reset = True

        if reset:
            self.reset()
            logger.info(f"AzimuthalIntegrator was reset. Current grazing parameters: incident_angle: {incident_angle}, tilt_angle: {tilt_angle}, sample_orientation: {sample_orientation}.")

            self._cache_parameters['incident_angle'] = incident_angle
            self._cache_parameters['tilt_angle'] = tilt_angle
            self._cache_parameters['sample_orientation'] = sample_orientation


    def integrate_fiber(self, data,
                        npt_ip, ip_range=None, unit_ip=None,
                        npt_oop=1000, oop_range=None, unit_oop=None,
                        vertical_integration = True,
                        npt_integrated=100, integrated_unit=units.Q_IP, integrated_unit_range=None,
                        sample_orientation=None,
                        filename=None,
                        correctSolidAngle=True,
                        mask=None, dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None,
                        method=("no", "histogram", "cython"),
                        normalization_factor=1.0, **kwargs):
        """Calculate the integrated profile curve along a specific FiberUnit

        :param ndarray data: 2D array from the Detector/CCD camera
        :param int npt_output: number of points in the output pattern
        :param pyFAI.units.UnitFiber output_unit: Output units
        :param output_unit_range: The lower and upper range of the output unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :param int npt_integrated: number of points to be integrated along integrated_unit
        :param pyFAI.units.UnitFiber integrated_unit: unit to be integrated along integrated_unit_range
        :param integrated_unit_range: The lower and upper range to be integrated. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :param sample_orientation: 1-4, four different orientation of the fiber axis regarding the detector main axis, from 1 to 4 is +90º
        :param str filename: output filename in 2/3 column ascii format
        :param bool correctSolidAngle: correct for solid angle of each pixel if True
        :param ndarray mask: array (same size as image) with 1 for masked pixels, and 0 for valid pixels
        :param float dummy: value for dead/masked pixels
        :param float delta_dummy: precision for dummy value
        :param float polarization_factor: polarization factor between -1 (vertical) and +1 (horizontal).
                * 0 for circular polarization or random,
                * None for no correction,
                * True for using the former correction
        :param ndarray dark: dark noise image
        :param ndarray flat: flat field image
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param float normalization_factor: Value of a normalization monitor
        :return: chi bins center positions and regrouped intensity
        :rtype: Integrate1dResult
        """

        if "npt_output" or "npt_integrated" in kwargs:
            logger.warning(f"npt_output and npt_integrated are deprecated parameters. Use npt_oop and npt_ip instead")

        if "output_unit" or "integrated_unit" in kwargs:
            logger.warning(f"output_unit and integrated_unit are deprecated parameters. Use unit_oop and unit_ip instead")

        if "output_unit_range" or "integrated_unit_range" in kwargs:
            logger.warning(f"output_unit_range and integrated_unit_range are deprecated parameters. Use oop_range and ip_range instead")

        unit_ip, unit_oop = self.parse_units(unit_ip=unit_ip, unit_oop=unit_oop,
                                             sample_orientation=sample_orientation)

        self.reset_integrator(incident_angle=unit_ip.incident_angle,
                              tilt_angle=unit_ip.tilt_angle,
                              sample_orientation=unit_ip.sample_orientation)

        if (isinstance(method, (tuple, list)) and method[0] != "no") or (isinstance(method, IntegrationMethod) and method.split != "no"):
            logger.warning(f"Method {method} is using a pixel-splitting scheme. GI integration should be use WITHOUT PIXEL-SPLITTING! The results could be wrong!")

        if vertical_integration:
            npt_integrated = npt_ip
            integrated_unit_range = ip_range
            integrated_unit = unit_ip
            npt_output = npt_oop
            output_unit_range = oop_range
            output_unit = unit_oop
        else:
            npt_integrated = npt_oop
            integrated_unit_range = oop_range
            integrated_unit = unit_oop
            npt_output = npt_ip
            output_unit_range = ip_range
            output_unit = unit_ip

        res = self.integrate2d_ng(data, npt_rad=npt_integrated, npt_azim=npt_output,
                                  correctSolidAngle=correctSolidAngle,
                                  mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                                  polarization_factor=polarization_factor,
                                  dark=dark, flat=flat, method=method,
                                  normalization_factor=normalization_factor,
                                  radial_range=integrated_unit_range,
                                  azimuth_range=output_unit_range,
                                  unit=(integrated_unit, output_unit))

        unit_scale = output_unit.scale
        sum_signal = res.sum_signal.sum(axis=-1)
        count = res.count.sum(axis=-1)
        sum_normalization = res._sum_normalization.sum(axis=-1)
        mask = numpy.where(count == 0)
        empty = dummy if dummy is not None else self._empty
        intensity = sum_signal / sum_normalization
        intensity[mask] = empty

        if res.sigma is not None:
            sum_variance = res.sum_variance.sum(axis=-1)
            sigma = numpy.sqrt(sum_variance) / sum_normalization
            sigma[mask] = empty
        else:
            sum_variance = None
            sigma = None
        result = Integrate1dResult(res.azimuthal * unit_scale, intensity, sigma)
        result._set_method_called("integrate_radial")
        result._set_unit(output_unit)
        result._set_sum_normalization(sum_normalization)
        result._set_count(count)
        result._set_sum_signal(sum_signal)
        result._set_sum_variance(sum_variance)
        result._set_has_dark_correction(dark is not None)
        result._set_has_flat_correction(flat is not None)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_method = res.method
        result._set_compute_engine = res.compute_engine

        if filename is not None:
            save_integrate_result(filename, result)

        return result

    def integrate_grazing_incidence(self, data,
                                    npt_ip, ip_range=None, unit_ip=None,
                                    npt_oop=1000, oop_range=None, unit_oop=None,
                                    vertical_integration = True,
                                    incident_angle=None, tilt_angle=None, sample_orientation=None,
                                    filename=None,
                                    correctSolidAngle=True,
                                    mask=None, dummy=None, delta_dummy=None,
                                    polarization_factor=None, dark=None, flat=None,
                                    method=("no", "histogram", "cython"),
                                    normalization_factor=1.0,
                                    **kwargs):
        """Calculate the integrated profile curve along a specific FiberUnit, additional inputs for incident angle and tilt angle

        :param ndarray data: 2D array from the Detector/CCD camera
        :param int npt_output: number of points in the output pattern
        :param pyFAI.units.UnitFiber output_unit: Output units
        :param output_unit_range: The lower and upper range of the output unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :param int npt_integrated: number of points to be integrated along integrated_unit
        :param pyFAI.units.UnitFiber integrated_unit: unit to be integrated along integrated_unit_range
        :param integrated_unit_range: The lower and upper range to be integrated. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
        :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
        :param sample_orientation: 1-4, four different orientation of the fiber axis regarding the detector main axis, from 1 to 4 is +90º
        :param str filename: output filename in 2/3 column ascii format
        :param bool correctSolidAngle: correct for solid angle of each pixel if True
        :param ndarray mask: array (same size as image) with 1 for masked pixels, and 0 for valid pixels
        :param float dummy: value for dead/masked pixels
        :param float delta_dummy: precision for dummy value
        :param float polarization_factor: polarization factor between -1 (vertical) and +1 (horizontal).
                * 0 for circular polarization or random,
                * None for no correction,
                * True for using the former correction
        :param ndarray dark: dark noise image
        :param ndarray flat: flat field image
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param float normalization_factor: Value of a normalization monitor
        :return: chi bins center positions and regrouped intensity
        :rtype: Integrate1dResult
        """
        if "npt_output" or "npt_integrated" in kwargs:
            logger.warning(f"npt_output and npt_integrated are deprecated parameters. Use npt_oop and npt_ip instead")

        if "output_unit" or "integrated_unit" in kwargs:
            logger.warning(f"output_unit and integrated_unit are deprecated parameters. Use unit_oop and unit_ip instead")

        if "output_unit_range" or "integrated_unit_range" in kwargs:
            logger.warning(f"output_unit_range and integrated_unit_range are deprecated parameters. Use oop_range and ip_range instead")

        unit_ip, unit_oop = self.parse_units(unit_ip=unit_ip, unit_oop=unit_oop,
                                             incident_angle=incident_angle,
                                             tilt_angle=tilt_angle,
                                             sample_orientation=sample_orientation)

        self.reset_integrator(incident_angle=unit_ip.incident_angle,
                              tilt_angle=unit_ip.tilt_angle,
                              sample_orientation=unit_ip.sample_orientation)

        return self.integrate_fiber(data=data,
                                    npt_oop=npt_oop, unit_oop=unit_oop, oop_range=oop_range,
                                    npt_ip=npt_ip, unit_ip=unit_ip, ip_range=ip_range,
                                    vertical_integration=vertical_integration,
                                    sample_orientation=sample_orientation,
                                    filename=filename,
                                    correctSolidAngle=correctSolidAngle,
                                    mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                                    polarization_factor=polarization_factor, dark=dark, flat=flat,
                                    method=method,
                                    normalization_factor=normalization_factor)


    def integrate2d_fiber(self, data,
                          npt_ip=1000, unit_ip=None, ip_range=None,
                          npt_oop=1000, unit_oop=None, oop_range=None,
                          sample_orientation=None,
                          filename=None,
                          correctSolidAngle=True,
                          mask=None, dummy=None, delta_dummy=None,
                          polarization_factor=None, dark=None, flat=None,
                          method=("no", "histogram", "cython"),
                          normalization_factor=1.0, **kwargs):

        if "npt_horizontal" in kwargs:
            logger.warning(f"npt_horizontal is a valid, but deprecated parameter. Use npt_ip instead")
            npt_ip = kwargs["npt_horizontal"]
        if "npt_vertical" in kwargs:
            logger.warning(f"npt_vertical is a valid, but deprecated parameter. Use npt_oop instead")
            npt_oop = kwargs["npt_vertical"]
        if "horizontal_unit" in kwargs:
            logger.warning(f"horizontal_unit is a valid, but deprecated parameter. Use unit_ip instead")
            unit_ip = kwargs["horizontal_unit"]
        if "vertical_unit" in kwargs:
            logger.warning(f"vertical_unit is a valid, but deprecated parameter. Use unit_oop instead")
            unit_oop = kwargs["vertical_unit"]
        if "horizontal_unit_range" in kwargs:
            logger.warning(f"horizontal_unit_range is a valid, but deprecated parameter. Use ip_range instead")
            ip_range = kwargs["horizontal_unit_range"]
        if "vertical_unit_range" in kwargs:
            logger.warning(f"vertical_unit_range is a valid, but deprecated parameter. Use oop_range instead")
            oop_range = kwargs["horizontal_unit_range"]

        unit_ip, unit_oop = self.parse_units(unit_ip=unit_ip, unit_oop=unit_oop,
                                             sample_orientation=sample_orientation)

        self.reset_integrator(incident_angle=unit_ip.incident_angle,
                              tilt_angle=unit_ip.tilt_angle,
                              sample_orientation=unit_ip.sample_orientation)

        if (isinstance(method, (tuple, list)) and method[0] != "no") or (isinstance(method, IntegrationMethod) and method.split != "no"):
            logger.warning(f"Method {method} is using a pixel-splitting scheme. GI integration should be use WITHOUT PIXEL-SPLITTING! The results could be wrong!")


        return self.integrate2d_ng(data, npt_rad=npt_ip, npt_azim=npt_oop,
                                  correctSolidAngle=correctSolidAngle,
                                  mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                                  polarization_factor=polarization_factor,
                                  dark=dark, flat=flat, method=method,
                                  normalization_factor=normalization_factor,
                                  radial_range=ip_range,
                                  azimuth_range=oop_range,
                                  unit=(unit_ip, unit_oop),
                                  filename=filename)


    def integrate2d_grazing_incidence(self, data,
                                      npt_ip=1000, unit_ip=None, ip_range=None,
                                      npt_oop=1000, unit_oop=None, oop_range=None,
                                      incident_angle=None, tilt_angle=None, sample_orientation=None,
                                      filename=None,
                                      correctSolidAngle=True,
                                      mask=None, dummy=None, delta_dummy=None,
                                      polarization_factor=None, dark=None, flat=None,
                                      method=("no", "histogram", "cython"),
                                      normalization_factor=1.0, **kwargs):

        if "npt_horizontal" in kwargs:
            logger.warning(f"npt_horizontal is a valid, but deprecated parameter. Use npt_ip instead")
            npt_ip = kwargs["npt_horizontal"]
        if "npt_vertical" in kwargs:
            logger.warning(f"npt_vertical is a valid, but deprecated parameter. Use npt_oop instead")
            npt_oop = kwargs["npt_vertical"]
        if "horizontal_unit" in kwargs:
            logger.warning(f"horizontal_unit is a valid, but deprecated parameter. Use unit_ip instead")
            unit_ip = kwargs["horizontal_unit"]
        if "vertical_unit" in kwargs:
            logger.warning(f"vertical_unit is a valid, but deprecated parameter. Use unit_oop instead")
            unit_oop = kwargs["vertical_unit"]
        if "horizontal_unit_range" in kwargs:
            logger.warning(f"horizontal_unit_range is a valid, but deprecated parameter. Use ip_range instead")
            ip_range = kwargs["horizontal_unit_range"]
        if "vertical_unit_range" in kwargs:
            logger.warning(f"vertical_unit_range is a valid, but deprecated parameter. Use oop_range instead")
            oop_range = kwargs["horizontal_unit_range"]

        unit_ip, unit_oop = self.parse_units(unit_ip=unit_ip, unit_oop=unit_oop,
                                             incident_angle=incident_angle,
                                             tilt_angle=tilt_angle,
                                             sample_orientation=sample_orientation)

        self.reset_integrator(incident_angle=unit_ip.incident_angle,
                              tilt_angle=unit_ip.tilt_angle,
                              sample_orientation=unit_ip.sample_orientation)

        if (isinstance(method, (tuple, list)) and method[0] != "no") or (isinstance(method, IntegrationMethod) and method.split != "no"):
            logger.warning(f"Method {method} is using a pixel-splitting scheme. GI integration should be use WITHOUT PIXEL-SPLITTING! The results could be wrong!")

        return self.integrate2d_fiber(data=data, npt_ip=npt_ip, npt_oop=npt_oop,
                                      unit_ip=unit_ip, unit_oop=unit_oop,
                                      ip_range=ip_range,
                                      oop_range=oop_range,
                                      sample_orientation=sample_orientation,
                                      filename=filename,
                                      correctSolidAngle=correctSolidAngle,
                                      mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                                      polarization_factor=polarization_factor, dark=dark, flat=flat,
                                      method=method,
                                      normalization_factor=normalization_factor,
                                      )

    integrate2d = integrate2d_grazing_incidence

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
from .. import units

class FiberIntegrator(AzimuthalIntegrator):
    def integrate_fiber(self, data,
                        npt_output, output_unit=units.Q_OOP, output_unit_range=None,
                        npt_integrated=100, integrated_unit=units.Q_IP, integrated_unit_range=None,
                        sample_orientation=None,
                        filename=None,
                        correctSolidAngle=True,
                        mask=None, dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None,
                        method=("no", "histogram", "cython"),
                        normalization_factor=1.0):
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
        if isinstance(integrated_unit, units.UnitFiber):
            sample_orientation = sample_orientation or integrated_unit.sample_orientation
        else:
            sample_orientation = sample_orientation or 1

        reset = False
        if isinstance(integrated_unit, units.UnitFiber):
            if integrated_unit.sample_orientation != sample_orientation:
                integrated_unit.set_sample_orientation(sample_orientation)
                logger.info(f"Sample orientation set to {sample_orientation} for unit {integrated_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            integrated_unit = units.to_unit(integrated_unit)
            integrated_unit.set_sample_orientation(sample_orientation)
            logger.info(f"Sample orientation set to {sample_orientation} for unit {integrated_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(output_unit, units.UnitFiber):
            if output_unit.sample_orientation != sample_orientation:
                output_unit.set_sample_orientation(sample_orientation)
                logger.info(f"Sample orientation set to {sample_orientation} for unit {output_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            output_unit = units.to_unit(output_unit)
            output_unit.set_sample_orientation(sample_orientation)
            logger.info(f"Sample orientation set to {sample_orientation} for unit {output_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if reset:
            self.reset()
            logger.info(f"AzimuthalIntegrator was reset. Current fiber orientation: {sample_orientation}.")


        if (isinstance(method, (tuple, list)) and method[0] != "no") or (isinstance(method, IntegrationMethod) and method.split != "no"):
            logger.warning(f"Method {method} is using a pixel-splitting scheme. GI integration should be use WITHOUT PIXEL-SPLITTING! The results could be wrong!")

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
                        npt_output, output_unit=units.Q_OOP, output_unit_range=None,
                        npt_integrated=100, integrated_unit=units.Q_IP, integrated_unit_range=None,
                        incident_angle=None, tilt_angle=None, sample_orientation=None,
                        filename=None,
                        correctSolidAngle=True,
                        mask=None, dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None,
                        method=("no", "histogram", "cython"),
                        normalization_factor=1.0):
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
        reset = False

        if isinstance(integrated_unit, units.UnitFiber):
            incident_angle = incident_angle or integrated_unit.incident_angle
        else:
            incident_angle = incident_angle or 0.0

        if isinstance(integrated_unit, units.UnitFiber):
            if integrated_unit.incident_angle != incident_angle:
                integrated_unit.set_incident_angle(incident_angle)
                logger.info(f"Incident angle set to {incident_angle} for unit {integrated_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            integrated_unit = units.to_unit(integrated_unit)
            integrated_unit.set_incident_angle(incident_angle)
            logger.info(f"Incident angle set to {incident_angle} for unit {integrated_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(output_unit, units.UnitFiber):
            if output_unit.incident_angle != incident_angle:
                output_unit.set_incident_angle(incident_angle)
                logger.info(f"Incident angle set to {incident_angle} for unit {output_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            output_unit = units.to_unit(output_unit)
            output_unit.set_incident_angle(incident_angle)
            logger.info(f"Incident angle set to {incident_angle} for unit {output_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(integrated_unit, units.UnitFiber):
            tilt_angle = tilt_angle or integrated_unit.tilt_angle
        else:
            tilt_angle = tilt_angle or 0.0

        if isinstance(integrated_unit, units.UnitFiber):
            if integrated_unit.tilt_angle != tilt_angle:
                integrated_unit.set_tilt_angle(tilt_angle)
                logger.info(f"Tilt angle set to {tilt_angle} for unit {integrated_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            integrated_unit = units.to_unit(integrated_unit)
            integrated_unit.set_tilt_angle(tilt_angle)
            logger.info(f"Tilt angle set to {tilt_angle} for unit {integrated_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(output_unit, units.UnitFiber):
            if output_unit.tilt_angle != tilt_angle:
                output_unit.set_tilt_angle(tilt_angle)
                logger.info(f"Tilt angle set to {tilt_angle} for unit {output_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            output_unit = units.to_unit(output_unit)
            output_unit.set_tilt_angle(tilt_angle)
            logger.info(f"Tilt angle set to {tilt_angle} for unit {output_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(integrated_unit, units.UnitFiber):
            sample_orientation = sample_orientation or integrated_unit.sample_orientation
        else:
            sample_orientation = sample_orientation or 1

        if isinstance(integrated_unit, units.UnitFiber):
            if integrated_unit.sample_orientation != sample_orientation:
                integrated_unit.set_sample_orientation(sample_orientation)
                logger.info(f"Sample orientation set to {sample_orientation} for unit {integrated_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            integrated_unit = units.to_unit(integrated_unit)
            integrated_unit.set_sample_orientation(sample_orientation)
            logger.info(f"Sample orientation set to {sample_orientation} for unit {integrated_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(output_unit, units.UnitFiber):
            if output_unit.sample_orientation != sample_orientation:
                output_unit.set_sample_orientation(sample_orientation)
                logger.info(f"Sample orientation set to {sample_orientation} for unit {output_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            output_unit = units.to_unit(output_unit)
            output_unit.set_sample_orientation(sample_orientation)
            logger.info(f"Sample orientation set to {sample_orientation} for unit {output_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if reset:
            self.reset()
            logger.info(f"AzimuthalIntegrator was reset. Current grazing parameters: incident_angle: {incident_angle}, tilt_angle: {tilt_angle}, sample_orientation: {sample_orientation}.")

        return self.integrate_fiber(data=data,
                                    npt_output=npt_output, output_unit=output_unit, output_unit_range=output_unit_range,
                                    npt_integrated=npt_integrated, integrated_unit=integrated_unit, integrated_unit_range=integrated_unit_range,
                                    sample_orientation=sample_orientation,
                                    filename=filename,
                                    correctSolidAngle=correctSolidAngle,
                                    mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                                    polarization_factor=polarization_factor, dark=dark, flat=flat,
                                    method=method,
                                    normalization_factor=normalization_factor)

    def integrate2d_fiber(self, data,
                          npt_horizontal, horizontal_unit=units.Q_IP, horizontal_unit_range=None,
                          npt_vertical=1000, vertical_unit=units.Q_OOP, vertical_unit_range=None,
                          sample_orientation=None,
                          filename=None,
                          correctSolidAngle=True,
                          mask=None, dummy=None, delta_dummy=None,
                          polarization_factor=None, dark=None, flat=None,
                          method=("no", "histogram", "cython"),
                          normalization_factor=1.0):
        if isinstance(vertical_unit, units.UnitFiber):
            sample_orientation = sample_orientation or vertical_unit.sample_orientation
        else:
            sample_orientation = sample_orientation or 1

        reset = False
        if isinstance(vertical_unit, units.UnitFiber):
            if vertical_unit.sample_orientation != sample_orientation:
                vertical_unit.set_sample_orientation(sample_orientation)
                logger.info(f"Sample orientation was set to {sample_orientation} for unit {vertical_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            vertical_unit = units.to_unit(vertical_unit)
            vertical_unit.set_sample_orientation(sample_orientation)
            logger.info(f"Sample orientation was set to {sample_orientation} for unit {vertical_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(horizontal_unit, units.UnitFiber):
            if horizontal_unit.sample_orientation != sample_orientation:
                horizontal_unit.set_sample_orientation(sample_orientation)
                logger.info(f"Sample orientation was set to {sample_orientation} for unit {horizontal_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            horizontal_unit = units.to_unit(horizontal_unit)
            horizontal_unit.set_sample_orientation(sample_orientation)
            logger.info(f"Sample orientation was set to {sample_orientation} for unit {horizontal_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if reset:
            self.reset()
            logger.info(f"AzimuthalIntegrator was reset. Current fiber parameters: sample_orientation: {sample_orientation}.")


        if (isinstance(method, (tuple, list)) and method[0] != "no") or (isinstance(method, IntegrationMethod) and method.split != "no"):
            logger.warning(f"Method {method} is using a pixel-splitting scheme. GI integration should be use WITHOUT PIXEL-SPLITTING! The results could be wrong!")

        return self.integrate2d_ng(data, npt_rad=npt_horizontal, npt_azim=npt_vertical,
                                  correctSolidAngle=correctSolidAngle,
                                  mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                                  polarization_factor=polarization_factor,
                                  dark=dark, flat=flat, method=method,
                                  normalization_factor=normalization_factor,
                                  radial_range=horizontal_unit_range,
                                  azimuth_range=vertical_unit_range,
                                  unit=(horizontal_unit, vertical_unit),
                                  filename=filename)

    def integrate2d_grazing_incidence(self, data,
                                      npt_horizontal, horizontal_unit=units.Q_IP, horizontal_unit_range=None,
                                      npt_vertical=1000, vertical_unit=units.Q_OOP, vertical_unit_range=None,
                                      incident_angle=None, tilt_angle=None, sample_orientation=None,
                                      filename=None,
                                      correctSolidAngle=True,
                                      mask=None, dummy=None, delta_dummy=None,
                                      polarization_factor=None, dark=None, flat=None,
                                      method=("no", "histogram", "cython"),
                                      normalization_factor=1.0):

        reset = False

        if isinstance(vertical_unit, units.UnitFiber):
            incident_angle = incident_angle or vertical_unit.incident_angle
        else:
            incident_angle = incident_angle or 0.0

        if isinstance(vertical_unit, units.UnitFiber):
            if vertical_unit.incident_angle != incident_angle:
                vertical_unit.set_incident_angle(incident_angle)
                logger.info(f"Incident angle set to {incident_angle} for unit {vertical_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            vertical_unit = units.to_unit(vertical_unit)
            vertical_unit.set_incident_angle(incident_angle)
            logger.info(f"Incident angle set to {incident_angle} for unit {vertical_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(horizontal_unit, units.UnitFiber):
            if horizontal_unit.incident_angle != incident_angle:
                horizontal_unit.set_incident_angle(incident_angle)
                logger.info(f"Incident angle set to {incident_angle} for unit {horizontal_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            horizontal_unit = units.to_unit(horizontal_unit)
            horizontal_unit.set_incident_angle(incident_angle)
            logger.info(f"Incident angle set to {incident_angle} for unit {horizontal_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(vertical_unit, units.UnitFiber):
            tilt_angle = tilt_angle or vertical_unit.tilt_angle
        else:
            tilt_angle = tilt_angle or 0.0

        if isinstance(vertical_unit, units.UnitFiber):
            if vertical_unit.tilt_angle != tilt_angle:
                vertical_unit.set_tilt_angle(tilt_angle)
                logger.info(f"Tilt angle set to {tilt_angle} for unit {vertical_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            vertical_unit = units.to_unit(vertical_unit)
            vertical_unit.set_tilt_angle(tilt_angle)
            logger.info(f"Tilt angle set to {tilt_angle} for unit {vertical_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(horizontal_unit, units.UnitFiber):
            if horizontal_unit.tilt_angle != tilt_angle:
                horizontal_unit.set_tilt_angle(tilt_angle)
                logger.info(f"Tilt angle set to {tilt_angle} for unit {horizontal_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            horizontal_unit = units.to_unit(horizontal_unit)
            horizontal_unit.set_tilt_angle(tilt_angle)
            logger.info(f"Tilt angle set to {tilt_angle} for unit {horizontal_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(vertical_unit, units.UnitFiber):
            sample_orientation = sample_orientation or vertical_unit.sample_orientation
        else:
            sample_orientation = sample_orientation or 1

        if isinstance(vertical_unit, units.UnitFiber):
            if vertical_unit.sample_orientation != sample_orientation:
                vertical_unit.set_sample_orientation(sample_orientation)
                logger.info(f"Sample orientation set to {sample_orientation} for unit {vertical_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            vertical_unit = units.to_unit(vertical_unit)
            vertical_unit.set_sample_orientation(sample_orientation)
            logger.info(f"Sample orientation set to {sample_orientation} for unit {vertical_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if isinstance(horizontal_unit, units.UnitFiber):
            if horizontal_unit.sample_orientation != sample_orientation:
                horizontal_unit.set_sample_orientation(sample_orientation)
                logger.info(f"Sample orientation set to {sample_orientation} for unit {horizontal_unit}. AzimuthalIntegrator will be reset.")
                reset = True
        else:
            horizontal_unit = units.to_unit(horizontal_unit)
            horizontal_unit.set_sample_orientation(sample_orientation)
            logger.info(f"Sample orientation set to {sample_orientation} for unit {horizontal_unit}. AzimuthalIntegrator will be reset.")
            reset = True

        if reset:
            self.reset()
            logger.info(f"AzimuthalIntegrator was reset. Current grazing parameters: incident_angle: {incident_angle}, tilt_angle: {tilt_angle}, sample_orientation: {sample_orientation}.")

        if (isinstance(method, (tuple, list)) and method[0] != "no") or (isinstance(method, IntegrationMethod) and method.split != "no"):
            logger.warning(f"Method {method} is using a pixel-splitting scheme. GI integration should be use WITHOUT PIXEL-SPLITTING! The results could be wrong!")

        return self.integrate2d_fiber(data=data, npt_horizontal=npt_horizontal, npt_vertical=npt_vertical,
                                      horizontal_unit=horizontal_unit, vertical_unit=vertical_unit,
                                      horizontal_unit_range=horizontal_unit_range,
                                      vertical_unit_range=vertical_unit_range,
                                      sample_orientation=sample_orientation,
                                      filename=filename,
                                      correctSolidAngle=correctSolidAngle,
                                      mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                                      polarization_factor=polarization_factor, dark=dark, flat=flat,
                                      method=method,
                                      normalization_factor=normalization_factor,
                                      )

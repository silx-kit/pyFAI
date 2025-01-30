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

__author__ = "Edgar Gutiérrez Fernández "
__contact__ = "edgar.gutierrez-fernandez@esr.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/11/2024"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger(__name__)
import numpy
from .azimuthal import AzimuthalIntegrator
from ..containers import Integrate1dResult
from ..method_registry import IntegrationMethod
from ..io import save_integrate_result
from ..units import parse_fiber_unit
from ..utils.decorators import deprecated_warning

def get_deprecated_params_1d(**kwargs) -> dict:
    deprecated = {}
    if "npt_output" in kwargs:
        deprecated_warning(type_=type(kwargs["npt_output"]), name="npt_output", replacement=("npt_oop, npt_ip, vertical_integration instead"), since_version="2024.11/12")
        deprecated['npt_oop'] = kwargs["npt_output"]
        deprecated['vertical_integration'] = True
    if "npt_integrated" in kwargs:
        deprecated_warning(type_=type(kwargs["npt_integrated"]), name="npt_integrated", replacement=("npt_oop, npt_ip, vertical_integration instead"), since_version="2024.11/12")
        deprecated['npt_ip'] = kwargs["npt_integrated"]
        deprecated['vertical_integration'] = True
    if "output_unit" in kwargs:
        deprecated_warning(type_=type(kwargs["output_unit"]), name="output_unit", replacement=("unit_oop, unit_ip, vertical_integration instead"), since_version="2024.11/12")
        deprecated['unit_oop'] = kwargs["output_unit"]
        deprecated['vertical_integration'] = True
    if "integrated_unit" in kwargs:
        deprecated_warning(type_=type(kwargs["integrated_unit"]), name="integrated_unit", replacement=("unit_oop, unit_ip, vertical_integration instead"), since_version="2024.11/12")
        deprecated['unit_ip'] = kwargs["integrated_unit"]
        deprecated['vertical_integration'] = True
    if "output_unit_range" in kwargs:
        deprecated_warning(type_=type(kwargs["output_unit_range"]), name="output_unit_range", replacement=("oop_range, ip_range, vertical_integration instead"), since_version="2024.11/12")
        deprecated['oop_range'] = kwargs["output_unit_range"]
        deprecated['vertical_integration'] = True
    if "integrated_unit_range" in kwargs:
        deprecated_warning(type_=type(kwargs["integrated_unit_range"]), name="integrated_unit_range", replacement=("oop_range, ip_range, vertical_integration instead"), since_version="2024.11/12")
        deprecated['ip_range'] = kwargs["integrated_unit_range"]
        deprecated['vertical_integration'] = True
    return deprecated

def get_deprecated_params_2d(**kwargs) -> dict:
    deprecated = {}
    if "npt_horizontal" in kwargs:
        deprecated_warning(type_=type(kwargs["npt_horizontal"]), name="npt_horizontal", replacement="npt_ip", since_version="2024.11/12")
        deprecated['npt_ip'] = kwargs["npt_horizontal"]
    if "npt_vertical" in kwargs:
        deprecated_warning(type_=type(kwargs["npt_vertical"]), name="npt_vertical", replacement="npt_oop", since_version="2024.11/12")
        deprecated['npt_oop'] = kwargs["npt_vertical"]
    if "horizontal_unit" in kwargs:
        deprecated_warning(type_=type(kwargs["horizontal_unit"]), name="horizontal_unit", replacement="unit_ip", since_version="2024.11/12")
        deprecated['unit_ip'] = kwargs["horizontal_unit"]
    if "vertical_unit" in kwargs:
        deprecated_warning(type_=type(kwargs["vertical_unit"]), name="vertical_unit", replacement="unit_oop", since_version="2024.11/12")
        deprecated['unit_oop'] = kwargs["vertical_unit"]
    if "horizontal_unit_range" in kwargs:
        deprecated_warning(type_=type(kwargs["horizontal_unit_range"]), name="horizontal_unit_range", replacement="ip_range", since_version="2024.11/12")
        deprecated['ip_range'] = kwargs["horizontal_unit_range"]
    if "vertical_unit_range" in kwargs:
        deprecated_warning(type_=type(kwargs["vertical_unit_range"]), name="vertical_unit_range", replacement="oop_range", since_version="2024.11/12")
        deprecated['oop_range'] = kwargs["vertical_unit_range"]
    return deprecated

class FiberIntegrator(AzimuthalIntegrator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_parameters = {}

    def __repr__(self, dist_unit="m", ang_unit="rad", wl_unit="m"):
        core_repr = super().__repr__(dist_unit=dist_unit, ang_unit=ang_unit, wl_unit=wl_unit)
        return f"{core_repr}\nIncident angle: {self.incident_angle:.2f}° Tilt angle {self.tilt_angle:.2f}° Sample orientation {self.sample_orientation}"

    @property
    def incident_angle(self) -> float:
        """Pitch angle: projection angle of the beam in the sample. Its rotation axis is the horizontal axis of the lab system.
        """
        return self._cache_parameters.get('incident_angle', 0.0)

    @property
    def tilt_angle(self) -> float:
        """Roll angle. Its rotation axis is the beam axis. Tilting of the horizon for grazing incidence in thin films.
        """
        return self._cache_parameters.get('tilt_angle', 0.0)

    @property
    def sample_orientation(self) -> int:
        """Orientation of the fiber axis according to EXIF orientation values

        Sample orientations
        1 - No changes are applied to the image
        2 - Image is mirrored (flipped horizontally)
        3 - Image is rotated 180 degrees
        4 - Image is rotated 180 degrees and mirrored
        5 - Image is mirrored and rotated 90 degrees counter clockwise
        6 - Image is rotated 90 degrees counter clockwise
        7 - Image is mirrored and rotated 90 degrees clockwise
        8 - Image is rotated 90 degrees clockwise
        """
        return self._cache_parameters.get('sample_orientation', 1)

    def reset_integrator(self, incident_angle, tilt_angle, sample_orientation):
        """Reset the cache values for the gi/fiber parameters
        :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
        :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
        :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def sample_orientation)
        """
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
                        npt_oop=None, unit_oop=None, oop_range=None,
                        npt_ip=None, unit_ip=None, ip_range=None,
                        vertical_integration = True,
                        sample_orientation=None,
                        filename=None,
                        correctSolidAngle=True,
                        mask=None, dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None,
                        method=("no", "histogram", "cython"),
                        normalization_factor=1.0,
                        **kwargs):
        """Calculate the integrated profile curve along a specific FiberUnit, additional input for sample_orientation

        :param ndarray data: 2D array from the Detector/CCD camera
        :param int npt_oop: number of points to be used along the out-of-plane axis
        :param pyFAI.units.UnitFiber/str unit_oop: unit to describe the out-of-plane axis. If not provided, it takes qoop_nm^-1
        :param list oop_range: The lower and upper range of the out-of-plane unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :param int npt_ip: number of points to be used along the in-plane axis
        :param pyFAI.units.UnitFiber/str unit_ip: unit to describe the in-plane axis. If not provided, it takes qip_nm^-1
        :param list ip_range: The lower and upper range of the in-plane unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :param bool vertical_integration: If True, integrates along unit_ip; if False, integrates along unit_oop
        :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
        :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
        :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def sample_orientation)
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
        deprecated_params = get_deprecated_params_1d(**kwargs)
        npt_oop = deprecated_params.get('npt_oop', None) or npt_oop
        npt_ip = deprecated_params.get('npt_ip', None) or npt_ip
        unit_oop = deprecated_params.get('unit_oop', None) or unit_oop
        unit_ip = deprecated_params.get('unit_ip', None) or unit_ip
        oop_range = deprecated_params.get('oop_range', None) or oop_range
        ip_range = deprecated_params.get('ip_range', None) or ip_range
        vertical_integration = deprecated_params.get('vertical_integration', None) or vertical_integration

        unit_ip = unit_ip or 'qip_nm^-1'
        unit_oop = unit_oop or 'qoop_nm^-1'
        unit_ip = parse_fiber_unit(unit=unit_ip,
                                   incident_angle=kwargs.get('incident_angle', None),
                                   tilt_angle=kwargs.get('tilt_angle', None),
                                   sample_orientation=sample_orientation)
        unit_oop = parse_fiber_unit(unit=unit_oop,
                                    incident_angle=unit_ip.incident_angle,
                                    tilt_angle=unit_ip.tilt_angle,
                                    sample_orientation=unit_ip.sample_orientation)

        self.reset_integrator(incident_angle=unit_ip.incident_angle,
                              tilt_angle=unit_ip.tilt_angle,
                              sample_orientation=unit_ip.sample_orientation)

        if (isinstance(method, (tuple, list)) and method[0] != "no") or (isinstance(method, IntegrationMethod) and method.split != "no"):
            logger.warning(f"Method {method} is using a pixel-splitting scheme. GI integration should be use WITHOUT PIXEL-SPLITTING! The results could be wrong!")


        if vertical_integration and npt_oop is None:
            raise RuntimeError("npt_oop (out-of-plane bins) is needed to do the integration")
        elif not vertical_integration and npt_ip is None:
            raise RuntimeError("npt_ip (in-plane bins) is needed to do the integration")

        npt_oop = npt_oop or 500
        npt_ip = npt_ip or 500

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
        mask_ = numpy.where(count == 0)
        empty = dummy if dummy is not None else self._empty
        intensity = sum_signal / sum_normalization
        intensity[mask_] = empty

        if res.sigma is not None:
            sum_variance = res.sum_variance.sum(axis=-1)
            sigma = numpy.sqrt(sum_variance) / sum_normalization
            sigma[mask_] = empty
        else:
            sum_variance = None
            sigma = None
        result = Integrate1dResult(res.azimuthal, intensity, sigma)
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

    integrate_grazing_incidence = integrate_fiber
    integrate1d_grazing_incidence = integrate_grazing_incidence
    integrate1d_fiber = integrate_fiber

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
        """Reshapes the data pattern as a function of two FiberUnits, additional inputs for sample_orientation

        :param ndarray data: 2D array from the Detector/CCD camera
        :param int npt_ip: number of points to be used along the in-plane axis
        :param pyFAI.units.UnitFiber/str unit_ip: unit to describe the in-plane axis. If not provided, it takes qip_nm^-1
        :param list ip_range: The lower and upper range of the in-plane unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :param int npt_oop: number of points to be used along the out-of-plane axis
        :param pyFAI.units.UnitFiber/str unit_oop: unit to describe the out-of-plane axis. If not provided, it takes qoop_nm^-1
        :param list oop_range: The lower and upper range of the out-of-plane unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
        :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
        :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def sample_orientation)
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
        :return: regrouped intensity and unit arrays
        :rtype: Integrate2dResult
        """
        deprecated_params = get_deprecated_params_2d(**kwargs)
        npt_oop = deprecated_params.get('npt_oop', None) or npt_oop
        npt_ip = deprecated_params.get('npt_ip', None) or npt_ip
        unit_oop = deprecated_params.get('unit_oop', None) or unit_oop
        unit_ip = deprecated_params.get('unit_ip', None) or unit_ip
        oop_range = deprecated_params.get('oop_range', None) or oop_range
        ip_range = deprecated_params.get('ip_range', None) or ip_range

        unit_ip = unit_ip or 'qip_nm^-1'
        unit_oop = unit_oop or 'qoop_nm^-1'
        unit_ip = parse_fiber_unit(unit=unit_ip,
                                   sample_orientation=sample_orientation,
                                   incident_angle=kwargs.get('incident_angle', None),
                                   tilt_angle=kwargs.get('tilt_angle', None),
                                   )
        unit_oop = parse_fiber_unit(unit=unit_oop,
                                    incident_angle=unit_ip.incident_angle,
                                    tilt_angle=unit_ip.tilt_angle,
                                    sample_orientation=unit_ip.sample_orientation)

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

    integrate2d_grazing_incidence = integrate2d_fiber

    def integrate2d_polar(self, polar_degrees=True, radial_unit="nm^-1", rotate=False, **kwargs):
        """Reshapes the data pattern as a function of polar angle=arctan(qOOP / qIP) versus q modulus.

        :param polar_degrees bool: if True, polar angle in degrees, else in radians
        :param radial_unit str: unit of q modulus:  nm^-1 or A^-1
        :param rotate bool: if False, polar_angle vs q, if True q vs polar_angle

        Calls method integrate2d_fiber ->
        """
        unit_oop = "chigi_deg" if polar_degrees else "chigi_rad"
        unit_ip = "qtot_A^-1" if radial_unit == "A^-1" else "qtot_nm^-1"

        if rotate:
            kwargs["unit_ip"] = unit_oop
            kwargs["unit_oop"] = unit_ip
        else:
            kwargs["unit_ip"] = unit_ip
            kwargs["unit_oop"] = unit_oop
        return self.integrate2d_fiber(**kwargs)

    integrate2d_polar.__doc__ += "\n" + integrate2d_fiber.__doc__

    def integrate2d_exitangles(self, angle_degrees=True, **kwargs):
        """Reshapes the data pattern as a function of exit angles with the origin at the sample horizon

        :param angle_degrees bool: if True, exit angles in degrees, else in radians

        Calls method integrate2d_fiber ->
        """
        if angle_degrees:
            unit_ip = "exit_angle_horz_deg"
            unit_oop = "exit_angle_vert_deg"
        else:
            unit_ip = "exit_angle_horz_rad"
            unit_oop = "exit_angle_vert_rad"

        kwargs["unit_ip"] = unit_ip
        kwargs["unit_oop"] = unit_oop
        return self.integrate2d_fiber(**kwargs)

    integrate2d_exitangles.__doc__ += "\n" + integrate2d_fiber.__doc__

    def integrate1d_polar(self, polar_degrees=True, radial_unit="nm^-1", radial_integration=False, **kwargs):
        """Calculate the integrated profile curve along the polar angle=arctan(qOOP / qIP) or as a function of the polar angle along q modulus

        :param polar_degrees bool: if True, polar angle in degrees, else in radians
        :param radial_unit str: unit of q modulus:  nm^-1 or A^-1
        :param radial_integration bool: if False, the output profile is I vs q, if True (I vs polar_angle)

        Calls method integrate_fiber ->
        """
        unit_oop = "chigi_deg" if polar_degrees else "chigi_rad"
        unit_ip = "qtot_A^-1" if radial_unit == "A^-1" else "qtot_nm^-1"
        kwargs["unit_ip"] = unit_ip
        kwargs["unit_oop"] = unit_oop
        if radial_integration:
            kwargs["vertical_integration"] = True
        else:
            kwargs["vertical_integration"] = False
        return self.integrate_fiber(**kwargs)

    integrate1d_polar.__doc__ += "\n" + integrate_fiber.__doc__

    def integrate1d_exitangles(self, angle_degrees=True, vertical_integration=True, **kwargs):
        """Calculate the integrated profile curve along the one of the exit angles (with the origin at the sample horizon)

        :param angle_degrees bool: if True, exit angles in degrees, else in radians
        :param vertical_integration bool: if True, the output profile is I vs vertical_angle, if False (I vs horizontal_angle)

        Calls method integrate_fiber ->
        """
        if angle_degrees:
            unit_ip = "exit_angle_horz_deg"
            unit_oop = "exit_angle_vert_deg"
        else:
            unit_ip = "exit_angle_horz_rad"
            unit_oop = "exit_angle_vert_rad"

        kwargs["unit_ip"] = unit_ip
        kwargs["unit_oop"] = unit_oop
        kwargs["vertical_integration"] = vertical_integration
        return self.integrate_fiber(**kwargs)

    integrate1d_exitangles.__doc__ += "\n" + integrate_fiber.__doc__

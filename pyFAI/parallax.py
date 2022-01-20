# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2021 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module for managing parallax correction"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/01/2022"
__status__ = "development"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger(__name__)
import numpy
import numexpr
import scipy.integrate, scipy.signal
from math import sin, cos, pi, log, sqrt
from .utils.decorators import timeit
EPS = numpy.finfo("float64").eps
OVERSAMPLING = 1024  # Default oversampling value


class Beam:
    """This class represents the properties of the incident beam"""

    def __init__(self, width=None, profile="gaussian"):
        """Constructor of the Beam
        
        :param width: FWHM or diameter of the beam in meters
        :param shape: can be "gaussian", "circle" or "square"
        """
        self.width = width
        self.profile = profile
        self.pos = None
        self.intensity = None
        self.funct = self.__getattribute__(self.profile)

    def __repr__(self):
        return f"Beam of shape {self.profile} with a width of {self.width}m"

    def get_config(self):
        """Gets the configuration as a dictionnary"""
        return {"class": self.__class__.__name__,
                "width": self.width,
                "profile":self.profile}

    def set_config(self, cfg):
        """Set the configuration from a dictionnary"""
        if "class" in cfg:
            assert cfg["class"] == self.__class__.__name__
        self.width = cfg.get("width")
        self.profile = cfg.get("profile")
        self.pos = None
        self.intensity = None
        self.funct = self.__getattribute__(self.profile)

    def gaussian(self, width, over=None):
        """Model the beam as a gaussian profile.

        :param width: FWHM of the gaussian curve
        :param over: oversampling factor, take that many points to describe the peak
        :return: position, intensity (normalized)
        """
        over = over or OVERSAMPLING
        if width < EPS:
            print("Warning, width too small")
            width = EPS
        step = width / over
        sigma = width / (2.0 * sqrt(2.0 * log(2.0)))
        nsteps = 2 * int(3 * sigma / step + 1) + 1
        pos = (numpy.arange(nsteps) - nsteps // 2) * step
        peak = numexpr.evaluate("exp(-pos**2/(2*(sigma**2)))")
        peak /= peak.sum()
        return pos, peak

    def square(self, width, over=None):
        """Model the beam as a square signal

        :param width: width of the signal
        :param over: oversampling factor, take that many points to describe the peak
        :return: position, intensity (normalized)
        """
        over = over or OVERSAMPLING
        if width < EPS:
            print("Warning, width too small")
            width = EPS
        step = width / over
        nsteps = 2 * int(2 * width / step + 1) + 1
        pos = (numpy.arange(nsteps) - nsteps // 2) * step
        peak = numexpr.evaluate("where(abs(pos)<=width/2, 1.0, 0.0)")
        peak /= peak.sum()
        return pos, peak

    def circle(self, width, over=None):
        """Model the beam as a circular signal

        :param width: Diameter of the beam
        :param over: oversampling factor, take that many points to describe the peak
        :return: position, intensity (normalized)
        """
        over = over or OVERSAMPLING
        if width < EPS:
            print("Warning, width too small")
            width = EPS
        step = width / over
        nsteps = 2 * int(width / step + 2) + 1
        pos = (numpy.arange(nsteps) - nsteps // 2) * step
        peak = numexpr.evaluate("where(abs(pos)<=width/2, sqrt(1.0-(2.0*pos/width)**2), 0.0)")
        peak /= peak.sum()
        return pos, peak

    def __call__(self, width=None, over=None):
        """Return the beam profile I=f(x)"""
        over = over or OVERSAMPLING
        if width is None:
            if self.pos is None:
                self.pos, self.intensity = self.funct(self.width)
            return self.pos, self.intensity
        else:
            return self.funct(width=width, over=over)


class BaseSensor:
    """
    This class represents the sensor material for an thick slab (thickness >> 1/mu) and a small beam .

    The absorption for the beam is assumed to follow Beer-Lambert's law, 
    i.e. the beam sees an exponential decay of the intensity with the traveled distance exp(-µx).
    The inverse of the absorption coefficient, 1/mu, is the "average" penetration distance.
    """

    def __init__(self, mu=None):
        """Constructor of the base-sensor.
        
        :param mu: 
        """
        self.mu = mu

    def measure_displacement(self, angle=None, beam=None, over=None):
        """Measures the displacement of the peak due to parallax effect
        :param width: FWHM of the peak, same unit as thickness. *Unused*
        :param angle: incidence angle in degrees
        :param beam: shape of the beam. *Unused*
        :param over: oversampling factor. *Unused*
        """
        return numpy.sin(angle) / self.mu

    def __repr__(self):
        return f"Thick sensor with µ={self.mu} 1/m"

    def get_config(self):
        """Gets the configuration as a dictionnary"""
        return {"class": self.__class__.__name__,
                "mu": self.mu, }

    def set_config(self, cfg):
        """Set the configuration from a dictionnary"""
        if "class" in cfg:
            assert cfg["class"] == self.__class__.__name__
        self.mu = cfg.get("mu")
        return self


ThickSensor = BaseSensor


class ThinSensor(BaseSensor):
    """
    This class represents the sensor material.
    
    The absorption for the beam is assumed to follow Beer-Lambert's law, 
    i.e. the beam sees an exponential decay of the intensity with the traveled distance.
    The inverse of the absorption coefficient, 1/mu, is the "average" penetration distance.
     
    The sensor is defined by its thickness (in meter) and the apparent (normal) efficiency
    of the material which is usually easier to obtain from the manufacturer than the 
    exect composition of the sensor or the absortion coefficient.
    
    Nota: the absortion coefficient depends on the wavelength, but the effect is expected to
    be negligeable when refining the wagelength in a calibration experiment. 
    """

    def __init__(self, thickness=None, efficiency=None):
        """Class to simulate the decay of the parallax effect

        :param thickness: thickness of the sensible layer, in meter
        :param efficiency: efficiency for the sensor material between 0 and 1
        """
        self.thickness = None
        self.efficiency = None
        if thickness is not None:
            self.thickness = float(thickness)
        if efficiency is not None:
            self.efficiency = float(efficiency)
        if self.thickness and self.efficiency:
            BaseSensor.__init__(self, -log(1.0 - self.efficiency) / self.thickness)
        else:
            BaseSensor.__init__(self, None)
        self.formula = numexpr.NumExpr("where(x<0, 0.0, mu*exp(-mu*x))")

    def __repr__(self):
        return f"Thin sensor with µ={self.mu} 1/m, thickness={self.thickness}m and efficiency={self.efficiency}"

    def get_config(self):
        """Gets the configuration as a dictionnary"""
        return {"class": self.__class__.__name__,
                "mu": self.mu,
                "thickness": self.thickness,
                "efficiency": self.efficiency}

    def set_config(self, cfg):
        """Set the configuration from a dictionnary"""
        if "class" in cfg:
            assert cfg["class"] == self.__class__.__name__
        self.mu = cfg.get("mu")
        self.thickness = float(cfg.get("thickness"))
        self.efficiency = float(cfg.get("efficiency"))
        return self

    def __call__(self, x):
        "Calculate the absorption at position x"
        return self.formula(self.mu, x)

    def integrate(self, x):
        """Integrate between 0 and x

        :param x: length of the path, same unit as thickness
        """
        return scipy.integrate.quad(self, 0.0, x)

    def test(self):
        """Validate the formula for lambda
        sum(decay) between 0 and thickness is efficiency"""
        value, error = self.integrate(self.thickness)
        assert abs(value - self.efficiency) < error
        return True

    def absorption(self, angle, over=None):
        """Calculate the absorption along the path for a beam inclined with the given angle
        
        :param angle: incidence angle in radian
        :param over: enforce oversampling factor
        :return position (along the detector), absorption (normalized)
        """
        over = over or OVERSAMPLING
        length = self.thickness / cos(angle)
        pos = numpy.linspace(0, length, over)
        decay = self.__call__(pos)
        decay /= decay.sum()  # Normalize the decay
        pos *= sin(angle)  # rotate the decay to have it in the detector plan:
        return pos, decay

    def convolve(self, angle, beam, over=None):
        """Calculate the line profile convoluted with parallax effect

        :param angle: incidence angle in radians
        :param beam: instance of Beam with width and shape
        :param over: oversampling factor for numerical integration
        :return: position, intensity(position)
        """
        over = over or OVERSAMPLING
        pos_dec, decay = self.absorption(angle, over)

        pos_peak, peak = beam(beam.width / cos(angle), over=over)
        # Interpolate grids ...
        pos_min = min(pos_dec[0], pos_peak[0])
        pos_max = max(pos_dec[-1], pos_peak[-1])
        step = min((pos_dec[-1] - pos_dec[0]) / (pos_dec.shape[0] - 1),
                   (pos_peak[-1] - pos_peak[0]) / (pos_dec.shape[0] - 1))
        if step < EPS:
            step = max((pos_dec[-1] - pos_dec[0]) / (pos_dec.shape[0] - 1),
                       (pos_peak[-1] - pos_peak[0]) / (pos_dec.shape[0] - 1))
        nsteps_2 = int(max(-pos_min, pos_max) / step + 0.5)

        max_steps = 1 << 20
        if nsteps_2 > max_steps:
            nsteps_2 = max_steps
            step = (pos_max - pos_min) / (max_steps - 1)

        pos = (numpy.arange(2 * nsteps_2 + 1) - nsteps_2) * step
        big_decay = numpy.interp(pos, pos_dec, decay, left=0.0, right=0.0)
        dsum = big_decay.sum()
        if dsum == 0:
            big_decay[numpy.argmin(abs(pos))] = 1.0
        else:
            big_decay /= dsum
        big_peak = numpy.interp(pos, pos_peak, peak, left=0.0, right=0.0)
        return pos, scipy.signal.convolve(big_peak, big_decay, "same")

    def plot_displacement(self, angle, beam, ax=None):
        """Plot the displacement of the peak depending on the FWHM and the incidence angle"""
        if ax is None:
            from matplotlib.pyplot import subplots
            _, ax = subplots()
        ax.set_xlabel("Radial displacement on the detector (mm)")
        c = self.absorption(angle)
        ax.plot(*c, label="Absorption")
        c = beam()
        ax.plot(*c, label=f"peak w={beam.width*1000} mm")
        c = beam(beam.width / cos(angle))
        ax.plot(*c, label=f"peak w={beam.width*1000} mm, inclined")

        c = self.convolve(angle, beam=beam)
        ax.plot(*c, label="Convolution")
        idx = numpy.argmax(c[1])
        maxi = self.measure_displacement(angle, beam=beam)
        ax.annotate(f"$\\delta r$={maxi*1000:.3f}mm", (maxi, c[1][idx]),
                   xycoords='data',
                   xytext=(0.8, 0.5), textcoords='axes fraction',
                   arrowprops=dict(facecolor='black', shrink=0.05),
                   horizontalalignment='right', verticalalignment='top',)
        ax.set_title(f"Profile {beam.profile}, width: {beam.width*1000}mm, angle: {180*angle/pi}°")
        ax.legend()
        return ax

    def measure_displacement(self, angle, beam, over=None):
        """Measures the displacement of the peak due to parallax effect
        :param angle: incidence angle in radians
        :param beam: Instance of Beam
        :param over: oversampling factor
        """
        over = over or OVERSAMPLING
        angle = abs(angle)
        if angle >= pi / 2.0:
            return 1.0 / self.mu

        x, y = self.convolve(angle, beam=beam, over=over)
        ymax = y.max()
        idx_max = numpy.where(y == ymax)[0]
        if len(idx_max) > 1:
            return x[idx_max].mean()
        else:
            idx = idx_max[0]
            if idx > 1 or idx < len(y) - 1:
                # Second order tailor expension
                f_prime = 0.5 * (y[idx + 1] - y[idx - 1])
                f_sec = (y[idx + 1] + y[idx - 1] - 2 * y[idx])
                if f_sec == 0:
                    print('f" is null')
                    return x[idx]
                delta = -f_prime / f_sec
                if abs(delta) > 1:
                    print("Too large displacement")
                    return x[idx]
                step = (x[-1] - x[0]) / (len(x) - 1)
                return x[idx] + delta * step
            return x[idx]


class Parallax:
    """Provides the displacement of the peak position
    due to parallax effect from the sine of the incidence angle
     
    """
    SIZE = 64  # <8k  best fits into L1 cache

    def __init__(self, sensor=None, beam=None):
        """Constructor for the Parallax class
        
        :param sensor: instance of the BaseSensor
        :param beam: instance of Beam
        """
        if sensor:
            assert isinstance(sensor, BaseSensor)
        if beam:
            assert isinstance(beam, Beam)
        self.sensor = sensor
        self.beam = beam
        self.displacement = None
        self.sin_incidence = None
        if self.sensor:
            self.init()

    @timeit
    def init(self, over=None):
        """Initialize actually the class...

        :param over: enforce the oversampling factor for numerical integration 
        """
        angles = numpy.linspace(0, pi / 2.0, self.SIZE)
        displacement = [self.sensor.measure_displacement(angle, beam=self.beam, over=over)
                        for angle in angles]
        self.sin_incidence = numpy.sin(angles)
        self.displacement = numpy.array(displacement)

    def __repr__(self):
        return f"Parallax correction for {self.beam} and {self.sensor}"

    def __call__(self, sin_incidence):
        """Calculate the displacement from the sine of the incidence angle"""
        return numpy.interp(sin_incidence, self.sin_incidence, self.displacement)

    def get_config(self):
        dico = {"class": self.__class__.__name__}
        dico["sensor"] = self.sensor.get_config() if self.sensor else None
        dico["beam"] = self.beam.get_config() if self.beam else None
        return dico

    def set_config(self, cfg):
        """Set the configuration from a dictionnary"""
        if "class" in cfg:
            assert cfg["class"] == self.__class__.__name__
        if "beam" in cfg:
            bfg = cfg["beam"]
            if bfg is None:
                self.beam = None
            else:
                if "class" in bfg:
                    classname = bfg["class"]
                    Klass = globals()[classname]
                else:
                    Klass = Beam
                self.beam = Klass()
                self.beam.set_config(bfg)
        if "sensor" in cfg:
            sfg = cfg["sensor"]
            if sfg is None:
                self.sensor = None
            else:
                if "class" in sfg:
                    classname = sfg["class"]
                    Klass = globals()[classname]
                else:
                    Klass = ThinSensor
                self.sensor = Klass()
                self.sensor.set_config(sfg)
        self.init()
        return self

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
__date__ = "23/08/2021"
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


class Sensor:
    """
    This class represents the sensor material.
    
    The absorption for the beam is assumed to follow Beer-Lambert's law, 
    i.e. the beam sees an exponential decay of the intensity with the traveled distance.
    The inverse of the absorption coefficient, 1/lambda, is the "average" penetration distance.
     
    The sensor is defined by its thickness (in meter) and the apparent (normal) efficiency
    of the material which is usually easier to obtain from the manufacturer than the 
    exect composition of the sensor or the absortion coefficient.
    
    Nota: the absortion coefficient depends on the wavelength, but the effect is expected to
    be negligeable when refining the wagelength in a calibration experiment. 
    """
    OVER = 1024 # Default oversampling value
    def __init__(self, thickness, efficiency):
        """Class to simulate the decay of the parallax effect

        :param thickness: thickness of the sensible layer, in meter or mm, µm...
        :param efficiency: efficiency for the sensor material between 0 and 1
        """
        self.thickness = float(thickness)
        self.efficiency = float(efficiency)
        self.lam = - log(1.0-efficiency)/thickness
        self.formula = numexpr.NumExpr("where(x<0, 0.0, l*exp(-l*x))")

    def __call__(self, x):
        "Calculate the absorption at position x"
        return self.formula(self.lam, x)

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

    def absorption(self, angle, over=None):
        """Calculate the absorption along the path for a beam inclined with the given angle
        
        :param angle: incidence angle
        :param over: enforce oversampling factor
        :return position (along the detector), absorption (normalized)
        """
        over = over or self.OVER
        angle_r = numpy.deg2rad(angle)
        length = self.thickness/cos(angle_r)
        pos = numpy.linspace(0, length, over)
        decay = self.__call__(pos)
        decay /= decay.sum()   # Normalize the decay
        pos *= sin(angle_r) # rotate the decay to have it in the detector plan:
        return pos, decay

    def gaussian(self, width, over=None):
        """Model the beam as a gaussian profile.

        :param width: FWHM of the gaussian curve
        :param over: oversampling factor, take that many points to describe the peak
        :return: position, intensity (normalized)
        """
        over = over or self.OVER
        if width<EPS:
            print("Warning, width too small")
            width = EPS
        step = width / over
        sigma = width/(2.0*sqrt(2.0*log(2.0)))
        nsteps = 2*int(3*sigma/step+1) + 1
        pos = (numpy.arange(nsteps) - nsteps//2) * step
        peak = numexpr.evaluate("exp(-pos**2/(2*(sigma**2)))")
        peak /= peak.sum()
        return pos, peak

    def square(self, width, over=None):
        """Model the beam as a square signal

        :param width: width of the signal
        :param over: oversampling factor, take that many points to describe the peak
        :return: position, intensity (normalized)
        """
        over = over or self.OVER
        if width<EPS:
            print("Warning, width too small")
            width = EPS
        step = width / over
        nsteps = 2*int(2*width/step+1) + 1
        pos = (numpy.arange(nsteps) - nsteps//2) * step
        peak = numexpr.evaluate("where(abs(pos)<=width/2, 1.0, 0.0)")
        peak /= peak.sum()
        return pos, peak

    def circle(self, width, over=None):
        """Model the beam as a circular signal

        :param width: Diameter of the beam
        :param over: oversampling factor, take that many points to describe the peak
        :return: position, intensity (normalized)
        """
        over = over or self.OVER
        if width<EPS:
            print("Warning, width too small")
            width = EPS
        step = width / over
        nsteps = 2*int(width/step+2) + 1
        pos = (numpy.arange(nsteps) - nsteps//2) * step
        peak = numexpr.evaluate("where(abs(pos)<=width/2, sqrt(1.0-(2.0*pos/width)**2), 0.0)")
        peak /= peak.sum()
        return pos, peak

    def convolve(self, width, angle, beam="gaussian", over=None):
        """Calculate the line profile convoluted with parallax effect

        :param width: FWHM of the peak, same unit as thickness
        :param angle: incidence angle in degrees
        :param beam: shape of the incident beam
        :param over: oversampling factor for numerical integration
        :return: position, intensity(position)
        """
        over = over or self.OVER
        angle_r = numpy.deg2rad(angle)
        pos_dec, decay = self.absorption(angle, over)
        peakf = self.__getattribute__(beam)
        pos_peak, peak = peakf(width/cos(angle_r), over=over)
        #Interpolate grids ...
        pos_min = min(pos_dec[0], pos_peak[0])
        pos_max = max(pos_dec[-1], pos_peak[-1])
        step = min((pos_dec[-1] - pos_dec[0])/(pos_dec.shape[0]-1),
                   (pos_peak[-1] - pos_peak[0])/(pos_dec.shape[0]-1))
        if step<EPS:
            step = max((pos_dec[-1] - pos_dec[0])/(pos_dec.shape[0]-1),
                       (pos_peak[-1] - pos_peak[0])/(pos_dec.shape[0]-1))
        nsteps_2 = int(max(-pos_min, pos_max)/step + 0.5)
        pos = (numpy.arange(2*nsteps_2+1) - nsteps_2) *  step
        big_decay = numpy.interp(pos, pos_dec, decay, left=0.0, right=0.0)
        dsum = big_decay.sum()
        if dsum == 0:
            big_decay[numpy.argmin(abs(pos))] = 1.0
        else:
            big_decay /= dsum
        big_peak = numpy.interp(pos, pos_peak, peak, left=0.0, right=0.0)
        return pos, scipy.signal.convolve(big_peak, big_decay, "same")

    def plot_displacement(self, width, angle, beam="gaussian", ax=None):
        """Plot the displacement of the peak depending on the FWHM and the incidence angle"""
        if ax is None:
            from matplotlib.pyplot import subplots
            _, ax = subplots()
        ax.set_xlabel("Radial displacement on the detector (mm)")
        c = self.absorption(angle)
        ax.plot(*c, label="Absorption")
        peakf = self.__getattribute__(beam)
        c = peakf(width)
        ax.plot(*c, label=f"peak w={width} mm")
        c = peakf(width/cos(angle*pi/180))
        ax.plot(*c, label=f"peak w={width} mm, inclined")

        c = self.convolve(width, angle, beam=beam)
        ax.plot(*c, label="Convolution")
        idx = numpy.argmax(c[1])
        maxi = self.measure_displacement(width, angle, beam=beam)
        ax.annotate(f"$\delta r$={maxi:.3f}", (maxi, c[1][idx]),
                   xycoords='data',
                   xytext=(0.8, 0.5), textcoords='axes fraction',
                   arrowprops=dict(facecolor='black', shrink=0.05),
                   horizontalalignment='right', verticalalignment='top',)
        ax.set_title(f"Profile {beam}, width: {width}mm, angle: {angle}°")
        ax.legend()
        return ax

    def measure_displacement(self, width, angle, beam="gaussian", over=None):
        """Measures the displacement of the peak due to parallax effect
        :param width: FWHM of the peak, same unit as thickness
        :param angle: incidence angle in degrees
        :param beam: shape of the beam
        :param over: oversampling factor
        """
        over = over or self.OVER
        if angle >=90:
            return 1.0/self.lam
        
        x,y = self.convolve(width, angle, beam=beam, over=over)
        ymax = y.max()
        idx_max = numpy.where(y==ymax)[0]
        if len(idx_max)>1:
            return x[idx_max].mean()

        idx = idx_max[0]
        if idx>1 or idx<len(y)-1:
            #Second order tailor expension
            f_prime = 0.5*(y[idx+1]-y[idx-1])
            f_sec = (y[idx+1]+y[idx-1]-2*y[idx])
            if f_sec == 0:
                print('f" is null')
                return x[idx]
            delta = -f_prime/f_sec
            if abs(delta)>1:
                print("Too large displacement")
                return x[idx]
            step = (x[-1]-x[0])/(len(x)-1)
            return x[idx] + delta*step
        return x[idx]


class Parallax:
    """Provides the displacement of the peak position
    due to parallax effect from the sine of the incidence angle
     
    """
    SIZE = 64 # <8k  best fits into L1 cache
    def __init__(self, sensor_thickness, sensor_efficiency, 
                 beam_size, beam_profile="gaussian"):
        """Constructor for the Parallax class
        
        :param sensor_thickness: Thickness of the sensor material in meter
        :param sensor_efficency: Normal absorption of the sensor (in 0..1) 
        :param beam_size: FHWM of the beam in meter (or diameter for circular beam) 
        :param beam_profile: "gaussian", "circle" or "square"
        """
        self.sensor = Sensor(sensor_thickness, sensor_efficiency)
        self.beam_size = beam_size
        self.beam_profile = beam_profile
        self.displacement = None
        self.sin_incidence = None
        self.init()
    
    @timeit
    def init(self):
        """Initialize actually the class...
        
        See doc for the constructor
        """
        
        displacement = [self.sensor.measure_displacement(self.beam_size, angle_d, beam=self.beam_profile)
            for angle_d in numpy.linspace(0, 90, self.SIZE)]
            
        self.sin_incidence = numpy.sin(numpy.linspace(0, pi/2.0, self.SIZE)) 
        self.displacement = numpy.array(displacement)
        
    def __call__(self, sin_incidence):
        """Calculate the displacement from the sine of the incidence angle"""
        return numpy.interp(sin_incidence, self.sin_incidence, self.displacement)
    
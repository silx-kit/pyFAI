# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import division, print_function, absolute_import

"""Jupyter helper functions
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/03/2018"
__status__ = "Production"
__docformat__ = 'restructuredtext'

import numpy
from pylab import subplots, legend
from matplotlib import lines


def display(img=None, cp=None, ai=None, label=None, sg=None, ax=None):
    """Display an image with the control points and the calibrated rings
    in Jupyter notebooks

    :param img: 2D numpy array with an image
    :param cp: ControlPoint instance
    :param ai: azimuthal integrator for iso-2th curves
    :param label: name of the curve
    :param sg: single geometry object regrouping img, cp and ai
    :param ax: subplot object to display in, if None, a new one is created.
    :return: Matplotlib subplot
    """
    if ax is None:
        _fig, ax = subplots()
    if sg is not None:
        if img is None:
            img = sg.image
        if cp is None:
            cp = sg.control_points
        if ai is None:
            ai = sg.geometry_refinement
        if label is None:
            label = sg.label

    ax.imshow(numpy.arcsinh(img).astype(numpy.float32), origin="lower", cmap="inferno")
    ax.set_title(label)
    if cp is not None:
        for lbl in cp.get_labels():
            pt = numpy.array(cp.get(lbl=lbl).points)
            if len(pt) > 0:
                ax.scatter(pt[:, 1], pt[:, 0], label=lbl)
        if ai is not None and cp.calibrant is not None:
            tth = cp.calibrant.get_2th()
            ttha = ai.twoThetaArray()
            ax.contour(ttha, levels=tth, cmap="autumn", linewidths=2, linestyles="dashed")
        legend()
    return ax


def plot1d(result, calibrant=None, label=None, ax=None):
    """Display the powder diffraction pattern in the jupyter notebook

    :param result: instance of Integrate1dResult
    :param calibrant: Calibrant instance to overlay diffraction lines
    :param label: (str) name of the curve
    :param ax: subplot object to display in, if None, a new one is created.
    :return: Matplotlib subplot
    """
    if ax is None:
        _fig, ax = subplots()

    unit = result.unit
    if result.sigma is not None:
        ax.errorbar(result.radial, result.intensity, result.sigma, label=label)
    else:
        ax.plot(result.radial, result.intensity, label=label)

    if label:
        ax.legend()
    if calibrant:
        x_values = calibrant.get_peaks(unit)
        if x_values is not None:
            for x in x_values:
                line = lines.Line2D([x, x], ax.axis()[2:4],
                                    color='red', linestyle='--')
                ax.add_line(line)

    ax.set_title("1D integration")
    ax.set_xlabel(unit.label)
    ax.set_ylabel("Intensity")

    return ax


def plot2d(result, calibrant=None, label=None, ax=None):
    """Display the caked image in the jupyter notebook

    :param result: instance of Integrate2dResult
    :param calibrant: Calibrant instance to overlay diffraction lines
    :param label: (str) name of the curve

    :param ax: subplot object to display in, if None, a new one is created.
    :return: Matplotlib subplot
    """
    img = result.intensity
    pos_rad = result.radial
    pos_azim = result.azimuthal
    if ax is None:
        _fig, ax = subplots()
    ax.imshow(numpy.arcsinh(img),
              origin="lower",
              extent=[pos_rad.min(), pos_rad.max(), pos_azim.min(), pos_azim.max()],
              aspect="auto",
              cmap="inferno")
    if label:
        ax.set_title("2D regrouping")
    else:
        ax.set_title(label)
    ax.set_xlabel(result.unit.label)
    ax.set_ylabel(r"Azimuthal angle $\chi$ ($^{o}$)")
    if calibrant:
        from pyFAI import units
        x_values = None
        twotheta = numpy.array([i for i in calibrant.get_2th() if i])  # in radian
        unit = result.unit
        if unit == units.TTH_DEG:
            x_values = numpy.rad2deg(twotheta)
        elif unit == units.TTH_RAD:
            x_values = twotheta
        elif unit == units.Q_A:
            x_values = (4.e-10 * numpy.pi / calibrant.wavelength) * numpy.sin(.5 * twotheta)
        elif unit == units.Q_NM:
            x_values = (4.e-9 * numpy.pi / calibrant.wavelength) * numpy.sin(.5 * twotheta)
        if x_values is not None:
            for x in x_values:
                line = lines.Line2D([x, x], [pos_azim.min(), pos_azim.max()],
                                    color='red', linestyle='--')
                ax.add_line(line)
    return ax

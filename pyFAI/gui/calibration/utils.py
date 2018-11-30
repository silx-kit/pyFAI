# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
#
# ###########################################################################*/

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "26/11/2018"


import numpy
import collections

from silx.gui import qt
from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from silx.gui.colors import Colormap

from pyFAI import units


def getFreeColorRange(colormap):
    name = colormap['name']
    colormap = Colormap(name)

    # extract all hues from colormap
    colors = colormap.getNColors()
    hues = []
    for c in colors:
        c = qt.QColor.fromRgb(c[0], c[1], c[2])
        hues.append(c.hueF())

    # search the bigger empty hue range
    current = (0, 0.0, 0.2)
    hues = filter(lambda x: x >= 0, set(hues))
    hues = list(sorted(hues))
    if len(hues) > 1:
        for i in range(len(hues)):
            h1 = hues[i]
            h2 = hues[(i + 1) % len(hues)]
            if h2 < h1:
                h2 = h2 + 1.0

            diff = h2 - h1
            if diff > 0.5:
                diff = 1.0 - diff

            if diff > current[0]:
                current = diff, h1, h2
    elif len(hues) == 1:
        h = (hues[0] + 0.5) % 1.0
        current = (0, h - 0.1, h + 0.1)
    else:
        pass

    h1, h2 = current[1:]
    delta = (h2 - h1) / 6.0

    # move the range from the colormap
    h1, h2 = h1 + delta, h2 - delta
    hmin = (h1 + h2) / 2.0

    # generate colors with 3 hsv control points
    # (h1, 1, 1), (hmid, 1, 0.5), (h2, 1, 1)
    colors = []
    for i in range(5):
        h = h1 + (hmin - h1) * (i / 5.0)
        v = 0.5 + 0.1 * (5 - i)
        c = qt.QColor.fromHsvF(h % 1.0, 1.0, v)
        colors.append(c)
    for i in range(5):
        h = hmin + (h2 - hmin) * (i / 5.0)
        v = 0.5 + 0.1 * (i)
        c = qt.QColor.fromHsvF(h % 1.0, 1.0, v)
        colors.append(c)
    return colors


def tthToRad(twoTheta, unit, wavelength=None, directDist=None):
    """
    Convert a two theta angle from original `unit` to radian.

    `directDist = ai.getFit2D()["directDist"]`
    """
    if isinstance(twoTheta, numpy.ndarray):
        pass
    elif isinstance(twoTheta, collections.Iterable):
        twoTheta = numpy.array(twoTheta)

    if unit == units.TTH_RAD:
        return twoTheta
    elif unit == units.TTH_DEG:
        return numpy.deg2rad(twoTheta)
    elif unit == units.Q_A:
        if wavelength is None:
            raise AttributeError("wavelength have to be specified")
        return numpy.arcsin((twoTheta * wavelength) / (4.e-10 * numpy.pi)) * 2.0
    elif unit == units.Q_NM:
        if wavelength is None:
            raise AttributeError("wavelength have to be specified")
        return numpy.arcsin((twoTheta * wavelength) / (4.e-9 * numpy.pi)) * 2.0
    elif unit == units.R_MM:
        if directDist is None:
            raise AttributeError("directDist have to be specified")
        # GF: correct formula?
        return numpy.arctan(twoTheta / directDist)
    elif unit == units.R_M:
        if directDist is None:
            raise AttributeError("directDist have to be specified")
        # GF: correct formula?
        return numpy.arctan(twoTheta / (directDist * 0.001))
    else:
        raise ValueError("Converting from 2th to unit %s is not supported", unit)


def from2ThRad(twoTheta, unit, wavelength=None, directDist=None, ai=None):
    if isinstance(twoTheta, numpy.ndarray):
        pass
    elif isinstance(twoTheta, collections.Iterable):
        twoTheta = numpy.array(twoTheta)

    if unit == units.TTH_DEG:
        return numpy.rad2deg(twoTheta)
    elif unit == units.TTH_RAD:
        return twoTheta
    elif unit == units.Q_A:
        return (4.e-10 * numpy.pi / wavelength) * numpy.sin(.5 * twoTheta)
    elif unit == units.Q_NM:
        return (4.e-9 * numpy.pi / wavelength) * numpy.sin(.5 * twoTheta)
    elif unit == units.R_MM:
        # GF: correct formula?
        if directDist is not None:
            beamCentre = directDist
        else:
            beamCentre = ai.getFit2D()["directDist"]  # in mm!!
        return beamCentre * numpy.tan(twoTheta)
    elif unit == units.R_M:
        # GF: correct formula?
        if directDist is not None:
            beamCentre = directDist
        else:
            beamCentre = ai.getFit2D()["directDist"]  # in mm!!
        return beamCentre * numpy.tan(twoTheta) * 0.001
    else:
        raise ValueError("Converting from 2th to unit %s is not supported", unit)


def createProcessingWidgetOverlay(parent):
    """Create a widget overlay to show that the application is processing data
    to update the plot.

    :param qt.QWidget widget: Widget containing the overlay
    :rtype: qt.QWidget
    """
    if hasattr(parent, "centralWidget"):
        parent = parent.centralWidget()
    button = WaitingPushButton(parent)
    button.setWaiting(True)
    button.setText("Processing...")
    button.setDown(True)
    position = parent.size()
    size = button.sizeHint()
    position = (position - size) / 2
    rect = qt.QRect(qt.QPoint(position.width(), position.height()), size)
    button.setGeometry(rect)
    button.setVisible(True)
    return button


_trueStrings = set(["yes", "true", "1"])
_falseStrings = set(["no", "false", "0"])


def stringToBool(string):
    """Returns a boolean from a string.

    :raise ValueError: If the string do not contains a boolean information.
    """
    lower = string.lower()
    if lower in _trueStrings:
        return True
    if lower in _falseStrings:
        return False
    raise ValueError("'%s' is not a valid boolean" % string)

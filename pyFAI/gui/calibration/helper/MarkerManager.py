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
__date__ = "23/08/2018"

import logging
import numpy

from ..model import MarkerModel
from .. import utils

_logger = logging.getLogger(__name__)


class MarkerManager(object):
    """Synchronize the display of markers from MarkerModel to a plot."""

    def __init__(self, plot, markerModel):
        self.__plot = plot
        self.__markerModel = markerModel
        self.__markerModel.changed.connect(self.__updateMarkers)
        self.__geometry = None
        self.__markers = []

    def updateProjection(self, geometry, radialUnit, wavelength, directDist):
        self.__geometry = geometry
        self.__radialUnit = radialUnit
        self.__wavelength = wavelength
        self.__directDist = directDist
        self.__updateMarkers()

    def __updateMarkers(self):
        for item in self.__markers:
            self.__plot.removeMarker(item.getLegend())

        template = "__markers__%s"

        for marker in self.__markerModel:
            if isinstance(marker, MarkerModel.PhysicalMarker):
                chiRad, tthRad = marker.physicalPosition()
            elif isinstance(marker, MarkerModel.PixelMarker):
                x, y = marker.pixelPosition()
                ax, ay = numpy.array([x]), numpy.array([y])
                chiRad = self.__geometry.chi(ay, ax)[0]
                tthRad = self.__geometry.tth(ay, ax)[0]
            else:
                _logger.debug("Unsupported logger %s", type(marker))
                continue

            tth = utils.from2ThRad(tthRad,
                                   unit=self.__radialUnit,
                                   wavelength=self.__wavelength,
                                   directDist=self.__directDist)
            chi = numpy.rad2deg(chiRad)

            legend = template % marker.name()
            self.__plot.addMarker(x=tth, y=chi, color="pink", legend=legend, text=marker.name())
            item = self.__plot._getMarker(legend)
            self.__markers.append(item)

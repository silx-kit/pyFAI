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
import functools
import numpy

from silx.gui import qt

from ..model import MarkerModel
from .. import utils

_logger = logging.getLogger(__name__)


class MarkerManager(object):
    """Synchronize the display of markers from MarkerModel to a plot."""

    _ITEM_TEMPLATE = "__markers__%s"

    def __init__(self, plot, markerModel, pixelBasedPlot=False):
        self.__plot = plot
        self.__markerModel = markerModel
        self.__markerModel.changed.connect(self.__updateMarkers)
        self.__geometry = None
        self.__markers = []
        self.__pixelBasedPlot = pixelBasedPlot

    def updateProjection(self, geometry, radialUnit, wavelength, directDist):
        if self.__pixelBasedPlot:
            raise RuntimeError("Invalide operation for this kind of plot")
        self.__geometry = geometry
        self.__radialUnit = radialUnit
        self.__wavelength = wavelength
        self.__directDist = directDist
        self.__updateMarkers()

    def __updateMarkers(self):
        for item in self.__markers:
            self.__plot.removeMarker(item.getLegend())

        if self.__pixelBasedPlot:
            self.__createMarkerOnPixelBased()
        else:
            self.__createMarkerOnPhysicalBased()

    def createMarkPixelAction(self, parent, mousePos):
        maskPixelAction = qt.QAction(parent)
        maskPixelAction.setText("Mark this pixel coord")
        maskPixelAction.triggered.connect(functools.partial(self.__createPixelMarker, mousePos))
        if not self.__pixelBasedPlot:
            maskPixelAction.setEnabled(self.__geometry is not None)
        return maskPixelAction

    def createMarkGeometryAction(self, parent, mousePos):
        maskGeometryAction = qt.QAction(parent)
        maskGeometryAction.setText(u"Mark this χ/2θ coord")
        maskGeometryAction.triggered.connect(functools.partial(self.__createGeometryMarker, mousePos))
        maskGeometryAction.setEnabled(self.__geometry is not None)
        return maskGeometryAction

    def dataToChiTth(self, data):
        """Returns chi and 2theta angles in radian from data coordinate"""
        try:
            tthRad = utils.tthToRad(data[0],
                                    unit=self.__radialUnit,
                                    wavelength=self.__wavelength,
                                    directDist=self.__directDist)
        except Exception:
            _logger.debug("Backtrace", exc_info=True)
            tthRad = None

        chiDeg = data[1]
        if chiDeg is not None:
            chiRad = numpy.deg2rad(chiDeg)
        else:
            chiRad = None

        return chiRad, tthRad

    def __createPixelMarker(self, pos):
        pos = self.__plot.pixelToData(pos.x(), pos.y())
        if self.__pixelBasedPlot:
            pixel = pos[1], pos[0]
        else:
            chiRad, tthRad = self.dataToChiTth(pos)
            pixel = utils.findPixel(self.__geometry, chiRad, tthRad)
        name = self.__findUnusedMarkerName()
        marker = MarkerModel.PixelMarker(name, pixel[1], pixel[0])
        self.__markerModel.add(marker)

    def __createGeometryMarker(self, pos):
        pos = self.__plot.pixelToData(pos.x(), pos.y())
        chiRad, tthRad = self.dataToChiTth(pos)
        name = self.__findUnusedMarkerName()
        marker = MarkerModel.PhysicalMarker(name, chiRad, tthRad)
        self.__markerModel.add(marker)

    def __findUnusedMarkerName(self):
        template = "mark%d"
        markerNames = set([m.name() for m in self.__markerModel])
        for i in range(0, 1000):
            name = template % i
            if name not in markerNames:
                return name
        # Returns something
        return "mark"

    def __createMarkerOnPixelBased(self):
        for marker in self.__markerModel:
            if isinstance(marker, MarkerModel.PhysicalMarker):
                # TODO: implement it
                continue
            elif isinstance(marker, MarkerModel.PixelMarker):
                x, y = marker.pixelPosition()
            else:
                _logger.debug("Unsupported logger %s", type(marker))
                continue
            legend = self._ITEM_TEMPLATE % marker.name()
            self.__plot.addMarker(x=x, y=y, color="pink", legend=legend, text=marker.name())
            item = self.__plot._getMarker(legend)
            self.__markers.append(item)

    def __createMarkerOnPhysicalBased(self):
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

            legend = self._ITEM_TEMPLATE % marker.name()
            self.__plot.addMarker(x=tth, y=chi, color="pink", legend=legend, text=marker.name())
            item = self.__plot._getMarker(legend)
            self.__markers.append(item)

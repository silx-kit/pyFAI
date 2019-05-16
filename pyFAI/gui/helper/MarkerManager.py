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
__date__ = "14/05/2019"

import logging
import functools
import numpy

from silx.gui import qt

from ..model import MarkerModel
from ..utils import unitutils
from ..utils import eventutils
from ..CalibrationContext import CalibrationContext
from pyFAI.ext.invert_geometry import InvertGeometry
import pyFAI.units


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
        self.__radialUnit = None
        self.__mustBeUpdated = False

        eventutils.createShowSignal(self.__plot)
        self.__plot.sigShown.connect(self.__plotIsShown)

    def updateProjection(self, geometry, radialUnit, wavelength, directDist, redraw=True):
        if self.__pixelBasedPlot:
            raise RuntimeError("Invalide operation for this kind of plot")
        self.__geometry = geometry
        self.__radialUnit = radialUnit
        self.__invertGeometry = InvertGeometry(
            self.__geometry.array_from_unit(typ="center", unit=self.__radialUnit, scale=True),
            numpy.rad2deg(self.__geometry.chiArray()))
        self.__wavelength = wavelength
        self.__directDist = directDist
        if redraw:
            self.__updateMarkers()

    def updatePhysicalMarkerPixels(self, geometry):
        self.__geometry = geometry
        if geometry is None:
            self.__wavelength = None
            self.__directDist = None
        else:
            self.__wavelength = geometry.wavelength
            try:
                self.__directDist = geometry.getFit2D()["directDist"]
            except Exception:
                # The geometry could not fit this param
                _logger.debug("Backtrace", exc_info=True)
                self.__directDist = None

        if geometry is not None:
            invertGeometry = InvertGeometry(
                geometry.array_from_unit(typ="center", unit=pyFAI.units.TTH_RAD, scale=False),
                geometry.chiArray())

        self.__markerModel.lockSignals()
        for marker in self.__markerModel:
            if not isinstance(marker, MarkerModel.PhysicalMarker):
                continue

            chiRad, tthRad = marker.physicalPosition()
            pixel = None
            if geometry is not None:
                pixel = invertGeometry(tthRad, chiRad, True)

                ax, ay = numpy.array([pixel[1]]), numpy.array([pixel[0]])
                tth = geometry.tth(ay, ax)[0]
                chi = geometry.chi(ay, ax)[0]

                error = numpy.sqrt((tthRad - tth) ** 2 + (chiRad - chi) ** 2)
                if error > 0.05:
                    # The identified pixel is far from the requested chi/tth. Marker ignored.
                    pixel = None
            if pixel is not None:
                marker.setPixelPosition(pixel[1], pixel[0])
            else:
                marker.removePixelPosition()
        # TODO: should be managed by the model
        self.__markerModel.wasChanged()
        self.__markerModel.unlockSignals()

    def __plotIsShown(self):
        if self.__mustBeUpdated:
            self.__updateMarkers()

    def __updateMarkers(self):
        if not self.__plot.isVisible():
            # Update the display later
            self.__mustBeUpdated = True
            return

        self.__mustBeUpdated = False

        for item in self.__markers:
            self.__plot.removeMarker(item.getLegend())

        color = CalibrationContext.instance().getMarkerColor(0, mode="html")
        for marker in self.__markerModel:
            position = self.getMarkerLocation(marker)
            if position is None:
                continue
            legend = self._ITEM_TEMPLATE % marker.name()
            self.__plot.addMarker(x=position[0], y=position[1], color=color, legend=legend, text=marker.name())
            item = self.__plot._getMarker(legend)
            self.__markers.append(item)

    def getMarkerLocation(self, marker):
        """
        Returns the location of the marker in the plot axes
        """
        if self.__pixelBasedPlot:
            if isinstance(marker, MarkerModel.PhysicalMarker):
                return marker.pixelPosition()
            elif isinstance(marker, MarkerModel.PixelMarker):
                return marker.pixelPosition()
            else:
                _logger.debug("Unsupported marker %s", type(marker))
                return None
        else:
            if isinstance(marker, MarkerModel.PhysicalMarker):
                chiRad, tthRad = marker.physicalPosition()
            elif isinstance(marker, MarkerModel.PixelMarker):
                if self.__geometry is None:
                    return None
                x, y = marker.pixelPosition()
                ax, ay = numpy.array([x]), numpy.array([y])
                chiRad = self.__geometry.chi(ay, ax)[0]
                tthRad = self.__geometry.tth(ay, ax)[0]
            else:
                _logger.debug("Unsupported marker %s", type(marker))
                return None

            if self.__radialUnit is None:
                return

            try:
                tth = unitutils.from2ThRad(tthRad,
                                           unit=self.__radialUnit,
                                           wavelength=self.__wavelength,
                                           directDist=self.__directDist)
                chi = numpy.rad2deg(chiRad)
            except Exception:
                _logger.debug("Backtrace", exc_info=True)
                return None
            return tth, chi

    def findClosestMarker(self, mousePos, delta=20):
        delta = delta ** 2.0
        if isinstance(mousePos, qt.QPoint):
            mousePos = mousePos.x(), mousePos.y()
        currentMarker = None
        currentDistance = float("inf")
        for marker in self.__markerModel:
            location = self.getMarkerLocation(marker)
            if location is None:
                continue
            location = self.__plot.dataToPixel(x=location[0], y=location[1], check=False)
            distance = (mousePos[0] - location[0]) ** 2.0 + (mousePos[1] - location[1]) ** 2.0
            if distance < currentDistance and distance < delta:
                currentDistance = distance
                currentMarker = marker
        return currentMarker

    def createRemoveClosestMaskerAction(self, parent, mousePos):
        action = qt.QAction(parent)

        marker = self.findClosestMarker(mousePos)
        if marker is None:
            return None

        action.setText("Remove marker '%s'" % marker.name())
        action.triggered.connect(functools.partial(self.__removeMarker, marker))
        return action

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

        if self.__pixelBasedPlot:
            x, y = data
            ax, ay = numpy.array([x]), numpy.array([y])
            chi = self.__geometry.chi(ay, ax)[0]
            tth = self.__geometry.tth(ay, ax)[0]
            return chi, tth
        else:
            try:
                tthRad = unitutils.tthToRad(data[0],
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

    def __removeMarker(self, marker):
        self.__markerModel.remove(marker)

    def __createPixelMarker(self, pos):
        pos = self.__plot.pixelToData(pos.x(), pos.y())
        if self.__pixelBasedPlot:
            pixel = pos[1], pos[0]
        else:
            pixel = self.__invertGeometry(pos[0], pos[1], True)

            # Check if the result is accurate
            # TODO: This could be avoided by checking it inside invertGeometry
            chiRad, tthRad = self.dataToChiTth(pos)
            if tthRad is not None and chiRad is not None:
                ax, ay = numpy.array([pixel[1]]), numpy.array([pixel[0]])
                tth = self.__geometry.tth(ay, ax)[0]
                chi = self.__geometry.chi(ay, ax)[0]

                error = numpy.sqrt((tthRad - tth) ** 2 + (chiRad - chi) ** 2)
                if error > 0.05:
                    _logger.error("The identified pixel is far from the requested chi/tth. Marker ignored.")
                    return

        name = self.__findUnusedMarkerName()
        marker = MarkerModel.PixelMarker(name, pixel[1], pixel[0])
        self.__markerModel.add(marker)

    def __createGeometryMarker(self, pos):
        pos = self.__plot.pixelToData(pos.x(), pos.y())
        chiRad, tthRad = self.dataToChiTth(pos)
        name = self.__findUnusedMarkerName()
        marker = MarkerModel.PhysicalMarker(name, chiRad, tthRad)

        if self.__pixelBasedPlot:
            marker.setPixelPosition(pos[0], pos[1])
        else:
            if self.__geometry is not None:
                pixel = self.__invertGeometry(pos[0], pos[1], True)
                marker.setPixelPosition(pixel[1], pixel[0])
            else:
                marker.removePixelPosition()

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

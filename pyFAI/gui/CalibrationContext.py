# coding: utf-8
# /*##########################################################################
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
# ############################################################################*/
"""Context shared through all the application"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "15/01/2019"

import logging
import numpy

from silx.gui import qt
from silx.gui.dialog.ColormapDialog import ColormapDialog
from silx.gui.colors import Colormap

from .model.CalibrationModel import CalibrationModel
from .model.DataModel import DataModel
from .utils import eventutils
from .utils import units
from .utils import colorutils
from .ApplicationContext import ApplicationContext


_logger = logging.getLogger(__name__)


class CalibrationContext(ApplicationContext):

    __instance = None

    @staticmethod
    def _releaseSingleton():
        ApplicationContext._releaseSingleton()
        CalibrationContext.__instance = None

    @staticmethod
    def instance():
        """
        :rtype: CalibrationContext
        """
        assert(CalibrationContext.__instance is not None)
        return CalibrationContext.__instance

    def __init__(self, settings=None):
        super(CalibrationContext, self).__init__(settings=settings)
        assert(CalibrationContext.__instance is None)
        self.__defaultColormapDialog = None
        CalibrationContext.__instance = self
        self.__calibrationModel = None
        self.__rawColormap = Colormap("inferno", normalization=Colormap.LOGARITHM)
        self.__settings = settings
        self.__angleUnit = DataModel()
        self.__angleUnit.setValue(units.Unit.RADIAN)
        self.__lengthUnit = DataModel()
        self.__lengthUnit.setValue(units.Unit.METER)
        self.__wavelengthUnit = DataModel()
        self.__wavelengthUnit.setValue(units.Unit.ANGSTROM)
        self.__markerColors = {}

        self.sigStyleChanged = self.__rawColormap.sigChanged

    def __restoreColormap(self, groupName, colormap):
        settings = self.__settings
        if settings is None:
            _logger.debug("Settings not set")
            return
        settings.beginGroup(groupName)
        byteArray = settings.value("default", None)
        if byteArray is not None:
            try:
                colormap.restoreState(byteArray)
            except Exception:
                _logger.debug("Backtrace", exc_info=True)
        settings.endGroup()

    def __saveColormap(self, groupName, colormap):
        settings = self.__settings
        if settings is None:
            _logger.debug("Settings not set")
            return

        if colormap is None:
            return
        settings.beginGroup(groupName)
        settings.setValue("default", colormap.saveState())
        settings.endGroup()

    def __restoreUnit(self, unitModel, settings, fieldName, defaultUnit):
        unitName = settings.value(fieldName, None)
        unit = None
        if unitName is not None:
            try:
                unit = getattr(units.Unit, unitName)
                if not isinstance(unit, units.Unit):
                    unit = None
            except Exception:
                _logger.error("Unit name '%s' is not an unit", unitName)
                unit = None
        if unit is None:
            unit = defaultUnit
        unitModel.setValue(unit)

    def restoreSettings(self):
        """Restore the settings of all the application"""
        settings = self.__settings
        if settings is None:
            _logger.debug("Settings not set")
            return
        self.__restoreColormap("raw-colormap", self.__rawColormap)

        settings.beginGroup("units")
        self.__restoreUnit(self.__angleUnit, settings, "angle-unit", units.Unit.RADIAN)
        self.__restoreUnit(self.__lengthUnit, settings, "length-unit", units.Unit.METER)
        self.__restoreUnit(self.__wavelengthUnit, settings, "wavelength-unit", units.Unit.ANGSTROM)
        settings.endGroup()

    def saveSettings(self):
        """Save the settings of all the application"""
        settings = self.__settings
        if settings is None:
            _logger.debug("Settings not set")
            return
        self.__saveColormap("raw-colormap", self.__rawColormap)

        settings.beginGroup("units")
        settings.setValue("angle-unit", self.__angleUnit.value().name)
        settings.setValue("length-unit", self.__lengthUnit.value().name)
        settings.setValue("wavelength-unit", self.__wavelengthUnit.value().name)
        settings.endGroup()

        # Synchronize the file storage
        super(CalibrationContext, self).saveSettings()

    def getCalibrationModel(self):
        if self.__calibrationModel is None:
            self.__calibrationModel = CalibrationModel()
        return self.__calibrationModel

    def getRawColormap(self):
        """Returns the user preference colormap used to display raw data
        images

        :rtype: Colormap
        """
        return self.__rawColormap

    def getColormapDialog(self):
        """Returns a shared color dialog.

        :rtype: ColorDialog
        """
        if self.__defaultColormapDialog is None:
            parent = self.parent()
            if parent is None:
                return None
            dialog = ColormapDialog(parent=parent)
            dialog.setModal(False)

            def dialogShown():
                dialog = self.__defaultColormapDialog
                self.restoreWindowLocationSettings("colormap-dialog", dialog)
                # Sync with the raw image
                data = self.getCalibrationModel().experimentSettingsModel().image().value()
                dialog.setData(data)

            def dialogHidden():
                dialog = self.__defaultColormapDialog
                self.saveWindowLocationSettings("colormap-dialog", dialog)

            eventutils.createShowSignal(dialog)
            eventutils.createHideSignal(dialog)

            dialog.sigShown.connect(dialogShown)
            dialog.sigHidden.connect(dialogHidden)
            self.__defaultColormapDialog = dialog

        return self.__defaultColormapDialog

    def getAngleUnit(self):
        return self.__angleUnit

    def getLengthUnit(self):
        return self.__lengthUnit

    def getWavelengthUnit(self):
        return self.__wavelengthUnit

    def getMarkerColor(self, index, mode="qt"):
        colors = self.markerColorList()
        color = colors[index % len(colors)]
        if mode == "html":
            return "#%02X%02X%02X" % (color.red(), color.green(), color.blue())
        elif mode == "numpy":
            return numpy.array([color.redF(), color.greenF(), color.blueF()])
        elif mode == "qt":
            return color
        else:
            raise ValueError("Mode '%s' not expected" % mode)

    def getLabelColor(self):
        """Returns the Qt color used to display text label

        :rtype: qt.QColor
        """
        markers = self.markerColorList()
        color = markers[0]
        return color

    def getMaskedColor(self):
        """Returns the Qt color used to display masked pixels

        :rtype: qt.QColor
        """
        return qt.QColor(255, 0, 255)

    def getBackgroundColor(self):
        """Returns the Qt color used to display part of the diagram outside of
        the data.

        :rtype: qt.QColor
        """
        colors = self.__rawColormap.getNColors()
        color = colors[0]
        color = 255 - ((255 - color) // 2)
        color = int(color.mean())
        color = [color, color, color]
        return qt.QColor(color[0], color[1], color[2])

    def getHtmlMarkerColor(self, index):
        colors = self.markerColorList()
        color = colors[index % len(colors)]
        "#%02X%02X%02X" % (color.red(), color.green(), color.blue())

    def markerColorList(self):
        colormap = self.getRawColormap()
        name = colormap['name']
        if name not in self.__markerColors:
            colors = self.createMarkerColors()
            self.__markerColors[name] = colors
        else:
            colors = self.__markerColors[name]
        return colors

    def createMarkerColors(self):
        colormap = self.getRawColormap()
        return colorutils.getFreeColorRange(colormap)

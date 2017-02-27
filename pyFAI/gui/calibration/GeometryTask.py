# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
__date__ = "27/02/2017"

import logging
import numpy

from pyFAI.gui import qt
from pyFAI.gui import icons
import pyFAI.utils
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask
from pyFAI.gui.calibration.RingCalibration import RingCalibration

import silx.gui.plot
from silx.gui.plot.PlotTools import PositionInfo
from silx.gui.plot import PlotActions
from . import utils

_logger = logging.getLogger(__name__)

_iconVariableFixed = None
_iconVariableConstrained = None
_iconVariableConstrainedOut = None


class FitParamView(qt.QObject):

    def __init__(self, parent, label, unit):
        qt.QObject.__init__(self, parent=parent)

        self.__label = qt.QLabel(parent)
        self.__label.setText(label)
        self.__lineEdit = qt.QLineEdit(parent)
        validator = qt.QDoubleValidator(self.__lineEdit)
        self.__lineEdit.setValidator(validator)
        self.__lineEdit.setAlignment(qt.Qt.AlignRight)
        self.__unit = qt.QLabel(parent)
        self.__unit.setText(unit)
        self.__contains = qt.QToolButton(parent)
        self.__contains.setAutoRaise(True)
        self.__model = None
        self.__constraintsModel = None

        global _iconVariableFixed, _iconVariableConstrained, _iconVariableConstrainedOut
        if _iconVariableFixed is None:
            _iconVariableFixed = icons.getQIcon("variable-fixed")
        if _iconVariableConstrained is None:
            _iconVariableConstrained = icons.getQIcon("variable-constrained")
        if _iconVariableConstrainedOut is None:
            _iconVariableConstrainedOut = icons.getQIcon("variable-constrained-out")

    def model(self):
        return self.__model

    def setModel(self, model):
        if self.__model is not None:
            self.__model.changed.disconnect(self.__modelChanged)
        self.__model = model
        if self.__model is not None:
            self.__model.changed.connect(self.__modelChanged)
            self.__modelChanged()

    def setConstraintsModel(self, model):
        if self.__constraintsModel is not None:
            self.__constraintsModel.changed.disconnect(self.__constraintsModelChanged)
        self.__constraintsModel = model
        if self.__constraintsModel is not None:
            self.__constraintsModel.changed.connect(self.__constraintsModelChanged)
            self.__constraintsModelChanged()

    def __modelChanged(self):
        old = self.__lineEdit.blockSignals(True)
        if self.__model is None:
            self.__lineEdit.setText("")
        else:
            value = self.__model.value()
            if value is None:
                value = ""
            self.__lineEdit.setText(str(value))
        self.__lineEdit.blockSignals(old)

    def __constraintsModelChanged(self):
        contraint = self.__constraintsModel
        if contraint.isFixed():
            icon = _iconVariableFixed
        else:
            icon = _iconVariableConstrained
        self.__contains.setIcon(icon)

    def widgets(self):
        return [self.__label, self.__lineEdit, self.__unit, self.__contains]


class _RingPlot(silx.gui.plot.PlotWidget):

    def __init__(self, parent=None):
        silx.gui.plot.PlotWidget.__init__(self, parent=parent)
        self.__markerColors = {}
        self.__ringLegends = []

    def markerColorList(self):
        colormap = self.getDefaultColormap()

        name = colormap['name']
        if name not in self.__markerColors:
            colors = self.createMarkerColors()
            self.__markerColors[name] = colors
        else:
            colors = self.__markerColors[name]
        return colors

    def createMarkerColors(self):
        colormap = self.getDefaultColormap()
        return utils.getFreeColorRange(colormap)

    def setRings(self, rings):
        for legend in self.__ringLegends:
            self.removeCurve(legend)
        self.__ringLegends = []

        colors = self.markerColorList()
        for ringId, polyline in enumerate(rings):
            color = colors[ringId % len(colors)]
            numpyColor = numpy.array([color.redF(), color.greenF(), color.blueF()])

            color = colors
            for lineId, line in enumerate(polyline):
                y, x = line[:, 0], line[:, 1]
                legend = "ring-%i-%i" % (ringId, lineId)
                self.addCurve(
                    x=x,
                    y=y,
                    legend=legend,
                    resetzoom=False,
                    color=numpyColor,
                    linewidth=3,
                    linestyle=":")
            self.__ringLegends.append(legend)


class GeometryTask(AbstractCalibrationTask):

    def __init__(self):
        super(GeometryTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-geometry.ui"), self)
        self.__dialogState = None

        self.__plot = self.__createPlot()

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.addWidget(self.__plot)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)

        layout = qt.QGridLayout(self._settings)
        self.__wavelength = FitParamView(self, "Wavelength:", u"Å")
        self.addParameterToLayout(layout, self.__wavelength)

        layout = qt.QGridLayout(self._geometry)
        self.__distance = FitParamView(self, "Distance:", "m")
        self.__poni1 = FitParamView(self, "PONI1:", u"m")
        self.__poni2 = FitParamView(self, "PONI2:", u"m")
        self.__rotation1 = FitParamView(self, "Rotation 1:", u"rad")
        self.__rotation2 = FitParamView(self, "Rotation 2:", u"rad")
        self.__rotation3 = FitParamView(self, "Rotation 3:", u"rad")
        self.addParameterToLayout(layout, self.__distance)
        self.addParameterToLayout(layout, self.__poni1)
        self.addParameterToLayout(layout, self.__poni2)
        self.addParameterToLayout(layout, self.__rotation1)
        self.addParameterToLayout(layout, self.__rotation2)
        self.addParameterToLayout(layout, self.__rotation3)

        self._fitButton.clicked.connect(self.__fitGeometry)
        self._resetButton.clicked.connect(self.__resetGeometry)
        self.__calibration = None

    def addParameterToLayout(self, layout, param):
        # an empty grid returns 1
        row = layout.rowCount()
        widgets = param.widgets()
        for i, widget in enumerate(widgets):
            layout.addWidget(widget, row, i)

    def __createPlot(self):
        plot = _RingPlot(parent=self._imageHolder)
        plot.setKeepDataAspectRatio(True)
        toolBar = self.__createPlotToolBar(plot)
        plot.addToolBar(toolBar)
        statusBar = self.__createPlotStatusBar(plot)
        plot.setStatusBar(statusBar)

        if isinstance(plot._backend, silx.gui.plot.BackendMatplotlib.BackendMatplotlib):
            # hide axes and viewbox rect
            plot._backend.ax.set_axis_off()
            plot._backend.ax2.set_axis_off()
            # remove external margins
            plot._backend.ax.set_position([0, 0, 1, 1])
            plot._backend.ax2.set_position([0, 0, 1, 1])

        colormap = {
            'name': "inferno",
            'normalization': 'log',
            'autoscale': True,
        }
        plot.setDefaultColormap(colormap)

        return plot

    def __createPlotToolBar(self, plot):
        toolBar = qt.QToolBar("Plot tools", plot)

        toolBar.addAction(PlotActions.ResetZoomAction(plot, toolBar))
        toolBar.addAction(PlotActions.ZoomInAction(plot, toolBar))
        toolBar.addAction(PlotActions.ZoomOutAction(plot, toolBar))
        toolBar.addSeparator()
        toolBar.addAction(PlotActions.ColormapAction(plot, toolBar))
        toolBar.addAction(PlotActions.PixelIntensitiesHistoAction(plot, toolBar))
        toolBar.addSeparator()
        toolBar.addAction(PlotActions.CopyAction(plot, toolBar))
        toolBar.addAction(PlotActions.SaveAction(plot, toolBar))
        toolBar.addAction(PlotActions.PrintAction(plot, toolBar))

        return toolBar

    def __createPlotStatusBar(self, plot):

        converters = [
            ('X', lambda x, y: x),
            ('Y', lambda x, y: y),
            ('Value', self.__getImageValue)]

        hbox = qt.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        info = PositionInfo(plot=plot, converters=converters)
        info.autoSnapToActiveCurve = True
        statusBar = qt.QStatusBar(plot)
        statusBar.setSizeGripEnabled(False)
        statusBar.addWidget(info)
        return statusBar

    def __createCalibration(self):
        image = self.model().experimentSettingsModel().image().value()
        calibrant = self.model().experimentSettingsModel().calibrantModel().calibrant()
        detector = self.model().experimentSettingsModel().detectorModel().detector()
        wavelength = self.model().experimentSettingsModel().wavelength().value()
        wavelength = wavelength / 1e10

        peaks = []
        for peakModel in self.model().peakSelectionModel():
            ringNumber = peakModel.ringNumber()
            for coord in peakModel.coords():
                peaks.append([coord[0], coord[1], ringNumber - 1])
        peaks = numpy.array(peaks)

        calibration = RingCalibration(image,
                                      calibrant,
                                      detector,
                                      wavelength,
                                      peaks=peaks,
                                      method="massif")

        return calibration

    def __getCalibration(self):
        if self.__calibration is None:
            self.__calibration = self.__createCalibration()
        return self.__calibration

    def __resetGeometry(self):
        calibration = self.__getCalibration()

        peaks = []
        for peakModel in self.model().peakSelectionModel():
            ringNumber = peakModel.ringNumber()
            for coord in peakModel.coords():
                peaks.append([coord[0], coord[1], ringNumber - 1])
        peaks = numpy.array(peaks)

        calibration.init(peaks, "massif")
        self.__updateDisplay()

    def __fitGeometry(self):
        calibration = self.__getCalibration()
        calibration.refine()
        self.__updateDisplay()

    def __updateDisplay(self):
        calibration = self.__getCalibration()

        rings = calibration.getRings()
        self.__plot.setRings(rings)

        center = calibration.getBeamCenter()
        if center is None:
            self.__plot.removeMarker(legend="center")
        else:
            color = self.__plot.markerColorList()[0]
            numpyColor = "#%02X%02X%02X" % (color.red(), color.green(), color.blue())
            self.__plot.addMarker(
                y=center[0],
                x=center[1],
                legend="center",
                color=numpyColor,
                symbol="+")

        model = self.model().fittedGeometry()
        calibration.toGeometryModel(model)

    def __getImageValue(self, x, y):
        """Get value of top most image at position (x, y).

        :param float x: X position in plot coordinates
        :param float y: Y position in plot coordinates
        :return: The value at that point or 'n/a'
        """
        value = 'n/a'

        image = self.__plot.getImage("image")
        if image is None:
            return value
        data, params = image[0], image[4]
        ox, oy = params['origin']
        sx, sy = params['scale']
        row, col = (y - oy) / sy, (x - ox) / sx
        if row >= 0 and col >= 0:
            # Test positive before cast otherwise issue with int(-0.5) = 0
            row, col = int(row), int(col)
            if (row < data.shape[0] and col < data.shape[1]):
                value = data[row, col]
        return value

    def _updateModel(self, model):
        settings = model.experimentSettingsModel()
        settings.image().changed.connect(self.__imageUpdated)

        geometry = model.fittedGeometry()

        self.__distance.setModel(geometry.distance())
        self.__wavelength.setModel(geometry.wavelength())
        self.__poni1.setModel(geometry.poni1())
        self.__poni2.setModel(geometry.poni2())
        self.__rotation1.setModel(geometry.rotation1())
        self.__rotation2.setModel(geometry.rotation2())
        self.__rotation3.setModel(geometry.rotation3())

        constrains = model.geometryConstraintsModel()
        self.__distance.setConstraintsModel(constrains.distance())
        self.__wavelength.setConstraintsModel(constrains.wavelength())
        self.__poni1.setConstraintsModel(constrains.poni1())
        self.__poni2.setConstraintsModel(constrains.poni2())
        self.__rotation1.setConstraintsModel(constrains.rotation1())
        self.__rotation2.setConstraintsModel(constrains.rotation2())
        self.__rotation3.setConstraintsModel(constrains.rotation3())

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().image().value()
        self.__plot.addImage(image, legend="image")
        if image is not None:
            self.__plot.setGraphXLimits(0, image.shape[0])
            self.__plot.setGraphYLimits(0, image.shape[1])

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
__date__ = "13/06/2017"

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
from . import validators

_logger = logging.getLogger(__name__)

_iconVariableFixed = None
_iconVariableConstrained = None
_iconVariableConstrainedOut = None


class FitParamView(qt.QObject):

    def __init__(self, parent, label, unit):
        qt.QObject.__init__(self, parent=parent)
        validator = validators.DoubleValidator(self)
        self.__label = qt.QLabel(parent)
        self.__label.setText(label)
        self.__lineEdit = qt.QLineEdit(parent)
        self.__lineEdit.setValidator(validator)
        self.__lineEdit.setAlignment(qt.Qt.AlignRight)
        self.__lineEdit.editingFinished.connect(self.__lineEditChanged)
        self.__unit = qt.QLabel(parent)
        self.__unit.setText(unit)
        self.__constraints = qt.QToolButton(parent)
        self.__constraints.setAutoRaise(True)
        self.__constraints.clicked.connect(self.__constraintsClicked)
        self.__model = None
        self.__wavelengthInvalidated = False
        self.__constraintsModel = None

        global _iconVariableFixed, _iconVariableConstrained, _iconVariableConstrainedOut
        if _iconVariableFixed is None:
            _iconVariableFixed = icons.getQIcon("variable-fixed")
        if _iconVariableConstrained is None:
            _iconVariableConstrained = icons.getQIcon("variable-constrained")
        if _iconVariableConstrainedOut is None:
            _iconVariableConstrainedOut = icons.getQIcon("variable-constrained-out")

    def __lineEditChanged(self):
        value = self.__lineEdit.text()
        try:
            value = float(value)
            self.__model.setValue(value)
        except ValueError:
            pass

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
        constraint = self.__constraintsModel
        if constraint.isFixed():
            icon = _iconVariableFixed
        else:
            icon = _iconVariableConstrained
        self.__constraints.setIcon(icon)

    def __constraintsClicked(self):
        constraint = self.__constraintsModel
        # FIXME implement popup with range
        constraint.setFixed(not constraint.isFixed())

    def widgets(self):
        return [self.__label, self.__lineEdit, self.__unit, self.__constraints]


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

    def setRings(self, rings, mask):
        for legend in self.__ringLegends:
            self.removeCurve(legend)
        self.__ringLegends = []

        colors = self.markerColorList()
        for ringId, polyline in enumerate(rings):
            color = colors[ringId % len(colors)]
            numpyColor = numpy.array([color.redF(), color.greenF(), color.blueF()])

            deltas = [(0.0, 0.0), (0.99, 0.0), (0.0, 0.99), (0.99, 0.99)]

            def filter_coord_over_mask(coord):
                for dx, dy in deltas:
                    if mask[int(coord[0] + dx), int(coord[1] + dy)] != 0:
                        return float("nan"), float("nan")
                return coord

            for lineId, line in enumerate(polyline):
                if mask is not None:
                    line = map(filter_coord_over_mask, line)
                    line = list(line)
                    line = numpy.array(line)
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
        self.initNextStep()
        self.widgetShow.connect(self.__widgetShow)

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
        self._fitButton.setDisabledWhenWaiting(True)
        self._resetButton.clicked.connect(self.__resetGeometry)
        self.__calibration = None
        self.__peaksInvalidated = False
        self.__fitting = False

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

        # FIXME Fix using silx 0.5
        if "BackendMatplotlib" in plot._backend.__class__.__name__:
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
        mask = self.model().experimentSettingsModel().mask().value()
        calibrant = self.model().experimentSettingsModel().calibrantModel().calibrant()
        detector = self.model().experimentSettingsModel().detector()
        wavelength = self.model().experimentSettingsModel().wavelength().value()
        wavelength = wavelength / 1e10

        peaks = []
        for peakModel in self.model().peakSelectionModel():
            ringNumber = peakModel.ringNumber()
            for coord in peakModel.coords():
                peaks.append([coord[0], coord[1], ringNumber - 1])
        peaks = numpy.array(peaks)

        calibration = RingCalibration(image,
                                      mask,
                                      calibrant,
                                      detector,
                                      wavelength,
                                      peaks=peaks,
                                      method="massif")

        return calibration

    def __invalidateWavelength(self):
        self.__wavelengthInvalidated = True

    def __getCalibration(self):
        if self.__calibration is None:
            self.__calibration = self.__createCalibration()

        # It have to be updated only if it changes
        image = self.model().experimentSettingsModel().image().value()
        calibrant = self.model().experimentSettingsModel().calibrantModel().calibrant()
        detector = self.model().experimentSettingsModel().detector()
        mask = self.model().experimentSettingsModel().mask().value()
        if self.__wavelengthInvalidated:
            self.__wavelengthInvalidated = False
            wavelength = self.model().experimentSettingsModel().wavelength().value()
            wavelength = wavelength / 1e10
        else:
            wavelength = None
        self.__calibration.update(image, mask, calibrant, detector, wavelength)

        return self.__calibration

    def __invalidatePeakSelection(self):
        self.__peaksInvalidated = True

    def __initGeometryFromPeaks(self):
        if self.__peaksInvalidated:
            # recompute the geometry from the peaks
            # FIXME numpy array can be allocated first
            peaks = []
            for peakModel in self.model().peakSelectionModel():
                ringNumber = peakModel.ringNumber()
                for coord in peakModel.coords():
                    peaks.append([coord[0], coord[1], ringNumber - 1])
            peaks = numpy.array(peaks)

            calibration = self.__getCalibration()
            calibration.init(peaks, "massif")
            calibration.toGeometryModel(self.model().peakGeometry())
            self.__peaksInvalidated = False

        self.model().fittedGeometry().setFrom(self.model().peakGeometry())

    def __resetGeometry(self):
        calibration = self.__getCalibration()
        self.__initGeometryFromPeaks()
        # write result to the fitted model
        model = self.model().fittedGeometry()
        calibration.toGeometryModel(model)
        self.__formatResidual()

    def __fitGeometry(self):
        self.__fitting = True
        self._fitButton.setWaiting(True)
        calibration = self.__getCalibration()
        if self.__peaksInvalidated:
            self.__initGeometryFromPeaks()
        else:
            calibration.fromGeometryModel(self.model().fittedGeometry())
        calibration.fromGeometryConstriansModel(self.model().geometryConstraintsModel())
        calibration.refine()
        # write result to the fitted model
        model = self.model().fittedGeometry()
        calibration.toGeometryModel(model)
        self._fitButton.setWaiting(False)
        self.__fitting = False

    def __formatResidual(self):
        calibration = self.__getCalibration()
        previousResidual = calibration.getPreviousResidual()
        residual = calibration.getResidual()
        text = '%.6e' % residual
        if previousResidual is not None:
            if residual == previousResidual:
                diff = "(no changes)"
            else:
                diff = '(%+.2e)' % (residual - previousResidual)
                if residual < previousResidual:
                    diff = '<font color="green">%s</font>' % diff
                else:
                    diff = '<font color="red">%s</font>' % diff
            text = '%s %s' % (text, diff)
        self._currentResidual.setText(text)

    def __geometryUpdated(self):
        calibration = self.__getCalibration()
        model = self.model().fittedGeometry()
        if model.isValid():
            resetResidual = self.__fitting is not True
            calibration.fromGeometryModel(model, resetResidual=resetResidual)
            self.__updateDisplay()
            self.__formatResidual()

    def __updateDisplay(self):
        calibration = self.__getCalibration()

        mask = self.model().experimentSettingsModel().mask().value()
        rings = calibration.getRings()
        self.__plot.setRings(rings, mask)

        center = calibration.getBeamCenter()
        if center is None:
            self.__plot.removeMarker(legend="center")
        else:
            color = self.__plot.markerColorList()[0]
            htmlColor = "#%02X%02X%02X" % (color.red(), color.green(), color.blue())
            self.__plot.addMarker(
                text="Beam",
                y=center[0],
                x=center[1],
                legend="center",
                color=htmlColor,
                symbol="+")

        poni = calibration.getPoni()
        if poni is None:
            self.__plot.removeMarker(legend="poni")
        else:
            color = self.__plot.markerColorList()[0]
            htmlColor = "#%02X%02X%02X" % (color.red(), color.green(), color.blue())
            self.__plot.addMarker(
                text="PONI",
                y=poni[0],
                x=poni[1],
                legend="poni",
                color=htmlColor,
                symbol="+")

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
        data = image.getData()
        ox, oy = image.getOrigin()
        sx, sy = image.getScale()
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
        settings.wavelength().changed.connect(self.__invalidateWavelength)

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

        model.fittedGeometry().changed.connect(self.__geometryUpdated)
        model.peakSelectionModel().changed.connect(self.__invalidatePeakSelection)

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().image().value()
        self.__plot.addImage(image, legend="image")
        if image is not None:
            self.__plot.setGraphXLimits(0, image.shape[0])
            self.__plot.setGraphYLimits(0, image.shape[1])
            self.__plot.resetZoom()

    def __widgetShow(self):
        if self.__peaksInvalidated:
            self.__initGeometryFromPeaks()
            self.__formatResidual()

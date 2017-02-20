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
__date__ = "20/02/2017"

import logging
import numpy
import functools
from pyFAI.gui import qt
from pyFAI.gui import icons
import pyFAI.utils
import pyFAI.massif
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask
from pyFAI.gui.calibration.model.PeakModel import PeakModel
import silx.gui.plot
from silx.gui.plot.PlotTools import PositionInfo
from silx.gui.plot import PlotActions

_logger = logging.getLogger(__name__)


class _DummyStdOut(object):

    def write(self, text):
        pass


class _PeakSelectionTableModel(qt.QAbstractTableModel):

    def __init__(self, parent, peakSelectionModel):
        super(_PeakSelectionTableModel, self).__init__(parent=parent)
        self.__peakSelectionModel = peakSelectionModel
        peakSelectionModel.changed.connect(self.__invalidateModel)
        self.__callbacks = []
        self.__invalidateModel()

    def __invalidateModel(self):
        self.beginResetModel()
        for callback in self.__callbacks:
            target, method = callback
            target.changed.disconnect(method)
        self.__callbacks = []
        for index, item in enumerate(self.__peakSelectionModel):
            callback = functools.partial(self.__invalidateItem, index)
            item.changed.connect(callback)
            self.__callbacks.append((item, callback))
        self.endResetModel()

    def __invalidateItem(self, index):
        index1 = self.index(index, 0, qt.QModelIndex())
        index2 = self.index(index, self.columnCount() - 1, qt.QModelIndex())
        self.dataChanged(index1, index2)

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        if orientation != qt.Qt.Horizontal:
            return None
        if role != qt.Qt.DisplayRole:
            return super(_PeakSelectionTableModel, self).headerData(section, orientation, role)
        if section == 0:
            return "Name"
        if section == 1:
            return "Number of points"
        if section == 2:
            return "Ring number"
        return None

    def flags(self, index):
        return (#qt.Qt.ItemIsEditable |
                qt.Qt.ItemIsEnabled |
                qt.Qt.ItemIsSelectable)

    def rowCount(self, parent=qt.QModelIndex()):
        return len(self.__peakSelectionModel)

    def columnCount(self, parent=qt.QModelIndex()):
        return 4

    def data(self, index=qt.QModelIndex(), role=qt.Qt.DisplayRole):
        peakModel = self.__peakSelectionModel[index.row()]
        column = index.column()
        if role == qt.Qt.DecorationRole:
            if column == 0:
                color = peakModel.color()
                pixmap = qt.QPixmap(16, 16)
                pixmap.fill(color)
                icon = qt.QIcon(pixmap)
                return icon
            else:
                return None
        if role == qt.Qt.DisplayRole or role == qt.Qt.EditRole:
            if column == 0:
                return peakModel.name()
            if column == 1:
                return len(peakModel.coords())
            if column == 2:
                return peakModel.ringNumber()
        return None


class PeakPickingTask(AbstractCalibrationTask):

    def __init__(self):
        super(PeakPickingTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-peakpicking.ui"), self)
        self.__dialogState = None

        layout = qt.QVBoxLayout(self._imageHolder)
        self.__plot = self.__createPlot()
        layout.addWidget(self.__plot)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)

        self._ringSelectionMode.setIcon(icons.getQIcon("search-ring"))
        self._peakSelectionMode.setIcon(icons.getQIcon("search-peak"))
        self.__plot.sigPlotSignal.connect(self.__onPlotEvent)

    def __createPlot(self):
        plot = silx.gui.plot.PlotWidget(parent=self._imageHolder)
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

    def __onPlotEvent(self, event):
        if event["event"] == "imageClicked":
            x, y, button = event["col"], event["row"], event["button"]
            if button == "left":
                image = self.model().experimentSettingsModel().image().value()
                massif = pyFAI.massif.Massif(image)
                points = massif.find_peaks([y, x], stdout=_DummyStdOut())
                if len(points) > 0:
                    peakModel = self.__createNewPeak(points)
                    self.model().peakSelectionModel().append(peakModel)

    def __createNewPeak(self, points):
        selection = self.model().peakSelectionModel()

        # FIXME support more than 'z' names
        name = chr(len(selection) + ord('a'))
        # FIXME improve color list
        colors = [qt.QColor(255, 0, 0), qt.QColor(0, 255, 0), qt.QColor(0, 0, 255)]
        color = colors[(len(selection)) % len(colors)]
        numpyColor = numpy.array([color.redF(), color.greenF(), color.blueF()])

        peakModel = PeakModel(self.model().peakSelectionModel())
        peakModel.setName(name)
        peakModel.setColor(color)
        peakModel.setCoords(points)

        y, x = points[0]
        self.__plot.addMarker(x=x, y=y,
                              legend="marker" + name,
                              text=name)
        y = map(lambda p: p[0], points)
        x = map(lambda p: p[1], points)
        self.__plot.addCurve(x=x, y=y,
                             legend="coord" + name,
                             linestyle=' ',
                             symbol='o',
                             color=numpyColor,
                             resetzoom=False)

        return peakModel

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
        settings.mask().changed.connect(self.__maskUpdated)
        self.__initPeakSelectionView(model)

    def __initPeakSelectionView(self, model):
        tableModel = _PeakSelectionTableModel(self, model.peakSelectionModel())
        self._peakSelection.setModel(tableModel)

        header = self._peakSelection.horizontalHeader()
        if qt.qVersion() < "5.0":
            setResizeMode = header.setResizeMode
        else:
            setResizeMode = header.setSectionResizeMode
        setResizeMode(0, qt.QHeaderView.Stretch)
        setResizeMode(1, qt.QHeaderView.ResizeToContents)
        setResizeMode(2, qt.QHeaderView.ResizeToContents)
        setResizeMode(3, qt.QHeaderView.ResizeToContents)

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().image().value()
        self.__plot.addImage(image, legend="image", selectable=True)
        if image is not None:
            self.__plot.setGraphXLimits(0, image.shape[0])
            self.__plot.setGraphYLimits(0, image.shape[1])

    def __maskUpdated(self):
        mask = self.model().experimentSettingsModel().mask().value()
        self.__maskPanel.setSelectionMask(mask)

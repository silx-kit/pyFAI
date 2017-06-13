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
import functools
from pyFAI.gui import qt
from pyFAI.gui import icons
import pyFAI.utils
import pyFAI.massif
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask
from pyFAI.gui.calibration.model.PeakModel import PeakModel
from pyFAI.gui.calibration.RingExtractor import RingExtractor
from collections import OrderedDict
import pyFAI.control_points

import silx.gui.plot
import os
from silx.gui.plot.PlotTools import PositionInfo
from silx.gui.plot import PlotActions
from . import utils

_logger = logging.getLogger(__name__)


class _DummyStdOut(object):

    def write(self, text):
        pass


class _PeakSelectionUndoCommand(qt.QUndoCommand):

    def __init__(self, parent, model, oldState, newState):
        super(_PeakSelectionUndoCommand, self).__init__(parent=parent)
        self.__peakPickingModel = model
        self.__oldState = list(oldState)
        self.__newState = list(newState)
        self.__redoInhibited = False

    def setRedoInhibited(self, isInhibited):
        """Allow to avoid to push the command into the QUndoStack without
        calling redo."""
        self.__redoInhibited = isInhibited

    def undo(self):
        peakPickingModel = self.__peakPickingModel
        peakPickingModel.clear()
        for peakModel in self.__oldState:
            peakPickingModel.append(peakModel)

    def redo(self):
        if self.__redoInhibited:
            return
        peakPickingModel = self.__peakPickingModel
        peakPickingModel.clear()
        for peakModel in self.__newState:
            peakPickingModel.append(peakModel)


class _PeakSelectionTableView(qt.QTableView):

    def __init__(self, parent):
        super(_PeakSelectionTableView, self).__init__(parent=parent)
        ringDelegate = _SpinBoxItemDelegate(self)
        toolDelegate = _PeakToolItemDelegate(self)
        self.setItemDelegateForColumn(2, ringDelegate)
        self.setItemDelegateForColumn(3, toolDelegate)

        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setVerticalScrollMode(qt.QAbstractItemView.ScrollPerPixel)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setFrameShape(qt.QFrame.NoFrame)

        self.horizontalHeader().setHighlightSections(False)
        self.verticalHeader().setVisible(False)
        self.setStyleSheet("QTableView {background: transparent;}")

    def setModel(self, model):
        if self.model() is not None:
            m = self.model()
            m.rowsInserted.disconnect(self.__onRowInserted)
            m.rowsRemoved.disconnect(self.__onRowRemoved)
            m.modelReset.disconnect(self.__openPersistantViewOnModelReset)

        super(_PeakSelectionTableView, self).setModel(model)

        if self.model() is not None:
            m = self.model()
            m.rowsInserted.connect(self.__onRowInserted)
            m.rowsRemoved.connect(self.__onRowRemoved)
            m.modelReset.connect(self.__openPersistantViewOnModelReset)
            self.__openPersistantViewOnModelReset()
            # it is not possible to set column constraints while there is no model
            self.__updateColumnConstraints()

    def sizeHint(self):
        """Size hint while grow according to the content of the view"""
        rowCount = self.model().rowCount()
        size = qt.QTableView.sizeHint(self)
        if rowCount <= 0:
            return size
        height = self.horizontalHeader().size().height()
        height = height + self.rowHeight(0) * rowCount
        if height < size.height():
            return size
        size = qt.QSize(size.width(), height)
        return size

    def __updateColumnConstraints(self):
        header = self.horizontalHeader()
        if qt.qVersion() < "5.0":
            setResizeMode = header.setResizeMode
        else:
            setResizeMode = header.setSectionResizeMode
        setResizeMode(0, qt.QHeaderView.Stretch)
        setResizeMode(1, qt.QHeaderView.ResizeToContents)
        setResizeMode(2, qt.QHeaderView.ResizeToContents)
        setResizeMode(3, qt.QHeaderView.Fixed)

    def __onRowRemoved(self, parent, start, end):
        self.updateGeometry()

    def __onRowInserted(self, parent, start, end):
        self.__openPersistantViewOnRowInserted(parent, start, end)
        self.updateGeometry()
        # It have to be done only on the 3, else the layout is wrong
        self.resizeColumnToContents(3)

    def __openPersistantViewOnRowInserted(self, parent, start, end):
        model = self.model()
        for row in range(start, end):
            index = model.index(row, 2, qt.QModelIndex())
            self.openPersistentEditor(index)
            index = model.index(row, 3, qt.QModelIndex())
            self.openPersistentEditor(index)

    def __openPersistantViewOnModelReset(self):
        model = self.model()
        index = qt.QModelIndex()
        row = model.rowCount(index)
        self.__onRowInserted(index, 0, row)


class _PeakSelectionTableModel(qt.QAbstractTableModel):

    requestRingChange = qt.Signal(object, int)

    requestRemovePeak = qt.Signal(object)

    def __init__(self, parent, peakSelectionModel):
        assert isinstance(parent, PeakPickingTask)
        super(_PeakSelectionTableModel, self).__init__(parent=parent)
        self.__peakSelectionModel = peakSelectionModel
        peakSelectionModel.changed.connect(self.__invalidateModel)
        self.__callbacks = []
        self.__invalidateModel()
        # QAbstractTableModel do not provide access to the parent
        self.__parent = parent

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
        self.dataChanged.emit(index1, index2)

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        if orientation != qt.Qt.Horizontal:
            return None
        if role != qt.Qt.DisplayRole:
            return super(_PeakSelectionTableModel, self).headerData(section, orientation, role)
        if section == 0:
            return "Name"
        if section == 1:
            return "Peaks"
        if section == 2:
            return "Ring number"
        return None

    def flags(self, index):
        if index.column() == 2:
            return (qt.Qt.ItemIsEditable |
                    qt.Qt.ItemIsEnabled |
                    qt.Qt.ItemIsSelectable)
        return (qt.Qt.ItemIsEnabled |
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

    def setData(self, index, value, role=qt.Qt.EditRole):
        if not index.isValid():
            return
        peakModel = self.__peakSelectionModel[index.row()]
        column = index.column()
        if column == 2:
            self.requestRingChange.emit(peakModel, value)

    def removeRows(self, row, count, parent=qt.QModelIndex()):
        # while the tablempdel is already connected to the data model
        self.__peakSelectionModel.changed.disconnect(self.__invalidateModel)

        self.beginRemoveRows(parent, row, row + count - 1)
        for i in reversed(range(count)):
            peakModel = self.__peakSelectionModel[row + i]
            self.requestRemovePeak.emit(peakModel)
        self.endRemoveRows()

        # while the tablempdel is already connected to the data model
        self.__peakSelectionModel.changed.connect(self.__invalidateModel)
        return True


class _PeakPickingPlot(silx.gui.plot.PlotWidget):

    def __init__(self, parent):
        super(_PeakPickingPlot, self).__init__(parent=parent)
        self.setKeepDataAspectRatio(True)

        # FIXME Fix using silx 0.5
        if "BackendMatplotlib" in self._backend.__class__.__name__:
            # hide axes and viewbox rect
            self._backend.ax.set_axis_off()
            self._backend.ax2.set_axis_off()
            # remove external margins
            self._backend.ax.set_position([0, 0, 1, 1])
            self._backend.ax2.set_position([0, 0, 1, 1])

        colormap = {
            'name': "inferno",
            'normalization': 'log',
            'autoscale': True,
        }
        self.setDefaultColormap(colormap)

        self.__peakSelectionModel = None
        self.__callbacks = {}
        self.__markerColors = {}

    def setModel(self, peakSelectionModel):
        assert self.__peakSelectionModel is None
        self.__peakSelectionModel = peakSelectionModel
        self.__peakSelectionModel.changed.connect(self.__invalidateModel)
        self.__invalidateModel()

    def __invalidateModel(self):
        added = set(self.__peakSelectionModel) - set(self.__callbacks.keys())
        removed = set(self.__callbacks.keys()) - set(self.__peakSelectionModel)

        # remove items
        for peakModel in removed:
            callback = self.__callbacks[peakModel]
            del self.__callbacks[peakModel]
            peakModel.changed.disconnect(callback)
            self.removePeak(peakModel)

        # add items
        for peakModel in added:
            callback = functools.partial(self.__invalidateItem, peakModel)
            peakModel.changed.connect(callback)
            self.addPeak(peakModel)
            self.__callbacks[peakModel] = callback

    def __invalidateItem(self, peakModel):
        self.updatePeak(peakModel)

    def removePeak(self, peakModel):
        legend = "marker" + peakModel.name()
        self.removeMarker(legend=legend)
        legend = "coord" + peakModel.name()
        self.removeCurve(legend=legend)

    def addPeak(self, peakModel):
        color = peakModel.color()
        numpyColor = numpy.array([color.redF(), color.greenF(), color.blueF()])
        points = peakModel.coords()
        name = peakModel.name()

        y, x = points[0]
        self.addMarker(x=x, y=y,
                       legend="marker" + name,
                       text=name)
        y = list(map(lambda p: p[0], points))
        x = list(map(lambda p: p[1], points))
        self.addCurve(x=x, y=y,
                      legend="coord" + name,
                      linestyle=' ',
                      symbol='o',
                      color=numpyColor,
                      resetzoom=False)

    def updatePeak(self, peakModel):
        self.removePeak(peakModel)
        self.addPeak(peakModel)

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


class _SpinBoxItemDelegate(qt.QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        if not index.isValid():
            return super(_SpinBoxItemDelegate, self).createEditor(parent, option, index)

        editor = qt.QSpinBox(parent=parent)
        editor.setMinimum(1)
        editor.setMaximum(999)
        editor.valueChanged.connect(lambda x: self.commitData.emit(editor))
        editor.setFocusPolicy(qt.Qt.StrongFocus)
        editor.setValue(index.data())
        editor.installEventFilter(self)
        return editor

    def eventFilter(self, widget, event):
        if event.type() == qt.QEvent.ChildPolished:
            # Fix issue relative to Qt4. after createEditor and setEditorData
            # The lineedit content is set selected without any reason.
            widget.lineEdit().deselect()
        return qt.QSpinBox.eventFilter(self, widget, event)

    def setEditorData(self, editor, index):
        value = index.data()
        if editor.value() == value:
            return
        old = editor.blockSignals(True)
        editor.setValue(value)
        editor.blockSignals(old)

    def setModelData(self, editor, model, index):
        editor.interpretText()
        value = editor.value()
        model.setData(index, value)

    def updateEditorGeometry(self, editor, option, index):
        """
        Update the geometry of the editor according to the changes of the view.

        :param qt.QWidget editor: Editor widget
        :param qt.QStyleOptionViewItem option: Control how the editor is shown
        :param qt.QIndex index: Index of the data to display
        """
        editor.setGeometry(option.rect)


class _PeakToolItemDelegate(qt.QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        if not index.isValid():
            return super(_PeakToolItemDelegate, self).createEditor(parent, option, index)

        editor = qt.QToolBar(parent=parent)
        editor.setIconSize(qt.QSize(32, 32))
        editor.setStyleSheet("QToolBar { border: 0px }")
        editor.setMinimumSize(32, 32)
        editor.setMaximumSize(32, 32)
        editor.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)

        remove = qt.QAction(editor)
        remove.setIcon(icons.getQIcon("remove-peak"))
        remove._customSignal = None
        persistantIndex = qt.QPersistentModelIndex(index)
        remove.triggered.connect(functools.partial(self.__removePeak, persistantIndex))
        editor.addAction(remove)
        return editor

    def updateEditorGeometry(self, editor, option, index):
        """
        Update the geometry of the editor according to the changes of the view.

        :param qt.QWidget editor: Editor widget
        :param qt.QStyleOptionViewItem option: Control how the editor is shown
        :param qt.QIndex index: Index of the data to display
        """
        editor.setGeometry(option.rect)

    def __removePeak(self, persistantIndex, checked):
        if not persistantIndex.isValid():
            return
        model = persistantIndex.model()
        model.removeRow(persistantIndex.row(), persistantIndex.parent())


class PeakPickingTask(AbstractCalibrationTask):

    def __init__(self):
        super(PeakPickingTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-peakpicking.ui"), self)
        self.initNextStep()
        self.__dialogState = None

        layout = qt.QVBoxLayout(self._imageHolder)
        self.__plot = _PeakPickingPlot(parent=self._imageHolder)
        toolBar = self.__createPlotToolBar(self.__plot)
        self.__plot.addToolBar(toolBar)
        statusBar = self.__createPlotStatusBar(self.__plot)
        self.__plot.setStatusBar(statusBar)

        layout.addWidget(self.__plot)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)

        self._ringSelectionMode.setIcon(icons.getQIcon("search-ring"))
        self._peakSelectionMode.setIcon(icons.getQIcon("search-peak"))
        self.__peakSelectionView = None
        self.__plot.sigPlotSignal.connect(self.__onPlotEvent)

        self.__undoStack = qt.QUndoStack(self)
        self._undoButton.setDefaultAction(self.__undoStack.createUndoAction(self, "Undo"))
        self._redoButton.setDefaultAction(self.__undoStack.createRedoAction(self, "Redo"))

        self.__mode = qt.QButtonGroup()
        self.__mode.setExclusive(True)
        self.__mode.addButton(self._peakSelectionMode)
        self.__mode.addButton(self._ringSelectionMode)
        self._ringSelectionMode.setChecked(True)

        self._extract.clicked.connect(self.__autoExtractRings)

    def __createSavePeakDialog(self):
        dialog = qt.QFileDialog(self)
        dialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        dialog.setWindowTitle("Save selected peaks")
        dialog.setModal(True)

        extensions = OrderedDict()
        extensions["Control point files"] = "*.npt"

        filters = []
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))

        dialog.setNameFilters(filters)
        return dialog

    def __createLoadPeakDialog(self):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Load peaks")
        dialog.setModal(True)

        extensions = OrderedDict()
        extensions["Control point files"] = "*.npt"

        filters = []
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))

        dialog.setNameFilters(filters)
        return dialog

    def __loadPeaksFromFile(self):
        dialog = self.__createLoadPeakDialog()

        if self.__dialogState is None:
            currentDirectory = os.getcwd()
            dialog.setDirectory(currentDirectory)
        else:
            dialog.restoreState(self.__dialogState)

        result = dialog.exec_()
        if not result:
            return

        self.__dialogState = dialog.saveState()

        filename = dialog.selectedFiles()[0]
        if os.path.exists(filename):
            try:
                controlPoints = pyFAI.control_points.ControlPoints(filename)
                oldState = self.__copyPeaks(self.__undoStack)
                self.model().peakSelectionModel().clear()
                for label in controlPoints.get_labels():
                    group = controlPoints.get(lbl=label)
                    peakModel = self.__createNewPeak(group.points)
                    peakModel.setRingNumber(group.ring)
                    peakModel.setName(label)
                    self.model().peakSelectionModel().append(peakModel)
                newState = self.__copyPeaks(self.__undoStack)
                command = _PeakSelectionUndoCommand(None, self.model().peakSelectionModel(), oldState, newState)
                command.setText("load rings")
                command.setRedoInhibited(True)
                self.__undoStack.push(command)
                command.setRedoInhibited(False)
            except Exception as e:
                _logger.error(e.args[0])
                _logger.error("Backtrace", exc_info=True)
                # FIXME Display error dialog
            except KeyboardInterrupt:
                raise

    def __savePeaksAsFile(self):
        dialog = self.__createSavePeakDialog()

        if self.__dialogState is None:
            currentDirectory = os.getcwd()
            dialog.setDirectory(currentDirectory)
        else:
            dialog.restoreState(self.__dialogState)

        result = dialog.exec_()
        if not result:
            return

        self.__dialogState = dialog.saveState()
        filename = dialog.selectedFiles()[0]
        if not os.path.exists(filename) and not filename.endswith(".npt"):
            filename = filename + ".npt"
        try:
            calibrant = self.model().experimentSettingsModel().calibrantModel().calibrant()
            wavelength = self.model().experimentSettingsModel().wavelength().value()
            wavelength = wavelength / 1e10
            controlPoints = pyFAI.control_points.ControlPoints(None, calibrant, wavelength)
            for peakModel in self.model().peakSelectionModel():
                ringNumber = peakModel.ringNumber()
                points = peakModel.coords()
                controlPoints.append(points=points, ring=ringNumber)
            controlPoints.save(filename)
        except Exception as e:
            _logger.error(e.args[0])
            _logger.error("Backtrace", exc_info=True)
            # FIXME Display error dialog
        except KeyboardInterrupt:
            raise

    def __createOptionsWidget(self):
        menu = qt.QMenu(self)

        # Load peak selection as file
        loadPeaksFromFile = qt.QAction(self)
        icon = icons.getQIcon('document-open')
        loadPeaksFromFile.setIcon(icon)
        loadPeaksFromFile.setText("Load peak selection from file")
        loadPeaksFromFile.triggered.connect(self.__loadPeaksFromFile)
        loadPeaksFromFile.setIconVisibleInMenu(True)
        menu.addAction(loadPeaksFromFile)

        # Save peak selection as file
        savePeaksAsFile = qt.QAction(self)
        icon = icons.getQIcon('document-save')
        savePeaksAsFile.setIcon(icon)
        savePeaksAsFile.setText("Save peak selection as file")
        savePeaksAsFile.triggered.connect(self.__savePeaksAsFile)
        savePeaksAsFile.setIconVisibleInMenu(True)
        menu.addAction(savePeaksAsFile)

        options = qt.QToolButton(self)
        icon = icons.getQIcon('options')
        options.setIcon(icon)
        options.setPopupMode(qt.QToolButton.InstantPopup)
        options.setMenu(menu)
        return options

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

        stretch = qt.QWidget(self)
        stretch.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        toolBar.addWidget(stretch)

        self.__options = self.__createOptionsWidget()
        toolBar.addWidget(self.__options)

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
                self.__plotClicked(x, y)

    def __plotClicked(self, x, y):
        image = self.model().experimentSettingsModel().image().value()
        massif = pyFAI.massif.Massif(image)
        points = massif.find_peaks([y, x], stdout=_DummyStdOut())
        if len(points) == 0:
            # toleration
            toleration = 3

            # clamp min to avoid negative values
            ymin = y - toleration
            if ymin < 0:
                ymin = 0
            ymax = y + toleration + 1
            xmin = x - toleration
            if xmin < 0:
                xmin = 0
            xmax = x + toleration + 1

            data = image[ymin:ymax, xmin:xmax]
            coord = numpy.argmax(data)
            coord = numpy.unravel_index(coord, data.shape)
            y, x = ymin + coord[0], xmin + coord[1]
            points = massif.find_peaks([y, x], stdout=_DummyStdOut())

        # filter peaks from the mask
        mask = self.model().experimentSettingsModel().mask().value()
        if mask is not None:
            points = filter(lambda coord: mask[int(coord[0]), int(coord[1])] != 1, points)
            points = list(points)

        if len(points) > 0:
            # reach bigger ring
            selection = self.model().peakSelectionModel()
            ringNumbers = [p.ringNumber() for p in selection]
            if ringNumbers == []:
                lastRingNumber = 0
            else:
                lastRingNumber = max(ringNumbers)

            if self._ringSelectionMode.isChecked():
                ringNumber = lastRingNumber + 1
            elif self._peakSelectionMode.isChecked():
                ringNumber = lastRingNumber
                points = points[0:1]
            else:
                raise ValueError("Picking mode unknown")

            peakModel = self.__createNewPeak(points)
            peakModel.setRingNumber(ringNumber)
            oldState = self.__copyPeaks(self.__undoStack)
            self.model().peakSelectionModel().append(peakModel)
            newState = self.__copyPeaks(self.__undoStack)
            command = _PeakSelectionUndoCommand(None, self.model().peakSelectionModel(), oldState, newState)
            command.setText("add peak %s" % peakModel.name())
            command.setRedoInhibited(True)
            self.__undoStack.push(command)
            command.setRedoInhibited(False)

    def __removePeak(self, peakModel):
        oldState = self.__copyPeaks(self.__undoStack)
        self.model().peakSelectionModel().remove(peakModel)
        newState = self.__copyPeaks(self.__undoStack)
        command = _PeakSelectionUndoCommand(None, self.model().peakSelectionModel(), oldState, newState)
        command.setText("remove peak %s" % peakModel.name())
        command.setRedoInhibited(True)
        self.__undoStack.push(command)
        command.setRedoInhibited(False)

    def __setRingNumber(self, peakModel, value):
        oldState = self.__copyPeaks(self.__undoStack)
        peakModel.setRingNumber(value)
        newState = self.__copyPeaks(self.__undoStack)
        command = _PeakSelectionUndoCommand(None, self.model().peakSelectionModel(), oldState, newState)
        command.setText("update ring number of %s" % peakModel.name())
        command.setRedoInhibited(True)
        self.__undoStack.push(command)
        command.setRedoInhibited(False)

    def __copyPeaks(self, parent):
        selection = self.model().peakSelectionModel()
        state = []
        for peakModel in selection:
            copy = peakModel.copy(parent)
            state.append(copy)
        return state

    def __createNewPeak(self, points):
        selection = self.model().peakSelectionModel()

        # reach the bigger name
        names = ["% 8s" % p.name() for p in selection]
        if len(names) > 0:
            names = list(sorted(names))
            bigger = names[-1].strip()
            number = 0
            for c in bigger:
                number = number * 26 + (ord(c) - ord('a'))
        else:
            number = -1
        number = number + 1

        # compute the next one
        name = ""
        if number == 0:
            name = "a"
        else:
            n = number
            while n > 0:
                c = n % 26
                n = n // 26
                name = chr(c + ord('a')) + name

        # FIXME improve color list
        colors = self.__plot.markerColorList()
        color = colors[number % len(colors)]

        peakModel = PeakModel(self.model().peakSelectionModel())
        peakModel.setName(name)
        peakModel.setColor(color)
        peakModel.setCoords(points)
        peakModel.setRingNumber(1)

        return peakModel

    def __autoExtractRings(self):
        maxRings = self._maxRingToExtract.value()
        pointPerDegree = self._numberOfPeakPerDegree.value()

        # extract peaks from settings info and current peaks
        image = self.model().experimentSettingsModel().image().value()
        mask = self.model().experimentSettingsModel().mask().value()
        calibrant = self.model().experimentSettingsModel().calibrantModel().calibrant()
        detector = self.model().experimentSettingsModel().detector()
        wavelength = self.model().experimentSettingsModel().wavelength().value()
        wavelength = wavelength / 1e10
        extractor = RingExtractor(image, mask, calibrant, detector, wavelength)

        # FIXME numpy array can be allocated first
        peaks = []
        for peakModel in self.model().peakSelectionModel():
            ringNumber = peakModel.ringNumber()
            for coord in peakModel.coords():
                peaks.append([coord[0], coord[1], ringNumber - 1])
        peaks = numpy.array(peaks)

        newPeaksRaw = extractor.extract(peaks=peaks,
                                        method="massif",
                                        maxRings=maxRings,
                                        pointPerDegree=pointPerDegree)

        # split peaks per rings
        newPeaks = {}
        for peak in newPeaksRaw:
            y, x, ringNumber = peak
            ringNumber = int(ringNumber) + 1
            if ringNumber not in newPeaks:
                newPeaks[ringNumber] = []
            newPeaks[ringNumber].append((y, x))

        # update the gui
        oldState = self.__copyPeaks(self.__undoStack)
        self.model().peakSelectionModel().clear()
        for ringNumber in sorted(newPeaks.keys()):
            coords = newPeaks[ringNumber]
            peakModel = self.__createNewPeak(coords)
            peakModel.setRingNumber(ringNumber)
            self.model().peakSelectionModel().append(peakModel)
        newState = self.__copyPeaks(self.__undoStack)
        command = _PeakSelectionUndoCommand(None, self.model().peakSelectionModel(), oldState, newState)
        command.setText("extract rings")
        command.setRedoInhibited(True)
        self.__undoStack.push(command)
        command.setRedoInhibited(False)

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
        settings.mask().changed.connect(self.__maskUpdated)
        self.__plot.setModel(model.peakSelectionModel())
        self.__initPeakSelectionView(model)
        self.__undoStack.clear()

    def __initPeakSelectionView(self, model):
        if self.__peakSelectionView is None:
            self.__peakSelectionView = _PeakSelectionTableView(self)
            layout = qt.QHBoxLayout(self._peakSelectionHolder)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.__peakSelectionView)
            self._peakSelectionHolder.setLayout(layout)

        tableModel = _PeakSelectionTableModel(self, model.peakSelectionModel())
        tableModel.requestRingChange.connect(self.__setRingNumber)
        tableModel.requestRemovePeak.connect(self.__removePeak)
        self.__peakSelectionView.setModel(tableModel)

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().image().value()
        self.__plot.addImage(image, legend="image", selectable=True)
        if image is not None:
            self.__plot.setGraphXLimits(0, image.shape[0])
            self.__plot.setGraphYLimits(0, image.shape[1])
            self.__plot.resetZoom()

    def __maskUpdated(self):
        _mask = self.model().experimentSettingsModel().mask().value()
        # FIXME maybe it is good to display the mask somewhere here

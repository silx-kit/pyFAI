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
__date__ = "17/12/2018"

import logging
import numpy
import functools
import os

from silx.gui import qt
from silx.gui import icons
import silx.gui.plot
from silx.gui.plot.tools import PositionInfo

import pyFAI.utils
import pyFAI.massif
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask
from pyFAI.gui.calibration.RingExtractor import RingExtractor
import pyFAI.control_points
from . import utils
from .helper.SynchronizeRawView import SynchronizeRawView
from .helper.SynchronizePlotBackground import SynchronizePlotBackground
from .CalibrationContext import CalibrationContext
from .helper.MarkerManager import MarkerManager
from ..utils import FilterBuilder
from ..utils import validators
from .helper import model_transform


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
        palette = qt.QPalette(self.palette())
        # make sure this value is not edited
        palette.setColor(qt.QPalette.Base, palette.base().color())
        ringDelegate.setPalette(palette)
        toolDelegate = _PeakToolItemDelegate(self)
        self.setItemDelegateForColumn(2, ringDelegate)
        self.setItemDelegateForColumn(3, toolDelegate)

        # self.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setVerticalScrollMode(qt.QAbstractItemView.ScrollPerPixel)
        self.setShowGrid(False)
        self.setWordWrap(False)
        # NoFrame glitchies on Debian8 Qt5
        # self.setFrameShape(qt.QFrame.NoFrame)

        self.horizontalHeader().setHighlightSections(False)
        self.verticalHeader().setVisible(False)

        palette = qt.QPalette(self.palette())
        palette.setColor(qt.QPalette.Base, qt.QColor(0, 0, 0, 0))
        self.setPalette(palette)
        self.setFrameShape(qt.QFrame.Panel)

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
        peakSelectionModel.structureChanged.connect(self.__invalidateModel)
        peakSelectionModel.contentChanged.connect(self.__invalidateContentModel)
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

    def __invalidateContentModel(self):
        for index, _ in enumerate(self.__peakSelectionModel):
            self.__invalidateItem(index)

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
            return False
        peakModel = self.__peakSelectionModel[index.row()]
        column = index.column()
        if column == 2:
            self.requestRingChange.emit(peakModel, value)
            return True
        return False

    def removeRows(self, row, count, parent=qt.QModelIndex()):
        # while the tablemodel is already connected to the data model
        self.__peakSelectionModel.structureChanged.disconnect(self.__invalidateModel)

        self.beginRemoveRows(parent, row, row + count - 1)
        for i in reversed(range(count)):
            peakModel = self.__peakSelectionModel[row + i]
            self.requestRemovePeak.emit(peakModel)
        self.endRemoveRows()

        # while the tablemodel is already connected to the data model
        self.__peakSelectionModel.structureChanged.connect(self.__invalidateModel)
        return True


class _PeakPickingPlot(silx.gui.plot.PlotWidget):

    def __init__(self, parent):
        super(_PeakPickingPlot, self).__init__(parent=parent)
        self.setKeepDataAspectRatio(True)
        self.setAxesDisplayed(False)

        self.__peakSelectionModel = None
        self.__callbacks = {}
        self.__processing = None

        markerModel = CalibrationContext.instance().getCalibrationModel().markerModel()
        self.__markerManager = MarkerManager(self, markerModel, pixelBasedPlot=True)

        handle = self.getWidgetHandle()
        handle.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        handle.customContextMenuRequested.connect(self.__plotContextMenu)

        colormap = CalibrationContext.instance().getRawColormap()
        self.setDefaultColormap(colormap)

        self.__plotBackground = SynchronizePlotBackground(self)

        if hasattr(self, "centralWidget"):
            self.centralWidget().installEventFilter(self)

    def eventFilter(self, widget, event):
        if event.type() == qt.QEvent.Enter:
            self.setCursor(qt.Qt.CrossCursor)
            return True
        elif event.type() == qt.QEvent.Leave:
            self.unsetCursor()
            return True
        return False

    def __plotContextMenu(self, pos):
        plot = self
        from silx.gui.plot.actions.control import ZoomBackAction
        zoomBackAction = ZoomBackAction(plot=plot, parent=plot)

        menu = qt.QMenu(self)

        menu.addAction(zoomBackAction)
        menu.addSeparator()
        menu.addAction(self.__markerManager.createMarkPixelAction(menu, pos))
        menu.addAction(self.__markerManager.createMarkGeometryAction(menu, pos))
        action = self.__markerManager.createRemoveClosestMaskerAction(menu, pos)
        if action is not None:
            menu.addAction(action)

        handle = plot.getWidgetHandle()
        menu.exec_(handle.mapToGlobal(pos))

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
        numpyColor = numpy.array([color.redF(), color.greenF(), color.blueF(), 0.5])
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
                      selectable=False,
                      symbol='o',
                      color=numpyColor,
                      resetzoom=False)

    def updatePeak(self, peakModel):
        self.removePeak(peakModel)
        self.addPeak(peakModel)

    def unsetProcessing(self):
        if self.__processing is not None:
            self.__processing.deleteLater()
        self.__processing = None

    def setProcessing(self):
        self.__processing = utils.createProcessingWidgetOverlay(self)


class _SpinBoxItemDelegate(qt.QStyledItemDelegate):

    def __init__(self, *args, **kwargs):
        qt.QStyledItemDelegate.__init__(self, *args, **kwargs)
        self.__palette = None

    def setPalette(self, palette):
        self.__palette = qt.QPalette(palette)

    def createEditor(self, parent, option, index):
        if not index.isValid():
            return super(_SpinBoxItemDelegate, self).createEditor(parent, option, index)

        editor = qt.QSpinBox(parent=parent)
        if self.__palette is not None:
            editor.setPalette(self.__palette)
        editor.setMinimum(1)
        editor.setMaximum(999)
        editor.valueChanged.connect(lambda x: self.commitData.emit(editor))
        editor.setFocusPolicy(qt.Qt.StrongFocus)
        editor.setValue(index.data())
        editor.installEventFilter(self)
        editor.setBackgroundRole(qt.QPalette.Background)
        editor.setAutoFillBackground(True)
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
        remove.setIcon(icons.getQIcon("pyfai:gui/icons/remove-peak"))
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

    def _initGui(self):
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-peakpicking.ui"), self)
        icon = silx.gui.icons.getQIcon("pyfai:gui/icons/task-identify-rings")
        self.setWindowIcon(icon)

        self.initNextStep()

        # Insert the plot on the layout
        holder = self._plotHolder
        self.__plot = _PeakPickingPlot(parent=holder)
        self.__plot.setObjectName("plot-picking")
        holderLayout = qt.QVBoxLayout(holder)
        holderLayout.setContentsMargins(1, 1, 1, 1)
        holderLayout.addWidget(self.__plot)

        # Insert the peak view on the layout
        holder = self._peakSelectionDummy.parent()
        self.__peakSelectionView = _PeakSelectionTableView(holder)
        holderLayout = holder.layout()
        holderLayout.replaceWidget(self._peakSelectionDummy, self.__peakSelectionView)

        self.__undoStack = qt.QUndoStack(self)

        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self._ringToolBarHolder.setLayout(layout)
        toolBar = self.__createRingToolBar()
        layout.addWidget(toolBar)

        self.__createPlotToolBar(self.__plot)
        statusBar = self.__createPlotStatusBar(self.__plot)
        self.__plot.setStatusBar(statusBar)

        self.__plot.sigPlotSignal.connect(self.__onPlotEvent)

        self._extract.setEnabled(False)
        self._extract.clicked.connect(self.__autoExtractRingsLater)

        validator = validators.DoubleValidator(self)
        self._numberOfPeakPerDegree.lineEdit().setValidator(validator)
        locale = qt.QLocale(qt.QLocale.C)
        self._numberOfPeakPerDegree.setLocale(locale)

        self.__synchronizeRawView = SynchronizeRawView()
        self.__synchronizeRawView.registerTask(self)
        self.__synchronizeRawView.registerPlot(self.__plot)

        self.__massif = None
        self.__massifReconstructed = None

    def __createSavePeakDialog(self):
        dialog = CalibrationContext.instance().createFileDialog(self)
        dialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        dialog.setWindowTitle("Save selected peaks")
        dialog.setModal(True)
        builder = FilterBuilder.FilterBuilder()
        builder.addFileFormat("Control point files", "npt")
        dialog.setNameFilters(builder.getFilters())
        return dialog

    def __createLoadPeakDialog(self):
        dialog = CalibrationContext.instance().createFileDialog(self)
        dialog.setWindowTitle("Load peaks")
        dialog.setModal(True)
        builder = FilterBuilder.FilterBuilder()
        builder.addFileFormat("Control point files", "npt")
        dialog.setNameFilters(builder.getFilters())
        return dialog

    def __loadPeaksFromFile(self):
        dialog = self.__createLoadPeakDialog()

        result = dialog.exec_()
        if not result:
            return

        filename = dialog.selectedFiles()[0]
        if os.path.exists(filename):
            try:
                controlPoints = pyFAI.control_points.ControlPoints(filename)
                oldState = self.__copyPeaks(self.__undoStack)
                peakSelectionModel = self.model().peakSelectionModel()
                model_transform.initPeaksFromControlPoints(peakSelectionModel, controlPoints)
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

        result = dialog.exec_()
        if not result:
            return

        filename = dialog.selectedFiles()[0]
        if not os.path.exists(filename) and not filename.endswith(".npt"):
            filename = filename + ".npt"
        try:
            controlPoints = model_transform.createControlPoints(self.model())
            controlPoints.save(filename)
        except Exception as e:
            _logger.error(e.args[0])
            _logger.error("Backtrace", exc_info=True)
            # FIXME Display error dialog
        except KeyboardInterrupt:
            raise

    def __createRingToolBar(self):
        toolBar = qt.QToolBar(self)

        action = qt.QAction(self)
        action.setIcon(icons.getQIcon("pyfai:gui/icons/search-full-ring"))
        action.setText("Ring")
        action.setCheckable(True)
        action.setToolTip("Extract peaks, beyond masked values")
        toolBar.addAction(action)
        self.__ringSelectionMode = action

        action = qt.QAction(self)
        action.setIcon(icons.getQIcon("pyfai:gui/icons/search-ring"))
        action.setText("Arc")
        action.setCheckable(True)
        action.setToolTip("Extract contiguous peaks")
        toolBar.addAction(action)
        self.__arcSelectionMode = action

        action = qt.QAction(self)
        action.setIcon(icons.getQIcon("pyfai:gui/icons/search-peak"))
        action.setText("Arc")
        action.setCheckable(True)
        action.setToolTip("Extract contiguous peaks")
        toolBar.addAction(action)
        self.__peakSelectionMode = action

        mode = qt.QActionGroup(self)
        mode.setExclusive(True)
        mode.addAction(self.__ringSelectionMode)
        mode.addAction(self.__arcSelectionMode)
        mode.addAction(self.__peakSelectionMode)
        self.__arcSelectionMode.setChecked(True)

        toolBar.addSeparator()

        # Load peak selection as file
        loadPeaksFromFile = qt.QAction(self)
        icon = icons.getQIcon('document-open')
        self.__icon = icon
        loadPeaksFromFile.setIcon(icon)
        loadPeaksFromFile.setText("Load peak selection from file")
        loadPeaksFromFile.triggered.connect(self.__loadPeaksFromFile)
        loadPeaksFromFile.setIconVisibleInMenu(True)
        toolBar.addAction(loadPeaksFromFile)

        # Save peak selection as file
        savePeaksAsFile = qt.QAction(self)
        icon = icons.getQIcon('document-save')
        savePeaksAsFile.setIcon(icon)
        savePeaksAsFile.setText("Save peak selection as file")
        savePeaksAsFile.triggered.connect(self.__savePeaksAsFile)
        savePeaksAsFile.setIconVisibleInMenu(True)
        toolBar.addAction(savePeaksAsFile)

        toolBar.addSeparator()
        style = qt.QApplication.style()

        action = self.__undoStack.createUndoAction(self, "Undo")
        icon = style.standardIcon(qt.QStyle.SP_ArrowBack)
        action.setIcon(icon)
        toolBar.addAction(action)

        action = self.__undoStack.createRedoAction(self, "Redo")
        icon = style.standardIcon(qt.QStyle.SP_ArrowForward)
        action.setIcon(icon)
        toolBar.addAction(action)

        return toolBar

    def __createPlotToolBar(self, plot):
        from silx.gui.plot import tools
        toolBar = tools.InteractiveModeToolBar(parent=self, plot=plot)
        plot.addToolBar(toolBar)
        toolBar = tools.ImageToolBar(parent=self, plot=plot)
        colormapDialog = CalibrationContext.instance().getColormapDialog()
        toolBar.getColormapAction().setColorDialog(colormapDialog)
        plot.addToolBar(toolBar)

    def __createPlotStatusBar(self, plot):

        converters = [
            ('X', lambda x, y: x),
            ('Y', lambda x, y: y),
            ('Value', self.__getImageValue)]

        hbox = qt.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        info = PositionInfo(plot=plot, converters=converters)
        info.setSnappingMode(True)
        statusBar = qt.QStatusBar(plot)
        statusBar.setSizeGripEnabled(False)
        statusBar.addWidget(info)
        return statusBar

    def __onPlotEvent(self, event):
        if event["event"] == "imageClicked":
            x, y, button = event["col"], event["row"], event["button"]
            if button == "left":
                self.__plotClicked(x, y)

    def __invalidateMassif(self):
        self.__massif = None
        self.__massifReconstructed = None

    def __widgetShow(self):
        pass

    def __createMassif(self, reconstruct=False):
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        experimentSettings = self.model().experimentSettingsModel()
        image = experimentSettings.image().value()
        mask = experimentSettings.mask().value()
        if image is None:
            return None
        massif = pyFAI.massif.Massif(image, mask)
        massif.get_labeled_massif(reconstruct=reconstruct)
        qt.QApplication.restoreOverrideCursor()
        return massif

    def __getMassif(self):
        if self.__ringSelectionMode.isChecked():
            if self.__massifReconstructed is None:
                self.__massifReconstructed = self.__createMassif(reconstruct=True)
            return self.__massifReconstructed
        elif self.__arcSelectionMode.isChecked() or self.__peakSelectionMode.isChecked():
            if self.__massif is None:
                self.__massif = self.__createMassif()
            return self.__massif
        else:
            assert(False)

    def __plotClicked(self, x, y):
        image = self.model().experimentSettingsModel().image().value()
        massif = self.__getMassif()
        if massif is None:
            # Nothing to pick
            return
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
            points = filter(lambda coord: mask[int(coord[0]), int(coord[1])] == 0, points)
            points = list(points)

        if len(points) > 0:
            # reach bigger ring
            peakSelectionModel = self.model().peakSelectionModel()
            ringNumbers = [p.ringNumber() for p in peakSelectionModel]
            if ringNumbers == []:
                lastRingNumber = 0
            else:
                lastRingNumber = max(ringNumbers)

            if self.__ringSelectionMode.isChecked() or self.__arcSelectionMode.isChecked():
                ringNumber = lastRingNumber + 1
            elif self.__peakSelectionMode.isChecked():
                if lastRingNumber == 0:
                    lastRingNumber = 1
                ringNumber = lastRingNumber
                points = points[0:1]
            else:
                raise ValueError("Picking mode unknown")

            peakModel = model_transform.createRing(points, peakSelectionModel)
            peakModel.setRingNumber(ringNumber)
            oldState = self.__copyPeaks(self.__undoStack)
            peakSelectionModel.append(peakModel)
            newState = self.__copyPeaks(self.__undoStack)
            command = _PeakSelectionUndoCommand(None, peakSelectionModel, oldState, newState)
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

    def __autoExtractRingsLater(self):
        self.__plot.setProcessing()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        self._extract.setWaiting(True)
        # Wait for Qt repaint first
        qt.QTimer.singleShot(1, self.__autoExtractRings)

    def __autoExtractRings(self):
        try:
            self.__autoExtractRingsCompute()
        finally:
            self.__plot.unsetProcessing()
            qt.QApplication.restoreOverrideCursor()
            self._extract.setWaiting(False)

    def __autoExtractRingsCompute(self):
        maxRings = self._maxRingToExtract.value()
        pointPerDegree = self._numberOfPeakPerDegree.value()

        # extract peaks from settings info and current peaks
        image = self.model().experimentSettingsModel().image().value()
        mask = self.model().experimentSettingsModel().mask().value()
        calibrant = self.model().experimentSettingsModel().calibrantModel().calibrant()
        detector = self.model().experimentSettingsModel().detector()
        wavelength = self.model().experimentSettingsModel().wavelength().value()

        if detector is None:
            self.__plot.unsetProcessing()
            qt.QApplication.restoreOverrideCursor()
            self._extract.setWaiting(False)
            qt.QMessageBox.critical(self, "Error", "No detector defined")
            return
        if calibrant is None:
            self.__plot.unsetProcessing()
            qt.QApplication.restoreOverrideCursor()
            self._extract.setWaiting(False)
            qt.QMessageBox.critical(self, "Error", "No calibrant defined")
            return
        if wavelength is None:
            self.__plot.unsetProcessing()
            qt.QApplication.restoreOverrideCursor()
            self._extract.setWaiting(False)
            qt.QMessageBox.critical(self, "Error", "No wavelength defined")
            return

        extractor = RingExtractor(image, mask, calibrant, detector, wavelength)

        # Constant dependant of the ui file
        FROM_PEAKS = 0
        FROM_FIT = 1

        geometrySourceIndex = self._geometrySource.currentIndex()
        if geometrySourceIndex == FROM_PEAKS:
            # FIXME numpy array can be allocated first
            peaks = []
            for peakModel in self.model().peakSelectionModel():
                ringNumber = peakModel.ringNumber()
                for coord in peakModel.coords():
                    peaks.append([coord[0], coord[1], ringNumber - 1])
            peaks = numpy.array(peaks)
            geometryModel = None
        elif geometrySourceIndex == FROM_FIT:
            peaks = None
            geometryModel = self.model().fittedGeometry()
            if not geometryModel.isValid():
                _logger.error("The fitted model is not valid. Extraction cancelled.")
                return
        else:
            assert(False)

        newPeaksRaw = extractor.extract(peaks=peaks,
                                        geometryModel=geometryModel,
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
        peakSelectionModel = self.model().peakSelectionModel()
        peakSelectionModel.clear()
        for ringNumber in sorted(newPeaks.keys()):
            coords = newPeaks[ringNumber]
            peakModel = model_transform.createRing(coords, peakSelectionModel)
            peakModel.setRingNumber(ringNumber)
            peakSelectionModel.append(peakModel)
        newState = self.__copyPeaks(self.__undoStack)
        command = _PeakSelectionUndoCommand(None, peakSelectionModel, oldState, newState)
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
        data = image.getData(copy=False)
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
        self.__synchronizeRawView.registerModel(model.rawPlotView())
        settings = model.experimentSettingsModel()
        settings.maskedImage().changed.connect(self.__imageUpdated)
        settings.image().changed.connect(self.__invalidateMassif)
        settings.mask().changed.connect(self.__invalidateMassif)
        model.peakSelectionModel().changed.connect(self.__peakSelectionChanged)
        self.__plot.setModel(model.peakSelectionModel())
        self.__initPeakSelectionView(model)
        self.__undoStack.clear()

        self.__imageUpdated()
        self.__peakSelectionChanged()

    def __peakSelectionChanged(self):
        peakCount = self.model().peakSelectionModel().peakCount()
        if peakCount < 3:
            self._extract.setEnabled(False)
            self.setToolTip("Select manually more peaks to auto extract peaks")
        else:
            self._extract.setEnabled(True)
            self.setToolTip("")

    def __initPeakSelectionView(self, model):
        tableModel = _PeakSelectionTableModel(self, model.peakSelectionModel())
        tableModel.requestRingChange.connect(self.__setRingNumber)
        tableModel.requestRemovePeak.connect(self.__removePeak)
        self.__peakSelectionView.setModel(tableModel)

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().maskedImage().value()
        if image is not None:
            self.__plot.addImage(image, legend="image", selectable=True, copy=False, z=-1)
            self.__plot.setGraphXLimits(0, image.shape[0])
            self.__plot.setGraphYLimits(0, image.shape[1])
            self.__plot.resetZoom()
        else:
            self.__plot.removeImage("image")

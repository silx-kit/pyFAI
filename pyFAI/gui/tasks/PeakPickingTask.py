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
__date__ = "16/05/2019"

import logging
import numpy
import functools
import os

from silx.gui import qt
from silx.gui import icons
from silx.gui import colors
import silx.gui.plot
from silx.gui.plot.tools import PositionInfo
from silx.gui.plot.items.shape import Shape

from pyFAI.third_party import six
import pyFAI.utils
import pyFAI.massif
import pyFAI.control_points
from .AbstractCalibrationTask import AbstractCalibrationTask
from ..helper.RingExtractor import RingExtractorThread
from ..helper.SynchronizeRawView import SynchronizeRawView
from ..helper.SynchronizePlotBackground import SynchronizePlotBackground
from ..CalibrationContext import CalibrationContext
from ..helper.MarkerManager import MarkerManager
from ..helper import ProcessingWidget
from ..utils import FilterBuilder
from ..utils import validators
from ..helper import model_transform
from ..widgets.ColoredCheckBox import ColoredCheckBox
from ..widgets.AdvancedSpinBox import AdvancedSpinBox
from ..dialog import MessageBox


_logger = logging.getLogger(__name__)


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
        enabledDelegate = _PeakEnabledItemDelegate(self)
        self.setItemDelegateForColumn(_PeakSelectionTableModel.ColumnRingNumber, ringDelegate)
        self.setItemDelegateForColumn(_PeakSelectionTableModel.ColumnControl, toolDelegate)
        self.setItemDelegateForColumn(_PeakSelectionTableModel.ColumnEnabled, enabledDelegate)

        self.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setVerticalScrollMode(qt.QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOn)
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

    def mousePressEvent(self, event):
        """
        :param qt.QMouseEvent event: Qt event
        """
        index = self.indexAt(event.pos())
        if index.isValid():
            selectionModel = self.selectionModel()
            if selectionModel.isSelected(index):
                selectionModel.clear()
                event.accept()
                return
        return super(_PeakSelectionTableView, self).mousePressEvent(event)

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
        setResizeMode(_PeakSelectionTableModel.ColumnName, qt.QHeaderView.Stretch)
        setResizeMode(_PeakSelectionTableModel.ColumnPeaksCount, qt.QHeaderView.ResizeToContents)
        setResizeMode(_PeakSelectionTableModel.ColumnRingNumber, qt.QHeaderView.ResizeToContents)
        setResizeMode(_PeakSelectionTableModel.ColumnEnabled, qt.QHeaderView.ResizeToContents)
        setResizeMode(_PeakSelectionTableModel.ColumnControl, qt.QHeaderView.Fixed)

    def __onRowRemoved(self, parent, start, end):
        self.updateGeometry()

    def __onRowInserted(self, parent, start, end):
        self.__openPersistantViewOnRowInserted(parent, start, end)
        self.updateGeometry()
        # It have to be done only on the 3, else the layout is wrong
        self.resizeColumnToContents(_PeakSelectionTableModel.ColumnControl)

    def __openPersistantViewOnRowInserted(self, parent, start, end):
        model = self.model()
        for row in range(start, end):
            index = model.index(row, _PeakSelectionTableModel.ColumnRingNumber, qt.QModelIndex())
            self.openPersistentEditor(index)
            index = model.index(row, _PeakSelectionTableModel.ColumnControl, qt.QModelIndex())
            self.openPersistentEditor(index)
            index = model.index(row, _PeakSelectionTableModel.ColumnEnabled, qt.QModelIndex())
            self.openPersistentEditor(index)

    def __openPersistantViewOnModelReset(self):
        model = self.model()
        index = qt.QModelIndex()
        row = model.rowCount(index)
        self.__onRowInserted(index, 0, row)


class _PeakSelectionTableModel(qt.QAbstractTableModel):

    requestRingChange = qt.Signal(object, int)

    requestRemovePeak = qt.Signal(object)

    requestChangeEnable = qt.Signal(object, bool)

    ColumnEnabled = 0
    ColumnName = 1
    ColumnPeaksCount = 2
    ColumnRingNumber = 3
    ColumnControl = 4

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
        if section == self.ColumnName:
            return "Name"
        elif section == self.ColumnPeaksCount:
            return "Peaks"
        elif section == self.ColumnRingNumber:
            return "Ring number"
        elif section == self.ColumnEnabled:
            return ""
        elif section == self.ColumnControl:
            return ""
        return None

    def flags(self, index):
        return (qt.Qt.ItemIsEnabled |
                qt.Qt.ItemIsSelectable)

    def rowCount(self, parent=qt.QModelIndex()):
        return len(self.__peakSelectionModel)

    def columnCount(self, parent=qt.QModelIndex()):
        return 5

    def peakObject(self, index):
        peakModel = self.__peakSelectionModel[index.row()]
        return peakModel

    def data(self, index=qt.QModelIndex(), role=qt.Qt.DisplayRole):
        if not index.isValid():
            return False
        peakModel = self.__peakSelectionModel[index.row()]
        column = index.column()
        if role == qt.Qt.DisplayRole or role == qt.Qt.EditRole:
            if column == self.ColumnName:
                return peakModel.name()
            elif column == self.ColumnPeaksCount:
                return len(peakModel)
            elif column == self.ColumnRingNumber:
                return peakModel.ringNumber()
            return ""
        return None

    def setData(self, index, value, role=qt.Qt.EditRole):
        if not index.isValid():
            return False
        peakModel = self.__peakSelectionModel[index.row()]
        column = index.column()
        if role == qt.Qt.CheckStateRole:
            if column == self.ColumnEnabled:
                if value == qt.Qt.Checked:
                    isChecked = True
                else:
                    isChecked = False
                self.requestChangeEnable.emit(peakModel, isChecked)
                return True
        elif role == qt.Qt.EditRole:
            if column == self.ColumnRingNumber:
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

    PEAK_SELECTION_MODE = 0
    ERASOR_MODE = 1
    BRUSH_MODE = 2

    sigPeakPicked = qt.Signal(int, int)
    """Emitted when a mouse interaction requesteing a peak selection."""

    sigShapeErased = qt.Signal(object)
    """Emitted when a mouse interaction requesteing to erase peaks on shape."""

    sigShapeBrushed = qt.Signal(object)
    """Emitted when a mouse interaction requesteing to brush peaks on shape."""

    def __init__(self, parent):
        super(_PeakPickingPlot, self).__init__(parent=parent)
        self.setKeepDataAspectRatio(True)
        self.setAxesDisplayed(False)

        self.__peakSelectionModel = None
        self.__callbacks = {}
        self.__selectedPeak = None
        self.__processing = None
        self.__mode = self.PEAK_SELECTION_MODE
        self.__mask = None

        self.sigPlotSignal.connect(self.__onPlotEvent)

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

    def setInteractiveMode(self, mode, color='black',
                           shape='polygon', label=None,
                           zoomOnWheel=True, source=None, width=None):
        """Override the function to allow to disable extrat interaction modes.
        """
        self.setPeakInteractiveMode(self.PEAK_SELECTION_MODE)
        silx.gui.plot.PlotWidget.setInteractiveMode(self, mode, color=color, shape=shape, label=label, zoomOnWheel=zoomOnWheel, source=source, width=width)

    def peakInteractiveMode(self):
        """Returns the peak interactive mode selected."""
        return self.__mode

    def setPeakInteractiveMode(self, mode):
        if self.__mode == mode:
            return
        self.__mode = mode

        if mode == self.PEAK_SELECTION_MODE:
            super(_PeakPickingPlot, self).setInteractiveMode('zoom')
        elif mode == self.ERASOR_MODE:
            color = "black"
            super(_PeakPickingPlot, self).setInteractiveMode('draw', shape='rectangle', source=self, color=color)
        elif mode == self.BRUSH_MODE:
            color = "black"
            super(_PeakPickingPlot, self).setInteractiveMode('draw', shape='rectangle', source=self, color=color)
        else:
            assert(False)

    def setSelectedPeak(self, name):
        if self.__selectedPeak == name:
            return
        self.__selectedPeak = name
        for peakModel in self.__peakSelectionModel:
            self.updatePeak(peakModel)

    def eventFilter(self, widget, event):
        if event.type() == qt.QEvent.Enter:
            if self.__mode == self.PEAK_SELECTION_MODE:
                self.setCursor(qt.Qt.CrossCursor)
            else:
                self.setCursor(qt.Qt.ArrowCursor)
            return True
        elif event.type() == qt.QEvent.Leave:
            self.unsetCursor()
            return True
        return False

    def event(self, event):
        """
        Dispatch Qt events to the widget.

        :param qt.QEvent event: Event from Qt
        """
        if event.type() == qt.QEvent.ToolTip:
            handle = self.getWidgetHandle()
            pos = handle.mapFromGlobal(event.globalPos())
            coord = self.pixelToData(pos.x(), pos.y())
            # About 4 pixels (screen relative)
            coord2 = self.pixelToData(pos.x() + 1, pos.y() + 1)
            ratio = abs(coord[0] - coord2[0]), abs(coord[1] - coord2[1])
            threshold = 2 * (ratio[0] + ratio[1])
            peak = self.__peakSelectionModel.closestGroup((coord[1], coord[0]), threshold=threshold)
            if peak is not None:
                message = "Group name: %s\nRing number: %d"
                message = message % (peak.name(), peak.ringNumber())
                qt.QToolTip.showText(event.globalPos(), message)
            else:
                qt.QToolTip.hideText()
                event.ignore()

            return True
        return super(_PeakPickingPlot, self).event(event)

    def __onPlotEvent(self, event):
        if self.__mode == self.PEAK_SELECTION_MODE:
            if event["event"] == "imageClicked":
                x, y, button = event["col"], event["row"], event["button"]
                if button == "left":
                    self.__plotClicked(x, y)
        elif self.__mode in [self.ERASOR_MODE, self.BRUSH_MODE]:
            if event['event'] == 'drawingFinished':
                # Convert from plot to array coords
                ox, oy = 0, 0
                sx, sy = 1.0, 1.0

                height = int(abs(event['height'] / sy))
                width = int(abs(event['width'] / sx))

                row = int((event['y'] - oy) / sy)
                if sy < 0:
                    row -= height

                col = int((event['x'] - ox) / sx)
                if sx < 0:
                    col -= width

                # Use a shape in case we generalize it
                # FIXME: This code should be done in silx
                shape = Shape('rectangle')
                points = numpy.array([[col, row], [col + width, row + height]])
                shape.setPoints(points, copy=False)
                if self.__mode == self.ERASOR_MODE:
                    self.sigShapeErased.emit(shape)
                elif self.__mode == self.BRUSH_MODE:
                    self.sigShapeBrushed.emit(shape)
                else:
                    assert(False)
        else:
            assert(False)

    def __plotClicked(self, x, y):
        self.sigPeakPicked.emit(x, y)

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
        try:
            self.removeMarker(legend=legend)
        except Exception:
            pass
        legend = "coord" + peakModel.name()
        self.removeCurve(legend=legend)

    def addPeak(self, peakModel):
        color = peakModel.color()
        if not peakModel.isEnabled():
            context = CalibrationContext.instance()
            color = context.disabledMarkerColor()
        numpyColor = numpy.array([color.redF(), color.greenF(), color.blueF(), 0.5])
        points = peakModel.coords()
        name = peakModel.name()

        if self.__selectedPeak is None:
            # Nothing selected
            symbol = 'o'
        elif name == self.__selectedPeak:
            # Selected marker
            symbol = 'o'
        else:
            # Unselected marker
            symbol = '+'

        if len(points) != 0:
            y, x = points[0] + 0.5
            self.addMarker(x=x, y=y,
                           legend="marker" + name,
                           text=name)
        yy = points[:, 0] + 0.5
        xx = points[:, 1] + 0.5
        self.addCurve(x=xx, y=yy,
                      legend="coord" + name,
                      linestyle=' ',
                      selectable=False,
                      symbol=symbol,
                      color=numpyColor,
                      resetzoom=False)

    def setProcessingLocation(self, mask):
        """Update the processing location over the image.

        :param numpy.ndarray mask: Mask of the location.
        """
        if mask is None:
            mask = numpy.empty(shape=(0, 0))

        if self.__mask is None:
            self.addImage(mask,
                          legend="processing-mask",
                          selectable=False,
                          copy=False,
                          z=-0.5,
                          resetzoom=False)
            self.__mask = self.getImage("processing-mask")
            colormap = colors.Colormap(name=None,
                                       colors=((0., 0., 0., 0.), (1., 1., 1., 0.5)),
                                       vmin=0,
                                       vmax=1)
            self.__mask.setColormap(colormap)

        mask = mask.astype(numpy.uint8)
        self.__mask.setData(mask)

    def updatePeak(self, peakModel):
        self.removePeak(peakModel)
        self.addPeak(peakModel)

    def unsetProcessing(self):
        if self.__processing is not None:
            self.__processing.deleteLater()
        self.__processing = None

    def setProcessing(self):
        self.__processing = ProcessingWidget.createProcessingWidgetOverlay(self)


class _SpinBoxItemDelegate(qt.QStyledItemDelegate):

    def __init__(self, *args, **kwargs):
        qt.QStyledItemDelegate.__init__(self, *args, **kwargs)
        self.__palette = None

    def setPalette(self, palette):
        self.__palette = qt.QPalette(palette)

    def createEditor(self, parent, option, index):
        if not index.isValid():
            return super(_SpinBoxItemDelegate, self).createEditor(parent, option, index)

        editor = AdvancedSpinBox(parent=parent)
        if self.__palette is not None:
            editor.setPalette(self.__palette)
        editor.setMouseWheelEnabled(False)
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


class _PeakEnabledItemDelegate(qt.QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        if not index.isValid():
            return super(_PeakToolItemDelegate, self).createEditor(parent, option, index)

        persistantIndex = qt.QPersistentModelIndex(index)

        editor = ColoredCheckBox(parent=parent)
        editor.toggled.connect(functools.partial(self.__toggleEnabled, persistantIndex))

        return editor

    def setEditorData(self, editor, index):
        """
        :param qt.QWidget editor: Editor widget
        :param qt.QIndex index: Index of the data to display
        """
        model = index.model()
        peak = model.peakObject(index)
        old = editor.blockSignals(True)
        editor.setChecked(peak.isEnabled())
        editor.blockSignals(old)
        editor.setBoxColor(peak.color())

    def __toggleEnabled(self, index):
        model = index.model()
        peak = model.peakObject(index)
        newValue = not peak.isEnabled()
        newValue = qt.Qt.Checked if newValue else qt.Qt.Unchecked
        model.setData(index, newValue, role=qt.Qt.CheckStateRole)

    def updateEditorGeometry(self, editor, option, index):
        # Center the widget to the cell
        size = editor.sizeHint()
        half = size / 2
        halfPoint = qt.QPoint(half.width(), half.height())
        pos = option.rect.center() - halfPoint
        editor.move(pos)


class _PeakToolItemDelegate(qt.QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        if not index.isValid():
            return super(_PeakToolItemDelegate, self).createEditor(parent, option, index)

        editor = qt.QToolBar(parent=parent)
        editor.setIconSize(qt.QSize(20, 20))
        editor.setStyleSheet("QToolBar { border: 0px }")

        persistantIndex = qt.QPersistentModelIndex(index)

        extract = qt.QAction(editor)
        extract.setToolTip("Re-extract peaks from this ring")
        extract.setIcon(icons.getQIcon("pyfai:gui/icons/extract-ring"))
        extract.triggered.connect(functools.partial(self.__extractPeak, persistantIndex))
        editor.addAction(extract)

        remove = qt.QAction(editor)
        remove.setToolTip("Remove this group of peaks")
        remove.setIcon(icons.getQIcon("pyfai:gui/icons/remove-peak"))
        remove.triggered.connect(functools.partial(self.__removePeak, persistantIndex))
        editor.addAction(remove)

        editor.setMinimumSize(editor.sizeHint())
        editor.setMaximumSize(editor.sizeHint())
        editor.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)
        return editor

    def updateEditorGeometry(self, editor, option, index):
        """
        Update the geometry of the editor according to the changes of the view.

        :param qt.QWidget editor: Editor widget
        :param qt.QStyleOptionViewItem option: Control how the editor is shown
        :param qt.QIndex index: Index of the data to display
        """
        editor.setGeometry(option.rect)

    def getTask(self):
        """
        :rtype: PeakPickingTask
        """
        widget = self
        while widget is not None:
            if isinstance(widget, PeakPickingTask):
                return widget
            widget = widget.parent()
        raise TypeError("PeakPickingTask not found")

    def __extractPeak(self, persistantIndex, checked):
        if not persistantIndex.isValid():
            return
        model = persistantIndex.model()
        peak = model.peakObject(persistantIndex)
        task = self.getTask()
        if task is not None:
            task.autoExtractSingleRing(peak)

    def __removePeak(self, persistantIndex, checked):
        if not persistantIndex.isValid():
            return
        model = persistantIndex.model()
        model.removeRow(persistantIndex.row(), persistantIndex.parent())


class _RingSelectionBehaviour(qt.QObject):
    """Manage behaviour relative to ring selection.

    This ensure coherence between many widgets.

    - If "always new ring" activated
        - The spinner is diabled
        - The spinner have to display the number of the next ring created.
        - The spinner value is decorelated from the hilighted ring
            of the table view
    - Else
        - The spinner is enabled
        - The value of the spinner have to be consistant with the hilighted
            ring from the table view.
    """

    def __init__(self, parent,
                 peakSelectionModel,
                 spinnerRing,
                 newRingOption,
                 ringSelectionModel,
                 plot):
        qt.QObject.__init__(self, parent)
        self.__peakSelectionModel = peakSelectionModel
        self.__spinnerRing = spinnerRing
        self.__newRingOption = newRingOption
        self.__ringSelectionModel = ringSelectionModel

        self.__peakSelectionModel.changed.connect(self.__peaksHaveChanged)
        self.__peakSelectionModel.structureChanged.connect(self.__peaksStructureHaveChanged)
        self.__spinnerRing.valueChanged.connect(self.__spinerRingChanged)
        self.__newRingOption.toggled.connect(self.__newRingToggeled)
        if self.__ringSelectionModel is not None:
            self.__ringSelectionModel.selectionChanged.connect(self.__ringSelectionChanged)
            self.__ringSelectionModel.selectionChanged.connect(self.__hightlightedRingChnaged)

        self.__plot = plot
        self.__initState()

    def incRing(self):
        """Select the next ring. The auto selectection will be disabled."""
        ringNumber = self.__spinnerRing.value()
        ringNumber = ringNumber + 1
        self.selectRing(ringNumber)

    def decRing(self):
        """Select the next ring. The auto selection will be disabled."""
        ringNumber = self.__spinnerRing.value()
        ringNumber = ringNumber - 1
        if ringNumber > 0:
            self.selectRing(ringNumber)

    def selectRing(self, ringNumber):
        """Select one of the rings.

        The tools are updated to edit/create this ring.

        :param int ringNumber: The ring number to select
        """
        if self.__newRingOption.isChecked():
            self.__newRingOption.trigger()
        self.__spinnerRing.setValue(ringNumber)

    def toggleNewRing(self):
        """Toggle the "new ring" modificator.
        """
        self.__newRingOption.trigger()

    def clear(self):
        self.__peakSelectionModel.changed.disconnect(self.__peaksHaveChanged)
        self.__peakSelectionModel.structureChanged.disconnect(self.__peaksStructureHaveChanged)
        self.__spinnerRing.valueChanged.disconnect(self.__spinerRingChanged)
        self.__newRingOption.toggled.disconnect(self.__newRingToggeled)
        if self.__ringSelectionModel is not None:
            self.__ringSelectionModel.selectionChanged.disconnect(self.__ringSelectionChanged)
            self.__ringSelectionModel.selectionChanged.disconnect(self.__hightlightedRingChnaged)

    def __initState(self):
        self.__newRingToggeled()

    def __peaksStructureHaveChanged(self):
        self.__plot.setSelectedPeak(None)

    def __peaksHaveChanged(self):
        if self.__newRingOption.isChecked():
            self.__updateNewRing()

        if not self.__ringSelectionModel.hasSelection():
            # Update the model selection if nothing was selected
            # TODO: It would be good to remove the timer,
            #       but this event is generated before the update of the model
            qt.QTimer.singleShot(0, self.__spinerRingChanged)

    def __updateNewRing(self):
        createNewRing = False
        indexes = self.__ringSelectionModel.selectedIndexes()
        if len(indexes) == 0:
            createNewRing = True

        peakSelectionModel = self.__peakSelectionModel
        if createNewRing or self.__newRingOption.isChecked():
            # reach bigger ring
            ringNumbers = [p.ringNumber() for p in peakSelectionModel]
            if ringNumbers == []:
                lastRingNumber = 0
            else:
                lastRingNumber = max(ringNumbers)
            ringNumber = lastRingNumber + 1
        else:
            assert(len(indexes))
            index = indexes[0]
            model = self.__ringSelectionModel.model()
            index = model.index(index.row(), 0)
            peak = model.peakObject(index)
            ringNumber = peak.ringNumber()

        self.__spinnerRing.valueChanged.disconnect(self.__spinerRingChanged)
        self.__spinnerRing.setValue(ringNumber)
        self.__spinnerRing.valueChanged.connect(self.__spinerRingChanged)

    def ringNumber(self):
        """Returns the targetted ring.

        :rtype: int
        """
        return self.__spinnerRing.value()

    def __hightlightedRingChnaged(self):
        indexes = self.__ringSelectionModel.selectedIndexes()
        model = self.__ringSelectionModel.model()
        if len(indexes) == 0:
            peak = None
        else:
            index = indexes[0]
            peak = model.peakObject(index)

        if peak is not None:
            name = peak.name()
            self.__plot.setSelectedPeak(name)
        else:
            self.__plot.setSelectedPeak(None)

    def __ringSelectionChanged(self):
        indexes = self.__ringSelectionModel.selectedIndexes()
        model = self.__ringSelectionModel.model()
        if len(indexes) == 0:
            peak = None
        else:
            index = indexes[0]
            peak = model.peakObject(index)

        if not self.__newRingOption.isChecked():
            # It have to be updated
            if peak is not None:
                self.__spinnerRing.valueChanged.disconnect(self.__spinerRingChanged)
                try:
                    ringNumber = peak.ringNumber()
                    self.__spinnerRing.setValue(ringNumber)
                finally:
                    self.__spinnerRing.valueChanged.connect(self.__spinerRingChanged)

    def __spinerRingChanged(self):
        """Called when the spinner displaying the selected ring changes."""
        ringNumber = self.__spinnerRing.value()
        if self.__ringSelectionModel is None:
            return

        model = self.__ringSelectionModel.model()
        self.__ringSelectionModel.selectionChanged.disconnect(self.__ringSelectionChanged)
        try:
            for i in range(model.rowCount()):
                index = model.index(i, 0)
                peak = model.peakObject(index)
                if peak.ringNumber() == ringNumber:
                    break
            else:
                i = None
            if i is not None:
                index = i
                indexStart = model.index(index, 0)
                indexEnd = model.index(index, model.columnCount() - 1)
                selection = qt.QItemSelection(indexStart, indexEnd)
                self.__ringSelectionModel.select(selection, qt.QItemSelectionModel.ClearAndSelect)
            else:
                self.__ringSelectionModel.clear()
        finally:
            self.__ringSelectionModel.selectionChanged.connect(self.__ringSelectionChanged)

    def __newRingToggeled(self):
        """Called when the new ring option is toggled."""
        newRingActivated = self.__newRingOption.isChecked()
        self.__spinnerRing.setEnabled(not newRingActivated)
        if self.__newRingOption.isChecked():
            self.__updateNewRing()
        else:
            self.__ringSelectionChanged()


class PeakPickingTask(AbstractCalibrationTask):

    def _initGui(self):
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-peakpicking.ui"), self)
        icon = silx.gui.icons.getQIcon("pyfai:gui/icons/task-identify-rings")
        self.setWindowIcon(icon)

        self.initNextStep()

        # Insert the plot on the layout
        holder = self._plotHolder
        self.__extractionThread = None
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
        toolBar = self.__createMainRingToolBar()
        layout.addWidget(toolBar)
        toolBar = self.__createRingToolBar()
        layout.addWidget(toolBar)

        self.__createPlotToolBar(self.__plot)
        statusBar = self.__createPlotStatusBar(self.__plot)
        self.__plot.setStatusBar(statusBar)

        self.__plot.sigPeakPicked.connect(self.__onPickPicked)
        self.__plot.sigShapeErased.connect(self.__onShapeErased)
        self.__plot.sigShapeBrushed.connect(self.__onShapeBrushed)
        self.__plot.sigInteractiveModeChanged.connect(self.__onPlotModeChanged)

        action = qt.QAction(self)
        action.setText("Extract rings until")
        action.setToolTip("Remove all the rings and extract it again")
        action.setIcon(icons.getQIcon("pyfai:gui/icons/extract-rings-to"))
        action.triggered.connect(self.__autoExtractRings)
        selectAction = self._extract.addDefaultAction(action)
        selectAction.triggered.connect(self.__updateOptionToExtractAgain)

        action = qt.QAction(self)
        action.setText("Extract already picked rings")
        action.setToolTip("Duplicated rings will be removed")
        action.setIcon(icons.getQIcon("pyfai:gui/icons/extract-current-rings"))
        action.triggered.connect(self.__autoExtractExistingRings)
        self._extract.addDefaultAction(action)

        action = qt.QAction(self)
        action.setText("Extract all reachable rings")
        action.setToolTip("Remove all the rings and extract everything possible")
        action.setIcon(icons.getQIcon("pyfai:gui/icons/extract-reachable-rings"))
        action.triggered.connect(self.__autoExtractReachableRings)
        self._extract.addDefaultAction(action)

        action = qt.QAction(self)
        action.setText("Extract more rings")
        action.setToolTip("Extract new rings after the last picked one")
        action.setIcon(icons.getQIcon("pyfai:gui/icons/extract-more-rings"))
        action.triggered.connect(self.__autoExtractMoreRings)
        selectAction = self._extract.addDefaultAction(action)
        selectAction.triggered.connect(self.__updateOptionToExtractMoreRings)
        self.__updateOptionToExtractMoreRings()
        moreAction = action

        action = qt.QAction(self)
        action.setText("Merge rings and sort")
        action.setToolTip("Merge the groups using the same ring number and sort them")
        action.setIcon(icons.getQIcon("silx:gui/icons/draw-brush"))
        action.triggered.connect(self.__cleanUpRings)
        self._extract.addAction(action)

        self._extract.setEnabled(False)
        self._extract.setDefaultAction(moreAction)

        validator = validators.DoubleValidator(self)
        self._numberOfPeakPerDegree.lineEdit().setValidator(validator)
        locale = qt.QLocale(qt.QLocale.C)
        self._numberOfPeakPerDegree.setLocale(locale)

        self.__synchronizeRawView = SynchronizeRawView()
        self.__synchronizeRawView.registerTask(self)
        self.__synchronizeRawView.registerPlot(self.__plot)

        self.__ringSelection = None
        self.__massif = None
        self.__massifReconstructed = None

        for i, key in enumerate(range(qt.Qt.Key_0, qt.Qt.Key_9 + 1)):
            if i == 0:
                i = 10
            action = qt.QAction(self)
            action.setText("Select ring %d" % i)

            def selectRing(ringNumber):
                self.__ringSelection.selectRing(ringNumber)
            action.triggered.connect(functools.partial(selectRing, i))
            action.setShortcut(qt.QKeySequence(key))
            self.addAction(action)

        action = qt.QAction(self)
        action.setText("Select the next ring")
        action.triggered.connect(lambda: self.__ringSelection.incRing())
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_Plus))
        self.addAction(action)

        action = qt.QAction(self)
        action.setText("Select the previous ring")
        action.triggered.connect(lambda: self.__ringSelection.decRing())
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_Minus))
        self.addAction(action)

        action = qt.QAction(self)
        action.setText("Toggle new ring tool")
        action.triggered.connect(lambda: self.__ringSelection.toggleNewRing())
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_Equal))
        self.addAction(action)

    def __onPlotModeChanged(self, owner):
        # TODO: This condition should not be reached like that
        if owner is not self.__plot:
            # Here a default plot tool is triggered
            # Set back the default tool
            if (not self.__arcSelectionMode.isChecked() and
                    not self.__ringSelectionMode.isChecked() and
                    not self.__peakSelectionMode.isChecked()):
                self.__arcSelectionMode.trigger()

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
                MessageBox.exception(self, "Error while loading peaks", e, _logger)
            except KeyboardInterrupt:
                raise

    def __savePeaksAsFile(self):
        dialog = self.__createSavePeakDialog()

        result = dialog.exec_()
        if not result:
            return

        filename = dialog.selectedFiles()[0]
        nameFilter = dialog.selectedNameFilter()
        isNptFilter = ".npt" in nameFilter

        if isNptFilter and not filename.endswith(".npt"):
            filename = filename + ".npt"
        try:
            controlPoints = model_transform.createControlPoints(self.model())
            controlPoints.save(filename)
        except Exception as e:
            MessageBox.exception(self, "Error while saving peaks", e, _logger)
        except KeyboardInterrupt:
            raise

    def __createMainRingToolBar(self):
        toolBar = qt.QToolBar(self)

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

        def createIcon(identifiyers):
            for i in identifiyers:
                if isinstance(i, six.string_types):
                    if qt.QIcon.hasThemeIcon(i):
                        return qt.QIcon.fromTheme(i)
                elif isinstance(i, qt.QIcon):
                    return i
                else:
                    return style.standardIcon(i)
            return qt.QIcon()

        action = self.__undoStack.createUndoAction(self, "Undo")
        action.setShortcut(qt.QKeySequence.Undo)
        icon = createIcon(["edit-undo", qt.QStyle.SP_ArrowBack])
        action.setIcon(icon)
        toolBar.addAction(action)

        action = self.__undoStack.createRedoAction(self, "Redo")
        action.setShortcut(qt.QKeySequence.Redo)
        icon = createIcon(["edit-redo", qt.QStyle.SP_ArrowForward])
        action.setIcon(icon)
        toolBar.addAction(action)

        return toolBar

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

        toolBar.addSeparator()

        action = qt.QAction(self)
        action.setIcon(icons.getQIcon("silx:gui/icons/draw-brush"))
        action.setText("Brush")
        action.setCheckable(True)
        action.setToolTip("Change the ring number to a set of already identified peaks")
        toolBar.addAction(action)
        self.__brushMode = action

        action = qt.QAction(self)
        action.setIcon(icons.getQIcon("silx:gui/icons/draw-rubber"))
        action.setText("Rubber")
        action.setCheckable(True)
        action.setToolTip("Remove a set of already identified peaks")
        toolBar.addAction(action)
        self.__erasorMode = action

        toolBar.addSeparator()

        action = qt.QAction(self)
        action.setIcon(icons.getQIcon("pyfai:gui/icons/new-ring"))
        action.setText("+")
        action.setCheckable(True)
        action.setChecked(True)
        action.setToolTip("Create always a new ring when a peak is picked")
        toolBar.addAction(action)
        self.__createNewRingOption = action

        spiner = qt.QSpinBox(self)
        spiner.setRange(1, 9999)
        spiner.setToolTip("Ring to edit")
        toolBar.addWidget(spiner)
        self.__selectedRingNumber = spiner

        mode = qt.QActionGroup(self)
        mode.setExclusive(True)
        mode.addAction(self.__ringSelectionMode)
        mode.addAction(self.__arcSelectionMode)
        mode.addAction(self.__peakSelectionMode)
        mode.addAction(self.__erasorMode)
        mode.addAction(self.__brushMode)
        mode.triggered.connect(self.__requestChangeMode)
        self.__arcSelectionMode.setChecked(True)

        return toolBar

    def __requestChangeMode(self, action):
        if action is self.__erasorMode:
            self.__plot.setPeakInteractiveMode(_PeakPickingPlot.ERASOR_MODE)
        elif action is self.__brushMode:
            self.__plot.setPeakInteractiveMode(_PeakPickingPlot.BRUSH_MODE)
        elif (action is self.__ringSelectionMode or
              action is self.__arcSelectionMode or
              action is self.__peakSelectionMode):
            self.__plot.setPeakInteractiveMode(_PeakPickingPlot.PEAK_SELECTION_MODE)
        else:
            assert(False)

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
            self.__massifReconstructed.log_info = False
            return self.__massifReconstructed
        elif self.__arcSelectionMode.isChecked() or self.__peakSelectionMode.isChecked():
            if self.__massif is None:
                self.__massif = self.__createMassif()
            self.__massif.log_info = False
            return self.__massif
        else:
            assert(False)

    def __findPeaks(self, x, y):
        """
        Returns peaks around the location x, y
        """
        image = self.model().experimentSettingsModel().image().value()
        massif = self.__getMassif()
        if massif is None:
            # Nothing to pick
            return
        points = massif.find_peaks([y, x], stdout=None)
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
            points = massif.find_peaks([y, x], stdout=None)

        # filter peaks from the mask
        mask = self.model().experimentSettingsModel().mask().value()
        if mask is not None:
            points = filter(lambda coord: mask[int(coord[0]), int(coord[1])] == 0, points)
            points = list(points)

        return points

    def __findSinglePeak(self, x, y):
        """
        Returns a single peak a location x, y
        """
        points = self.__findPeaks(x, y)
        if len(points) > 1:
            points = points[0:1]
        return points

    def __onPickPicked(self, x, y):

        if self.__peakSelectionMode.isChecked():
            points = self.__findSinglePeak(x, y)
        else:
            points = self.__findPeaks(x, y)

        if len(points) == 0:
            return

        peakSelectionModel = self.model().peakSelectionModel()
        ringNumber = self.__ringSelection.ringNumber()
        points = numpy.array(points)
        peakModel = model_transform.createRing(points, peakSelectionModel, ringNumber=ringNumber)
        oldState = self.__copyPeaks(self.__undoStack)
        peak = peakSelectionModel.peakFromRingNumber(ringNumber)
        if peak is None:
            peakSelectionModel.append(peakModel)
        else:
            peak.mergeCoords(peakModel)
        newState = self.__copyPeaks(self.__undoStack)
        command = _PeakSelectionUndoCommand(None, peakSelectionModel, oldState, newState)
        command.setText("add peak %s" % peakModel.name())
        command.setRedoInhibited(True)
        self.__undoStack.push(command)
        command.setRedoInhibited(False)

    def __onShapeBrushed(self, shape):
        """
        Callback when brushing peaks is requested.

        :param Shape shape: A shape containing peaks to erase
        """
        if shape.getType() == "rectangle":
            points = shape.getPoints()
            minCoord = points.min(axis=0)
            maxCoord = points.max(axis=0)

            def erasePeaksFromRect(x, y):
                return not (minCoord[1] < x < maxCoord[1] and
                            minCoord[0] < y < maxCoord[0])

            brushedPeaks = []
            oldState = self.__copyPeaks(self.__undoStack)
            peakSelectionModel = self.model().peakSelectionModel()
            model_transform.filterControlPoints(erasePeaksFromRect,
                                                peakSelectionModel,
                                                removedPeaks=brushedPeaks)
            if len(brushedPeaks) == 0:
                _logger.debug("No peak to brush")
                return
            brushedPeaks = numpy.array(brushedPeaks)
            ringNumber = self.__ringSelection.ringNumber()
            peak = peakSelectionModel.peakFromRingNumber(ringNumber)
            if peak is None:
                peakModel = model_transform.createRing(brushedPeaks, peakSelectionModel, ringNumber=ringNumber)
                peakSelectionModel.append(peakModel)
            else:
                peak.mergeCoords(brushedPeaks)

            newState = self.__copyPeaks(self.__undoStack)
            command = _PeakSelectionUndoCommand(None, peakSelectionModel, oldState, newState)
            command.setText("erase peaks with rubber tool")
            command.setRedoInhibited(True)
            self.__undoStack.push(command)
            command.setRedoInhibited(False)
        else:
            assert(False)

    def __onShapeErased(self, shape):
        """
        Callback when erasing peaks is requested.

        :param Shape shape: A shape containing peaks to erase
        """
        if shape.getType() == "rectangle":
            points = shape.getPoints()
            minCoord = points.min(axis=0)
            maxCoord = points.max(axis=0)

            def erasePeaksFromRect(x, y):
                return not (minCoord[1] < x < maxCoord[1] and
                            minCoord[0] < y < maxCoord[0])

            removedPeaks = []
            oldState = self.__copyPeaks(self.__undoStack)
            peakSelectionModel = self.model().peakSelectionModel()
            model_transform.filterControlPoints(erasePeaksFromRect,
                                                peakSelectionModel,
                                                removedPeaks=removedPeaks)
            if len(removedPeaks) == 0:
                _logger.debug("No peak to remove")
                return
            newState = self.__copyPeaks(self.__undoStack)
            command = _PeakSelectionUndoCommand(None, peakSelectionModel, oldState, newState)
            command.setText("erase peaks with rubber tool")
            command.setRedoInhibited(True)
            self.__undoStack.push(command)
            command.setRedoInhibited(False)
        else:
            assert(False)

    def __removePeak(self, peakModel):
        oldState = self.__copyPeaks(self.__undoStack)
        self.model().peakSelectionModel().remove(peakModel)
        newState = self.__copyPeaks(self.__undoStack)
        command = _PeakSelectionUndoCommand(None, self.model().peakSelectionModel(), oldState, newState)
        command.setText("remove peak %s" % peakModel.name())
        command.setRedoInhibited(True)
        self.__undoStack.push(command)
        command.setRedoInhibited(False)

    def __setRingEnable(self, peakModel, value):
        oldState = self.__copyPeaks(self.__undoStack)
        peakModel.setEnabled(value)
        newState = self.__copyPeaks(self.__undoStack)
        command = _PeakSelectionUndoCommand(None, self.model().peakSelectionModel(), oldState, newState)
        action = "enable" if value else "disable"
        command.setText("%s ring %s" % (action, peakModel.name()))
        command.setRedoInhibited(True)
        self.__undoStack.push(command)
        command.setRedoInhibited(False)

    def __setRingNumber(self, peakModel, value):
        oldState = self.__copyPeaks(self.__undoStack)
        context = CalibrationContext.instance()
        color = context.getMarkerColor(value - 1)
        with peakModel.lockContext():
            peakModel.setRingNumber(value)
            peakModel.setColor(color)
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

    def _createRingExtractor(self, ring=None, existingRings=False, reachableRings=False, moreRings=None):
        """Create a ring extractor according to some params.

        :param Union[int,None] ring: If set the extraction is only executed on
            a single ring
        :param bool existingRings: If true, the extractor is configured to only
            extract existing rings
        :param Union[int,None] moreRings: If defined, extract more rings that
            was not yet extracted
        :param bool reachableRings: If true, reach all reachable rings
        """
        extractor = RingExtractorThread(self)
        experimentSettings = self.model().experimentSettingsModel()
        extractor.setExperimentSettings(experimentSettings, copy=False)
        extractor.sigProcessLocationChanged.connect(self.__autoExtractLocationChanged)

        # Constant dependant of the ui file
        FROM_PEAKS = 0
        FROM_FIT = 1

        if reachableRings:
            maxRings = None
            ringNumbers = None
        elif moreRings is not None:
            maxRings = None
            peaksModel = self.model().peakSelectionModel()
            ringNumbers = [p.ringNumber() for p in peaksModel]
            maxRing = max(ringNumbers)
            ringNumbers = list(range(maxRing + 1, maxRing + 1 + moreRings))
        elif existingRings:
            maxRings = None
            peaksModel = self.model().peakSelectionModel()
            ringNumbers = [p.ringNumber() for p in peaksModel]
            ringNumbers = set(ringNumbers)
            ringNumbers = list(ringNumbers)
            ringNumbers = sorted(ringNumbers)
        elif ring is None:
            maxRings = self._maxRingToExtract.value()
            ringNumbers = None
        else:
            maxRings = self._maxRingToExtract.value()
            ringNumbers = [ring.ringNumber()]

        pointPerDegree = self._numberOfPeakPerDegree.value()
        extractor.setMaxRings(maxRings)
        extractor.setRingNumbers(ringNumbers)
        extractor.setPointPerDegree(pointPerDegree)

        geometrySourceIndex = self._geometrySource.currentIndex()
        if geometrySourceIndex == FROM_PEAKS:
            peaksModel = self.model().peakSelectionModel()
            extractor.setPeaksModel(peaksModel)
        elif geometrySourceIndex == FROM_FIT:
            geometryModel = self.model().fittedGeometry()
            extractor.setGeometryModel(geometryModel)
        else:
            assert(False)

        return extractor

    EXTRACT_ALL = "extract-all"
    EXTRACT_SINGLE = "extract-single"
    EXTRACT_EXISTING = "extract-existing"
    EXTRACT_MORE = "extract-more"

    def __autoExtractRings(self):
        thread = self._createRingExtractor(ring=None)
        thread.setUserData("ROLE", self.EXTRACT_ALL)
        thread.setUserData("TEXT", "extract rings")
        self.__startExtractThread(thread)

    def __autoExtractReachableRings(self):
        thread = self._createRingExtractor(reachableRings=True)
        thread.setUserData("ROLE", self.EXTRACT_ALL)
        thread.setUserData("TEXT", "extract reachable rings")
        self.__startExtractThread(thread)

    def __autoExtractExistingRings(self):
        thread = self._createRingExtractor(existingRings=True)
        thread.setUserData("ROLE", self.EXTRACT_EXISTING)
        thread.setUserData("TEXT", "extract existing rings")
        self.__startExtractThread(thread)

    def autoExtractSingleRing(self, ring):
        thread = self._createRingExtractor(ring=ring)
        thread.setUserData("ROLE", self.EXTRACT_SINGLE)
        thread.setUserData("TEXT", "extract ring %d" % ring.ringNumber())
        thread.setUserData("RING", ring)
        self.__startExtractThread(thread)

    def __autoExtractMoreRings(self):
        value = self._moreRingToExtract.value()
        thread = self._createRingExtractor(moreRings=value)
        thread.setUserData("ROLE", self.EXTRACT_MORE)
        thread.setUserData("TEXT", "extract %s more rings" % value)
        self.__startExtractThread(thread)

    def __startExtractThread(self, thread):
        if self.__extractionThread is not None:
            _logger.error("A task to extract rings is already processing")
            return
        thread.started.connect(self.__extractionStarted)
        thread.finished.connect(functools.partial(self.__extractionFinishedSafe, thread))
        thread.finished.connect(thread.deleteLater)
        thread.start()
        self.__extractionThread = thread

    def __autoExtractLocationChanged(self, mask):
        self.__plot.setProcessingLocation(mask)

    def __updateOptionToExtractAgain(self):
        self._moreRingToExtractTitle.setVisible(False)
        self._moreRingToExtract.setVisible(False)
        self._maxRingToExtractTitle.setVisible(True)
        self._maxRingToExtract.setVisible(True)

    def __updateOptionToExtractMoreRings(self):
        self._moreRingToExtractTitle.setVisible(True)
        self._moreRingToExtract.setVisible(True)
        self._maxRingToExtractTitle.setVisible(False)
        self._maxRingToExtract.setVisible(False)

    def __extractionStarted(self):
        self.__plot.setProcessing()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        self._extract.setWaiting(True)

    def __extractionFinishedSafe(self, thread):
        """
        Compute the result of the processing

        :param RingExtractorThread thread: A ring ring extractor processing
        """
        errorMessage = None
        if thread.isAborted():
            errorMessage = thread.errorString()
        else:
            try:
                self.__extractionFinished(thread)
            except Exception as e:
                _logger.debug("Backtrace", exc_info=True)
                errorMessage = str(e)

        self.__plot.setProcessingLocation(None)
        self.__plot.unsetProcessing()
        qt.QApplication.restoreOverrideCursor()
        self._extract.setWaiting(False)
        if errorMessage is not None:
            qt.QMessageBox.critical(self, "Error", errorMessage)
        self.__extractionThread = None

    def __extractionFinished(self, thread):
        """
        Compute the result of the processing

        :param RingExtractorThread thread: A ring ring extractor processing
        """
        newPeaks = thread.resultPeaks()

        # update the gui
        oldState = self.__copyPeaks(self.__undoStack)
        peakSelectionModel = self.model().peakSelectionModel()
        role = thread.userData("ROLE")
        if role == self.EXTRACT_ALL:
            # Remove everything and recreate everything
            disabledRings = set([p.ringNumber() for p in peakSelectionModel if not p.isEnabled()])
            peakSelectionModel.clear()
            for ringNumber in sorted(newPeaks.keys()):
                coords = newPeaks[ringNumber]
                peakModel = model_transform.createRing(coords, peakSelectionModel, ringNumber=ringNumber)
                if ringNumber in disabledRings:
                    peakModel.setEnabled(False)
                peakSelectionModel.append(peakModel)
        elif role == self.EXTRACT_EXISTING:
            # Remove everything and recreate everything with the same name/color...
            ringNumbers = sorted(newPeaks.keys())
            disabledRings = set([p.ringNumber() for p in peakSelectionModel if not p.isEnabled()])
            peaks = [peakSelectionModel.peakFromRingNumber(n) for n in ringNumbers]
            peakSelectionModel.clear()
            for prevousRing in peaks:
                coords = newPeaks[prevousRing.ringNumber()]
                ringNumber = prevousRing.ringNumber()
                peakModel = model_transform.createRing(coords, peakSelectionModel, ringNumber=ringNumber)
                peakModel.setName(prevousRing.name())
                if prevousRing.ringNumber() in disabledRings:
                    peakModel.setEnabled(False)
                peakSelectionModel.append(peakModel)
        elif role == self.EXTRACT_MORE:
            # Append the extracted rings to the current ones
            for ringNumber in sorted(newPeaks.keys()):
                coords = newPeaks[ringNumber]
                peakModel = model_transform.createRing(coords, peakSelectionModel, ringNumber=ringNumber)
                peakSelectionModel.append(peakModel)
        elif role == self.EXTRACT_SINGLE:
            # Update coord of a single ring
            ringObject = thread.userData("RING")
            empty = numpy.empty(shape=(0, 2))
            coords = newPeaks.get(ringObject.ringNumber(), empty)
            ringObject.setCoords(coords)
        else:
            assert(False)
        newState = self.__copyPeaks(self.__undoStack)
        command = _PeakSelectionUndoCommand(None, peakSelectionModel, oldState, newState)
        text = thread.userData("TEXT")
        command.setText(text)
        command.setRedoInhibited(True)
        self.__undoStack.push(command)
        command.setRedoInhibited(False)

    def __cleanUpRings(self):
        """Clean up the picked rings"""
        oldState = self.__copyPeaks(self.__undoStack)
        peakSelectionModel = self.model().peakSelectionModel()

        peaks = {}
        for p in peakSelectionModel:
            ringNumber = p.ringNumber()
            if ringNumber in peaks:
                peaks[ringNumber].append(p)
            else:
                peaks[ringNumber] = [p]

        peakSelectionModel.clear()
        for ringNumber in sorted(peaks.keys()):
            peak = peaks[ringNumber][0]
            for p in peaks[ringNumber][1:]:
                peak.mergeCoords(p)
            peakSelectionModel.append(peak)

        newState = self.__copyPeaks(self.__undoStack)
        command = _PeakSelectionUndoCommand(None, peakSelectionModel, oldState, newState)
        command.setText("Clean up")
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
        if self.__ringSelection is not None:
            self.__ringSelection.clear()
        self.__ringSelection = _RingSelectionBehaviour(self,
                                                       self.model().peakSelectionModel(),
                                                       self.__selectedRingNumber,
                                                       self.__createNewRingOption,
                                                       self.__peakSelectionView.selectionModel(),
                                                       self.__plot)

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
        tableModel.requestChangeEnable.connect(self.__setRingEnable)
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

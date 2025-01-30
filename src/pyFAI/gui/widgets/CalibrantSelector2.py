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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "16/10/2020"

import time
import logging
from typing import List

from silx.gui import qt
from pyFAI.calibrant import Calibrant
from ..model.CalibrantModel import CalibrantModel
from .model.CalibrantFilterProxyModel import CalibrantFilterProxyModel
from .model.CalibrantItemModel import CalibrantItemModel
from ...utils import get_ui_file


_logger = logging.getLogger(__name__)


class _CalibrantItemView(qt.QAbstractItemView):
    """
    Custom view  used as popup view for the main combobox.
    """

    sigLoadFileRequested = qt.Signal()

    def __init__(self, parent=None):
        super(_CalibrantItemView, self).__init__(parent=parent)
        filename = get_ui_file("calibrant-selector2.ui")
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.__ui = qt.loadUi(filename)
        self.__ui.setParent(self)
        layout.addWidget(self.__ui)

        self.__lastUsed = []
        self.__dropTime = None

        self.__filter = CalibrantFilterProxyModel(self)
        self.__ui.listView.setModel(self.__filter)
        self.__ui.allFilter.clicked.connect(self.__selectAll)
        self.__ui.lastFilter.clicked.connect(self.__selectLast)
        self.__ui.userFilter.clicked.connect(self.__selectUser)
        self.__ui.defaultFilter.clicked.connect(self.__selectDefault)
        self.__ui.loadButton.clicked.connect(self.__loadFileRequested)
        self.__ui.listView.clicked.connect(self.__currentChanged)
        self.__ui.listView.activated.connect(self.__currentChanged)

        # self.setFocusProxy(self.__ui.listView)

    def focusInEvent(self, event: qt.QEvent):
        # Update the size when the model was initialized
        self.adjustSize()
        size = self.size()
        h = self.__ui.listView.sizeHintForRow(0)
        size.setHeight(h * 8)
        self.setFixedSize(size)

        if len(self.__lastUsed) > 0:
            self.__selectLast()
        else:
            self.__selectAll()
        self.__dropTime = time.time()
        self.__ui.listView.setFocus()

    def setRecentCalibrants(self, calibrants: List[str]):
        self.__lastUsed = calibrants
        self._syncLastUsed()

    def recentCalibrants(self) -> List[str]:
        return self.__lastUsed

    def touchCalibrant(self, calibrant):
        filename = calibrant.filename
        if filename is not None:
            try:
                self.__lastUsed.remove(filename)
            except ValueError:
                pass
            self.__lastUsed.insert(0, filename)
            self.__lastUsed = self.__lastUsed[:8]

    def __currentChanged(self, current: qt.QModelIndex):
        if time.time() - self.__dropTime < 0.200:
            # When a mouse press is performed on the combobox, directly followed
            # a mouse release, if the first item was selected, the change is
            # triggered
            # Here is a mitigation
            return
        sourceIndex = self.__filter.mapToSource(current)
        self.setCurrentIndex(sourceIndex)
        calibrant = sourceIndex.data(CalibrantItemModel.CALIBRANT_ROLE)
        self.touchCalibrant(calibrant)
        self.accept()

    def accept(self):
        """Send event to close and accept the popup"""
        event = qt.QKeyEvent(qt.QKeyEvent.KeyPress, qt.Qt.Key_Enter, qt.Qt.NoModifier, "x")
        qt.QApplication.sendEvent(self, event);

    def reject(self):
        """Send event to close and reject the popup"""
        event = qt.QKeyEvent(qt.QKeyEvent.KeyPress, qt.Qt.Key_Escape, qt.Qt.NoModifier, "x")
        qt.QApplication.sendEvent(self, event);

    def __loadFileRequested(self):
        self.sigLoadFileRequested.emit()

    def restoreState(self, state: qt.QByteArray) -> bool:
        stream = qt.QDataStream(state, qt.QIODevice.ReadOnly)
        version = stream.readUInt32()
        if version != 0:
            _logger.warning("Serial version mismatch. Found %d." % version)
            return False

        nb = stream.readUInt32()
        names = []
        for _ in range(nb):
            name = stream.readQString()
            names.append(name)
        self.__lastUsed = names
        self._syncLastUsed()
        return True

    def saveState(self) -> qt.QByteArray:
        data = qt.QByteArray()
        stream = qt.QDataStream(data, qt.QIODevice.WriteOnly)
        stream.writeUInt32(0)
        stream.writeUInt32(len(self.__lastUsed))
        for c in self.__lastUsed:
            stream.writeQString(c)
        return data

    def __clearAll(self):
        self.__ui.allFilter.blockSignals(True)
        self.__ui.allFilter.setChecked(False)
        self.__ui.allFilter.blockSignals(False)

        self.__ui.lastFilter.blockSignals(True)
        self.__ui.lastFilter.setChecked(False)
        self.__ui.lastFilter.blockSignals(False)

        self.__ui.userFilter.blockSignals(True)
        self.__ui.userFilter.setChecked(False)
        self.__ui.userFilter.blockSignals(False)

        self.__ui.defaultFilter.blockSignals(True)
        self.__ui.defaultFilter.setChecked(False)
        self.__ui.defaultFilter.blockSignals(False)

    def __selectAll(self):
        self.__clearAll()
        self.__filter.setFilter(displayResource=True, displayUser=True)
        self.__ui.allFilter.blockSignals(True)
        self.__ui.allFilter.setChecked(True)
        self.__ui.allFilter.blockSignals(False)

    def __selectLast(self):
        self.__clearAll()
        self.__filter.setFilter(displayResource=False, displayUser=False, filenames=set(self.__lastUsed))
        self.__ui.lastFilter.blockSignals(True)
        self.__ui.lastFilter.setChecked(True)
        self.__ui.lastFilter.blockSignals(False)

    def __selectUser(self):
        self.__clearAll()
        self.__filter.setFilter(displayResource=False, displayUser=True)
        self.__ui.userFilter.blockSignals(True)
        self.__ui.userFilter.setChecked(True)
        self.__ui.userFilter.blockSignals(False)

    def __selectDefault(self):
        self.__clearAll()
        self.__filter.setFilter(displayResource=True, displayUser=False)
        self.__ui.defaultFilter.blockSignals(True)
        self.__ui.defaultFilter.setChecked(True)
        self.__ui.defaultFilter.blockSignals(False)

    def visualRegionForSelection(self, selection: qt.QItemSelection):
        return qt.QRegion()

    def visualRect(self, index: qt.QModelIndex):
        return qt.QRect()

    def moveCursor(self, cursorAction, modifiers) -> qt.QModelIndex:
        return qt.QModelIndex()

    def scrollTo(self, index, hint):
        self.__ui.listView.scrollTo(index, hint)

    def indexAt(self, point: qt.QPoint) -> qt.QModelIndex:
        return self.__ui.listView.indexAt(point)

    def setModel(self, model: qt.QStandardItemModel):
        self.__filter.setSourceModel(model)
        qt.QAbstractItemView.setModel(self, model)
        self._syncLastUsed()

    def _syncLastUsed(self):
        model = self.model()
        if model is None:
            return
        for f in self.__lastUsed:
            calibrant = Calibrant(f)
            index = model.indexFromCalibrant(calibrant)
            if not index.isValid():
                model.appendCalibrant(calibrant)

    def horizontalOffset(self):
        return 0

    def verticalOffset(self):
        return 0


class CalibrantSelector2(qt.QComboBox):
    """Dropdown widget to select a calibrant.

    It is a view on top of a calibrant model (see :meth:`setCalibrantModel`,
    :meth:`modelCalibrant`)

    The calibrant can be selected from a list of calibrant known by pyFAI.
    """

    sigLoadFileRequested = qt.Signal()

    def __init__(self, parent=None):
        super(CalibrantSelector2, self).__init__(parent=parent)
        model = CalibrantItemModel(self)
        self.setModel(model)
        self.setCurrentIndex(-1)

        self.__calibrantModel: CalibrantModel = None
        self.setCalibrantModel(CalibrantModel())

        view = _CalibrantItemView(self)
        view.sigLoadFileRequested.connect(self.__loadFileRequested)
        self.setView(view)

        self.activated.connect(self.__calibrantWasSelected)

    def __calibrantWasSelected(self):
        index = self.currentIndex()
        if index == -1:
            calibrant = None
        else:
            calibrant = self.itemData(index, role=CalibrantItemModel.CALIBRANT_ROLE)
        self.__calibrantModel.setCalibrant(calibrant)

    def __modelChanged(self):
        calibrant = self.__calibrantModel.calibrant()
        model = self.model()
        if calibrant is None or model is None:
            self.setCurrentIndex(-1)
        else:
            index = model.indexFromCalibrant(calibrant)
            if not index.isValid():
                model.appendCalibrant(calibrant)
                index = model.indexFromCalibrant(calibrant)
            if index.isValid():
                self.view().touchCalibrant(calibrant)
                self.setCurrentIndex(index.row())

    def setCalibrantModel(self, model: CalibrantModel):
        if self.__calibrantModel is not None:
            self.__calibrantModel.changed.disconnect(self.__modelChanged)
        self.__calibrantModel = model
        if self.__calibrantModel is not None:
            self.__calibrantModel.changed.connect(self.__modelChanged)
        self.__modelChanged()

    def calibrantModel(self) -> CalibrantModel:
        return self.__calibrantModel

    def recentCalibrants(self):
        return self.view().recentCalibrants()

    def setRecentCalibrants(self, recentCalibrants: List[str]):
        return self.view().setRecentCalibrants(recentCalibrants)

    def restoreState(self, state: qt.QByteArray) -> bool:
        return self.view().restoreState(state)

    def saveState(self) -> qt.QByteArray:
        return self.view().saveState()

    def __loadFileRequested(self):
        self.sigLoadFileRequested.emit()

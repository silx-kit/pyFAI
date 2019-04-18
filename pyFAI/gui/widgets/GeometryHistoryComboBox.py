# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2019 European Synchrotron Radiation Facility
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
__date__ = "18/04/2019"

import logging

from silx.gui import qt
from pyFAI.utils import stringutil
from .AdvancedComboBox import AdvancedComboBox


_logger = logging.getLogger(__name__)


class _GeometryListModel(qt.QAbstractItemModel):

    TimeColumn = 0
    RmsColumn = 1

    def __init__(self, parent, historyModel):
        qt.QAbstractItemModel.__init__(self, parent=parent)
        self.__data = []
        self.__historyModel = historyModel
        self.__historyModel.changed[object].connect(self.__historyChanged)

    def __historyChanged(self, events):
        if events.hasOnlyStructuralEvents():
            if len(events) == 1 and events[0].added:
                index = events[0].index
                self.beginInsertRows(qt.QModelIndex(), index, index)
                self.endInsertRows()
            else:
                # Not optimized, cause not often called
                self.beginResetModel()
                self.endResetModel()
        else:
            # Not optimized, cause never called
            self.beginResetModel()
            self.endResetModel()

    def appendGeometryData(self, time, rms):
        self.__data.append((time, rms))

    def parent(self, index):
        return qt.QModelIndex()

    def index(self, row, column, parent=qt.QModelIndex()):
        if row < 0 or column < 0:
            return qt.QModelIndex()
        if row >= self.rowCount() or column >= self.columnCount():
            return qt.QModelIndex()
        return self.createIndex(row, column)

    def rowCount(self, parent=qt.QModelIndex()):
        return len(self.__historyModel)

    def columnCount(self, parent=qt.QModelIndex()):
        return 2

    def data(self, index, role=qt.Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        item = self.__historyModel[row]
        if role == qt.Qt.DisplayRole:
            column = index.column()
            if column == self.RmsColumn:
                value = stringutil.to_scientific_unicode(item.rms())
                return value
            elif column == self.TimeColumn:
                time = item.time()
                return time.strftime("%Hh%M %Ss")
            else:
                return ""


class GeometryHistoryComboBox(AdvancedComboBox):

    def __init__(self, parent=None):
        super(GeometryHistoryComboBox, self).__init__(parent)
        self.setDisplayedDataCallback(self.__displayedData)

    def __displayedData(self, widget, row, role=qt.Qt.DisplayRole):
        if row == -1:
            if role == qt.Qt.DisplayRole:
                return "No history"
            return None
        if role == qt.Qt.DisplayRole:
            model = widget.model()
            index0 = model.index(row, 0)
            index1 = model.index(row, 1)
            text = index0.data() + ":  RMS: " + index1.data()
            return text
        elif role == qt.Qt.DecorationRole:
            return None
        return None

    def setHistoryModel(self, historyModel):
        model = _GeometryListModel(self, historyModel)
        self.setModel(model)

        tableView = qt.QTableView(self)
        tableView.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        tableView.setShowGrid(False)
        header = tableView.verticalHeader()
        header.setVisible(False)
        header = tableView.horizontalHeader()
        header.setVisible(False)
        header.setStretchLastSection(True)
        self.setView(tableView)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Loïc Huder (loic.huder@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""Tool to visualize diffraction maps."""

__author__ = "Loïc Huder"
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/03/2024"
__status__ = "development"

from silx.gui import qt
from silx.gui.widgets.FloatEdit import FloatEdit


class RoiRangeWidget(qt.QWidget):
    updated = qt.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = qt.QHBoxLayout()

        min_layout = qt.QHBoxLayout()
        self._min_edit = FloatEdit()
        self._min_edit.editingFinished.connect(self._onMinEdition)
        min_layout.addWidget(qt.QLabel("Min", self))
        min_layout.addWidget(self._min_edit)

        max_layout = qt.QHBoxLayout()
        self._max_edit = FloatEdit()
        self._max_edit.editingFinished.connect(self._onMaxEdition)
        max_layout.addWidget(qt.QLabel("Max", self))
        max_layout.addWidget(self._max_edit)

        title_label = qt.QLabel("ROI bounds", self)
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)
        layout.addLayout(min_layout)
        layout.addLayout(max_layout)
        self.setLayout(layout)

    @property
    def minValue(self):
        return self._min_edit.value()

    @property
    def maxValue(self):
        return self._max_edit.value()

    def setRange(self, new_min: float, new_max: float):
        self._min_edit.setValue(new_min)
        self._max_edit.setValue(new_max)

    def _onMinEdition(self):
        new_min = self.minValue
        current_max = self.maxValue

        if new_min >= current_max:
            new_min = current_max

        self.updated.emit(new_min, current_max)

    def _onMaxEdition(self):
        current_min = self.minValue
        new_max = self.maxValue

        if current_min >= new_max:
            new_max = current_min

        self.updated.emit(current_min, new_max)

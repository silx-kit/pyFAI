#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2026 European Synchrotron Radiation Facility, Grenoble, France
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

"""Controls for in-process histogram background estimation."""

from silx.gui import qt

from ..background import auto_smoothness
from .RoiRangeWidget import RoiRangeWidget


class BackgroundDialog(qt.QDialog):
    subtractionChanged = qt.Signal(bool)
    smoothnessChanged = qt.Signal()
    visibilityChanged = qt.Signal(bool)

    def __init__(self, fit_roi, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histogram background")

        self.fit_range = RoiRangeWidget(self, title="")
        self.fit_range.layout().setContentsMargins(0, 0, 0, 0)
        self.fit_range.updated.connect(fit_roi.setRange)
        fit_roi.sigRegionChanged.connect(
            lambda: self.fit_range.setRange(*fit_roi.getRange())
        )

        self._fit_point_count = 0
        self.automatic = qt.QPushButton("Auto", self)
        self.automatic.setCheckable(True)
        self.automatic.setChecked(True)
        self.smoothness = qt.QDoubleSpinBox(self)
        self.smoothness.setRange(0, 20)
        self.smoothness.setSingleStep(0.1)
        self.smoothness.setDecimals(2)
        self.smoothness.setEnabled(False)
        self.automatic.toggled.connect(self.smoothness.setDisabled)
        self.automatic.toggled.connect(self.updateAutomaticSmoothness)
        self.automatic.toggled.connect(
            lambda checked=False: self.smoothnessChanged.emit()
        )
        self.smoothness.valueChanged.connect(
            lambda value=0.0: self.smoothnessChanged.emit()
        )

        self.subtract = qt.QCheckBox("Subtract estimated background", self)
        self.subtract.toggled.connect(self.subtractionChanged)

        smoothness = qt.QHBoxLayout()
        smoothness.addWidget(self.automatic)
        smoothness.addWidget(self.smoothness)

        layout = qt.QFormLayout(self)
        layout.setLabelAlignment(qt.Qt.AlignmentFlag.AlignLeft)
        layout.addRow("2θ fit bounds", self.fit_range)
        layout.addRow("Smoothness", smoothness)
        layout.addRow(self.subtract)
        self.resize(self.sizeHint())

    def setFitPointCount(self, count):
        self._fit_point_count = count
        self.updateAutomaticSmoothness()

    def updateAutomaticSmoothness(self, *args):
        if not self.automatic.isChecked() or self._fit_point_count < 3:
            return
        blocked = self.smoothness.blockSignals(True)
        self.smoothness.setValue(auto_smoothness(self._fit_point_count))
        self.smoothness.blockSignals(blocked)

    def showEvent(self, event):
        super().showEvent(event)
        self.visibilityChanged.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.visibilityChanged.emit(False)

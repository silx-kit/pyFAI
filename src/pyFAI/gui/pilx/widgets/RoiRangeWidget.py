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

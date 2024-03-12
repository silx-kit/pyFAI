from __future__ import annotations

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.actions.control import (
    ColormapAction,
    KeepAspectRatioAction,
    ResetZoomAction,
)
from silx.gui.plot.actions.io import SaveAction
from silx.gui.plot.ColorBar import ColorBarWidget
from silx.gui.plot.tools import PositionInfo


class ImageToolbar(qt.QToolBar):
    def __init__(self, plot):
        super().__init__(plot)
        self.addAction(ResetZoomAction(plot, self))
        self.addAction(ColormapAction(plot, self))
        self.addAction(KeepAspectRatioAction(plot, self))


class ImagePlotWidget(PlotWidget):
    plotClicked = qt.Signal(float, float)

    def __init__(self, parent=None, backend=None):
        super().__init__(parent, backend)
        self._initColorbar()
        self._initStatusBar()
        self._initCentralWidget()
        self._toolbar = self._initToolbar()
        self._toolbar.addSeparator()
        self._toolbar.addAction(SaveAction(self, self._toolbar))
        self.addToolBar(self._toolbar)
        self.sigPlotSignal.connect(self.emitMouseClickSignal)

    def _initToolbar(self):
        return ImageToolbar(self)

    def _initColorbar(self):
        self._colorBarWidget = ColorBarWidget(
            plot=self,
            parent=self,
        )
        # Make ColorBarWidget background white by changing its palette
        self._colorBarWidget.setAutoFillBackground(True)
        palette = self._colorBarWidget.palette()
        palette.setColor(qt.QPalette.Window, qt.Qt.GlobalColor.white)
        self._colorBarWidget.setPalette(palette)

    def _initCentralWidget(self):
        gridLayout = qt.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.addWidget(self.getWidgetHandle(), 0, 0)
        gridLayout.addWidget(self._colorBarWidget, 0, 1)
        gridLayout.addWidget(self._statusBar, 1, 0, 1, -1)

        gridLayout.setRowStretch(0, 1)
        gridLayout.setColumnStretch(0, 1)
        centralWidget = qt.QWidget(self)
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)

    def _initStatusBar(self):
        converters = (
            ("X", lambda x, y: x),
            ("Y", lambda x, y: y),
            ("Data", self._dataConverter),
        )
        self._statusBar = PositionInfo(plot=self, converters=converters)

    def _dataConverter(self, x, y):
        raise NotImplementedError()

    def getColorBarWidget(self):
        """Public method needed for ColorBarAction"""
        return self._colorBarWidget

    def emitMouseClickSignal(self, signal_data):
        if signal_data["event"] != "mouseClicked":
            return

        self.plotClicked.emit(signal_data["x"], signal_data["y"])

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.items import Curve
from silx.gui.plot.tools import PositionInfo
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.actions.control import ResetZoomAction
from silx.gui.plot.actions.io import SaveAction
from silx.gui.plot.actions.mode import ZoomModeAction, PanModeAction


from ..HorizontalRangeROI import HorizontalRangeROI
from ..models import ROI_COLOR
from .RoiModeAction import RoiModeAction
from .RoiRangeWidget import RoiRangeWidget


class IntegratedPatternPlotWidget(PlotWidget):
    def __init__(self, parent=None, backend=None):
        super().__init__(parent, backend)
        self.sigPlotSignal.connect(self.onRectDraw)

        self._roi_manager = RegionOfInterestManager(parent=self)
        self.roi = self._initRoi()
        self._roi_manager.addRoi(self.roi)

        self._roi_range = RoiRangeWidget(self)
        # Interconnect the ROI and the ROI range widget
        self._roi_range.updated.connect(self.roi.setRange)
        self.roi.sigRegionChanged.connect(self.updateRoiRangeWidget)

        self._toolbar = self._initToolbar()
        self.addToolBar(self._toolbar)

        self._statusBar = self._initStatusBar()
        centralWidget = self._initCentralWidget(self._statusBar)
        self.setCentralWidget(centralWidget)

    def _initRoi(self):
        roi = HorizontalRangeROI()
        roi.setColor(ROI_COLOR)
        roi.setEditable(True)

        return roi

    def _initToolbar(self):
        toolbar = qt.QToolBar()
        toolbar.addAction(ResetZoomAction(self, toolbar))
        toolbar.addSeparator()
        toolbar.addAction(PanModeAction(self, toolbar))
        toolbar.addAction(ZoomModeAction(self, toolbar))
        roiAction = RoiModeAction(self, toolbar)
        toolbar.addAction(roiAction)
        # Start in ROI mode
        roiAction._actionTriggered()

        toolbar.addSeparator()
        toolbar.addAction(SaveAction(self, toolbar))
        return toolbar

    def _initStatusBar(self):
        converters = (
            ("X", lambda x, y: x),
            ("Data", self._dataConverter),
        )
        return PositionInfo(plot=self, converters=converters)

    def _initCentralWidget(self, status_bar: qt.QWidget):
        gridLayout = qt.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.addWidget(self.getWidgetHandle(), 0, 0)
        gridLayout.addWidget(status_bar, 1, 0, 1, -1)
        gridLayout.addWidget(self._roi_range, 2, 0)

        gridLayout.setRowStretch(0, 1)
        centralWidget = qt.QWidget(self)
        centralWidget.setLayout(gridLayout)
        return centralWidget

    def _dataConverter(self, x_data, y_data):
        curves = self.getAllCurves()
        if not curves:
            return
        curve_item = curves[0]
        assert isinstance(curve_item, Curve)
        pixel_x, pixel_y = self.dataToPixel(x_data, y_data)
        picking_result = curve_item.pick(pixel_x, pixel_y)
        if picking_result is None:
            return
        indices_x = picking_result.getIndices(copy=False)
        curve_data = curve_item.getYData(copy=False)
        return curve_data[indices_x[0]]

    def onRectDraw(self, signal_data):
        if signal_data["event"] != "drawingFinished":
            return

        v_min, v_max = signal_data["xdata"]
        if v_max < v_min:
            v_min, v_max = v_max, v_min
        self.roi.setRange(v_min, v_max)

    def updateRoiRangeWidget(self):
        v_min, v_max = self.roi.getRange()
        if v_min is None or v_max is None:
            return

        self._roi_range.setRange(v_min, v_max)

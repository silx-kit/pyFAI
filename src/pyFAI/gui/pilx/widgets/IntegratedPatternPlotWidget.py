#!/usr/bin/env python
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
from __future__ import annotations

__author__ = "Loïc Huder"
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/03/2024"
__status__ = "development"

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.actions.control import ResetZoomAction
from silx.gui.plot.actions.io import SaveAction
from silx.gui.plot.actions.mode import PanModeAction, ZoomModeAction
from silx.gui.plot.items import Curve
from silx.gui.plot.tools import PositionInfo
from silx.gui.plot.tools.roi import RegionOfInterestManager

from ..HorizontalRangeROI import HorizontalRangeROI
from ..models import ROI_COLOR
from .RoiModeAction import RoiModeAction
from .RoiRangeWidget import RoiRangeWidget


class IntegratedPatternPlotWidget(PlotWidget):
    rgbRoiChanged = qt.Signal()
    roiModeChanged = qt.Signal(str)
    activeRoiChanged = qt.Signal()
    rgbChannelChanged = qt.Signal(str)

    def __init__(self, parent=None, backend=None):
        super().__init__(parent, backend)
        self.sigPlotSignal.connect(self.onRectDraw)

        self._roi_manager = RegionOfInterestManager(parent=self)
        self.roi = self._initRoi()
        self._roi_manager.addRoi(self.roi, useManagerColor=False)
        self.rgb_rois = {}
        for channel, color in (
            ("R", "#ff0000"),
            ("G", "#00a000"),
            ("B", "#0060ff"),
        ):
            roi = HorizontalRangeROI()
            self._roi_manager.addRoi(roi)
            roi.setColor(color)
            for marker in roi.getItems():
                marker.setColor(color)
            roi.setEditable(False)
            roi.setVisible(False)
            roi.sigRegionChanged.connect(self.updateRoiRangeWidget)
            roi.sigRangeCommitted.connect(self.rgbRoiChanged.emit)
            self.rgb_rois[channel] = roi
        self._roi_mode = "single"
        self._active_rgb_channel = "R"
        self._rgb_rois_initialized = False

        self._roi_range = RoiRangeWidget(self)
        # Interconnect the ROI and the ROI range widget
        self._roi_range.updated.connect(self.setActiveRoiRange)
        self.roi.sigRegionChanged.connect(self.updateRoiRangeWidget)

        self._toolbar = self._initToolbar()
        self.addToolBar(self._toolbar)

        self._statusBar = self._initStatusBar()
        centralWidget = self._initCentralWidget(self._statusBar)
        self.setCentralWidget(centralWidget)

    def __iter__(self):
        yield from self.getAllCurves(just_legend=True)


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
        self._roi_action = RoiModeAction(self, self.roi, toolbar)
        roi_menu = qt.QMenu(toolbar)
        mode_group = qt.QActionGroup(roi_menu)
        mode_group.setExclusive(True)
        self._single_roi_action = roi_menu.addAction("Single channel")
        self._single_roi_action.setCheckable(True)
        self._single_roi_action.setChecked(True)
        self._single_roi_action.triggered.connect(
            lambda checked=False: self.setRoiMode("single") if checked else None
        )
        mode_group.addAction(self._single_roi_action)
        self._rgb_roi_action = roi_menu.addAction("RGB")
        self._rgb_roi_action.setCheckable(True)
        self._rgb_roi_action.triggered.connect(
            lambda checked=False: self.setRoiMode("rgb") if checked else None
        )
        mode_group.addAction(self._rgb_roi_action)
        roi_menu.addSeparator()
        channel_group = qt.QActionGroup(roi_menu)
        channel_group.setExclusive(True)
        self._rgb_channel_actions = {}
        for channel, text in (("R", "Red"), ("G", "Green"), ("B", "Blue")):
            action = roi_menu.addAction(text)
            action.setCheckable(True)
            action.setEnabled(False)
            action.setChecked(channel == "R")
            action.triggered.connect(
                lambda checked=False, selected=channel: (
                    self.setActiveRgbChannel(selected) if checked else None
                )
            )
            channel_group.addAction(action)
            self._rgb_channel_actions[channel] = action
        self._roi_action.setMenu(roi_menu)
        toolbar.addAction(self._roi_action)
        roi_button = toolbar.widgetForAction(self._roi_action)
        if isinstance(roi_button, qt.QToolButton):
            roi_button.setPopupMode(qt.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        # Start in ROI mode
        self._roi_action.trigger()

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
        if not isinstance(curve_item, Curve):
            raise RuntimeError("`curve` is not a `silx.gui.plot.items.curve.Curve` instance")
        tmp = self.dataToPixel(x_data, y_data)
        if tmp:
            pixel_x, pixel_y = tmp
            picking_result = curve_item.pick(pixel_x, pixel_y)
        else:
            picking_result = None
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
        self.activeRoi().setRange(v_min, v_max)

    def activeRoi(self):
        if self._roi_mode == "rgb":
            return self.rgb_rois[self._active_rgb_channel]
        return self.roi

    def setActiveRoiRange(self, v_min, v_max):
        self.activeRoi().setRange(v_min, v_max)

    def setRoiMode(self, mode):
        if mode not in {"single", "rgb"}:
            raise ValueError(f"Unsupported ROI mode: {mode}")
        if mode == "rgb" and not self._rgb_rois_initialized:
            v_min, v_max = self.roi.getRange()
            if v_min is not None and v_max is not None and v_max > v_min:
                for roi in self.rgb_rois.values():
                    blocked = roi.blockSignals(True)
                    roi.setRange(v_min, v_max)
                    roi.blockSignals(blocked)
                self._rgb_rois_initialized = True
        self._roi_mode = mode
        rgb = mode == "rgb"
        self._single_roi_action.setChecked(not rgb)
        self._rgb_roi_action.setChecked(rgb)
        for action in self._rgb_channel_actions.values():
            action.setEnabled(rgb)
        rois = list(self.rgb_rois.values()) if rgb else [self.roi]
        self._roi_action.setRois(rois, self.activeRoi())
        self.updateRoiRangeWidget()
        self.roiModeChanged.emit(mode)

    def setActiveRgbChannel(self, channel):
        if channel not in self.rgb_rois:
            raise ValueError(f"Unsupported RGB channel: {channel}")
        self._active_rgb_channel = channel
        self._rgb_channel_actions[channel].setChecked(True)
        if self._roi_mode == "rgb":
            self._roi_action.setRois(list(self.rgb_rois.values()), self.activeRoi())
            self.updateRoiRangeWidget()
            self.activeRoiChanged.emit()
            self.rgbChannelChanged.emit(channel)

    def updateRoiRangeWidget(self):
        roi = self.activeRoi()
        if self.sender() is not None and self.sender() is not roi:
            return
        v_min, v_max = roi.getRange()
        if v_min is None or v_max is None:
            return

        title = "ROI bounds"
        if self._roi_mode == "rgb":
            title = f"ROI bounds ({self._active_rgb_channel})"
        self._roi_range.setTitle(title)
        self._roi_range.setRange(v_min, v_max)

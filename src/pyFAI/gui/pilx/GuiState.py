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

"""Save and restore the state of the diffmap viewer."""

from datetime import datetime, timezone
import json
import logging
import os

from silx.gui import icons, qt

from .models import ImageIndices
from .point import Point


logger = logging.getLogger(__name__)


class GuiStateManager(qt.QObject):
    VERSION = 1

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self._last_path = None

    def createButton(self, parent):
        button = qt.QToolButton(parent)
        button.setIcon(icons.getQIcon("document-save"))
        button.setToolTip("Save or load GUI state")
        button.setPopupMode(qt.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = qt.QMenu(button)
        save_action = menu.addAction(
            icons.getQIcon("document-save"), "Save GUI state…"
        )
        save_action.triggered.connect(self.saveAs)
        load_action = menu.addAction(
            icons.getQIcon("document-open"), "Load GUI state…"
        )
        load_action.triggered.connect(self.loadFrom)
        menu.addSeparator()
        autosave_action = menu.addAction("Load autosaved GUI state")
        autosave_action.triggered.connect(self.loadAutosave)
        menu.aboutToShow.connect(
            lambda: autosave_action.setEnabled(os.path.isfile(self.autosavePath()))
        )
        button.setMenu(menu)
        return button

    def autosavePath(self):
        directory = qt.QStandardPaths.writableLocation(
            qt.QStandardPaths.StandardLocation.GenericConfigLocation
        )
        return os.path.join(directory, "pyfai", "diffmap-view", "autosave.json")

    def _colormapState(self, colormap):
        if colormap is None:
            return None
        vmin = colormap.getVMin()
        vmax = colormap.getVMax()
        nan_color = colormap.getNaNColor().name(qt.QColor.NameFormat.HexArgb)
        state = {
            "name": colormap.getName(),
            "normalization": colormap.getNormalization(),
            "vmin": None if vmin is None else float(vmin),
            "vmax": None if vmax is None else float(vmax),
            "autoscale_mode": colormap.getAutoscaleMode(),
            "autoscale_percentiles": [
                float(value) for value in colormap.getAutoscalePercentiles()
            ],
            "gamma": float(colormap.getGammaNormalizationParameter()),
            "nan_color": nan_color,
        }

        return state

    def _restoreColormap(self, colormap, state):
        if colormap is None or not state:
            return
        blocked = colormap.blockSignals(True)
        try:
            if state.get("name") is not None:
                colormap.setName(state["name"])
            colormap.setNormalization(state.get("normalization", "linear"))
            colormap.setVRange(state.get("vmin"), state.get("vmax"))
            colormap.setAutoscaleMode(state.get("autoscale_mode", "minmax"))
            percentiles = state.get("autoscale_percentiles")
            if percentiles is not None:
                colormap.setAutoscalePercentiles(tuple(percentiles))
            colormap.setGammaNormalizationParameter(state.get("gamma", 2.0))
            nan_color = state.get("nan_color")
            if nan_color is not None:
                colormap.setNaNColor(qt.QColor(nan_color))
        finally:
            colormap.blockSignals(blocked)
        colormap.sigChanged.emit()

    def _clampRange(self, values, radial):
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError("ROI bounds must contain two numbers")
        minimum = max(float(radial[0]), min(float(radial[-1]), float(values[0])))
        maximum = max(minimum, min(float(radial[-1]), float(values[1])))
        return minimum, maximum

    def state(self):
        window = self._window
        pattern = window._integrated_plot_widget
        background = window._background_dialog
        rgb_colormaps = {}
        if window._rgb_map_plot_widget is not None:
            rgb_colormaps = {
                channel: self._colormapState(colormap)
                for channel, colormap in window._rgb_map_plot_widget._rgb_colormaps.items()
            }

        current = None
        if window._unfixed_indices is not None:
            current = [
                int(window._unfixed_indices.row),
                int(window._unfixed_indices.col),
            ]
        background_point = None
        if window._background_point is not None:
            background_point = [
                int(window._background_point.indices.row),
                int(window._background_point.indices.col),
            ]

        state = {
            "version": self.VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "file_name": window._file_name,
                "dataset_path": window._dataset_path,
                "nxprocess_path": window._nxprocess_path,
            },
            "pattern": {
                "roi_mode": pattern._roi_mode,
                "roi": [float(value) for value in pattern.roi.getRange()],
                "rgb_rois": {
                    channel: [float(value) for value in roi.getRange()]
                    for channel, roi in pattern.rgb_rois.items()
                },
                "rgb_rois_initialized": pattern._rgb_rois_initialized,
                "active_rgb_channel": pattern._active_rgb_channel,
                "y_scale": "signed_sqrt" if pattern._sqrt_mode else pattern.getYAxis().getScale(),
                "observed_lines": pattern._observed_lines.isChecked(),
                "observed_circles": pattern._observed_circles.isChecked(),
            },
            "background": {
                "fit_bounds": [
                    float(value) for value in pattern.background_fit_roi.getRange()
                ],
                "subtract": background.subtract.isChecked(),
                "automatic": background.automatic.isChecked(),
                "smoothness": background.smoothness.value(),
                "dialog_visible": background.isVisible(),
                "dialog_geometry": bytes(
                    background.saveGeometry().toBase64()
                ).decode("ascii"),
            },
            "colormaps": {
                "detector": self._colormapState(
                    window._image_plot_widget._image_item.getColormap()
                ),
                "roi_map": self._colormapState(
                    window._map_plot_widget._scatter_item.getColormap()
                ),
                "rgb": rgb_colormaps,
            },
            "selection": {
                "current": current,
                "fixed": [
                    [int(indices.row), int(indices.col)]
                    for indices in sorted(window._fixed_indices)
                ],
                "background": background_point,
            },
            "view": {
                "window_geometry": bytes(window.saveGeometry().toBase64()).decode("ascii"),
                "plot_splitter": window._plot_splitter.sizes(),
                "right_splitter": window._right_splitter.sizes(),
                "active_map_tab": window._map_tab_widget.currentIndex(),
                "plot_limits": {
                    "detector": [
                        [
                            float(value)
                            for value in window._image_plot_widget.getGraphXLimits()
                        ],
                        [
                            float(value)
                            for value in window._image_plot_widget.getGraphYLimits()
                        ],
                    ],
                    "roi_map": [
                        [
                            float(value)
                            for value in window._map_plot_widget.getGraphXLimits()
                        ],
                        [
                            float(value)
                            for value in window._map_plot_widget.getGraphYLimits()
                        ],
                    ],
                    "pattern": [
                        [float(value) for value in pattern.getGraphXLimits()],
                        [float(value) for value in pattern.getGraphYLimits()],
                    ],
                },
            },
        }

        return state

    def save(self, path):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = json.dumps(self.state(), indent=2, sort_keys=True, allow_nan=False)
        output = qt.QSaveFile(path)
        mode = qt.QIODevice.OpenModeFlag.WriteOnly | qt.QIODevice.OpenModeFlag.Text
        if not output.open(mode):
            raise OSError(output.errorString())
        encoded = data.encode("utf-8")
        if output.write(encoded) != len(encoded):
            error = output.errorString()
            output.cancelWriting()
            raise OSError(error)
        if not output.commit():
            raise OSError(output.errorString())
        self._last_path = path

    def load(self, path):
        path = os.path.abspath(path)
        with open(path, "r", encoding="utf-8") as stream:
            state = json.load(stream)
        if not isinstance(state, dict):
            raise ValueError("GUI state must be a JSON object")
        if state.get("version") != self.VERSION:
            raise ValueError(
                f"Unsupported GUI state version: {state.get('version')!r}"
            )

        window = self._window
        source = state.get("source", {})
        file_name = source.get("file_name")
        dataset_path = source.get("dataset_path")
        nxprocess_path = source.get("nxprocess_path")
        if not file_name or not dataset_path or not nxprocess_path:
            raise ValueError("GUI state does not contain complete HDF5 source paths")
        if not os.path.isfile(file_name):
            raise FileNotFoundError(file_name)

        for indices in window._fixed_indices.copy():
            window.removeMapPoint(indices)
        window.removeMapMarker("BG_LOCATION")
        window._background_point = None
        window._unfixed_indices = None
        blocked = window._background_dialog.subtract.blockSignals(True)
        window._background_dialog.subtract.setChecked(False)
        window._background_dialog.subtract.blockSignals(blocked)
        window._subtract_background = False
        window.initData(file_name, dataset_path, nxprocess_path)

        pattern_state = state.get("pattern", {})
        pattern = window._integrated_plot_widget
        radial = window._background_radial_values
        roi_range = pattern_state.get("roi")
        if roi_range is not None:
            roi_range = self._clampRange(roi_range, radial)
            blocked = pattern.roi.blockSignals(True)
            pattern.roi.setRange(*roi_range)
            pattern.roi.blockSignals(blocked)
            pattern.updateRoiRangeWidget()
        rgb_ranges = pattern_state.get("rgb_rois", {})
        for channel, roi_range in rgb_ranges.items():
            if channel not in pattern.rgb_rois:
                continue
            roi_range = self._clampRange(roi_range, radial)
            roi = pattern.rgb_rois[channel]
            blocked = roi.blockSignals(True)
            roi.setRange(*roi_range)
            roi.blockSignals(blocked)
        rgb_initialized = pattern_state.get("rgb_rois_initialized")
        if rgb_initialized is None:
            rgb_initialized = len(rgb_ranges) == len(pattern.rgb_rois) and all(
                roi.getRange()[1] > roi.getRange()[0]
                for roi in pattern.rgb_rois.values()
            )
        pattern._rgb_rois_initialized = bool(rgb_initialized)

        background_state = state.get("background", {})
        fit_bounds = background_state.get("fit_bounds")
        if fit_bounds is not None:
            fit_bounds = self._clampRange(fit_bounds, radial)
            fit_roi = pattern.background_fit_roi
            blocked = fit_roi.blockSignals(True)
            fit_roi.setRange(*fit_bounds)
            fit_roi.blockSignals(blocked)
            window._background_dialog.fit_range.setRange(*fit_roi.getRange())
            window.setBackgroundFitRange()

        background = window._background_dialog
        automatic = bool(background_state.get("automatic", True))
        blocked = background.automatic.blockSignals(True)
        background.automatic.setChecked(automatic)
        background.automatic.blockSignals(blocked)
        background.smoothness.setDisabled(automatic)
        blocked = background.smoothness.blockSignals(True)
        background.smoothness.setValue(
            float(background_state.get("smoothness", background.smoothness.value()))
        )
        background.smoothness.blockSignals(blocked)
        if automatic:
            background.updateAutomaticSmoothness()

        pattern._observed_lines.setChecked(
            bool(pattern_state.get("observed_lines", True))
        )
        pattern._observed_circles.setChecked(
            bool(pattern_state.get("observed_circles", False))
        )
        y_scale = pattern_state.get("y_scale", "linear")
        if y_scale in pattern._y_scale_actions:
            pattern.setYAxisScale(y_scale)

        active_channel = pattern_state.get("active_rgb_channel", "R")
        if active_channel not in pattern.rgb_rois:
            active_channel = "R"
        pattern._active_rgb_channel = active_channel
        pattern._rgb_channel_actions[active_channel].setChecked(True)
        window._rgb_map_channel = active_channel
        roi_mode = pattern_state.get("roi_mode", "single")
        blocked = pattern.blockSignals(True)
        pattern.setRoiMode(roi_mode if roi_mode in {"single", "rgb"} else "single")
        pattern.blockSignals(blocked)

        selection = state.get("selection", {})
        current = selection.get("current")
        if current is None:
            current_indices = ImageIndices(0, 0)
        else:
            current_indices = ImageIndices(*current)
        rows, columns = window._map_plot_widget._map_shape
        if not (0 <= current_indices.row < rows and 0 <= current_indices.col < columns):
            current_indices = ImageIndices(0, 0)
        window._unfixed_indices = current_indices

        background_indices = selection.get("background")
        if background_indices is not None:
            background_indices = ImageIndices(*background_indices)
            if 0 <= background_indices.row < rows and 0 <= background_indices.col < columns:
                window._background_point = Point(
                    background_indices,
                    url_nxdata_path=(
                        f"{window._file_name}?{window._nxprocess_path}/result"
                    ),
                )

        window._fixed_indices.clear()
        for values in selection.get("fixed", []):
            indices = ImageIndices(*values)
            if 0 <= indices.row < rows and 0 <= indices.col < columns:
                window._fixed_indices.add(indices)

        subtract = bool(background_state.get("subtract", False))
        blocked = background.subtract.blockSignals(True)
        background.subtract.setChecked(subtract)
        background.subtract.blockSignals(blocked)
        window._subtract_background = subtract
        window._background_baseline = None

        window.displayPatternAtIndices(current_indices, legend="INTEGRATE")
        window.displayImageAtIndices(current_indices)
        window.setMapMarker(
            current_indices,
            color=window.getCurveColor("INTEGRATE"),
            symbol="o",
            legend="MAP_LOCATION",
        )
        for indices in window._fixed_indices:
            legend = f"INTEGRATE_{indices.row}_{indices.col}"
            window.displayPatternAtIndices(indices, legend=legend)
            window.setMapMarker(
                indices,
                color=window.getCurveColor(legend),
                symbol="d",
                legend=f"MAP_LOCATION_{indices.row}_{indices.col}",
            )
        if window._background_point is not None:
            window.setMapMarker(
                window._background_point.indices,
                color="black",
                symbol="x",
                legend="BG_LOCATION",
            )

        roi_minimum, roi_maximum = pattern.roi.getRange()
        window.displayAverageMap(roi_minimum, roi_maximum)
        rgb_state = state.get("colormaps", {}).get("rgb", {})
        if roi_mode == "rgb" or rgb_state:
            window.displayRgbMap()

        colormaps = state.get("colormaps", {})
        self._restoreColormap(
            window._image_plot_widget._image_item.getColormap(),
            colormaps.get("detector"),
        )
        self._restoreColormap(
            window._map_plot_widget._scatter_item.getColormap(),
            colormaps.get("roi_map"),
        )
        if window._rgb_map_plot_widget is not None:
            for channel, colormap_state in rgb_state.items():
                self._restoreColormap(
                    window._rgb_map_plot_widget._rgb_colormaps.get(channel),
                    colormap_state,
                )
            window._rgb_map_plot_widget.setRgbChannel(active_channel)

        view = state.get("view", {})
        geometry = view.get("window_geometry")
        if geometry:
            window.restoreGeometry(qt.QByteArray.fromBase64(geometry.encode("ascii")))
        if "plot_splitter" in view:
            window._plot_splitter.setSizes(view["plot_splitter"])
        if "right_splitter" in view:
            window._right_splitter.setSizes(view["right_splitter"])
        plots = {
            "detector": window._image_plot_widget,
            "roi_map": window._map_plot_widget,
            "pattern": pattern,
        }
        for name, limits in view.get("plot_limits", {}).items():
            plot = plots.get(name)
            if plot is not None and len(limits) == 2:
                plot.setGraphXLimits(*limits[0])
                plot.setGraphYLimits(*limits[1])
        tab_index = int(view.get("active_map_tab", 0))
        tab_index = max(0, min(window._map_tab_widget.count() - 1, tab_index))
        window._map_tab_widget.setCurrentIndex(tab_index)

        dialog_geometry = background_state.get("dialog_geometry")
        if dialog_geometry:
            background.restoreGeometry(
                qt.QByteArray.fromBase64(dialog_geometry.encode("ascii"))
            )
        background.setVisible(bool(background_state.get("dialog_visible", False)))
        window.drawContoursOnImage()
        pattern._updateObservedStyle()
        self._last_path = path

    def saveAs(self):
        if self._last_path is not None:
            suggested = self._last_path
        elif self._window._file_name is not None:
            stem = os.path.splitext(os.path.basename(self._window._file_name))[0]
            suggested = os.path.join(
                os.path.dirname(self._window._file_name), f"{stem}.gui-state.json"
            )
        else:
            suggested = "pyFAI-diffmap-view.gui-state.json"
        path, _ = qt.QFileDialog.getSaveFileName(
            self._window, "Save GUI state", suggested, "JSON files (*.json)"
        )
        if not path:
            return
        try:
            self.save(path)
        except Exception as error:
            qt.QMessageBox.critical(
                self._window, "Cannot save GUI state", str(error)
            )

    def loadFrom(self):
        directory = self._last_path or os.path.dirname(
            self._window._file_name or ""
        )
        path, _ = qt.QFileDialog.getOpenFileName(
            self._window, "Load GUI state", directory, "JSON files (*.json)"
        )
        if path:
            self._loadWithMessage(path)

    def loadAutosave(self):
        self._loadWithMessage(self.autosavePath())

    def _loadWithMessage(self, path):
        try:
            self.load(path)
        except Exception as error:
            qt.QMessageBox.critical(
                self._window, "Cannot load GUI state", str(error)
            )

    def autosave(self):
        try:
            self.save(self.autosavePath())
        except Exception:
            logger.exception("Cannot autosave diffmap viewer GUI state")

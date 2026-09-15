#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023-2026 European Synchrotron Radiation Facility, Grenoble, France
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

__authors__ = ["Loïc Huder", "E. Gutierrez-Fernandez", "Jérôme Kieffer"]
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/01/2026"
__status__ = "development"

import json
import logging
import os.path
import posixpath
from string import digits

import h5py
import numpy
from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.plot.items.image import ImageBase
from silx.image.marchingsquares import find_contours

from ...io.diffmap_config import DiffmapConfig
from ...io.integration_config import WorkerConfig
from ...utils.mathutil import binning
from .background import arpls
from .GuiState import GuiStateManager
from .models import ImageIndices
from .point import Point
from .utils import (
    compute_radial_values,
    get_axes_dataset,
    get_axes_index,
    get_dataset,
    get_indices_from_values,
    get_mask_image,
    get_radial_dataset,
    get_signal_dataset,
)
from .widgets.BackgroundDialog import BackgroundDialog
from .widgets.DiffractionImagePlotWidget import DiffractionImagePlotWidget
from .widgets.IntegratedPatternPlotWidget import IntegratedPatternPlotWidget
from .widgets.MapPlotWidget import MapPlotWidget

logger = logging.getLogger(__name__)


class MainWindow(qt.QMainWindow):
    sigFileChanged = qt.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._file_name: str | None = None
        self._dataset_path: str | None = None
        self._nxprocess_path: str | None = None
        self._unfixed_indices = None
        self._fixed_indices = set()
        self._background_point = None
        self._subtract_background = False
        self._background_baseline = None
        self._background_radial_values = None
        self._map_plot_widgets = []
        self._rgb_map_plot_widget = None
        self._rgb_map_channel = "R"

        self.setWindowTitle("PyFAI-diffmap viewer")

        self._image_plot_widget = DiffractionImagePlotWidget(self)
        self._image_plot_widget.setDefaultColormap(
            Colormap("gray", normalization="log")
        )
        self._image_plot_widget.setKeepDataAspectRatio(True)
        self._image_plot_widget.plotClicked.connect(self.onMouseClickOnImage)

        self._map_tab_widget = qt.QTabWidget(self)
        self._map_tab_widget.setTabsClosable(True)
        self._map_tab_widget.tabCloseRequested.connect(self.removeMapTab)
        self._gui_state = GuiStateManager(self)
        self._map_tab_widget.setCornerWidget(
            self._gui_state.createButton(self._map_tab_widget),
            qt.Qt.Corner.TopRightCorner,
        )
        self._map_plot_widget = self.addMapTab("2θ ROI", closable=False)
        self._map_plot_widget.setDefaultColormap(
            Colormap("viridis", normalization="log")
        )

        self._integrated_plot_widget = IntegratedPatternPlotWidget(self)
        self._background_dialog = BackgroundDialog(
            self._integrated_plot_widget.background_fit_roi, self
        )
        self._integrated_plot_widget.roi.sigRangeCommitted.connect(self.onRoiEdition)
        self._integrated_plot_widget.roi.sigRangeCommitted.connect(self.drawContoursOnImage)
        self._integrated_plot_widget.rgbRoiChanged.connect(self.displayRgbMap)
        self._integrated_plot_widget.rgbRoiChanged.connect(self.drawContoursOnImage)
        self._integrated_plot_widget.roiModeChanged.connect(self.roiModeChanged)
        self._integrated_plot_widget.activeRoiChanged.connect(
            self.drawContoursOnImage
        )
        self._integrated_plot_widget.rgbChannelChanged.connect(
            self.setRgbMapChannel
        )
        self._background_dialog.subtractionChanged.connect(
            self.setBackgroundSubtraction
        )
        self._background_dialog.smoothnessChanged.connect(
            self.setBackgroundSmoothness
        )
        self._background_dialog.visibilityChanged.connect(
            self._integrated_plot_widget.setBackgroundFitVisible
        )
        self._integrated_plot_widget.backgroundRequested.connect(
            self._background_dialog.show
        )
        self._integrated_plot_widget.background_fit_roi.sigRangeCommitted.connect(
            self.setBackgroundFitRange
        )

        self._central_widget = qt.QWidget()
        self._right_splitter = qt.QSplitter(qt.Qt.Orientation.Vertical, self)
        self._right_splitter.addWidget(self._map_tab_widget)
        self._right_splitter.addWidget(self._integrated_plot_widget)
        self._right_splitter.setChildrenCollapsible(False)
        self._right_splitter.setHandleWidth(6)
        self._right_splitter.setStretchFactor(0, 1)
        self._right_splitter.setStretchFactor(1, 1)

        self._plot_splitter = qt.QSplitter(qt.Qt.Orientation.Horizontal, self)
        self._plot_splitter.addWidget(self._image_plot_widget)
        self._plot_splitter.addWidget(self._right_splitter)
        self._plot_splitter.setChildrenCollapsible(False)
        self._plot_splitter.setHandleWidth(6)
        self._plot_splitter.setStretchFactor(0, 1)
        self._plot_splitter.setStretchFactor(1, 1)

        layout = qt.QVBoxLayout(self._central_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_splitter, 1)
        self._central_widget.setLayout(layout)
        self.setCentralWidget(self._central_widget)

        self.worker_config = None

        # declaration of instance variables
        self._map_ptr = None # This is the map of the indices of input frame

    def addMapTab(
        self,
        title: str,
        image: numpy.ndarray | None = None,
        x: numpy.ndarray | None = None,
        y: numpy.ndarray | None = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        closable: bool = True,
    ) -> MapPlotWidget:
        map_plot_widget = MapPlotWidget(self._map_tab_widget)
        map_plot_widget.setDefaultColormap(Colormap("viridis"))
        map_plot_widget.clearPointsSignal.connect(self.clearPoints)
        map_plot_widget.plotClicked.connect(self.selectMapPoint)
        map_plot_widget.pinContextEntrySelected.connect(self.fixMapPoint)
        map_plot_widget.setBackgroundClicked.connect(self.setNewBackgroundCurve)
        self.sigFileChanged.connect(map_plot_widget.onFileChange)
        if self._file_name is not None:
            map_plot_widget.onFileChange(self._file_name)
        if image is not None:
            map_plot_widget.setScatterData(image, x, y, xlabel, ylabel)

        index = self._map_tab_widget.addTab(map_plot_widget, title)
        self._map_plot_widgets.append(map_plot_widget)
        self._map_tab_widget.setCurrentIndex(index)
        if not closable:
            tab_bar = self._map_tab_widget.tabBar()
            tab_bar.setTabButton(
                index, qt.QTabBar.ButtonPosition.LeftSide, None
            )
            tab_bar.setTabButton(
                index, qt.QTabBar.ButtonPosition.RightSide, None
            )

        if image is not None:
            if self._unfixed_indices is not None:
                self.setMapMarker(
                    self._unfixed_indices,
                    color=self.getCurveColor(legend="INTEGRATE"),
                    symbol="o",
                    legend="MAP_LOCATION",
                )
            for indices in self._fixed_indices:
                legend = f"INTEGRATE_{indices.row}_{indices.col}"
                self.setMapMarker(
                    indices,
                    color=self.getCurveColor(legend=legend),
                    symbol="d",
                    legend=f"MAP_LOCATION_{indices.row}_{indices.col}",
                )
            if self._background_point is not None:
                self.setMapMarker(
                    self._background_point.indices,
                    color="black",
                    symbol="x",
                    legend="BG_LOCATION",
                )

        return map_plot_widget

    def removeMapTab(self, index: int):
        if index == 0:
            return
        map_plot_widget = self._map_tab_widget.widget(index)
        self._map_tab_widget.removeTab(index)
        self._map_plot_widgets.remove(map_plot_widget)
        if map_plot_widget is self._rgb_map_plot_widget:
            self._rgb_map_plot_widget = None
        map_plot_widget.deleteLater()

    def setMapMarker(self, indices: ImageIndices, **kwargs):
        for map_plot_widget in self._map_plot_widgets:
            coordinates = map_plot_widget.getMapPointCoordinates(indices)
            if coordinates is not None:
                map_plot_widget.addMarker(*coordinates, **kwargs)

    def removeMapMarker(self, legend: str):
        for map_plot_widget in self._map_plot_widgets:
            map_plot_widget.removeMarker(legend=legend)

    def initData(self,
                 file_name: str,
                 dataset_path: str="/entry_0000/measurement/images_0001",
                 nxprocess_path: str="/entry_0000/pyFAI",
                 ):

        self._file_name = os.path.abspath(file_name)
        self._dataset_path = dataset_path
        while self._map_tab_widget.count() > 1:
            self.removeMapTab(1)
        self._dataset_paths = {}
        self._nxprocess_path = nxprocess_path

        self.sigFileChanged.emit(self._file_name)

        with h5py.File(self._file_name, "r") as h5file:
            nxprocess = h5file[self._nxprocess_path]
            nxdata = nxprocess["result"]
            map_dataset = get_signal_dataset(nxdata, default="intensity")
            axes_index = get_axes_index(map_dataset)
            map_data  = map_dataset[()].sum(axis=axes_index.radial)
            try:
                slow = get_axes_dataset(nxdata, dim=axes_index.slow, default="slow")
            except (KeyError, RuntimeError):
                slow_label = slow_values = None
            else:
                slow_label = slow.attrs.get("long_name", "Y")
                slow_values = slow[()]
            try:
                fast = get_axes_dataset(nxdata, dim=axes_index.fast, default="fast")
            except (KeyError, RuntimeError):
                fast_values = fast_label = None
            else:
                fast_label = fast.attrs.get("long_name", "X")
                fast_values = fast[()]

            pyFAI_config_as_str = get_dataset(
                parent=nxprocess,
                path="configuration/data")[()]
            pyFAI_config_as_dict = json.loads(pyFAI_config_as_str)
            if "diffmap_config_version" in pyFAI_config_as_dict:
                diffmap_config = DiffmapConfig.from_dict(pyFAI_config_as_dict, inplace=True)
                self.worker_config = diffmap_config.ai
            else:
                self.worker_config = WorkerConfig.from_dict(pyFAI_config_as_dict, inplace=True)

            radial_dset = get_radial_dataset(nxdata, size=self.worker_config.nbpt_rad)
            radial_values = radial_dset[()]
            delta_radial = (radial_values[-1] - radial_values[0]) / len(radial_values)

            if "offset" in nxprocess:
                self._offset = nxprocess["offset"][()]
            else:
                self._offset = 0

            try:
                self._map_ptr = get_dataset(nxdata, "map_ptr")[()]
            except (KeyError, RuntimeError):
                logger.warning("No `map_ptr` dataset in NXdata: guessing the frame indices !")
                self._map_ptr = numpy.arange(self._offset, self._offset + map_data.size)
                self._map_ptr.shape = map_data.shape

            _dataset_path = dataset_path.rstrip(digits)
            path, base = posixpath.split(_dataset_path)

            try:
                image_grp = h5file[path]
            except KeyError:
                self.warning(f"Cannot access diffraction images at {path}: no such path.")
            else:
                if isinstance(image_grp, h5py.Group):
                    lst = []
                    for key in image_grp:
                        try:
                            ds = image_grp[key]
                        except KeyError:
                            self.warning(f"Cannot access diffraction images at {path}/{key}: not a valid dataset.")
                        else:
                            if key.startswith(base) and isinstance(ds, h5py.Dataset):
                                lst.append(key)

                    lst.sort()
                    for key in lst:
                        self._dataset_paths[posixpath.join(path, key)] = len(image_grp[key])
                else:
                    self.warning(f"Cannot access diffraction images at {path}: not a group.")

        self._radial_matrix = compute_radial_values(self.worker_config)
        self._delta_radial_over_2 = delta_radial / 2
        self._background_radial_values = radial_values
        self._background_baseline = None
        fit_roi = self._integrated_plot_widget.background_fit_roi
        radial_minimum = float(radial_values[0])
        radial_span = float(radial_values[-1]) - radial_minimum
        fit_minimum = radial_minimum + 0.05 * radial_span
        fit_maximum = radial_minimum + 0.95 * radial_span
        blocked = fit_roi.blockSignals(True)
        fit_roi.setRange(fit_minimum, fit_maximum)
        fit_roi.blockSignals(blocked)
        self._background_dialog.fit_range.setRange(fit_minimum, fit_maximum)
        start = numpy.searchsorted(radial_values, fit_minimum, side="left")
        stop = numpy.searchsorted(radial_values, fit_maximum, side="right")
        self._background_dialog.setFitPointCount(stop - start)

        roi_minimum = radial_minimum + 0.45 * radial_span
        roi_maximum = radial_minimum + 0.55 * radial_span
        pattern = self._integrated_plot_widget
        blocked = pattern.roi.blockSignals(True)
        pattern.roi.setRange(roi_minimum, roi_maximum)
        pattern.roi.blockSignals(blocked)
        for channel, (minimum, maximum) in zip(
            "RGB", ((0.35, 0.40), (0.45, 0.50), (0.55, 0.60))
        ):
            roi = pattern.rgb_rois[channel]
            blocked = roi.blockSignals(True)
            roi.setRange(
                radial_minimum + minimum * radial_span,
                radial_minimum + maximum * radial_span,
            )
            roi.blockSignals(blocked)
        pattern._rgb_rois_initialized = True
        pattern.updateRoiRangeWidget()

        self._map_plot_widget.setScatterData(map_data, fast_values, slow_values, fast_label, slow_label)
        self.displayAverageMap(roi_minimum, roi_maximum)
        # BUG: selectMapPoint(0, 0) does not work at first render cause the picking fails
        initial_indices = ImageIndices(0, 0)
        self._unfixed_indices = initial_indices
        self.displayPatternAtIndices(initial_indices, legend="INTEGRATE")
        self.displayImageAtIndices(initial_indices)
        self.drawContoursOnImage()
        self.setMapMarker(
            initial_indices,
            color=self.getCurveColor(legend="INTEGRATE"),
            symbol="o",
            legend="MAP_LOCATION",
        )

    def getRoiRadialRange(self) -> tuple[float | None, float | None]:
        return self._integrated_plot_widget.roi.getRange()

    def displayPatternAtIndices(self,
                                indices: ImageIndices,
                                legend: str,
                                color: str=None):
        if self._file_name is None:
            return
        point = Point(indices,
                      url_nxdata_path=f"{self._file_name}?{self._nxprocess_path}/result"
        )

        if self._background_point:
            curve = point.get_curve() - self._background_point.get_curve()
        else:
            curve = point.get_curve()
        if self._subtract_background:
            lower, upper = self._integrated_plot_widget.background_fit_roi.getRange()
            start = numpy.searchsorted(point._radial_curve, lower, side="left")
            stop = numpy.searchsorted(point._radial_curve, upper, side="right")
            if stop - start >= 3:
                smoothness = (
                    None
                    if self._background_dialog.automatic.isChecked()
                    else self._background_dialog.smoothness.value()
                )
                curve = curve[start:stop] - arpls(
                    curve[start:stop], smoothness=smoothness
                )
                radial_curve = point._radial_curve[start:stop]
            else:
                radial_curve = point._radial_curve
        else:
            radial_curve = point._radial_curve

        self._integrated_plot_widget.addDataCurve(
            x=radial_curve,
            y=curve,
            legend=legend,
            selectable=False,
            resetzoom=self._integrated_plot_widget.getGraphXLimits() == (0, 100),
        )
        self._integrated_plot_widget.setGraphXLabel(point._x_name)
        self._integrated_plot_widget.setDataYLabel(point._y_name)

    def getMask(self, image, maskfile=None):
        """returns a 2D array of boolean with invalid pixels masked,
        combination of Detector mask, static & dynamic mask.
        Handles detector/mask image binning on the fly

        :param image: 2D array image with data, used for dynamic masking
        :param maskfile: filename or URL pointing to a static mask
        :return: 2D array
        """
        if maskfile:
            mask_image = get_mask_image(maskfile, image.shape)
        else:
            mask_image = None

        detector = self.worker_config.poni.detector
        if not detector:
            return mask_image

        detector_mask = detector.mask
        if detector.shape != image.shape:
            detector.guess_binning(image)
            detector_mask = binning(detector_mask, detector.binning)

        if mask_image is None:
            detector.mask = detector_mask
        elif detector_mask is None:
            detector.mask = mask_image
        else:
            detector.mask = numpy.logical_or(mask_image, detector_mask)

        return detector.dynamic_mask(image)

    def displayImageAtIndices(self, indices: ImageIndices):
        if self._file_name is None:
            return
        row = indices.row
        col = indices.col

        with h5py.File(self._file_name, "r") as h5file:
            nxprocess = h5file[self._nxprocess_path]
            map_dataset = get_signal_dataset(nxprocess, "result", default="intensity")
            axes_index = get_axes_index(map_dataset)
            map_shape = map_dataset.shape
            if self._map_ptr is None:
                logger.warning("No `map_ptr` defined: guessing the frame indices !")
                image_index = row * map_shape[axes_index.fast] + col + self._offset
            else:
                image_index = self._map_ptr[row, col]

            if self._dataset_paths:
                for dataset_path, size in self._dataset_paths.items():
                    if image_index < size:
                        break
                    else:
                        image_index -= size
            else:
                self.warning(f"No diffraction data images found in {self._file_name}")
                return
            try:
                image_dset = get_dataset(h5file, dataset_path)
            except KeyError:
                image_link = h5file.get(dataset_path, getlink=True)
                self.warning(f"Cannot access diffraction images at {image_link}")
                return

            if image_index >= len(image_dset):
                return

            image = image_dset[image_index]

            if "maskfile" in h5file[self._nxprocess_path]:
                maskfile = bytes.decode(h5file[self._nxprocess_path]["maskfile"][()])
            else:
                maskfile = None

        image_base = ImageBase(data=image, mask=self.getMask(image, maskfile))
        title = f"{image_dset.file.filename}\n::{image_dset.name}\n#{image_index}"
        self._image_plot_widget.setImageData(image_base.getValueData(), title)

    def selectMapPoint(self, x: float, y: float):
        map_plot_widget = self.sender()
        if not isinstance(map_plot_widget, MapPlotWidget):
            map_plot_widget = self._map_plot_widget
        indices = map_plot_widget.getImageIndices(x, y)
        if indices is None:
            return

        if indices == self._unfixed_indices:
            return
        else:
            self._unfixed_indices = indices

        self.displayPatternAtIndices(indices, legend="INTEGRATE")
        self.displayImageAtIndices(indices)
        self.setMapMarker(
            indices,
            color=self.getCurveColor(legend="INTEGRATE"),
            symbol="o",
            legend="MAP_LOCATION",
        )

    def fixMapPoint(self, x: float, y: float):
        map_plot_widget = self.sender()
        if not isinstance(map_plot_widget, MapPlotWidget):
            map_plot_widget = self._map_plot_widget
        indices = map_plot_widget.getImageIndices(x, y)

        if indices is None:
            return
        # Remove curve and marker if the fixing point is the last clicked
        if indices == self._unfixed_indices:
            self._unfixed_point = None
            self._integrated_plot_widget.removeCurve(legend="INTEGRATE")
            self.removeMapMarker(legend="MAP_LOCATION")

        # Unfix is the fixing point is already fixed
        if indices in self._fixed_indices:
            self.removeMapPoint(indices=indices)

        # Fix the point
        else:
            self._fixed_indices.add(indices)

            legend = f"INTEGRATE_{indices.row}_{indices.col}"
            self.displayPatternAtIndices(
                indices,
                legend=legend,
            )
            used_color = self._integrated_plot_widget.getCurve(legend=legend).getColor()

            self.displayImageAtIndices(indices)
            self.setMapMarker(
                indices,
                color=used_color,
                symbol="d",
                legend=f"MAP_LOCATION_{indices.row}_{indices.col}",
            )

    def removeMapPoint(self, indices: ImageIndices):
        self._fixed_indices.remove(indices)
        self._integrated_plot_widget.removeCurve(
            legend=f"INTEGRATE_{indices.row}_{indices.col}"
        )
        self.removeMapMarker(legend=f"MAP_LOCATION_{indices.row}_{indices.col}")

    def setBackgroundSubtraction(self, enabled):
        self._subtract_background = enabled
        if self._unfixed_indices is not None:
            self.displayPatternAtIndices(self._unfixed_indices, legend="INTEGRATE")
        for indices in self._fixed_indices:
            self.displayPatternAtIndices(
                indices, legend=f"INTEGRATE_{indices.row}_{indices.col}"
            )
        self.onRoiEdition()
        if self._rgb_map_plot_widget is not None:
            self.displayRgbMap()

    def setBackgroundFitRange(self):
        radial = self._background_radial_values
        if radial is None:
            return
        roi = self._integrated_plot_widget.background_fit_roi
        lower, upper = roi.getRange()
        lower = max(float(radial[0]), min(float(radial[-1]), lower))
        upper = max(lower, min(float(radial[-1]), upper))
        if (lower, upper) != roi.getRange():
            roi.setRange(lower, upper)
            return
        self._background_baseline = None
        start = numpy.searchsorted(radial, lower, side="left")
        stop = numpy.searchsorted(radial, upper, side="right")
        self._background_dialog.setFitPointCount(stop - start)
        if self._subtract_background:
            self.setBackgroundSubtraction(True)

    def setBackgroundSmoothness(self):
        self._background_baseline = None
        if self._subtract_background:
            self.setBackgroundSubtraction(True)

    def _backgroundMap(self, full_map, axes_index, radial):
        lower, upper = self._integrated_plot_widget.background_fit_roi.getRange()
        start = numpy.searchsorted(radial, lower, side="left")
        stop = numpy.searchsorted(radial, upper, side="right")
        key = (self._file_name, self._nxprocess_path, full_map.shape, start, stop)
        if stop - start < 3:
            return None, start, stop
        if self._background_baseline is None or self._background_baseline[0] != key:
            selection = [slice(None)] * full_map.ndim
            selection[axes_index.radial] = slice(start, stop)
            data = numpy.asarray(full_map[tuple(selection)], dtype=float)
            moved = numpy.moveaxis(data, axes_index.radial, -1)
            histograms = moved.reshape(-1, moved.shape[-1])
            smoothness = (
                None
                if self._background_dialog.automatic.isChecked()
                else self._background_dialog.smoothness.value()
            )
            baselines = numpy.array(
                [arpls(hist, smoothness=smoothness) for hist in histograms]
            )
            background = numpy.moveaxis(
                baselines.reshape(moved.shape), -1, axes_index.radial
            )
            self._background_baseline = (key, background)
        return self._background_baseline[1], start, stop

    def onRoiEdition(self):
        v_min, v_max = self.getRoiRadialRange()
        if v_min is None or v_max is None:
            return

        self.displayAverageMap(v_min, v_max)

    def roiModeChanged(self, mode):
        if mode == "rgb":
            self.displayRgbMap()
            if self._rgb_map_plot_widget is not None:
                self._map_tab_widget.setCurrentWidget(self._rgb_map_plot_widget)
        else:
            self._map_tab_widget.setCurrentWidget(self._map_plot_widget)
        self.drawContoursOnImage()

    def setRgbMapChannel(self, channel):
        self._rgb_map_channel = channel
        if self._rgb_map_plot_widget is not None:
            self._rgb_map_plot_widget.setRgbChannel(channel)

    def drawContoursOnImage(self):
        roi = self._integrated_plot_widget.activeRoi()
        v_min, v_max = roi.getRange()
        if v_min is None or v_max is None:
            return
        color = roi.getColor()
        self._image_plot_widget.clearCurves()

        min_contours = find_contours(self._radial_matrix, v_min)
        for i, contour in enumerate(min_contours):
            self._image_plot_widget.addContour(
                contour, legend=f"min_contour_{i}", color=color
            )

        center_contours = find_contours(
            self._radial_matrix, v_min + (v_max - v_min) / 2
        )
        for i, contour in enumerate(center_contours):
            self._image_plot_widget.addContour(
                contour, legend=f"center_contour_{i}", color=color, linestyle=":"
            )

        max_contours = find_contours(self._radial_matrix, v_max)
        for i, contour in enumerate(max_contours):
            self._image_plot_widget.addContour(
                contour,
                legend=f"max_contour_{i}",
                color=color,
            )

    def displayRgbMap(self):
        title = "2θ RGB"
        created = self._rgb_map_plot_widget is None
        if created:
            self._rgb_map_plot_widget = self.addMapTab(title)
        if self._file_name is None:
            return
        ranges = [
            self._integrated_plot_widget.rgb_rois[channel].getRange()
            for channel in "RGB"
        ]
        if any(v_min is None or v_max is None for v_min, v_max in ranges):
            return

        maps = []
        with h5py.File(self._file_name, "r") as h5file:
            nxdata = h5file[self._nxprocess_path + "/result"]
            radial = get_radial_dataset(
                nxdata, size=self.worker_config.nbpt_rad
            )[()]
            full_map = get_signal_dataset(nxdata, default="intensity")
            axes_index = get_axes_index(full_map)
            background = None
            if self._subtract_background:
                background, background_start, background_stop = self._backgroundMap(
                    full_map, axes_index, radial
                )

            for v_min, v_max in ranges:
                i_min, i_max = get_indices_from_values(v_min, v_max, radial)
                i_min = max(0, i_min)
                i_max = min(len(radial), i_max)
                if i_min >= i_max:
                    return
                if axes_index.radial == 2:
                    map_data = full_map[:, :, i_min:i_max].mean(axis=2)
                else:
                    map_data = full_map[i_min:i_max, :, :].mean(axis=0)
                if (
                    background is not None
                    and background_start <= i_min < i_max <= background_stop
                ):
                    if axes_index.radial == 2:
                        map_data = map_data - background[
                            :, :, i_min - background_start:i_max - background_start
                        ].mean(axis=2)
                    else:
                        map_data = map_data - background[
                            i_min - background_start:i_max - background_start, :, :
                        ].mean(axis=0)
                maps.append(numpy.asarray(map_data, dtype=float))

            fast = get_axes_dataset(nxdata, dim=axes_index.fast, default="fast")
            slow = get_axes_dataset(nxdata, dim=axes_index.slow, default="slow")
            fast_name = fast.attrs.get("long_name", "X")
            fast_values = fast[()]
            slow_name = slow.attrs.get("long_name", "Y")
            slow_values = slow[()]

        rgb = numpy.stack(maps, axis=2)

        self._rgb_map_plot_widget.setRgbData(
            rgb, fast_values, slow_values, fast_name, slow_name
        )
        self._rgb_map_plot_widget.setRgbChannel(self._rgb_map_channel)
        if created and self._unfixed_indices is not None:
            coordinates = self._rgb_map_plot_widget.getMapPointCoordinates(
                self._unfixed_indices
            )
            if coordinates is not None:
                self._rgb_map_plot_widget.addMarker(
                    *coordinates,
                    color=self.getCurveColor(legend="INTEGRATE"),
                    symbol="o",
                    legend="MAP_LOCATION",
                )
        if created and self._background_point is not None:
            coordinates = self._rgb_map_plot_widget.getMapPointCoordinates(
                self._background_point.indices
            )
            if coordinates is not None:
                self._rgb_map_plot_widget.addMarker(
                    *coordinates, color="black", symbol="x", legend="BG_LOCATION"
                )
        index = self._map_tab_widget.indexOf(self._rgb_map_plot_widget)
        self._map_tab_widget.setTabText(index, title)

    def displayAverageMap(self, v_min: float, v_max: float):
        if self._file_name is None:
            return

        with h5py.File(self._file_name, "r") as h5file:
            nxprocess = h5file.get(self._nxprocess_path)
            nxdata = nxprocess["result"]
            radial = get_radial_dataset(nxdata, size=self.worker_config.nbpt_rad)[()]
            i_min, i_max = get_indices_from_values(v_min, v_max, radial)
            i_min = max(0, i_min)
            i_max = min(len(radial), i_max)
            # An unset or sub-bin ROI can select no samples. Keep the current
            # map rather than replacing it with the mean of an empty slice.
            if i_min >= i_max:
                return
            full_map = get_signal_dataset(nxdata, default="intensity")
            axes_index = get_axes_index(full_map)
            if axes_index.radial == 2:
                map_data = full_map[:,:, i_min:i_max].mean(axis=2)
            else:
                map_data = full_map[i_min:i_max, :, : ].mean(axis=0)
            if self._subtract_background:
                background, background_start, background_stop = self._backgroundMap(
                    full_map, axes_index, radial
                )
                if background_start <= i_min < i_max <= background_stop:
                    if axes_index.radial == 2:
                        map_data = map_data - background[
                            :, :, i_min - background_start:i_max - background_start
                        ].mean(axis=2)
                    else:
                        map_data = map_data - background[
                            i_min - background_start:i_max - background_start, :, :
                        ].mean(axis=0)
            fast = get_axes_dataset(nxdata, dim=axes_index.fast, default="fast")
            slow = get_axes_dataset(nxdata, dim=axes_index.slow, default="slow")
            fast_name = fast.attrs.get("long_name", "X")
            fast_values = fast[()]
            slow_name = slow.attrs.get("long_name", "Y")
            slow_values = slow[()]
        self._map_plot_widget.setScatterData(map_data, fast_values, slow_values, fast_name, slow_name)

    def onMouseClickOnImage(self, x: float, y: float):
        indices = self._image_plot_widget.getImageIndices(x, y)
        if indices is None:
            return
        radial_value = self._radial_matrix[indices.row, indices.col]
        self._integrated_plot_widget.setActiveRoiRange(
            radial_value - self._delta_radial_over_2,
            radial_value + self._delta_radial_over_2,
        )

    def getCurveColor(self, legend: str):
        curve = self._integrated_plot_widget.getCurve(legend=legend)
        if curve:
            return curve.getColor()
        else:
            return

    def getAvailableColor(self, legend: str):
        if self._integrated_plot_widget.getCurve(legend=legend):
            color = self._integrated_plot_widget.getCurve(legend=legend).getColor()
        else:
            color, _style = self._integrated_plot_widget._getColorAndStyle()
        return color

    def setNewBackgroundCurve(self, x: float, y: float):
        map_plot_widget = self.sender()
        if not isinstance(map_plot_widget, MapPlotWidget):
            map_plot_widget = self._map_plot_widget
        new_indices = map_plot_widget.getImageIndices(x, y)
        if new_indices is None or self._file_name is None:
            return

        new_background_point = Point(
            new_indices,
            url_nxdata_path=f"{self._file_name}?{self._nxprocess_path}/result"
        )

        # Unset the background if it's the same pixel and delete markers
        if (
            self._background_point
            and self._background_point.indices == new_background_point.indices
        ):
            self.removeMapMarker(legend="BG_LOCATION")
            self._background_point = None
        else:
            self._background_point = new_background_point
            self.setMapMarker(
                new_indices,
                color="black",
                symbol="x",
                legend="BG_LOCATION",
            )

        # Refresh displayed curves
        if self._unfixed_indices:
            self.displayPatternAtIndices(self._unfixed_indices, legend="INTEGRATE")

        for indices in self._fixed_indices:
            self.displayPatternAtIndices(
                indices, legend=f"INTEGRATE_{indices.row}_{indices.col}"
            )

    def clearPoints(self):
        for indices in self._fixed_indices.copy():
            self.removeMapPoint(indices=indices)

    def warning(self, error_msg):
        """Log a warning both in the terminal and in the status bar if possible

        :param error_msg: string with the message
        """
        logger.warning(error_msg)
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(error_msg)

    def closeEvent(self, event):
        self._gui_state.autosave()
        super().closeEvent(event)

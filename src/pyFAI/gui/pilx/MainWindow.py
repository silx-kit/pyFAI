#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023-2025 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "27/01/2025"
__status__ = "development"

from typing import Tuple
import json
import h5py
import logging
import os.path
import posixpath
from silx.gui import qt
from silx.gui.colors import Colormap
from silx.image.marchingsquares import find_contours
from silx.io.url import DataUrl
from silx.io import get_data
from silx.gui.plot.items.image import ImageBase

from .models import ImageIndices
from .point import Point
from .utils import (
    compute_radial_values,
    get_dataset,
    get_indices_from_values,
    get_radial_dataset,
)
from .widgets.DiffractionImagePlotWidget import DiffractionImagePlotWidget
from .widgets.IntegratedPatternPlotWidget import IntegratedPatternPlotWidget
from .widgets.MapPlotWidget import MapPlotWidget
from .widgets.TitleWidget import TitleWidget
from ...io.integration_config import WorkerConfig


class MainWindow(qt.QMainWindow):
    sigFileChanged = qt.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._file_name: str | None = None

        self.setWindowTitle("PyFAI-diffmap viewer")

        self._image_plot_widget = DiffractionImagePlotWidget(self)
        self._image_plot_widget.setDefaultColormap(
            Colormap("gray", normalization="log")
        )
        self._image_plot_widget.setKeepDataAspectRatio(True)
        self._image_plot_widget.plotClicked.connect(self.onMouseClickOnImage)

        self._map_plot_widget = MapPlotWidget(self)
        self._map_plot_widget.clearPointsSignal.connect(self.clearPoints)
        self._map_plot_widget.setDefaultColormap(
            Colormap("viridis", normalization="log")
        )
        self._map_plot_widget.plotClicked.connect(self.selectMapPoint)
        self._map_plot_widget.pinContextEntrySelected.connect(self.fixMapPoint)
        self._map_plot_widget.setBackgroundClicked.connect(self.setNewBackgroundCurve)

        self.sigFileChanged.connect(self._map_plot_widget.onFileChange)

        self._integrated_plot_widget = IntegratedPatternPlotWidget(self)
        self._integrated_plot_widget.roi.sigRegionChanged.connect(self.onRoiEdition)
        self._integrated_plot_widget.roi.sigRangeChanged.connect(
            self.drawContoursOnImage
        )

        self._title_widget = TitleWidget(self)

        self._central_widget = qt.QWidget()
        layout = qt.QGridLayout(self._central_widget)
        layout.setSpacing(0)
        layout.addWidget(self._title_widget, 0, 0, 1, 2)
        layout.addWidget(self._image_plot_widget, 1, 0, 2, 1)
        layout.addWidget(self._map_plot_widget, 1, 1)
        layout.addWidget(self._integrated_plot_widget, 2, 1)
        self._central_widget.setLayout(layout)
        self.setCentralWidget(self._central_widget)

        self._unfixed_indices = None
        self._fixed_indices = set()
        self._background_point = None
        self.worker_config = None

    def initData(self,
                 file_name: str,
                 dataset_path: str="/entry_0000/measurement/images_0001",
                 nxprocess_path: str="/entry_0000/pyFAI",
                 ):

        self._file_name = os.path.abspath(file_name)
        self._dataset_paths = {}
        self._nxprocess_path = nxprocess_path

        self.sigFileChanged.emit(self._file_name)

        with h5py.File(self._file_name, "r") as h5file:
            nxprocess = h5file[self._nxprocess_path]
            map_data = get_dataset(nxprocess, "result/intensity")[()].sum(axis=-1)
            try:
                slow = get_dataset(nxprocess, "result/slow")
            except Exception:
                slow_label = slow_values = None
            else:
                slow_label = slow.attrs.get("long_name", "Y")
                slow_values = slow[()]
            try:
                fast = get_dataset(nxprocess, "result/fast")
            except Exception:
                fast_values = fast_label = None
            else:
                fast_label = fast.attrs.get("long_name", "X")
                fast_values = fast[()]

            pyFAI_config_as_str = get_dataset(
                parent=nxprocess,
                path=f"configuration/data"
            )[()]
            self.worker_config = WorkerConfig.from_dict(json.loads(pyFAI_config_as_str), inplace=True)

            radial_dset = get_radial_dataset(
                h5file, nxdata_path=f"{self._nxprocess_path}/result",
                size=self.worker_config.nbpt_rad
            )
            delta_radial = (radial_dset[-1] - radial_dset[0]) / len(radial_dset)

            if "offset" in h5file[self._nxprocess_path]:
                self._offset = h5file[f"{self._nxprocess_path}/offset"][()]
            else:
                self._offset = 0

            # Find source dataset paths
            cnt = 0
            for char in dataset_path[-1::-1]:
                if char.isdigit():
                    cnt += 1
                else:
                    break

            _dataset_path = dataset_path[:-cnt]
            path, base = posixpath.split(_dataset_path)

            try:
                image_grp = h5file[path]
            except KeyError:
                error_msg = f"Cannot access diffraction images at {path}: no such path."
                logging.warning(error_msg)
                status_bar = self.statusBar()
                if status_bar:
                    status_bar.showMessage(error_msg)
            else:
                if not isinstance(image_grp, h5py.Group):
                    error_msg = f"Cannot access diffraction images at {path}: not a group."
                    logging.warning(error_msg)
                    status_bar = self.statusBar()
                    if status_bar:
                        status_bar.showMessage(error_msg)
                else:
                    lst = []
                    for key in image_grp:
                        if key.startswith(base) and isinstance(image_grp[key], h5py.Dataset):
                            lst.append(key)
                    lst.sort()
                    for key in lst:
                        self._dataset_paths[posixpath.join(path, key)] = len(image_grp[key])

        self._radial_matrix = compute_radial_values(self.worker_config)
        self._delta_radial_over_2 = delta_radial / 2

        self._title_widget.setText(os.path.basename(file_name))
        self._map_plot_widget.setScatterData(map_data, fast_values, slow_values, fast_label, slow_label)
        # BUG: selectMapPoint(0, 0) does not work at first render cause the picking fails
        initial_indices = ImageIndices(0, 0)
        self._unfixed_indices = initial_indices
        self.displayPatternAtIndices(initial_indices, legend="INTEGRATE")
        self.displayImageAtIndices(initial_indices)
        self._map_plot_widget.addMarker(
            0,
            0,
            color=self.getCurveColor(legend="INTEGRATE"),
            symbol="o",
            legend="MAP_LOCATION",
        )

    def getRoiRadialRange(self) -> Tuple[float | None, float | None]:
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

        self._integrated_plot_widget.addCurve(
            x=point._radial_curve,
            y=curve,
            legend=legend,
            selectable=False,
            resetzoom=self._integrated_plot_widget.getGraphXLimits() == (0, 100),
        )
        self._integrated_plot_widget.setGraphXLabel(point._x_name)
        self._integrated_plot_widget.setGraphYLabel(point._y_name)

    def displayImageAtIndices(self, indices: ImageIndices):
        if self._file_name is None:
            return
        row = indices.row
        col = indices.col

        with h5py.File(self._file_name, "r") as h5file:
            nxprocess = h5file[self._nxprocess_path]
            map_shape = get_dataset(nxprocess, "result/intensity").shape
            image_index = row * map_shape[1] + col + self._offset
            for dataset_path, size in self._dataset_paths.items():
                if image_index < size:
                    break
                else:
                    image_index -= size
            try:
                image_dset = get_dataset(h5file, dataset_path)
            except KeyError:
                image_link = h5file.get(dataset_path, getlink=True)
                error_msg = f"Cannot access diffraction images at {image_link}"
                logging.warning(error_msg)
                status_bar = self.statusBar()
                if status_bar:
                    status_bar.showMessage(error_msg)
                return

            if image_index >= len(image_dset):
                return

            image = image_dset[image_index]

            if "maskfile" in h5file[self._nxprocess_path]:
                maskfile = bytes.decode(h5file[self._nxprocess_path]["maskfile"][()])
            else:
                maskfile = None

        if maskfile:
            mask_image = get_data(url=DataUrl(maskfile))
            if mask_image.shape != image.shape:
                mask_image = None
        else:
            mask_image = None

        image_base = ImageBase(data=image, mask=mask_image)
        self._image_plot_widget.setImageData(image_base.getValueData())
        self._image_plot_widget.setGraphTitle(f"{posixpath.basename(dataset_path)} #{image_index}")

    def selectMapPoint(self, x: float, y: float):
        indices = self._map_plot_widget.getImageIndices(x, y)
        if indices is None:
            return

        if indices == self._unfixed_indices:
            return
        else:
            self._unfixed_indices = indices

        self.displayPatternAtIndices(indices, legend="INTEGRATE")
        self.displayImageAtIndices(indices)
        pixel_center_coords = self._map_plot_widget.findCenterOfNearestPixel(x, y)
        self._map_plot_widget.addMarker(
            *pixel_center_coords,
            color=self.getCurveColor(legend="INTEGRATE"),
            symbol="o",
            legend="MAP_LOCATION",
        )

    def fixMapPoint(self, x: float, y: float):
        indices = self._map_plot_widget.getImageIndices(x, y)

        if indices is None:
            return
        # Remove curve and marker if the fixing point is the last clicked
        if indices == self._unfixed_indices:
            self._unfixed_point = None
            self._integrated_plot_widget.removeCurve(legend="INTEGRATE")
            self._map_plot_widget.removeMarker(legend="MAP_LOCATION")

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
            pixel_center_coords = self._map_plot_widget.findCenterOfNearestPixel(x, y)
            self._map_plot_widget.addMarker(
                *pixel_center_coords,
                color=used_color,
                symbol="d",
                legend=f"MAP_LOCATION_{indices.row}_{indices.col}",
            )

    def removeMapPoint(self, indices: ImageIndices):
        self._fixed_indices.remove(indices)
        self._integrated_plot_widget.removeCurve(
            legend=f"INTEGRATE_{indices.row}_{indices.col}"
        )
        self._map_plot_widget.removeMarker(
            legend=f"MAP_LOCATION_{indices.row}_{indices.col}"
        )

    def onRoiEdition(self):
        v_min, v_max = self.getRoiRadialRange()
        if v_min is None or v_max is None:
            return

        self.displayAverageMap(v_min, v_max)

    def drawContoursOnImage(self):
        v_min, v_max = self.getRoiRadialRange()
        if v_min is None or v_max is None:
            return
        self._image_plot_widget.clearCurves()

        min_contours = find_contours(self._radial_matrix, v_min)
        for i, contour in enumerate(min_contours):
            self._image_plot_widget.addContour(contour, legend=f"min_contour_{i}")

        center_contours = find_contours(
            self._radial_matrix, v_min + (v_max - v_min) / 2
        )
        for i, contour in enumerate(center_contours):
            self._image_plot_widget.addContour(
                contour, legend=f"center_contour_{i}", linestyle=":"
            )

        max_contours = find_contours(self._radial_matrix, v_max)
        for i, contour in enumerate(max_contours):
            self._image_plot_widget.addContour(
                contour,
                legend=f"max_contour_{i}",
            )

    def displayAverageMap(self, v_min: float, v_max: float):
        if self._file_name is None:
            return

        with h5py.File(self._file_name, "r") as h5file:
            radial = get_radial_dataset(h5file, nxdata_path=f"{self._nxprocess_path}/result")[
                ()
            ]
            i_min, i_max = get_indices_from_values(v_min, v_max, radial)
            map_data = get_dataset(h5file, f"{self._nxprocess_path}/result/intensity")[:,:, i_min:i_max
            ].mean(axis=2)
            fast = get_dataset(h5file, f"{self._nxprocess_path}/result/fast")
            slow = get_dataset(h5file, f"{self._nxprocess_path}/result/slow")
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
        self._integrated_plot_widget.roi.setRange(
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
            color, style = self._integrated_plot_widget._getColorAndStyle()
        return color

    def setNewBackgroundCurve(self, x: float, y: float):
        new_indices = self._map_plot_widget.getImageIndices(x, y)
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
            self._map_plot_widget.removeMarker(legend="BG_LOCATION")
            self._background_point = None
        else:
            self._background_point = new_background_point
            pixel_center_coords = self._map_plot_widget.findCenterOfNearestPixel(x, y)
            self._map_plot_widget.addMarker(
                *pixel_center_coords, color="black", symbol="x", legend=f"BG_LOCATION"
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

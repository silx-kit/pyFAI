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

from collections import namedtuple
from .utils import (
    get_dataset,
    get_dataset_name,
    get_radial_dataset,
)
import h5py
import copy

ROI_COLOR = "orange"

ImageIndices = namedtuple("ImageIndices", ["row", "col"])

class DiffMapViewPoint():
    def __init__(self, row, col, file_name, legend="", color=None):
        self._row = row
        self._col = col
        self._file_name = file_name
        self._legend = legend
        self._curve = None
        self._color = color
        self._set_curve()

    @property
    def legend(self):
        return self._legend

    def __eq__(self, other) -> bool:
        if isinstance(other, DiffMapViewPoint):
            return (self._row, self._col) == (other._row, other._col)
        else:
            return False

    def __repr__(self) -> str:
        return str((self._row, self._col))

    def __add__(self, other):
        return self._curve + other._curve

    def __sub__(self, other):
        return self._curve - other._curve

    def _set_curve(self):
        with h5py.File(self._file_name, "r") as h5file:
            radial_dset = get_radial_dataset(
                h5file, nxdata_path="/entry_0000/pyFAI/result"
            )
            self._radial_curve = radial_dset[()]
            self._x_name = get_dataset_name(radial_dset)
            intensity_dset = get_dataset(h5file, "/entry_0000/pyFAI/result/intensity")
            self._intensity_curve = intensity_dset[self._row, self._col, :]
            self._y_name = intensity_dset.attrs.get("long_name", "Intensity")

    def get_curve(self):
        return self._intensity_curve


class DisplayedPoint(DiffMapViewPoint):
    def __init__(self, row, col, file_name, legend="", background_point=None, color=None) -> None:
        super().__init__(
            row=row,
            col=col,
            file_name=file_name,
            legend=legend,
            color=color,
        )
        self.set_background(background_point)

    def set_background(self, background_point:DiffMapViewPoint):
        self._background = background_point

    def get_curve(self):
        if self._background:
            return copy.copy(self._intensity_curve) - self._background.get_curve()
        else:
            return copy.copy(self._intensity_curve)

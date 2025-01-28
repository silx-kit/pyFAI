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
__author__ = "Loïc Huder"
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/01/2025"
__status__ = "development"

from typing import Iterable, Optional
import logging
logger = logging.getLogger(__name__)
import json
import h5py
import numpy
import os.path
from ...integrator.azimuthal import AzimuthalIntegrator
from ...detectors import Detector
from ...io.integration_config import WorkerConfig


def compute_radial_values(worker_config: WorkerConfig) -> numpy.ndarray:
    ai = AzimuthalIntegrator.sload(worker_config.poni)
    scaled_values = ai.center_array(worker_config.shape, worker_config.unit)
    return scaled_values


def get_indices_from_values(vmin: float,
                            vmax: float,
                            radial_values: numpy.ndarray
                            ) -> tuple[int, int]:
    step = (radial_values[-1] - radial_values[0]) / len(radial_values)
    init_val = radial_values[0]

    return int((vmin - init_val) / step), int((vmax - init_val) / step)


def get_dataset(parent: h5py.Group | h5py.File, path: str) -> h5py.Dataset:
    dset = parent[path]
    assert isinstance(dset, h5py.Dataset)
    return dset


def get_radial_dataset(parent: h5py.Group,
                       nxdata_path: str,
                       size: Optional[int]=None) -> h5py.Dataset:
    nxdata = parent[nxdata_path]
    assert isinstance(nxdata, h5py.Group)
    assert nxdata.attrs["NX_class"] == "NXdata"
    if size is None:
        if "intensity" in nxdata:
            size = nxdata["intensity"].shape[-1]
        else:
            size = nxdata[nxdata.attrs["signal"]].shape[0]
    axes = nxdata.attrs["axes"]
    if isinstance(axes, Iterable):
        radial_path = axes[0]
        if size is not None:
            for idx in [0, -1, 1, -2]:
                radial_path = axes[idx]
                if radial_path in nxdata and isinstance(nxdata[radial_path], h5py.Dataset):
                    ds = get_dataset(nxdata, radial_path)
                    if ds.shape[0] == size:
                        break
            else:
                logger.warning("No dataset matchs radial size !")
    else:
        radial_path = axes
    return get_dataset(nxdata, radial_path)


def get_dataset_name(dataset: h5py.Dataset):
    return dataset.attrs.get("long_name", os.path.basename(dataset.name))


def guess_axis_path(existing_axis_path: str, parent: h5py.Group) -> str | None:
    options: set[str] = {"x", "y", "z"}
    axis_template, existing_axis = existing_axis_path[:-1], existing_axis_path[-1:]
    if existing_axis in options:
        options.remove(existing_axis)

    for axis in options:
        guessed_axis_path = axis_template + axis
        if guessed_axis_path in parent:
            return guessed_axis_path

    return None

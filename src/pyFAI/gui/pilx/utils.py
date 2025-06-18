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
__date__ = "18/06/2025"
__status__ = "development"

from collections import namedtuple
import logging
from typing import Iterable, Optional, Tuple
import os.path
logger = logging.getLogger(__name__)


import h5py
import numpy
from silx.io import get_data
from silx.io.url import DataUrl

from ...integrator.azimuthal import AzimuthalIntegrator
from ...io.integration_config import WorkerConfig
from ...utils.mathutil import binning

AxisIndex = namedtuple("AxisIndex", "slow fast radial")

def get_axes_index(dataset):
    """Calculate the indices of the axis according to the interpretation of the dataset"""
    if dataset.attrs["interpretation"] == "image":
        res = AxisIndex(1, 2, 0)
    elif dataset.attrs["interpretation"] == "spectrum":
        res = AxisIndex(0, 1, 2)
    else:
        logger.warning(f"No interpretation for NXdata '{dataset.name}', guessing !")
        res = AxisIndex(0, 1, 2)
    return res

def compute_radial_values(worker_config: WorkerConfig) -> numpy.ndarray:
    ai = AzimuthalIntegrator.sload(worker_config.poni)
    if worker_config.shape is not None:
        ai.detector.guess_binning(worker_config.shape)
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
    if not isinstance(dset, h5py.Dataset):
        raise TypeError("dataset is not a `h5py.Dataset` instance")
    return dset


def get_NXdata(parent: h5py.Group | h5py.File,
               path: str | None = None):
    "Return a NXdata group with minimum checks"
    nxdata = parent[path] if path else parent

    if not isinstance(nxdata, h5py.Group):
        raise TypeError("NXdata is not a `h5py.Group` instance")
    attrs = nxdata.attrs
    if attrs.get("NX_class") != "NXdata":
        logger.warning(f"Expected a NXdata class for {nxdata.name}")
    return nxdata

def get_signal_dataset(parent: h5py.Group | h5py.File,
                       path: str | None = None,
                       default: str="intensity") -> h5py.Dataset:
    "Read the `signal` dataset associated to a NXdata, if no signal is provided, use the default one"
    nxdata = get_NXdata(parent, path)
    attrs = nxdata.attrs
    dset = nxdata[attrs.get("signal",default)]
    if not isinstance(dset, h5py.Dataset):
        raise TypeError(f"dataset '{dset}' is not a `h5py.Dataset` instance")
    return dset

def get_axes_dataset(parent: h5py.Group | h5py.File,
                     path: str | None = None,
                     dim: int = 0,
                     default: str="x") -> h5py.Dataset:
    "Read the `axes` dataset associated to a NXdata, if nothing along that axes, use the default one"
    nxdata = get_NXdata(parent, path)
    attrs = nxdata.attrs
    axes = attrs.get("axes")
    if axes is None:
        dset_name = default
    else:
        if dim<0:
            dim += len(axes)
        if dim>=len(axes):
            dset_name = default
        else:
            dset_name = axes[dim]
    if dset_name not in nxdata:
        raise RuntimeError(f"dataset `{dset_name}` does not exist in NXdata `{nxdata.name}`.")
    dset = nxdata[dset_name]
    if not isinstance(dset, h5py.Dataset):
        raise RuntimeError("dataset is not a `h5py.Dataset` instance")
    return dset


def get_radial_dataset(parent: h5py.Group,
                       nxdata_path: str | None = None,
                       size: Optional[int]=None) -> h5py.Dataset:
    nxdata = get_NXdata(parent, nxdata_path)
    if size is None:
        if "intensity" in nxdata:
            dset = nxdata["intensity"]
        else:
            dset = nxdata[nxdata.attrs.get("signal")]
        axes_index = get_axes_index(dset)
        size = dset.shape[axes_index.radial]
    else:
        axes_index = AxisIndex(1,2,0)
    axes = nxdata.attrs["axes"]
    if isinstance(axes, Iterable):
        radial_path = axes[axes_index.radial]
        if size is not None:
            for idx in [axes_index.radial, 0, -1, 1, -2]:
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


def get_mask_image(maskfile: str, image_shape: Tuple[int, int]) -> numpy.ndarray | None:
    """Retrieves mask image from the URL. Rebin to match"""
    mask_image = get_data(url=DataUrl(maskfile))
    if not isinstance(mask_image, numpy.ndarray):
        raise RuntimeError("Mask is expected to be a numpy array")
    if mask_image.shape == image_shape:
        return mask_image

    # If mismatched shapes, try to rebin
    bin_size = [m // i for i, m in zip(image_shape, mask_image.shape)]
    if bin_size[0] == 0 or bin_size[1] == 0:
        return None

    return binning(mask_image, bin_size)

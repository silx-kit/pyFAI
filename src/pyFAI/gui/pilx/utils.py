from __future__ import annotations
import json
from typing import Iterable
import h5py
import numpy
import os.path
import pyFAI
import pyFAI.detectors
import pyFAI.units


def compute_radial_values(pyFAI_config_as_str: str) -> numpy.ndarray:
    pyFAI_config: dict = json.loads(pyFAI_config_as_str)
    ai = pyFAI.load(pyFAI_config)
    if "detector" not in pyFAI_config:
        ai.detector = pyFAI.detectors.Detector(
            pixel1=pyFAI_config.get("pixel1"),
            pixel2=pyFAI_config.get("pixel2"),
            splineFile=pyFAI_config.get("splineFile"),
            max_shape=pyFAI_config.get("max_shape"),
        )

    # Scale manually for now
    # https://github.com/silx-kit/pyFAI/issues/1996
    unscaled_values = ai.center_array(
        pyFAI_config["shape"], pyFAI_config["unit"], scale=False
    )
    pyFAI_unit = pyFAI.units.to_unit(pyFAI_config["unit"])
    return unscaled_values * pyFAI_unit.scale


def get_indices_from_values(
    vmin: float, vmax: float, radial_values: numpy.ndarray
) -> tuple[int, int]:
    step = (radial_values[-1] - radial_values[0]) / len(radial_values)
    init_val = radial_values[0]

    return int((vmin - init_val) / step), int((vmax - init_val) / step)


def get_dataset(parent: h5py.Group | h5py.File, path: str) -> h5py.Dataset:
    dset = parent[path]
    assert isinstance(dset, h5py.Dataset)
    return dset


def get_radial_dataset(parent: h5py.Group, nxdata_path: str) -> h5py.Dataset:
    nxdata = parent[nxdata_path]
    assert isinstance(nxdata, h5py.Group)
    assert nxdata.attrs["NX_class"] == "NXdata"
    axes = nxdata.attrs["axes"]
    radial_path = axes[-1] if isinstance(axes, Iterable) else axes
    return get_dataset(nxdata, radial_path)


def get_dataset_name(dataset: h5py.Dataset):
    return dataset.attrs.get("long_name", os.path.basename(dataset.name))


def guess_axis_path(existing_axis_path: str, parent: h5py.Group) -> str | None:
    options: set[str] = {"x", "y", "z"}

    axis_template, existing_axis = existing_axis_path[:-1], existing_axis_path[-1:]
    options.remove(existing_axis)

    for axis in options:
        guessed_axis_path = axis_template + axis
        if guessed_axis_path in parent:
            return guessed_axis_path

    return None

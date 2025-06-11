# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2021 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Module for writing spots in HDF5 in the Nexus style"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/05/2025"
__status__ = "production"
__docformat__ = 'restructuredtext'

import sys
import os
import json
import posixpath
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
import numpy
import fabio
from .. import version
from ..units import to_unit
from ._json import UnitEncoder
from .nexus import Nexus, get_isotime, h5py

try:
    import hdf5plugin
except ImportError:
    cmp = {"chunks":True,
           "compression": "gzip",
           "compression_opts":1}
else:
    cmp = hdf5plugin.Bitshuffle()


def _stack_frames(fimg):
    """return a stack of images from a single or multiframe fabio object
    :param fimg: opened fabio image
    :return: 3d array
    """
    shape = (fimg.nframes,) + fimg.shape
    stack = numpy.empty(shape, dtype=fimg.dtype)
    for i, f in enumerate(fimg):
        stack[i] = f.data
    return stack


def save_spots_nexus(filename, spots, beamline="beamline", ai=None, source=None, extra={}, grid=None, powder=False):
    """Write the list of spots per frame into a HDF5 file with the Nexus convention

    :param filename: name of the file
    :param spots: list of spots per frame (as built by peakfinder)
    :param beamline: name of the beamline as text
    :param ai: Instance of geometry or azimuthal integrator
    :param source: list of input files
    :param extra: dict with extra metadata
    :param grid: 2-tuple with grid shape and if it was acquired in zigzag mode
    :param powder: unused
    :return: None
    """
    if len(spots) == 0:
        raise RuntimeError("No spot provided to save")
    spots_per_frame = numpy.array([len(s) for s in spots], dtype=numpy.int32)
    with Nexus(filename, mode="w", creator="pyFAI_%s" % version) as nexus:

        instrument = nexus.new_instrument(instrument_name=beamline)
        entry = instrument.parent
        peaks_grp = nexus.new_class(entry, "peaks", class_type="NXdata")
        entry.attrs["default"] = posixpath.relpath(peaks_grp.name, entry.name)
        if grid and grid[0] and len(grid[0]) > 1:
            img = spots_per_frame.reshape(grid[0])
            if grid[1]:
                img[1::2,:] = img[1::2, -1::-1]  # flip one line out of 2
            spot_ds = peaks_grp.create_dataset("spots", data=img)
            spot_ds.attrs["interpretation"] = "image"
        else:
            spot_ds = peaks_grp.create_dataset("spots", data=spots_per_frame)
            spot_ds.attrs["interpretation"] = "spectrum"
        peaks_grp.attrs["signal"] = "spots"

        peaks_grp["frame_ptr"] = numpy.concatenate(([0], numpy.cumsum(spots_per_frame))).astype(dtype=numpy.int32)
        index = numpy.concatenate([i["index"] for i in spots])
        peaks_grp.create_dataset("index", data=index, **cmp)

        intensity = numpy.concatenate([i["intensity"] for i in spots])
        peaks_grp.create_dataset("intensity", data=intensity, **cmp)

        sigma = numpy.concatenate([i["sigma"] for i in spots])
        peaks_grp.create_dataset("sigma", data=sigma, **cmp)

        # to have pos1 and pos2 along same dim as poni1 and poni2
        pos0 = numpy.concatenate([i["pos0"] for i in spots])
        peaks_grp.create_dataset("pos1", data=pos0, **cmp).attrs["dir"] = "y"

        pos1 = numpy.concatenate([i["pos1"] for i in spots])
        peaks_grp.create_dataset("pos2", data=pos1, **cmp).attrs["dir"] = "x"

        sparsify_grp = nexus.new_class(entry, "peakfinder", class_type="NXprocess")
        sparsify_grp["program"] = "pyFAI"
        sparsify_grp["sequence_index"] = 1
        sparsify_grp["version"] = version
        sparsify_grp["date"] = get_isotime()
        # sparsify_grp.create_dataset("command",
        #                        data=numpy.array(sys.argv, dtype=h5py.special_dtype(vlen=str)),
        #                        ).attrs["hint"] = "argv"

        sparsify_grp.create_dataset("argv", data=numpy.array(sys.argv, dtype=h5py.special_dtype(vlen=str))).attrs["help"] = "Command line arguments"
        sparsify_grp.create_dataset("cwd", data=os.getcwd()).attrs["help"] = "Working directory"
        if source is not None:
            sparsify_grp.create_dataset("source", data=numpy.array(source, dtype=h5py.special_dtype(vlen=str)))
        if ai is not None:
            config_grp = nexus.new_class(sparsify_grp, "configuration", class_type="NXnote")
            config_grp["type"] = "text/json"
            parameters = OrderedDict([("geometry", ai.get_config()),
                                      ("peakfinder", extra)])
            config_grp["data"] = json.dumps(parameters, indent=2, separators=(",\r\n", ": "), cls=UnitEncoder)

            detector_grp = nexus.new_class(instrument, ai.detector.name.replace(" ", "_"), "NXdetector")
            dist_ds = detector_grp.create_dataset("distance", data=ai.dist)
            dist_ds.attrs["units"] = "m"
            xpix_ds = detector_grp.create_dataset("x_pixel_size", data=ai.pixel2)
            xpix_ds.attrs["units"] = "m"
            ypix_ds = detector_grp.create_dataset("y_pixel_size", data=ai.pixel1)
            ypix_ds.attrs["units"] = "m"
            f2d = ai.getFit2D()
            xbc_ds = detector_grp.create_dataset("beam_center_x", data=f2d["centerX"])
            xbc_ds.attrs["units"] = "pixel"
            ybc_ds = detector_grp.create_dataset("beam_center_y", data=f2d["centerY"])
            ybc_ds.attrs["units"] = "pixel"
            if ai.wavelength is not None:
                monochromator_grp = nexus.new_class(instrument, "monchromator", "NXmonochromator")
                wl_ds = monochromator_grp.create_dataset("wavelength", data=numpy.float32(ai.wavelength * 1e10))
                wl_ds.attrs["units"] = "Å"


def save_spots_cxi(filename, spots, beamline="beamline", ai=None, source=None, extra={}, grid=None, powder=None):
    """Write the list of spots per frame into a HDF5 file with the CXI convention
    https://raw.githubusercontent.com/cxidb/CXI/master/cxi_file_format.pdf

    :param filename: name of the file
    :param spots: list of spots per frame (as built by peakfinder)
    :param beamline: name of the beamline as text
    :param ai: Instance of geometry or azimuthal integrator
    :param source: list of input files
    :param extra: dict with extra metadata
    :param grid: unused
    :param powder: provide the position in radial range to activate the calculation/saving of the pseudo-powder pattern
    :return: None
    """
    if len(spots) == 0:
        raise RuntimeError("No spots provided to save")
    spots_per_frame = numpy.array([len(s) for s in spots], dtype=numpy.int32)
    nframes = len(spots)
    max_spots = spots_per_frame.max()
    cxi = ai.getCXI() if ai else {}
    detector = None
    with h5py.File(filename, mode="w") as h5:
        h5["cxi_version"] = cxi.get("cxi_version", 100)

        entry = h5.create_group("entry_1")
        entry.attrs["NX_class"] = "NXentry"
        h5.attrs["default"] = entry.name.strip("/")
        result = entry.create_group("result_1")
        result.attrs["NX_class"] = "NXdata"
        result.attrs["signal"] = "nPeaks"
        nPeaks_ds = result.create_dataset("nPeaks", data=spots_per_frame)
        nPeaks_ds.attrs["interpretation"] = "spectrum"

        if grid and grid[0] and len(grid[0]) > 1:
            img = spots_per_frame.reshape(grid[0])
            if grid[1]:
                img[1::2,:] = img[1::2, -1::-1]  # flip one line out of 2
            spot_ds = result.create_dataset("map", data=img)
            spot_ds.attrs["interpretation"] = "image"

        total_int = numpy.zeros((nframes, max_spots), dtype=numpy.float32)
        xpos = numpy.zeros((nframes, max_spots), dtype=numpy.float32)
        ypos = numpy.zeros((nframes, max_spots), dtype=numpy.float32)
        snr = numpy.zeros((nframes, max_spots), dtype=numpy.float32)
        entry.attrs["default"] = posixpath.relpath(result.name, entry.name)

        for i, s in enumerate(spots):
            l = len(s)
            total_int[i,:l] = s["intensity"]
            xpos[i,:l] = s["pos1"]
            ypos[i,:l] = s["pos0"]
            snr[i,:l] = s["intensity"] / s["sigma"]
        result.create_dataset("peakTotalIntensity", data=total_int, **cmp)
        result.create_dataset("peakXPosRaw", data=xpos, **cmp)
        result.create_dataset("peakYPosRaw", data=ypos, **cmp)
        result.create_dataset("peakSNR", data=snr, **cmp)
        process = result.create_group("process_1")
        process.attrs["NX_class"] = "NXprocess"
        if extra:
            process["metadata"] = json.dumps(extra, indent=2, cls=UnitEncoder)
            process["metadata"].attrs["type"] = "text/json"
        process.create_dataset("command",
                               data=numpy.array(sys.argv, dtype=h5py.special_dtype(vlen=str)),
                               ).attrs["hint"] = "argv"
        process.create_dataset("cwd", data=os.getcwd()).attrs["help"] = "Working directory"
        process["date"] = get_isotime()
        process["program"] = "pyFAI"
        process["version"] = version

        if powder is not None:
            unit = to_unit(extra.get("unit", "r_mm"))
            r1d = powder * unit.scale
            dr = (r1d[1:] - r1d[:-1]).mean()
            rng = [r1d[0] - dr * 0.5, r1d[-1] + dr * 0.5]

            r2d = ai.array_from_unit(typ="center", unit=unit, scale=True)
            from ..ext.bilinear import Bilinear
            bl = Bilinear(r2d)
            histo = None
            for s in spots:
                r = bl.many((s["pos0"], s["pos1"]))
                res = numpy.histogram(r, bins=len(r1d), range=rng, weights=s["intensity"])
                if histo is None:
                    histo = res[0]
                else:
                    histo += res[0]

            powder = entry.create_group("powder_1")
            powder.attrs["signal"] = "I"
            powder.attrs["NX_class"] = "NXdata"
            powder['I'] = histo

            powder.attrs["axes"] = unit.name
            powder[unit.name] = r1d
            powder[unit.name].attrs["long_name"] = unit.label

            if unit.name.startswith("q"):
                name = unit.name.replace("q", "d").replace("^-1", "")
                r1d = numpy.pi * 2.0 / r1d
                unit = name.split('_')[-1]
                if unit == "A":
                    unit = "$\\AA$"
                label = f"$d$-spacing ({unit})"
                powder[name] = r1d
                powder[name].attrs["long_name"] = label

        if beamline:
            instrument = entry.create_group("instrument_1")
            instrument.attrs["NX_class"] = "NXinstrument"
            instrument["name"] = beamline
            if ai:

                detector = instrument.create_group("detector_1")
                detector.attrs["NX_class"] = "NXdetector"
                detector["description"] = str(ai.detector.aliases[0] if ai.detector.aliases else ai.detector.__class__.__name__)

                detector["distance"] = ai.dist
                detector["distance"].attrs["units"] = "m"
                if ai.detector.mask is not None:
                    detector.create_dataset("mask", data=ai.detector.mask, **cmp)
                detector["x_pixel_size"] = ai.detector.pixel2
                detector["y_pixel_size"] = ai.detector.pixel1
                detector["x_pixel_size"].attrs["units"] = "m"
                detector["y_pixel_size"].attrs["units"] = "m"
                # Probably useless: wastes disk space for nothing.
                # detector.create_dataset("corner_position",
                #                         data=ai.position_array(corners=True, dtype=numpy.float32, use_cython=True, do_parallax=False)[:,:,0,:],
                #                         **cmp).attrs["order"] = "zyx"#                                                                    ^ only first corner !
                ai_cxi = ai.getCXI()
                geo_cxi = ai_cxi["detector_1"]["geometry_1"]
                geo = detector.create_group("geometry_1")
                geo.attrs["NX_class"] = "NXgeometry"
                geo["translation"] = geo_cxi["translation"]
                geo["orientation"] = geo_cxi["orientation"]
                if ai.wavelength is not None:
                    beam = instrument.create_group("beam_1")
                    beam.attrs["NX_class"] = "NXbeam"
                    beam["incident_wavelength"] = ai.wavelength
                    beam["incident_wavelength"].attrs["units"] = "m"
        if source:
            for idx, file_src in enumerate(source):
                datagrp = entry.create_group(f"data_{idx+1}")
                datagrp.attrs["NX_class"] = "NXdata"
                datagrp.attrs["signal"] = "data"
                with fabio.open(file_src) as fimg:
                    try:
                        lst_ds = fimg.dataset
                    except Exception as err:
                        logger.error(f"file {file_src} of type {type(fimg)} has not dataset attribute, skipping ({type(err)}: {err})")
                        datagrp.create_dataset("data", data=_stack_frames(fimg), **cmp).attrs["interpretation"] = "image"
                    else:
                        if isinstance(lst_ds, h5py.Dataset):
                            lst_ds = [lst_ds]
                        if len(lst_ds) == 1:
                            datagrp["data"] = h5py.ExternalLink(file_src, lst_ds[0].name)
                            # datagrp[f"data"].attrs["interpretation"] = "image"
                        else:
                            for j, ds in enumerate(fimg.dataset):
                                datagrp[f"data_{j+1}"] = h5py.ExternalLink(file_src, ds.name)
                                # datagrp[f"data_{j+1}"].attrs["interpretation"] = "image"

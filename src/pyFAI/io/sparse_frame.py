# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module for writing sparse frames in HDF5 in the Nexus style"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/05/2025"
__status__ = "production"
__docformat__ = 'restructuredtext'

import os
import sys
import json
import posixpath
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
import numpy
import fabio
from .. import version
from .nexus import Nexus, get_isotime, h5py
from ._json import UnitEncoder

try:
    import hdf5plugin
except ImportError:
    cmp = {"chunks":True,
           "compression": "gzip",
           "compression_opts":1}
else:
    cmp = hdf5plugin.Bitshuffle()


def _generate_densify_script(integer):
    "Provide a script to densify those data"
    res = """#python
import numpy
frames = []
masked = numpy.where(numpy.logical_not(numpy.isfinite(mask)))
for idx, bg in enumerate(background_avg):
    dense = numpy.interp(mask, radius, bg)
    flat = dense.ravel()
    start, stop = frame_ptr[idx:idx+2]
    flat[index[start:stop]] = intensity[start:stop]"""
    if integer:
        res += """
    dense = numpy.round(dense)
    dense[masked] = dummy"""
    else:
        res += """
    dense[masked] = numpy.nan"""
    res += """
    frames.append(dense.astype(intensity.dtype))
"""
    return res


def save_sparse(filename, frames, beamline="beamline", ai=None, source=None, extra={}, start_time=None):
    """Write the list of frames into a HDF5 file

    :param filename: name of the file
    :param frames: list of sparse frames (as built by sparsify)
    :param beamline: name of the beamline as text
    :param ai: Instance of geometry or azimuthal integrator
    :param source: list of input files
    :param extra: dict with extra metadata
    :param start_time: float with the time of start of the processing
    :return: None
    """
    if len(frames) == 0:
        raise RuntimeError("No frame provided to save")
    with Nexus(filename, mode="w", creator="pyFAI_%s" % version, start_time=start_time) as nexus:
        instrument = nexus.new_instrument(instrument_name=beamline)
        entry = instrument.parent
        sparse_grp = nexus.new_class(entry, "sparse_frames", class_type="NXdata")
        entry.attrs["default"] = posixpath.relpath(sparse_grp.name, entry.name)
        entry.attrs["pyFAI_sparse_frames"] = sparse_grp.name
        sparse_grp["frame_ptr"] = numpy.concatenate(([0], numpy.cumsum([i.intensity.size for i in frames]))).astype(dtype=numpy.uint32)
        index = numpy.concatenate([i.index for i in frames]).astype(numpy.uint32)
        intensity = numpy.concatenate([i.intensity for i in frames])
        is_integer = numpy.issubdtype(intensity.dtype, numpy.integer)
        sparse_grp["script"] = _generate_densify_script(is_integer)
        sparse_grp.create_dataset("index", data=index, **cmp)
        sparse_grp.create_dataset("intensity", data=intensity, **cmp)
        radius = frames[0].radius
        mask = frames[0].mask
        dummy = frames[0].dummy
        unit = frames[0].unit
        if dummy is None:
            if is_integer:
                dummy = 0
            else:
                dummy = numpy.nan
        sparse_grp.create_dataset("dummy", data=dummy)
        rds = sparse_grp.create_dataset("radius", data=radius * unit.scale, dtype=numpy.float32)
        rds.attrs["interpretation"] = "spectrum"
        rds.attrs["unit"] = str(unit)
        rds.attrs["long_name"] = unit.label
        mskds = sparse_grp.create_dataset("mask", data=mask * unit.scale, **cmp)
        mskds.attrs["interpretation"] = "image"
        rds.attrs["long_name"] = unit.label
        mskds.attrs["unit"] = str(unit)
        background_avg = numpy.vstack([f.background_avg for f in frames])
        background_std = numpy.vstack([f.background_std for f in frames])
        bgavgds = sparse_grp.create_dataset("background_avg", data=background_avg, **cmp)
        bgavgds.attrs["interpretation"] = "spectrum"
        bgavgds.attrs["signal"] = 1
        bgavgds.attrs["long_name"] = "Average value of background"
        bgstdds = sparse_grp.create_dataset("background_std", data=background_std, **cmp)
        sparse_grp["errors"] = bgstdds
        bgstdds.attrs["interpretation"] = "spectrum"
        bgstdds.attrs["long_name"] = "Standard deviation of background"
        sparse_grp.attrs["signal"] = "background_avg"
        try:
            sparse_grp.attrs["axes"] = [".", "radius"]
        except TypeError:
            logger.error("Please upgrade your installation of h5py !!!")

        if frames[0].peaks is not None:
            peaks = [f.peaks for f in frames]
            spots_per_frame = numpy.array([len(s) for s in peaks], dtype=numpy.int32)
            nframes = len(frames)
            max_spots = spots_per_frame.max()

            peak_grp = nexus.new_class(entry, "peaks", class_type="NXdata")
            peak_grp.attrs["NX_class"] = "NXdata"
            peak_grp.attrs["signal"] = "nPeaks"
            peak_grp.create_dataset("nPeaks", data=spots_per_frame).attrs["interpretation"] = "spectrum"

            total_int = numpy.zeros((nframes, max_spots), dtype=numpy.float32)
            xpos = numpy.zeros((nframes, max_spots), dtype=numpy.float32)
            ypos = numpy.zeros((nframes, max_spots), dtype=numpy.float32)
            snr = numpy.zeros((nframes, max_spots), dtype=numpy.float32)
            #entry.attrs["default"] = posixpath.relpath(peak_grp.name, entry.name)  # prevents the densify from working

            for i, s in enumerate(peaks):
                l = len(s)
                total_int[i,:l] = s["intensity"]
                xpos[i,:l] = s["pos1"]
                ypos[i,:l] = s["pos0"]
                snr[i,:l] = s["intensity"] / s["sigma"]
            peak_grp.create_dataset("peakTotalIntensity", data=total_int, **cmp)
            peak_grp.create_dataset("peakXPosRaw", data=xpos, **cmp)
            peak_grp.create_dataset("peakYPosRaw", data=ypos, **cmp)
            peak_grp.create_dataset("peakSNR", data=snr, **cmp)

        if ai is not None:
            if extra.get("correctSolidAngle") or (extra.get("polarization_factor") is not None):
                if extra.get("correctSolidAngle"):
                    normalization = ai.solidAngleArray()
                else:
                    normalization = None
                pf = extra.get("polarization_factor")
                if pf:
                    if normalization is None:
                        normalization = ai.polarization(factor=pf)
                    else:
                        normalization *= ai.polarization(factor=pf)
                nrmds = sparse_grp.create_dataset("normalization", data=normalization, **cmp)
                nrmds.attrs["interpretation"] = "image"

            sparsify_grp = nexus.new_class(entry, "sparsify", class_type="NXprocess")
            sparsify_grp["program"] = "pyFAI"
            sparsify_grp["sequence_index"] = 1
            sparsify_grp["version"] = version
            sparsify_grp["date"] = get_isotime()
            sparsify_grp.create_dataset("argv", data=numpy.array(sys.argv, h5py.string_dtype("utf8"))).attrs["help"] = "Command line arguments"
            sparsify_grp.create_dataset("cwd", data=os.getcwd()).attrs["help"] = "Working directory"

            config_grp = nexus.new_class(sparsify_grp, "configuration", class_type="NXnote")
            config_grp["type"] = "text/json"
            parameters = OrderedDict([("geometry", ai.get_config()),
                                      ("sparsify", extra)])
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
                # wl_ds.attrs["resolution"] = 0.014
#                 nrj_ds = monochromator_grp.create_dataset("energy", data=numpy.floaself.energy)
#                 nrj_ds.attrs["units"] = "keV"
#                 #nrj_ds.attrs["resolution"] = 0.014

        if source is not None:
            dat_grp = nexus.new_class(entry, "data", "NXdata")
            idx = 1
            for fn in source:
                with fabio.open(fn) as fimg:
                    if "dataset" in dir(fimg):
                        for ds in fimg.dataset:
                            actual_filename = ds.file.filename
                            rel_path = os.path.relpath(os.path.abspath(actual_filename), os.path.dirname(os.path.abspath(filename)))
                            dat_grp[f"data_{idx:04d}"] = h5py.ExternalLink(rel_path, ds.name)
                            idx += 1
                    else:
                        logger.error("Only HDF5 files readable with FabIO can be linked")
            sparsify_grp["source"] = source

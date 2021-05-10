# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2020 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "10/05/2021"
__status__ = "production"
__docformat__ = 'restructuredtext'

import json
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
import numpy
from .. import version
from .nexus import Nexus, get_isotime

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
    dense[masked] = numpy.NaN"""
    res += """
    frames.append(dense.astype(intensity.dtype))
"""
    return res


def save_sparse(filename, frames, beamline="beamline", ai=None, source=None, extra={}):
    """Write the list of frames into a HDF5 file
    
    :param filename: name of the file
    :param frames: list of sparse frames (as built by sparsify)
    :param beamline: name of the beamline as text
    :param ai: Instance of geometry or azimuthal integrator
    :param source: list of input files
    :param extra: dict with extra metadata 
    :return: None
    """
    assert len(frames)
    with Nexus(filename, mode="w", creator="pyFAI_%s" % version) as nexus:
        instrument = nexus.new_instrument(instrument_name=beamline)
        entry = instrument.parent
        sparse_grp = nexus.new_class(entry, "sparse_frames", class_type="NXdata")
        entry.attrs["default"] = sparse_grp.name
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
        if dummy is None:
            if is_integer:
                dummy = 0
            else:
                dummy = numpy.NaN
        sparse_grp.create_dataset("dummy", data=dummy)
        rds = sparse_grp.create_dataset("radius", data=radius, dtype=numpy.float32)
        rds.attrs["interpretation"] = "spectrum"
        mskds = sparse_grp.create_dataset("mask", data=mask, **cmp)
        mskds.attrs["interpretation"] = "image"
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
            if source is not None:
                sparsify_grp["source"] = source
            config_grp = nexus.new_class(sparsify_grp, "configuration", class_type="NXnote")
            config_grp["type"] = "text/json"
            parameters = OrderedDict([("geometry", ai.get_config()),
                                      ("sparsify", extra)])
            config_grp["data"] = json.dumps(parameters, indent=2, separators=(",\r\n", ": "))

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

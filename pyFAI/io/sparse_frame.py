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
__date__ = "11/08/2020"
__status__ = "production"
__docformat__ = 'restructuredtext'

import json
import numpy
from .. import version
from .nexus import Nexus, get_isotime

try:
    import hdf5plugin
except:
    cmp = {}
else:
    cmp = hdf5plugin.Bitshuffle()
    

def _generate_densify_script(integer):
    "Provide a script to densify those data"
    res = """#python
import numpy
frames = []
for idx, bg in enumerate(background_avg):
    dense = numpy.interp(mask, radius, bg)
    flat = dense.ravel()
    start = frame_ptr[idx]
    stop = frame_ptr[idx+1]
    flat[index[start:stop] = intensity[start:stop]
    masked = numpy.where(numpy.logical_not(numpy.isfinite(mask)))"""
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

def save_sparse(filename, frames, beamline="ESRF_ID00", ai=None):
    "Write the list of frames in HDF5"
    assert len(frames)
    with Nexus(filename, mode="w", creator="pyFAI_%s"%version) as nexus:
        instrument = nexus.new_instrument(instrument_name=beamline)
        sparse_grp = nexus.new_class(instrument, "sparse_frames", class_type="NXcollection")
        sparse_grp["frame_ptr"] = numpy.concatenate(([0],numpy.cumsum([i.intensity.size for i in frames])),dtype=numpy.int32)
        index = numpy.concatenate([i.index for i in frames]).astype("int32")
        intensity = numpy.concatenate([i.intensity for i in frames])
        sparse_grp["script"] = _generate_densify_script(numpy.issubdtype(frames[0].dtype, numpy.integer))
        sparse_grp.create_dataset("index", data=index, **cmp)
        sparse_grp.create_dataset("intensity", data=intensity, **cmp)
        radius = frames[0].radius
        mask = frames[0].mask
        sparse_grp.create_dataset("radius", data=radius, dtype=numpy.float32)
        sparse_grp.create_dataset("mask", data=mask, **cmp)
        background_avg = numpy.vstack([f.background_avg for f in frames])
        background_std = numpy.vstack([f.background_std for f in frames])
        sparse_grp.create_dataset("background_avg", data=background_avg, **cmp)
        sparse_grp.create_dataset("background_std", data=background_std, **cmp)
        
        if ai is not None:
            sparsify_grp = nexus.new_class(instrument, "sparsify", class_type="NXprocess")
            sparsify_grp["program"] = "pyFAI"
            sparsify_grp["sequence_index"] = 1
            sparsify_grp["version"] = version
            sparsify_grp["date"] = get_isotime()
            config_grp = nexus.new_class(sparsify_grp, "configuration", class_type="NXnote")
            config_grp["type"] = "text/json"
            config_grp["data"] = json.dumps(ai.get_config(), indent=2, separators=(",\r\n", ": "))
        
            detector_grp = nexus.new_class(instrument, str(ai.detector), "NXdetector")
            dist_ds = detector_grp.create_dataset("distance", data=ai.dist)
            dist_ds.attrs["units"] = "m"
            xpix_ds = detector_grp.create_dataset("x_pixel_size", data=ai.pixel2)
            xpix_ds.attrs["units"] = "m"
            ypix_ds = detector_grp.create_dataset("y_pixel_size", data= ai.pixel1)
            ypix_ds.attrs["units"] ="m"
            f2d = ai.getFit2D()
            xbc_ds = detector_grp.create_dataset("beam_center_x", data=f2d["centerX"])
            xbc_ds.attrs["units"] = "pixel"
            ybc_ds = detector_grp.create_dataset("beam_center_y", data=f2d["centerY"])
            ybc_ds.attrs["units"] = "pixel"

            
            
        
    
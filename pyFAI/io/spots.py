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
__date__ = "02/12/2021"
__status__ = "production"
__docformat__ = 'restructuredtext'

import sys
import os
import json
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
import numpy
from .. import version
from .nexus import Nexus, get_isotime, h5py

try:
    import hdf5plugin
except ImportError:
    cmp = {"chunks":True,
           "compression": "gzip",
           "compression_opts":1}
else:
    cmp = hdf5plugin.Bitshuffle()



def save_spots(filename, spots, beamline="beamline", ai=None, source=None, extra={}, grid=None):
    """Write the list of spots per frame into a HDF5 file
    
    :param filename: name of the file
    :param frames: list of spots per frame (as built by peakfinder)
    :param beamline: name of the beamline as text
    :param ai: Instance of geometry or azimuthal integrator
    :param source: list of input files
    :param extra: dict with extra metadata
    :param grid: 2-tuple with grid shape and if it was acquired in zigzag mode
    :return: None
    """
    assert len(spots)
    spots_per_frame = numpy.array([len(s) for s in spots], dtype=numpy.int32)
    with Nexus(filename, mode="w", creator="pyFAI_%s" % version) as nexus:
        
        instrument = nexus.new_instrument(instrument_name=beamline)
        entry = instrument.parent
        peaks_grp = nexus.new_class(entry, "peaks", class_type="NXdata")
        entry.attrs["default"] = peaks_grp.name
        if grid and grid[0] and len(grid[0])>1:
            img = spots_per_frame.reshape(grid[0])
            if grid[1]:
                img[1::2,:] = img[1::2,-1::-1] # flip one line out of 2 
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

        #to have pos1 and pos2 along same dim as poni1 and poni2
        pos0 = numpy.concatenate([i["pos0"] for i in spots])
        peaks_grp.create_dataset("pos1", data=pos0, **cmp).attrs["dir"] = "y"

        pos1 = numpy.concatenate([i["pos1"] for i in spots])
        peaks_grp.create_dataset("pos2", data=pos1, **cmp).attrs["dir"] = "x"


        sparsify_grp = nexus.new_class(entry, "peakfinder", class_type="NXprocess")
        sparsify_grp["program"] = "pyFAI"
        sparsify_grp["sequence_index"] = 1
        sparsify_grp["version"] = version
        sparsify_grp["date"] = get_isotime()
        sparsify_grp.create_dataset("argv", data=numpy.array(sys.argv, h5py.string_dtype("utf8"))).attrs["help"] = "Command line arguments" 
        sparsify_grp.create_dataset("cwd", data=os.getcwd()).attrs["help"] = "Working directory"
        if source is not None:
            sparsify_grp.create_dataset("source", data=numpy.array(source, h5py.string_dtype("utf8")))
        if ai is not None:
            config_grp = nexus.new_class(sparsify_grp, "configuration", class_type="NXnote")
            config_grp["type"] = "text/json"
            parameters = OrderedDict([("geometry", ai.get_config()),
                                      ("peakfinder", extra)])
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

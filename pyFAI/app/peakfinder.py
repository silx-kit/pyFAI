#!/usr/bin/env python3
# coding: utf-8
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2022 European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer <Jerome.Kieffer@ESRF.eu>
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
#

"""peakfinder: Count the number of Bragg-peaks on an image.

Bragg peaks are local maxima of the background subtracted signal. 
Peaks are integrated and variance propagated. The centroids are reported.

Background is calculated by an iterative sigma-clipping in the polar space. 
The number of iteration, the clipping value and the number of radial bins could be adjusted.

This program requires OpenCL. The device needs be properly selected.
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "03/02/2022"
__status__ = "production"

import os
import sys
import argparse
import time
from collections import OrderedDict
import numexpr
import logging
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
import glob
try:
    import hdf5plugin  # noqa
except ImportError:
    logger.debug("Unable to load hdf5plugin, backtrace:", exc_info=True)

import fabio
from .. import version, load
from ..units import to_unit
from ..opencl import ocl

if ocl is None:
    logger.error("Peakfinder requires a valid OpenCL stack to be installed")
else:
    from ..opencl.peak_finder import OCL_PeakFinder
from ..utils.shell import ProgressBar
from ..io.spots import save_spots
# Define few constants:
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ARGUMENT_FAILURE = 2


def expand_args(args):
    """
    Takes an argv and expand it (under Windows, cmd does not convert *.tif into
    a list of files.

    :param list args: list of files or wildcards
    :return: list of actual args
    """
    new = []
    for afile in args:
        if glob.has_magic(afile):
            new += glob.glob(afile)
        else:
            new.append(afile)
    return new


def parse():
    epilog = "Current status of the program: " + __status__
    parser = argparse.ArgumentParser(prog="peakfinder",
                                     description=__doc__,
                                     epilog=epilog)
    parser.add_argument("IMAGE", nargs="*",
                        help="File with input images. All results are concatenated into a single HDF5 file.")
    parser.add_argument("-V", "--version", action='version', version=version,
                        help="output version and exit")
    parser.add_argument("-v", "--verbose", action='store_true', dest="verbose", default=False,
                        help="Show information for each frame")
    parser.add_argument("--debug", action='store_true', dest="debug", default=False,
                        help="Show debug information")
    parser.add_argument("--profile", action='store_true', dest="profile", default=False,
                        help="Show profiling information")
    group = parser.add_argument_group("main arguments")
#     group.add_argument("-l", "--list", action="store_true", dest="list", default=None,
#                        help="show the list of available formats and exit")
    group.add_argument("-o", "--output", default='spots.h5', type=str,
                       help="Output filename")
    group.add_argument("--save-source", action='store_true', dest="save_source", default=False,
                       help="Save the path for all source files")

    group = parser.add_argument_group("Scan options")
    group.add_argument("--grid-size", nargs=2, type=int, dest="grid_size", default=None,
                       help="Grid along which the data was acquired, disabled by default")
    group.add_argument("--zig-zag", action='store_true', dest="zig_zag", default=False,
                       help="Build the 2D image considering the scan was performed with a zig-zag pattern")
# TODO: implement those
#     group = parser.add_argument_group("optional behaviour arguments")
#     group.add_argument("-f", "--force", dest="force", action="store_true", default=False,
#                        help="if an existing destination file cannot be" +
#                        " opened, remove it and try again (this option" +
#                        " is ignored when the -n option is also used)")
#     group.add_argument("-n", "--no-clobber", dest="no_clobber", action="store_true", default=False,
#                        help="do not overwrite an existing file (this option" +
#                        " is ignored when the -i option is also used)")
#     group.add_argument("--remove-destination", dest="remove_destination", action="store_true", default=False,
#                        help="remove each existing destination file before" +
#                        " attempting to open it (contrast with --force)")
#     group.add_argument("-u", "--update", dest="update", action="store_true", default=False,
#                        help="copy only when the SOURCE file is newer" +
#                        " than the destination file or when the" +
#                        " destination file is missing")
#     group.add_argument("-i", "--interactive", dest="interactive", action="store_true", default=False,
#                        help="prompt before overwrite (overrides a previous -n" +
#                        " option)")
#     group.add_argument("--dry-run", dest="dry_run", action="store_true", default=False,
#                        help="do everything except modifying the file system")

    group = parser.add_argument_group("Experimental setup options")
    group.add_argument("-b", "--beamline", type=str, default="beamline",
                       help="Name of the instument (for the HDF5 NXinstrument)")
    group.add_argument("-p", "--poni", type=str, default=None,
                       help="Geometry description file: Mandatory")
    group.add_argument("-m", "--mask", type=str, default=None,
                       help="Mask to be used for invalid pixels")
    group.add_argument("--dummy", type=float, default=None,
                       help="Value of dynamically masked pixels (disabled by default)")
    group.add_argument("--delta-dummy", type=float, default=None,
                       help="Precision for dummy value")
    group.add_argument("--radial-range", dest="radial_range", nargs=2, type=float, default=None,
                       help="radial range as a 2-tuple of number of pixels (all available range by default)")
    group.add_argument("-P", "--polarization", type=float, default=None,
                       help="Polarization factor of the incident beam [-1:1] (off by default, 0.99 is a good guess on synchrotrons")
    group.add_argument("-A", "--solidangle", action='store_true', default=None,
                       help="Correct for solid-angle correction (important if the detector is not mounted normally to the incident beam, off by default")
    group = parser.add_argument_group("Sigma-clipping options")
    group.add_argument("--bins", type=int, default=800,
                       help="Number of radial bins to consider (800 by default)")
    group.add_argument("--unit", type=str, default="r_m",
                       help="radial unit to perform the calculation (r_m by default)")
    group.add_argument("--cycle", type=int, default=5,
                       help="Number of cycles for the sigma-clipping (5 by default)")
    group.add_argument("--cutoff-clip", dest="cutoff_clip", type=float, default=0.0,
                       help="SNR threshold for considering a pixel outlier when performing the sigma-clipping (0 by default: fallback on Chauvenet criterion)")
    group.add_argument("--error-model", dest="error_model", type=str, default="poisson",
                       help="Statistical model for the signal error, may be `poisson`(default) or `azimuthal` (slower) or even a simple formula like '5*I+8'")
    group = parser.add_argument_group("Peak finding options")
    group.add_argument("--cutoff-pick", dest="cutoff_pick", type=float, default=3.0,
                       help="SNR threshold for considering a pixel high when searching for peaks (3 by default)")
    group.add_argument("--noise", type=float, default=1.0,
                       help="Noise added quadratically to the background (1 by default")
    group.add_argument("--patch-size", type=int, default=5,
                       help="Size of the neighborhood patch for integration (5 by default)")
    group.add_argument("--connected", type=int, default=3,
                       help="Number of high pixels in neighborhood to be considered as a peak (3 by default)")
    group = parser.add_argument_group("Opencl setup options")
    group.add_argument("--workgroup", type=int, default=None,
                       help="Enforce the workgroup size for OpenCL kernel. Impacts only on the execution speed, not on the result.")
    group.add_argument("--device", nargs=2, type=int, default=None,
                       help="definition of the platform and device identifier: 2 integers. Use `clinfo` to get a description of your system")
    group.add_argument("--device-type", type=str, default="all",
                       help="device type like `cpu` or `gpu` or `acc`. Can help to select the proper device.")
    try:
        args = parser.parse_args()

        if args.debug:
            logger.setLevel(logging.DEBUG)

        if len(args.IMAGE) == 0:
            raise argparse.ArgumentError(None, "No input file specified.")
        if ocl is None:
            raise RuntimeError("sparsify-Brgg requires _really_ a valide OpenCL environment. Please install pyopencl !")

    except argparse.ArgumentError as e:
        logger.error(e.message)
        logger.debug("Backtrace", exc_info=True)
        return EXIT_ARGUMENT_FAILURE
    else:
        # the upper case IMAGE is used for the --help auto-documentation
        args.images = expand_args(args.IMAGE)
        args.images.sort()
        return args


def process(options):
    """Perform actually the processing

    :param options: The argument parsed by agrparse.
    :return: EXIT_SUCCESS or EXIT_FAILURE
    """
    if options.verbose:
        pb = None
    else:
        pb = ProgressBar("Peak-finder", 100, 30)

    logger.debug("Count the number of frames")
    if pb:
        pb.update(0, message="Count the number of frames")
    dense = [fabio.open(f) for f in options.images]
    nframes = sum([f.nframes for f in dense])

    logger.debug("Initialize the geometry")
    if pb:
        pb.update(0, message="Initialize the geometry", max_value=nframes)
    ai = load(options.poni)
    if options.mask is not None:
        mask = fabio.open(options.mask).data
        ai.detector.mask = mask
    else:
        mask = ai.detector.mask
    shape = dense[0].shape

    unit = to_unit(options.unit)
    if options.radial_range is not None:
        rrange = [ float(i) for i in options.radial_range]
    else:
        rrange = None

    integrator = ai.setup_CSR(shape,
                              options.bins,
                              mask=mask,
                              pos0_range=rrange,
                              unit=unit,
                              split="no",
                              scale=False)

    logger.debug("Initialize the OpenCL device")
    if pb:
        pb.update(0, message="Initialize the OpenCL device")

    if options.device is not None:
        ctx = ocl.create_context(platformid=options.device[0], deviceid=options.device[1],)
    else:
        ctx = ocl.create_context(devicetype=options.device_type)

    logger.debug("Initialize the azimuthal integrator")
    pf = OCL_PeakFinder(integrator.lut,
                        image_size=shape[0] * shape[1],
                        empty=options.dummy,
                        unit=unit,
                        bin_centers=integrator.bin_centers,
                        radius=ai._cached_array[unit.name.split("_")[0] + "_center"],
                        mask=mask,
                        ctx=ctx,
                        profile=options.profile,
                        block_size=options.workgroup)

    logger.debug("Start peak search")
    frames = []

    cnt = 0
    if "I" in options.error_model:
        variance = numexpr.NumExpr(options.error_model)
        error_model = None
    else:
        error_model = options.error_model
        variance = None

    parameters = OrderedDict([("dummy", options.dummy),
                              ("delta_dummy", options.delta_dummy),
                              ("safe", False),
                              ("error_model", error_model),
                              ("cutoff_clip", options.cutoff_clip),
                              ("cycle", options.cycle),
                              ("noise", options.noise),
                              ("cutoff_pick", options.cutoff_pick),
                              ("radial_range", rrange),
                              ('patch_size', options.patch_size),
                              ("connected", options.connected)])
    if options.solidangle:
        parameters["solidangle"], parameters["solidangle_checksum"] = ai.solidAngleArray(with_checksum=True)
    if options.polarization is not None:
        parameters["polarization"], parameters["polarization_checksum"] = ai.polarization(factor=options.polarization, with_checksum=True)
    t0 = time.perf_counter()
    for fabioimage in dense:
        for frame in fabioimage:
            intensity = frame.data
            current = pf.peakfinder8(intensity,
                                     variance=None if variance is None else variance(intensity),
                                     **parameters)
            frames.append(current)
            if pb:
                pb.update(cnt, message="%s: %i pixels" % (os.path.basename(fabioimage.filename), len(current)))
            else:
                print("%s frame #%d, found %d intense pixels" % (fabioimage.filename, fabioimage.currentframe, len(current)))
            cnt += 1
    t1 = time.perf_counter()
    if pb:
        pb.update(nframes, message="Saving: " + options.output)
        pb.clear()
    else:
        print("Saving: " + options.output)
    logger.debug("Save data")

    parameters["unit"] = unit.name.split("_")[0]
    parameters["error_model"] = options.error_model

    if options.polarization is not None:
        parameters.pop("polarization")
        parameters.pop("polarization_checksum")
        parameters["polarization_factor"] = options.polarization
    if options.solidangle:
        parameters.pop("solidangle")
        parameters.pop("solidangle_checksum")
        parameters["correctSolidAngle"] = True
    if options.mask is not None:
        parameters["mask"] = True

    save_spots(options.output,
                frames,
                beamline=options.beamline,
                ai=ai,
                source=options.images if options.save_source else None,
                extra=parameters,
                grid=(options.grid_size, options.zig_zag))

    if options.profile:
        try:
            pf.log_profile(True)
        except Exception:
            pf.log_profile()
    if pb:
        pb.clear()
    logger.info(f"Total peakfinder time: %.3fs \t (%.3f fps)", t1 - t0, cnt / (t1 - t0))

    return EXIT_SUCCESS


def main():
    options = parse()
    if options == EXIT_ARGUMENT_FAILURE:
        sys.exit(EXIT_ARGUMENT_FAILURE)
    res = process(options)
    sys.exit(res)


if __name__ == "__main__":
    main()

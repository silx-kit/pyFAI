#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2013-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""
pyFAI-calib

A tool for determining the geometry of a detector using a reference sample.

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "13/05/2025"
__status__ = "production"

import os
import sys
import time
import logging
logger = logging.getLogger(__name__)

import math
import numpy
from silx.image import marchingsquares
from scipy.stats import linregress
import fabio
from fabio.fabioutils import exists as fabio_exists

from argparse import ArgumentParser
from urllib.parse import urlparse

from .matplotlib import pylab, matplotlib
from .utils import update_fig
from . import utils as gui_utils
from ..detectors import detector_factory, Detector
from ..geometryRefinement import GeometryRefinement
from .peak_picker import PeakPicker
from .. import units
from .. import average
from ..utils import measure_offset, expand_args, \
            readFloatFromKeyboard, FixedParameters, round_fft, \
            win32
from ..integrator.azimuthal import AzimuthalIntegrator
from ..units import hc
from .. import version as PyFAI_VERSION
from .. import date as PyFAI_DATE
from ..calibrant import Calibrant, CALIBRANT_FACTORY, get_calibrant
from .mpl_calib_qt import QtMplCalibWidget
try:
    from ..ext._convolution import gaussian_filter
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    from scipy.ndimage.filters import gaussian_filter

try:
    from ..ext import morphology
    pyFAI_morphology = True
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    from scipy.ndimage import morphology
    pyFAI_morphology = False


def get_detector(detector, datafiles=None):
    """
    Detector factory taking into account the binning knowing the datafiles

    :param detector: string or detector or other junk
    :param datafiles: can be a list of images to be opened and their shape used
    :return: pyFAI.detector.Detector instance
    :raise RuntimeError: If no detector found
    """
    res = None
    if isinstance(detector, (str,)):
        try:
            res = detector_factory(detector)
        except RuntimeError:
            raise RuntimeError("Not a valid detector: %s" % detector)
    elif isinstance(detector, Detector):
        res = detector
    else:
        res = Detector()
    if datafiles and fabio_exists(datafiles[0]):
        with fabio.open(datafiles[0]) as fimg:
            shape = fimg.shape
        res.guess_binning(shape)
    return res


class AbstractCalibration(object):

    """
    Everything that is common to Calibration and Recalibration
    """

    win_error = "We are under windows with a 32 bit version of python,"\
                " matplotlib is not able to"\
                " display too many images without crashing, this"\
                " is why the window showing the diffraction image"\
                " is closed"
    _HELP = {"help": "Try to get the help of a given action, like 'refine?'. Use done when finished. "
             "Most command are composed of 'action parameter value' like 'set wavelength 1 A'.",
             "get": "print he value of a parameter",
             "set": "set the value of a parameter to the given value, i.e 'set wavelength 0.1 nm', units are optional",
             'fix': "fixes the value of a parameter so that its value will not be optimized, i.e. 'fix wavelength'",
             'free': "frees the parameter so that the value can be optimized, i.e. 'free wavelength'",
             'bound': "sets the upper and lower bound of a parameter: 'bound dist 0.1 0.2'",
             'bounds': "sets the upper and lower bound of all parameters",
             'refine': "performs a new cycle of refinement",
             'recalib': "extract a new set of rings and re-perform the calibration. One can specify how many rings to extract and the algorithm to use (blob, massif, watershed) and the nb_pts_per_deg in azimuth",
             'done': "finishes the processing, performs an integration and quits",
             'validate': "plot the offset between the calibrated image and the back-projected image",
             'validate2': "measures the offset of the center as function of azimuthal angle by cross-correlation of 2 plots, 180 deg appart. Option: number of azimuthal sliced, default: 36",
             'integrate': "perform the azimuthal integration and display results",
             'abort': "quit immediately, discarding any unsaved changes",
             'show': "Just print out the current parameter set. Optional parameters are units for length, rotation and wavelength, i.e. 'show mm deg A'",
             'reset': "Reset the geometry to the initial guess (rotation to zero, distance to 0.1m, poni at the center of the image)",
             'assign': "Change the assignment of a group of points to a rings",
             "weight": "toggle from weighted to unweighted mode...",
             "define": "Re-define the value for a constant internal parameter of the program like max_iter, nPt_1D, nPt_2D_azim, nPt_2D_rad, integrator_method, error_model. Warning: attribute change may be harmful !",
             "chiplot": "plot control point radial error as function of azimuthal angle, optional parameters: the rings for which this need to be plotted",
             "delete": "delete a group of points, provide the letter."
             }
    PARAMETERS = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"]
    UNITS = {"dist": "meter", "poni1": "meter", "poni2": "meter", "rot1": "radian",
             "rot2": "radian", "rot3": "radian", "wavelength": "meter"}
    VALID_URL = ["", 'file', 'hdf5', "nxs", "h5"]
    PTS_PER_DEG = 0.3

    def __init__(self, img=None, mask=None, detector=None, wavelength=None, calibrant=None):
        """Constructor of AbstractCalibration

        :param img: 2d numpy array with the data
        :param mask: 2d array with invalid pixels marked
        :param detector: Detector name or instance
        :param wavelength: radiation wavelength in meter
        :param calibrant: pyFAI.calibrant.Calibrant instance
        """

        self.cutBackground = None
        self.peakPicker = None
        self.img = img

        if detector:
            detector = detector_factory(detector)

        if mask is None:
            if detector is not None:
                mask = detector.mask
        else:
            if detector is not None:
                mask = numpy.logical_or(mask, detector.mask)
                detector.mask = mask

        self.mask = mask
        self.detector = detector
        self.ai = AzimuthalIntegrator(dist=0.1, detector=detector)
        self.wavelength = wavelength
        if wavelength:
            self.ai.wavelength = wavelength

        self.data = None
        self.basename = None
        self.geoRef = None
        self.reconstruct = False
        if calibrant:
            if isinstance(calibrant, Calibrant):
                self.calibrant = calibrant
            elif calibrant in CALIBRANT_FACTORY:
                self.calibrant = get_calibrant(calibrant)
            elif os.path.isfile(calibrant) and os.path.isfile(calibrant):
                self.calibrant = Calibrant(calibrant)
            else:
                logger.error("Unable to handle such calibrant %s", calibrant)
                self.calibrant = None

            if self.calibrant and wavelength:
                self.calibrant.setWavelength_change2th(wavelength)
        else:
            self.calibrant = None

        self.gaussianWidth = None
        self.pointfile = None
        self.saturation = 0
        self.fixed = FixedParameters(["wavelength", "rot3"])  # parameter fixed during optimization
        self.max_rings = None
        self.max_iter = 1000
        self.gui = True
        self.interactive = True
        self.filter = "mean"
        self.weighted = False
        self.polarization_factor = None
        self.parser = None
        self.nPt_1D = 1024
        self.nPt_2D_azim = 360
        self.nPt_2D_rad = 400
        self.unit = units.to_unit("2th_deg")
        self.keep = True
        self.check_calib = None
        self.fig_integrate = self.ax_xrpd_1d = self.ax_xrpd_2d = None
        self.fig_chiplot = self.ax_chiplot = None
        self.fig_center = self.ax_center = None
        self.integrator_method = ("full", "historgam", "cython")
        self.error_model = ""

    def __repr__(self):
        lst = [f"{self.__class__.__name__} object:"]
        if self.fixed:
            lst.append("fixed=" + ", ".join(self.fixed))
        else:
            lst.append("fixed= None")
        lst.append(self.detector.__repr__())
        return os.linesep.join(lst)

    def configure_parser(self, version="calibration from pyFAI  version %s: %s" % (PyFAI_VERSION, PyFAI_DATE),
                         usage="pyFAI-calib [options] input_image.edf",
                         description=None, epilog=None):
        """Common configuration for parsers
        """
        self.parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
        self.parser.add_argument("-V", "--version", action='version', version=version)
        self.parser.add_argument("args", metavar="FILE", help="List of files to calibrate", nargs='+')
        self.parser.add_argument("-o", "--out", dest="outfile",
                                 help="Filename where processed image is saved", metavar="FILE",
                                 default="merged.edf")
        self.parser.add_argument("-v", "--verbose",
                                 action="store_true", dest="debug", default=False,
                                 help="switch to debug/verbose mode")
        self.parser.add_argument("-c", "--calibrant", dest="spacing", metavar="FILE",
                                 help="Calibrant name or file containing d-spacing of the reference sample (MANDATORY, case sensitive !)",
                                 default=None)
        self.parser.add_argument("-w", "--wavelength", dest="wavelength", type=float,
                                 help="wavelength of the X-Ray beam in Angstrom. Mandatory ", default=None)
        self.parser.add_argument("-e", "--energy", dest="energy", type=float,
                                 help="energy of the X-Ray beam in keV (hc=%skeV.A)." % hc, default=None)
        self.parser.add_argument("-P", "--polarization", dest="polarization_factor",
                                 type=float, default=None,
                                 help="polarization factor, from -1 (vertical) to +1 (horizontal),"
                                 " default is None (no correction), synchrotrons are around 0.95")
        self.parser.add_argument("-i", "--poni", dest="poni", metavar="FILE",
                                 help="file containing the diffraction parameter (poni-file). MANDATORY for pyFAI-recalib!",
                                 default=None)
        self.parser.add_argument("-b", "--background", dest="background",
                                 help="Automatic background subtraction if no value are provided",
                                 default=None)
        self.parser.add_argument("-d", "--dark", dest="dark",
                                 help="list of comma separated dark images to average and subtract", default=None)
        self.parser.add_argument("-f", "--flat", dest="flat",
                                 help="list of comma separated flat images to average and divide", default=None)
        self.parser.add_argument("-s", "--spline", dest="spline",
                                 help="spline file describing the detector distortion", default=None)
        self.parser.add_argument("-D", "--detector", dest="detector_name",
                                 help="Detector name (instead of pixel size+spline)", default=None)
        self.parser.add_argument("-m", "--mask", dest="mask",
                                 help="file containing the mask (for image reconstruction)", default=None)
        self.parser.add_argument("-n", "--pt", dest="npt",
                                 help="file with datapoints saved. Default: basename.npt", default=None)
        self.parser.add_argument("--filter", dest="filter",
                                 help="select the filter, either mean(default), max or median",
                                 default="mean")
        self.parser.add_argument("-l", "--distance", dest="distance", type=float,
                                 help="sample-detector distance in millimeter. Default: 100mm", default=None)
        self.parser.add_argument("--dist", dest="dist", type=float,
                                 help="sample-detector distance in meter. Default: 0.1m", default=None)
        self.parser.add_argument("--poni1", dest="poni1", type=float,
                                 help="poni1 coordinate in meter. Default: center of detector", default=None)
        self.parser.add_argument("--poni2", dest="poni2", type=float,
                                 help="poni2 coordinate in meter. Default: center of detector", default=None)
        self.parser.add_argument("--rot1", dest="rot1", type=float,
                                 help="rot1 in radians. default: 0", default=None)
        self.parser.add_argument("--rot2", dest="rot2", type=float,
                                 help="rot2 in radians. default: 0", default=None)
        self.parser.add_argument("--rot3", dest="rot3", type=float,
                                 help="rot3 in radians. default: 0", default=None)

        self.parser.add_argument("--fix-dist", dest="fix_dist",
                                 help="fix the distance parameter", default=None, action="store_true")
        self.parser.add_argument("--free-dist", dest="fix_dist",
                                 help="free the distance parameter. Default: Activated", default=None, action="store_false")

        self.parser.add_argument("--fix-poni1", dest="fix_poni1",
                                 help="fix the poni1 parameter", default=None, action="store_true")
        self.parser.add_argument("--free-poni1", dest="fix_poni1",
                                 help="free the poni1 parameter. Default: Activated", default=None, action="store_false")

        self.parser.add_argument("--fix-poni2", dest="fix_poni2",
                                 help="fix the poni2 parameter", default=None, action="store_true")
        self.parser.add_argument("--free-poni2", dest="fix_poni2",
                                 help="free the poni2 parameter. Default: Activated", default=None, action="store_false")

        self.parser.add_argument("--fix-rot1", dest="fix_rot1",
                                 help="fix the rot1 parameter", default=None, action="store_true")
        self.parser.add_argument("--free-rot1", dest="fix_rot1",
                                 help="free the rot1 parameter. Default: Activated", default=None, action="store_false")

        self.parser.add_argument("--fix-rot2", dest="fix_rot2",
                                 help="fix the rot2 parameter", default=None, action="store_true")
        self.parser.add_argument("--free-rot2", dest="fix_rot2",
                                 help="free the rot2 parameter. Default: Activated", default=None, action="store_false")

        self.parser.add_argument("--fix-rot3", dest="fix_rot3",
                                 help="fix the rot3 parameter", default=None, action="store_true")
        self.parser.add_argument("--free-rot3", dest="fix_rot3",
                                 help="free the rot3 parameter. Default: Activated", default=None, action="store_false")

        self.parser.add_argument("--fix-wavelength", dest="fix_wavelength",
                                 help="fix the wavelength parameter. Default: Activated", default=True, action="store_true")
        self.parser.add_argument("--free-wavelength", dest="fix_wavelength",
                                 help="free the wavelength parameter. Default: Deactivated ", default=True, action="store_false")

        self.parser.add_argument("--tilt", dest="tilt",
                                 help="Allow initially detector tilt to be refined (rot1, rot2, rot3). Default: Activated", default=None, action="store_true")
        self.parser.add_argument("--no-tilt", dest="tilt",
                                 help="Deactivated tilt refinement and set all rotation to 0", default=None, action="store_false")

        self.parser.add_argument("--saturation", dest="saturation",
                                 help="consider all pixel>max*(1-saturation) as saturated and "
                                 "reconstruct them, default: 0 (deactivated)",
                                 default=0, type=float)
        self.parser.add_argument("--weighted", dest="weighted",
                                 help="weight fit by intensity, by default not.",
                                 default=False, action="store_true")
        self.parser.add_argument("--npt", dest="nPt_1D",
                                 help="Number of point in 1D integrated pattern, Default: 1024", type=int,
                                 default=1024)
        self.parser.add_argument("--npt-azim", dest="nPt_2D_azim",
                                 help="Number of azimuthal sectors in 2D integrated images. Default: 360", type=int,
                                 default=360)
        self.parser.add_argument("--npt-rad", dest="nPt_2D_rad",
                                 help="Number of radial bins in 2D integrated images. Default: 400", type=int,
                                 default=400)
        self.parser.add_argument("--unit", dest="unit",
                                 help="Valid units for radial range: 2th_deg, 2th_rad, q_nm^-1,"
                                 " q_A^-1, r_mm. Default: 2th_deg", type=str, default="2th_deg")
        self.parser.add_argument("--no-gui", dest="gui",
                                 help="force the program to run without a Graphical interface",
                                 default=True, action="store_false")
        self.parser.add_argument("--no-interactive", dest="interactive",
                                 help="force the program to run and exit without prompting"
                                 " for refinements", default=True, action="store_false")

    def analyse_options(self, options=None, args=None, sysargv=None):
        """Analyzes options and arguments

        :return: option,arguments
        """
        if (options is None) and (args is None):
            options = self.parser.parse_args(sysargv)
            args = options.args
        if options.debug:
            logger.setLevel(logging.DEBUG)
        self.outfile = options.outfile
        if options.dark:
            self.darkFiles = [f for f in options.dark.split(",") if os.path.isfile(f)]
            if not self.darkFiles:  # empty container !!!
                logger.error("No dark file exists !!!")
                self.darkFiles = None
        if options.flat:
            self.flatFiles = [f for f in options.flat.split(",") if os.path.isfile(f)]
            if not self.flatFiles:  # empty container !!!
                logger.error("No flat file exists !!!")
                self.flatFiles = None

        if options.detector_name:
            self.detector = get_detector(options.detector_name, args)
            self.ai.detector = self.detector
        if options.spline:
            if "Pilatus" in self.detector.name:
                self.detector.set_splineFile(options.spline)  # is as 2-tuple of path
            elif os.path.isfile(options.spline):
                self.detector.set_splineFile(os.path.abspath(options.spline))
            else:
                logger.error("Unknown spline file %s", options.spline)

        if options.mask and os.path.isfile(options.mask):
            with fabio.open(options.mask) as fimg:
                self.mask = (fimg.data != 0)
        else:  # Use default mask provided by detector
            self.mask = self.detector.mask

        self.pointfile = options.npt
        if options.spacing:
            if options.spacing in CALIBRANT_FACTORY:
                self.calibrant = get_calibrant(options.spacing)
            elif os.path.isfile(options.spacing):
                self.calibrant = Calibrant(options.spacing)
            else:
                logger.error("No such Calibrant / d-Spacing file: %s", options.spacing)

        if self.calibrant is None:
            self.read_dSpacingFile(True)

        if options.poni:
            self.ai.load(options.poni)

        if options.wavelength:
            self.ai.wavelength = self.wavelength = 1e-10 * options.wavelength
        elif options.energy:
            self.ai.wavelength = self.wavelength = 1e-10 * hc / options.energy
#        else:
            # This should be read from the poni. It it is missing; it is called in preprocess.
#            self.read_wavelength()
#            pass
        if options.distance:
            self.ai.dist = 1e-3 * options.distance
        if options.dist:
            self.ai.dist = options.dist

        if options.tilt is False:
            self.ai.rot1 = 0.0
            self.ai.rot2 = 0.0
            self.ai.rot3 = 0.0
        if options.poni1 is not None:
            self.ai.poni1 = options.poni1
        if options.poni2 is not None:
            self.ai.poni2 = options.poni2
        if options.rot1 is not None:
            self.ai.rot1 = options.rot1
        if options.rot2 is not None:
            self.ai.rot2 = options.rot2
        if options.rot3 is not None:
            self.ai.rot3 = options.rot3
        self.dataFiles = expand_args(args)
        if not self.dataFiles:
            raise RuntimeError("Please provide some calibration images ... "
                               "if you want to analyze them. Try also the "
                               "--help option to see all options!")
        self.fixed = FixedParameters()
        if options.tilt is not None:
            for key in ["rot1", "rot2", "rot3"]:
                self.fixed.add_or_discard(key, not(options.tilt))

        self.fixed.add_or_discard("dist", options.fix_dist)
        self.fixed.add_or_discard("poni1", options.fix_poni1)
        self.fixed.add_or_discard("poni2", options.fix_poni2)
        self.fixed.add_or_discard("rot1", options.fix_rot1)
        self.fixed.add_or_discard("rot2", options.fix_rot2)
        self.fixed.add_or_discard("rot3", options.fix_rot3)
        self.fixed.add_or_discard("wavelength", options.fix_wavelength)
        print(self.fixed)
        self.saturation = options.saturation

        self.gui = options.gui
        self.interactive = options.interactive
        self.filter = options.filter
        self.weighted = options.weighted
        self.polarization_factor = options.polarization_factor
        self.detector = self.ai.detector
        self.nPt_1D = options.nPt_1D
        self.nPt_2D_azim = options.nPt_2D_azim
        self.nPt_2D_rad = options.nPt_2D_rad
        self.unit = units.to_unit(options.unit)
        if options.background is not None:
            try:
                self.cutBackground = float(options.background)
            except Exception:
                self.cutBackground = True

        return options, args

    def get_pixelSize(self, ans):
        """convert a comma separated sting into pixel size"""
        sp = ans.split(",")
        if len(sp) >= 2:
            try:
                pixelSizeXY = [float(i) * 1e-6 for i in sp[:2]]
            except Exception:
                logger.error("error in reading pixel size_2")
                return
        elif len(sp) == 1:
            px = sp[0]
            try:
                pixelSizeXY = [float(px) * 1e-6, float(px) * 1e-6]
            except Exception:
                logger.error("error in reading pixel size_1")
                return
        else:
            logger.error("error in reading pixel size_0")
            return
        self.detector.pixel1 = pixelSizeXY[1]
        self.detector.pixel2 = pixelSizeXY[0]

    def read_pixelsSize(self):
        """Read the pixel size from prompt if not available"""
        if (self.detector.pixel1 is None) and (self.detector.splineFile is None):
            pixelSize = [15, 15]
            ans = input("Please enter the pixel size (in micron, comma separated X,Y "
                        " i.e. %.2e,%.2e) or a spline file: " % tuple(pixelSize)).strip()
            if os.path.isfile(ans):
                self.detector.splineFile = ans
            else:
                self.get_pixelSize(ans)

    def read_dSpacingFile(self, verbose=True):
        """Read the name of the calibrant / file with d-spacing"""
        if (self.calibrant is None):
            comments = ["pyFAI calib has changed !!!",
                        "Instead of entering the 2theta value, which was tedious,"
                        "the program takes a calibrant name or a d-spacing file in input "
                        "(just a serie of number representing the inter-planar "
                        "distance in Angstrom)",
                        "and an associated wavelength",
                        "You will be asked to enter the ring number,"
                        " which is usually a simpler than the 2theta value."]
            if verbose:
                print(os.linesep.join(comments))
            valid = False
            while valid:
                ans = input("Please enter the calibrant name or the file"
                            " containing the d-spacing:\t").strip()
                if ans in CALIBRANT_FACTORY:
                    self.calibrant = get_calibrant(ans)
                    valid = True
                elif os.path.isfile(ans):
                    self.calibrant = Calibrant(ans)
                    valid = True

    def read_wavelength(self):
        """Read the wavelength"""
        while not self.wavelength:
            ans = input("Please enter wavelength in Angstrom:\t").strip()
            try:
                self.wavelength = self.ai.wavelength = 1e-10 * float(ans)
            except Exception:
                self.wavelength = None

    def preprocess(self):
        """
        Initialize peakpicker
        """
        self.peakPicker = PeakPicker(self.img, reconst=self.reconstruct, mask=self.mask,
                                     pointfile=self.pointfile, calibrant=self.calibrant,
                                     wavelength=self.ai.wavelength, detector=self.detector)

        if self.gaussianWidth is not None:
            self.peakPicker.massif.valley_size = self.gaussianWidth
        else:
            self.peakPicker.massif.init_valley_size()

    def extract_cpt(self, method="massif", pts_per_deg=1.0, max_rings=numpy.iinfo(int).max):
        """
        Performs an automatic keypoint extraction:
        Can be used in recalib or in calib after a first calibration has been performed.

        :param method: method for keypoint extraction
        :param pts_per_deg: number of control points per azimuthal degree (increase for better precision)
        :param max_rings: extract at most max_rings
        """

        logger.info("in extract_cpt with method %s", method)
        if self.ai is None:
            raise RuntimeError("AzimuthalIntegrator is not defined (None)")
        if self.calibrant is None:
            raise RuntimeError("Calibrant is not defined (None)")
        if self.peakPicker is None:
            raise RuntimeError("PeakPicker is not defined (None)")
        self.peakPicker.reset()
        self.peakPicker.init(method, False)
        if self.geoRef:
            self.ai.setPyFAI(**self.geoRef.getPyFAI())
        tth = numpy.array([i for i in self.calibrant.get_2th() if i is not None])
        tth = numpy.unique(tth)
        tth_min = numpy.zeros_like(tth)
        tth_max = numpy.zeros_like(tth)
        delta = (tth[1:] - tth[:-1]) / 4.0
        tth_max[:-1] = delta
        tth_max[-1] = delta[-1]
        tth_min[1:] = -delta
        tth_min[0] = -delta[0]
        tth_max += tth
        tth_min += tth

        if self.geoRef:
            ttha = self.geoRef.get_ttha()
            chia = self.geoRef.get_chia()
            if (ttha is None) or (ttha.shape != self.peakPicker.data.shape):
                ttha = self.geoRef.twoThetaArray(self.peakPicker.data.shape)
            if (chia is None) or (chia.shape != self.peakPicker.data.shape):
                chia = self.geoRef.chiArray(self.peakPicker.data.shape)
        else:
            ttha = self.ai.twoThetaArray(self.peakPicker.data.shape)
            chia = self.ai.chiArray(self.peakPicker.data.shape)
        rings = 0
        self.peakPicker.sync_init()
        if self.max_rings is None:
            self.max_rings = tth.size

        ms = marchingsquares.MarchingSquaresMergeImpl(ttha, self.mask, use_minmax_cache=True)
        for i in range(tth.size):
            if rings >= min(self.max_rings, max_rings):
                break
            mask = numpy.logical_and(ttha >= tth_min[i], ttha < tth_max[i])
            if self.mask is not None:
                mask = numpy.logical_and(mask, numpy.logical_not(self.mask))

            size = mask.sum(dtype=int)
            if (size > 0):
                rings += 1
                self.peakPicker.massif_contour(mask)
                # if self.gui:
                #     self.peakPicker.widget.update()
                sub_data = self.peakPicker.data.ravel()[numpy.where(mask.ravel())]
                mean = sub_data.mean(dtype=numpy.float64)
                std = sub_data.std(dtype=numpy.float64)
                upper_limit = mean + std
                mask2 = numpy.logical_and(self.peakPicker.data > upper_limit, mask)
                size2 = mask2.sum(dtype=int)
                if size2 < 1000:
                    upper_limit = mean
                    mask2 = numpy.logical_and(self.peakPicker.data > upper_limit, mask)
                    size2 = mask2.sum()
                # length of the arc:
                points = ms.find_pixels(tth[i])
                seeds = set((i[0], i[1]) for i in points if mask2[i[0], i[1]])
                # max number of points: 360 points for a full circle
                azimuthal = chia[points[:, 0].clip(0, self.peakPicker.data.shape[0]), points[:, 1].clip(0, self.peakPicker.data.shape[1])]
                nb_deg_azim = numpy.unique(numpy.rad2deg(azimuthal).round()).size
                keep = int(nb_deg_azim * pts_per_deg)
                if keep == 0:
                    continue
                dist_min = len(seeds) / 2.0 / keep
                # why 3.0, why not ?

                logger.info("Extracting datapoint for ring %s (2theta = %.2f deg); "
                            "searching for %i pts out of %i with I>%.1f, dmin=%.1f" %
                            (i, numpy.degrees(tth[i]), keep, size2, upper_limit, dist_min))
                _res = self.peakPicker.peaks_from_area(mask=mask2, Imin=upper_limit, keep=keep, method=method, ring=i, dmin=dist_min, seed=seeds)

        if self.basename:
            self.peakPicker.points.save(self.basename + ".npt")
        if self.weighted:
            self.data = self.peakPicker.points.getWeightedList(self.peakPicker.data)
        else:
            self.data = self.peakPicker.points.getList()
        if self.geoRef:
            self.geoRef.data = numpy.array(self.data, dtype=numpy.float64)

    def refine(self, maxiter=1000000, fixed=None):
        """
        Contains the common geometry refinement part

        :param maxiter: number of iteration to run for in the minimizer
        :param fixed: a list of parameters for maintain fixed during the refinement. self.fixed by default.
        :return: nothing, object updated in place
        """

        fixed = self.fixed if fixed is None else fixed

        if win32 and self.peakPicker is not None:
            logging.info(self.win_error)
            self.peakPicker.closeGUI()
        if self.geoRef is None:
            self.geoRef = self.initgeoRef()

        print("Before refinement, the geometry is:")
        print(self.geoRef)
        previous = sys.maxsize
        finished = False
        fig2 = None
        while not finished:
            count = 0
            if "wavelength" in fixed:
                while (previous > self.geoRef.chi2()) and (count < self.max_iter):
                    if (count == 0):
                        previous = sys.maxsize
                    else:
                        previous = self.geoRef.chi2()
                    self.geoRef.refine2(maxiter, fix=fixed)
                    print(self.geoRef)
                    count += 1
            else:
                while previous > self.geoRef.chi2_wavelength() and (count < self.max_iter):
                    if (count == 0):
                        previous = sys.maxsize
                    else:
                        previous = self.geoRef.chi2_wavelength()
                    self.geoRef.refine2_wavelength(maxiter, fix=fixed)
                    print(self.geoRef)
                    count += 1
                self.peakPicker.points.setWavelength_change2th(self.geoRef.wavelength)
            if self.basename:
                self.geoRef.save(self.basename + ".poni")
            self.geoRef.del_ttha()
            self.geoRef.del_dssa()
            self.geoRef.del_chia()
            tth = self.geoRef.twoThetaArray(self.peakPicker.shape)
            dsa = self.geoRef.solidAngleArray(self.peakPicker.shape)
            # self.geoRef.chiArray(self.peakPicker.shape)
            # self.geoRef.corner_array(self.peakPicker.shape, unit=units.TTH_RAD, scale=False)
            if win32:
                logger.info(self.win_error)
            else:
                if self.gui:
                    self.peakPicker.contour(tth)
                    if self.interactive:
                        if fig2 is None:
                            fig2 = pylab.plt.figure()
                            sp = fig2.add_subplot(111)
                            im = sp.imshow(dsa, origin="lower")
                            _cbar = fig2.colorbar(im)  # Add color bar
                            sp.set_title("Pixels solid-angle (relative to PONI)")
                        else:
                            im.set_array(dsa)
                            im.autoscale()

                        fig2.show()
                        update_fig(fig2)

            if self.interactive:
                finished = self.prompt()
            else:
                finished = True
            if not finished:
                previous = sys.maxsize

    def prompt(self):
        """
        prompt for commands to guide the calibration process

        :return: True when the user is happy with what he has, False to request another refinement
        """

        while True:
            req_help = False
            print("Fixed: " + ", ".join(self.fixed))
            ans = input("Modify parameters (or ? for help)?\t ").strip()
            if "?" in ans:
                req_help = True
            if not ans:
                print("'done' to continue")
                continue
            words = ans.lower().split()
            action = words[0]
            if action in ["help", "?"]:
                req_help = True
            if req_help:
                for what in self._HELP.keys():
                    if action.startswith(what):
                        print("Help on %s" % what)
                        print(self._HELP[what])
                        break
                else:
                    print("Help on commands")
                    print(self._HELP["help"])
                    print("Valid actions: " + ", ".join(self._HELP.keys()))
                    print("Valid parameters: " + ", ".join(self.PARAMETERS))
            elif action == "get":  # get wavelength
                if (len(words) >= 2):
                    for param in words[1:]:
                        if param in self.PARAMETERS:
                            print("Value of parameter %s: %s  %s" % (param, self.geoRef.__getattribute__(param), self.UNITS[param]))
                        else:
                            print("No a parameter: %s" % param)
                else:
                    print(self._HELP[action])

            elif action == "set":  # set wavelength 1e-10
                if (len(words) in (3, 4)) and words[1] in self.PARAMETERS:
                    param = words[1]
                    try:
                        value = float(words[2])
                    except ValueError:
                        logger.warning("invalid value")
                    else:
                        scale = 1.0
                        if len(words) == 4:
                            unit = units.to_unit(words[3], units.LENGTH_UNITS + units.ANGLE_UNITS)
                            if unit:
                                scale = unit.scale
                        setattr(self.geoRef, param, value / scale)
                else:
                    print(self._HELP[action])
            elif action == "fix":  # fix wavelength
                if (len(words) >= 2):
                    for param in words[1:]:
                        if param in self.PARAMETERS:
                            print("Value of parameter %s: %s %s" % (param, self.geoRef.__getattribute__(param), self.UNITS[param]))
                            self.fixed.add(param)
                        else:
                            print("No a parameter: %s" % param)
                else:
                    print(self._HELP[action])
            elif action == "free":  # free wavelength
                if (len(words) >= 2):
                    for param in words[1:]:
                        if param in self.PARAMETERS:
                            print("Value of parameter %s: %s %s" % (param, self.geoRef.__getattribute__(param), self.UNITS[param]))
                            self.fixed.discard(param)
                        else:
                            print("No a parameter: %s" % param)
                else:
                    print(self._HELP[action])

            elif action == "recalib":
                max_rings = None
                pts_per_deg = self.PTS_PER_DEG
                if len(words) >= 2:
                    try:
                        max_rings = int(words[1])
                    except Exception:
                        logger.warning("specify the number of rings to extract")
                        max_rings = None
                    else:
                        self.max_rings = max_rings
                else:
                    self.max_rings = None
                if len(words) >= 3 and words[2] in PeakPicker.VALID_METHODS:
                    method = words[2]
                else:
                    method = "blob"
                if len(words) >= 4:
                    try:
                        pts_per_deg = float(words[3])
                    except ValueError:
                        pts_per_deg = self.PTS_PER_DEG
                self.extract_cpt(method, pts_per_deg)
                self.geoRef.data = numpy.array(self.data, dtype=numpy.float64)
                return False
            elif action == "bound":  # bound dist
                if len(words) >= 2 and words[1] in self.PARAMETERS:
                    param = words[1]
                    if len(words) == 2:
                        text = ("Enter %s in %s " % (param, self.UNITS[param]) +
                                "(or %s_min[%.3f] %s[%.3f] %s_max[%.3f]):\t " % (
                                    param, self.geoRef.__getattribute__("get_%s_min" % param)(),
                                    param, self.geoRef.__getattribute__("get_%s" % param)(),
                                    param, self.geoRef.__getattribute__("get_%s_max" % param)()))
                        values = {
                            1: [self.geoRef.__getattribute__("set_%s" % param)],
                            2: [self.geoRef.__getattribute__("set_%s_min" % param),
                                self.geoRef.__getattribute__("set_%s_max" % param)],
                            3: [self.geoRef.__getattribute__("set_%s_min" % param),
                                self.geoRef.__getattribute__("set_%s" % param),
                                self.geoRef.__getattribute__("set_%s_max" % param)]}
                        readFloatFromKeyboard(text, values)
                    elif len(words) == 3:
                        try:
                            value = float(words[2])
                        except ValueError:
                            logger.warning("invalid value")
                        else:
                            self.geoRef.__getattribute__("set_%s" % param)(value)
                    elif len(words) == 4:
                        try:
                            value_min = float(words[2])
                            value_max = float(words[3])
                        except ValueError:
                            logger.warning("invalid value")
                        else:
                            self.geoRef.__getattribute__("set_%s_min" % param)(value_min)
                            self.geoRef.__getattribute__("set_%s_max" % param)(value_max)
                    elif len(words) == 5:
                        try:
                            value_min = float(words[2])
                            value = float(words[3])
                            value_max = float(words[4])
                        except ValueError:
                            logger.warning("invalid value")
                        else:
                            self.geoRef.__getattribute__("set_%s_min" % param)(value_min)
                            self.geoRef.__getattribute__("set_%s" % param)(value)
                            self.geoRef.__getattribute__("set_%s_max" % param)(value_max)
                    else:
                        print(self._HELP[action])
                else:
                    print(self._HELP[action])
            elif action == "bounds":
                readFloatFromKeyboard("Enter Distance in meter "
                                      "(or dist_min[%.3f] dist[%.3f] dist_max[%.3f]):\t " %
                                      (self.geoRef.dist_min, self.geoRef.dist, self.geoRef.dist_max),
                                      {1: [self.geoRef.set_dist], 2: [self.geoRef.set_dist_min, self.geoRef.set_dist_max],
                                       3: [self.geoRef.set_dist_min, self.geoRef.set_dist, self.geoRef.set_dist_max]})
                readFloatFromKeyboard("Enter Poni1 in meter "
                                      "(or poni1_min[%.3f] poni1[%.3f] poni1_max[%.3f]):\t " %
                                      (self.geoRef.poni1_min, self.geoRef.poni1, self.geoRef.poni1_max),
                                      {1: [self.geoRef.set_poni1], 2: [self.geoRef.set_poni1_min, self.geoRef.set_poni1_max],
                                       3: [self.geoRef.set_poni1_min, self.geoRef.set_poni1, self.geoRef.set_poni1_max]})
                readFloatFromKeyboard("Enter Poni2 in meter "
                                      "(or poni2_min[%.3f] poni2[%.3f] poni2_max[%.3f]):\t " %
                                      (self.geoRef.poni2_min, self.geoRef.poni2, self.geoRef.poni2_max),
                                      {1: [self.geoRef.set_poni2], 2: [self.geoRef.set_poni2_min, self.geoRef.set_poni2_max],
                                       3: [self.geoRef.set_poni2_min, self.geoRef.set_poni2, self.geoRef.set_poni2_max]})
                readFloatFromKeyboard("Enter Rot1 in rad "
                                      "(or rot1_min[%.3f] rot1[%.3f] rot1_max[%.3f]):\t " %
                                      (self.geoRef.rot1_min, self.geoRef.rot1, self.geoRef.rot1_max),
                                      {1: [self.geoRef.set_rot1], 2: [self.geoRef.set_rot1_min, self.geoRef.set_rot1_max],
                                       3: [self.geoRef.set_rot1_min, self.geoRef.set_rot1, self.geoRef.set_rot1_max]})
                readFloatFromKeyboard("Enter Rot2 in rad "
                                      "(or rot2_min[%.3f] rot2[%.3f] rot2_max[%.3f]):\t " %
                                      (self.geoRef.rot2_min, self.geoRef.rot2, self.geoRef.rot2_max),
                                      {1: [self.geoRef.set_rot2], 2: [self.geoRef.set_rot2_min, self.geoRef.set_rot2_max],
                                       3: [self.geoRef.set_rot2_min, self.geoRef.set_rot2, self.geoRef.set_rot2_max]})
                readFloatFromKeyboard("Enter Rot3 in rad "
                                      "(or rot3_min[%.3f] rot3[%.3f] rot3_max[%.3f]):\t " %
                                      (self.geoRef.rot3_min, self.geoRef.rot3, self.geoRef.rot3_max),
                                      {1: [self.geoRef.set_rot3], 2: [self.geoRef.set_rot3_min, self.geoRef.set_rot3_max],
                                       3: [self.geoRef.set_rot3_min, self.geoRef.set_rot3, self.geoRef.set_rot3_max]})
            elif action == "done":
                self.postProcess()
                return True
            elif action == "quit":
                return True
            elif action == "refine":
                return False
            elif action == "fit":
                return False
            elif action == "validate":
                self.validate_calibration()
            elif action == "validate2":
                if len(words) > 1:
                    nb = int(words[1])
                else:
                    nb = 36
                self.validate_center(nb)
            elif action == "integrate":
                self.postProcess()
            elif action == "abort":
                sys.exit()
            elif action == "show":
                args = []
                print("The current parameter set is:")
                if len(words) > 1:
                    args = ans.split()[1:]
                print(self.geoRef.__repr__(*args))
            elif action == "reset":
                if len(words) > 1:
                    how = words[1]
                else:
                    how = "center"
                self.reset_geometry(how)
            elif action == "assign":
                # Re assign a group of point to a ring ...
                if self.peakPicker and self.peakPicker.points:
                    control_points = self.peakPicker.points
                    control_points.readRingNrFromKeyboard()
                    control_points.save(self.basename + ".npt")
                    if self.weighted:
                        self.data = self.peakPicker.points.getWeightedList(self.peakPicker.data)
                    else:
                        self.data = self.peakPicker.points.getList()
                    self.geoRef.data = numpy.array(self.data, dtype=numpy.float64)
            elif action == "weight":
                old = self.weighted
                if len(words) == 2:
                    value = words[1].lower()
                    if value in ("0", "off", "no", "none", "false"):
                        self.weighted = False
                    elif value in ("1", "on", "yes", "true"):
                        self.weighted = True
                    else:
                        logger.warning("Unrecognized argument for weight: %s", value)
                        continue
                print("Weights: %s" % self.weighted)
                if (old != self.weighted):
                    if self.weighted:
                        self.data = self.peakPicker.points.getWeightedList(self.peakPicker.data)
                    else:
                        self.data = self.peakPicker.points.getList()
                    self.geoRef.data = numpy.array(self.data, dtype=numpy.float64)
            elif action == "define":
                if len(words) == 3:
                    param = words[1]
                    sval = words[2]
                    for cs_param in dir(self):
                        if cs_param.lower() == param:
                            oldval = self.__getattribute__(cs_param)
                            t = type(oldval)
                            print("constant %s was %s of type %s, setting to %s" % (cs_param, oldval, t, sval))
                            try:
                                newval = t(sval)
                            except Exception as err:
                                print("Unable to convert type")
                                logger.warning(err)
                            self.__setattr__(cs_param, newval)
                            break
                    else:
                        print("No such parameter %s" % param)
                else:
                    print(self._HELP[action])
            elif action == "chiplot":
                    print(self._HELP[action])
                    rings = None
                    if len(words) > 1:
                        try:
                            rings = [int(i) for i in words[1:]]
                        except ValueError:
                            print("Please provide ring numbers ... ")
                    self.chiplot(rings)
            elif action == "delete":
                if len(words) < 2:
                    print(self._HELP[action])
                else:
                    for code in words[1:]:
                        self.peakPicker.remove_grp(code)
                    self.data = self.peakPicker.points.getList()
                    self.geoRef.data = numpy.array(self.data, dtype=numpy.float64)
            else:
                logger.warning("Unrecognized action: %s, type 'quit' to leave ", action)

    def chiplot(self, rings=None):
        """
        plot delta_2theta/2theta = f(chi) and fit the curve.

        :param rings: list of rings to consider
        """
        from scipy.optimize import leastsq
        model = lambda x, mean, amp, phase: mean + amp * numpy.sin(x + phase)
        error = lambda param, xdata, ydata: model(xdata, *param) - ydata

        def jacob(param, xdata, ydata):
            j = numpy.ones((param.size, xdata.size))
            j[1,:] = numpy.sin(xdata + param[2])
            j[2,:] = param[1] * numpy.cos(xdata + param[2])
            return j

        sqrt2 = math.sqrt(2.)
        ttha = self.geoRef.twoThetaArray(self.detector.shape)
        resolution = numpy.rad2deg(max(abs(ttha[1:] - ttha[:-1]).max(),
                                       abs(ttha[:, 1:] - ttha[:,:-1]).max()))
        if self.gui:
            if self.fig_chiplot:
                self.fig_chiplot.clf()
            else:
                self.fig_chiplot = pylab.plt.figure()
            self.ax_chiplot = self.fig_chiplot.add_subplot(1, 1, 1)
            self.ax_chiplot.set_xlim(-180, 180)
            self.ax_chiplot.set_xticks(numpy.linspace(-180, 180, 9))
            self.ax_chiplot.set_xlabel(r"Azimuthal angle $\chi$ ($^o$)")
            self.ax_chiplot.set_ylabel(r"Error in Radial angle $\Delta$ 2$\theta$/2$\theta$*10$^4$")
            self.ax_chiplot.set_title("Chi plot")

        else:
            print("chiplot display only possible with GUI")
        if rings is None:
            rings = list(set(int(i[2]) for i in self.data))
            rings.sort()
        for ring in rings:
            ref_2th = numpy.rad2deg(self.calibrant.get_2th()[ring])
            print("Fitting ring #%s (2th=%.3fdeg)" % (ring, ref_2th))
            d1 = []
            d2 = []
            for i in self.data:
                if i[2] == ring:
                    d1.append(i[0])
                    d2.append(i[1])
            if len(d1) < 5:
                print(" Skip group of length %i" % len(d1))
                continue
            d1 = numpy.array(d1)
            d2 = numpy.array(d2)
            tth = numpy.rad2deg(self.geoRef.tth(d1, d2))
            err4 = (tth - ref_2th) / ref_2th * 10000
            chi = self.geoRef.chi(d1, d2)
            mean = err4.mean()
            amp = err4.std() * sqrt2
            phase = 0.0
            param = numpy.array([mean, amp, phase])
            print(r" guessed err4 = %.3f + %.3f *sin($\chi$+ %.3f )" % (mean, amp, phase))
            res = leastsq(error, param, (chi, err4), jacob, col_deriv=True)
            popt = res[0]
            str_res = r"%.3f + %.3f *sin($\chi$+ %.3f )" % tuple(popt)
            print(" fitted err4 = " + str_res)
            chi = numpy.rad2deg(chi)
            if self.ax_chiplot:
                color = list(matplotlib.colors.cnames.keys())[ring]
                self.ax_chiplot.plot(chi, err4, "o", color=color, label="ring #%i (%.3f$^o$)" % (ring, ref_2th))
                chi2 = numpy.linspace(-180, 180, 360)
                self.ax_chiplot.plot(chi2, model(numpy.deg2rad(chi2), *popt), color=color, label=str_res)

        self.ax_chiplot.legend()
        if not gui_utils.main_loop:
            self.fig_chiplot.show()
        update_fig(self.fig_chiplot)
        logger.info("One pixel = %.3e deg", resolution)

    def postProcess(self):
        """
        Common part: shows the result of the azimuthal integration in 1D and 2D
        """
        if self.geoRef is None:
            self.refine()
        if "wavelength" not in self.fixed:
            self.peakPicker.points.setWavelength_change2th(self.geoRef.wavelength)
        self.peakPicker.points.save(self.basename + ".npt")
        self.geoRef.save(self.basename + ".poni")
        self.geoRef.mask = self.mask
        self.geoRef.del_ttha()
        self.geoRef.del_dssa()
        self.geoRef.del_chia()
        t0 = time.perf_counter()
        _tth = self.geoRef.twoThetaArray(self.peakPicker.shape)
        t1 = time.perf_counter()
        _dsa = self.geoRef.solidAngleArray(self.peakPicker.shape)
        t2 = time.perf_counter()
        self.geoRef.chiArray(self.peakPicker.shape)
        t2a = time.perf_counter()
        self.geoRef.corner_array(self.peakPicker.shape, units.TTH_DEG,
                                 scale=False)
        t2b = time.perf_counter()
        if self.gui:
            if self.fig_integrate is None:
                self.fig_integrate = pylab.plt.figure()
                self.ax_xrpd_1d = self.fig_integrate.add_subplot(1, 2, 1)
                self.ax_xrpd_2d = self.fig_integrate.add_subplot(1, 2, 2)
            else:
                self.ax_xrpd_1d.cla()
                self.ax_xrpd_2d.cla()
                update_fig(self.fig_integrate)

        t3 = time.perf_counter()
        res1 = self.geoRef.integrate1d_ng(self.peakPicker.data, self.nPt_1D,
                                          filename=self.basename + ".xy",
                                          unit=self.unit,
                                          polarization_factor=self.polarization_factor,
                                          method=self.integrator_method,
                                          error_model=self.error_model)
        t4 = time.perf_counter()
        res2 = self.geoRef.integrate2d(self.peakPicker.data,
                                          self.nPt_2D_rad, self.nPt_2D_azim,
                                          filename=self.basename + ".azim",
                                          unit=self.unit,
                                          polarization_factor=self.polarization_factor,
                                          method=self.integrator_method,
                                          error_model=self.error_model)
        t5 = time.perf_counter()
        logger.info(os.linesep.join([f"Timings ({self.integrator_method}):",
                                     f" * two theta array generation {(t1 - t0):.3f}s",
                                     f" * diff Solid Angle           {(t2 - t1):.3f}s",
                                     f" * chi array generation       {(t2a - t2):.3f}s",
                                     f" * corner coordinate array    {(t2b - t2a):.3f}s",
                                     f" * 1D Azimuthal integration   {(t4 - t3):.3f}s",
                                     f" * 2D Azimuthal integration   {(t5 - t4):.3f}s"]))

        if self.gui:
            self.ax_xrpd_1d.plot(res1.radial, res1.intensity)
            # GF: Add vertical line for each used calibration ring:
            xValues = None
            twoTheta = numpy.array([i for i in self.peakPicker.points.calibrant.get_2th() if i])  # in radian
            if self.unit == units.TTH_DEG:
                xValues = numpy.rad2deg(twoTheta)
            elif self.unit == units.TTH_RAD:
                xValues = twoTheta
            elif self.unit == units.Q_A:
                xValues = (4.e-10 * numpy.pi / self.wavelength) * numpy.sin(.5 * twoTheta)
            elif self.unit == units.Q_NM:
                xValues = (4.e-9 * numpy.pi / self.wavelength) * numpy.sin(.5 * twoTheta)
            elif self.unit == units.R_MM:
                # GF: correct formula?
                dBeamCentre = self.geoRef.getFit2D()["directDist"]  # in mm!!
                xValues = dBeamCentre * numpy.tan(twoTheta)
            else:
                logger.warning("Unknown unit %s, do not plot calibration rings", self.unit)
            if xValues is not None:
                for x in xValues:
                    line = matplotlib.lines.Line2D([x, x], self.ax_xrpd_1d.axis()[2:4],
                                                   color='red', linestyle='--')
                    self.ax_xrpd_1d.add_line(line)
            self.ax_xrpd_1d.set_title("1D integration")
            self.ax_xrpd_1d.set_xlabel(self.unit.label)
            self.ax_xrpd_1d.set_ylabel("Intensity")
            img = res2.intensity
            pos_rad = res2.radial
            pos_azim = res2.azimuthal
            self.ax_xrpd_2d.imshow(numpy.log(img - img.min() + 1e-3), origin="lower",
                                   extent=[pos_rad.min(), pos_rad.max(), pos_azim.min(), pos_azim.max()],
                                   aspect="auto")
            self.ax_xrpd_2d.set_title("2D regrouping")
            self.ax_xrpd_2d.set_xlabel(self.unit.label)
            self.ax_xrpd_2d.set_ylabel(r"Azimuthal angle $\chi$ ($^{o}$)")
            if not gui_utils.main_loop:
                self.fig_integrate.show()
            update_fig(self.fig_integrate)

    def validate_calibration(self):
        """
        Validate the calibration and calculate the offset in the diffraction image
        """
        if not self.check_calib:
            self.check_calib = CheckCalib()
        if self.geoRef:
            self.ai.setPyFAI(**self.geoRef.getPyFAI())
            self.ai.wavelength = self.geoRef.wavelength
        self.check_calib.ai = self.ai
        self.check_calib.img = self.peakPicker.data
        self.check_calib.mask = self.peakPicker.mask
        self.check_calib.wavelength = self.check_calib.wavelength
        self.check_calib.integrate()
        self.check_calib.rebuild()
        self.check_calib.show()

    def validate_center(self, slices=36):
        """
        Validate the position of the center by cross-correlating two spectra 180 deg appart.
        Output values are in micron.

        Designed for orthogonal setup with centered beam...

        :param slices: number of slices on which perform
        """
        if slices <= 0:
            logger.warning("The number of slices should be strictly positive")
            slices = 1
        if slices % 2 == 1:
            logger.warning("Validate assumes the number of slices is even. adding one")
            slices += 1
        half_slices = slices // 2
        npt = round_fft(int(math.sqrt(self.peakPicker.data.shape[0] ** 2 + self.peakPicker.data.shape[1] ** 2) + 1))

        if self.geoRef:
            self.ai.setPyFAI(**self.geoRef.getPyFAI())
            self.ai.wavelength = self.geoRef.wavelength
        logger.info("Performing autocorrelation on %sx%s, Fourier analysis may take some time", slices, npt)
        img, tth, chi = self.ai.integrate2d(self.peakPicker.data, npt, slices, azimuth_range=(-180, 180), unit="r_mm", method="splitpixel")
        ft = numpy.fft.fft(img, npt * 2, axis=-1)
        crosscor = numpy.fft.ifft(ft[:half_slices,:] * (ft[half_slices:,:].conj()), axis=-1).real
        centered = numpy.empty_like(crosscor)
        centered[:,:npt] = crosscor[:, npt:]
        centered[:, npt:] = crosscor[:,:npt]

        center = numpy.zeros(slices)  # in micron
        dr = (tth[1] - tth[0]) * 1000.0  # ouput in r(mm) -> micron

        # sub-bin precision obtained by second order expantion of peak
        range_half_slices = range(half_slices)
        x0 = centered.argmax(axis=-1)
        f_x = centered[(range_half_slices, x0)]
        f_xm1 = centered[(range_half_slices, x0 - 1)]
        f_xp1 = centered[(range_half_slices, x0 + 1)]
        f_prime = (f_xp1 - f_xm1) / 2.0
        f_second = (f_xp1 + f_xm1 - 2.0 * f_x)
        dx = -f_prime / f_second
        if (abs(dx) >= 0.5).any():
            msk = abs(dx) > 1
            logger.info("Correction is important ! %s", msk)
            dx[msk] = 0.0
        center[half_slices:] = (x0 + dx - npt) * dr
        center[:half_slices] = -center[half_slices:]
        if self.gui:
            if self.fig_center:
                self.fig_center.clf()
            else:
                self.fig_center = pylab.plt.figure()
            self.ax_center = self.fig_center.add_subplot(1, 1, 1)
            self.ax_center.set_xlim(-180, 180)
            self.ax_center.set_xticks(numpy.linspace(-180, 180, 9))
            self.ax_center.set_xlabel(r"Azimuthal angle $\chi$ ($^o$)")
            self.ax_center.set_ylabel(r"Error of the center position along radius ($\mu$m)")
            self.ax_center.set_title("Center plot")
            self.ax_center.plot(chi, center, label="From pattern cross-correlation")
            self.fig_center.show()
            update_fig(self.fig_center)

    def set_data(self, data):
        """call-back function for the peak-picker

        :param data: list of point with ring index
        :return: associated azimuthal integrator
        """
        if self.weighted:
            self.data = numpy.array(data)
        else:
            self.data = numpy.array(data)[:,:3]
        self.refine()
        return self.geoRef

    def reset_geometry(self, how="center", refine=False):
        """
        Reset the geometry: no tilt in all cases

        :param how: multiple options
            * center: set the PONI at the center of the detector
            * ring: center the poni at the middle of the inner-most ring
            * best: try both option and keeps the best (this option is not available)
        :param refine: launch the refinement (argument not used)
        """
        if how not in ["center", "ring"]:  # ,"best"]:
            logger.warning("unknow geometry reset method: %s, fall back on detector center", how)
            how = "center"
        if self.data is None:
            logger.warning("No datapoint: fall back on detector center")
            how = "center"
        # this is true for all:
        self.ai.rot1 = 0.0
        self.ai.rot2 = 0.0
        self.ai.rot3 = 0.0

        if how == "ring":
            inner_ring = min(set(i[2] for i in self.data))
            print("inner ring: %s" % inner_ring)
            data = numpy.array([[i[0], i[1]] for i in self.data if i[2] == inner_ring])
            center = data.mean(axis=0)
            self.ai.poni1, self.ai.poni2 = data.mean(axis=0)
            tth = self.calibrant.get_2th()[int(inner_ring)]
            dist = (data - center)
            d = numpy.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2).mean()
            self.ai.dist = d / numpy.tan(tth)
        elif how == "center":
            self.ai.dist = 0.1
            try:
                p1, p2, _p3 = self.detector.calc_cartesian_positions()
                self.ai.poni1 = p1.max() / 2.0
                self.ai.poni2 = p2.max() / 2.0
            except Exception as err:
                logger.warning(err)
                self.ai.poni1 = self.detector.pixel1 * (self.peakPicker.shape[0] / 2.)
                self.ai.poni2 = self.detector.pixel2 * (self.peakPicker.shape[1] / 2.)

        if self.geoRef:
            # reset geoRef object
            self.geoRef.set_dist_min(0)
            self.geoRef.set_dist_max(100)
            self.geoRef.set_dist(self.ai.dist)

            self.geoRef.set_poni1_min(-10.0 * self.ai.poni1)
            self.geoRef.set_poni1_max(10.0 * self.ai.poni1)
            self.geoRef.set_poni1(self.ai.poni1)

            self.geoRef.set_poni2_min(-10.0 * self.ai.poni2)
            self.geoRef.set_poni2_max(10.0 * self.ai.poni2)
            self.geoRef.set_poni2(self.ai.poni2)

            self.geoRef.set_rot1_min(-math.pi)
            self.geoRef.set_rot1_max(math.pi)
            self.geoRef.set_rot1(self.ai.rot1)

            self.geoRef.set_rot2_min(-math.pi)
            self.geoRef.set_rot2_max(math.pi)
            self.geoRef.set_rot2(self.ai.rot2)

            self.geoRef.set_rot3_min(-math.pi)
            self.geoRef.set_rot3_max(math.pi)
            self.geoRef.set_rot3(self.ai.rot3)

    def initgeoRef(self, defaults=None):
        """
        Tries to initialise the GeometryRefinement (dist, poni, rot)

        :param: default parameters as a dict to be passed to constructor of GeometryRefinement
        :return: initialized geometry refinement
        """
        if defaults is None:
            defaults = {"dist": 0.1, "poni1": 0.0, "poni2": 0.0,
                        "rot1": 0.0, "rot2": 0.0, "rot3": 0.0}
        else:
            defaults = defaults.copy()
        if self.detector:
            if not (defaults.get("poni1") or defaults.get("poni2")):
                try:
                    p1, p2, _p3 = self.detector.calc_cartesian_positions()
                    defaults["poni1"] = p1.max() / 2.
                    defaults["poni2"] = p2.max() / 2.
                except Exception as err:
                    logger.warning(err)
            defaults["detector"] = self.detector
        if self.ai:
            for key in defaults.keys():  # not PARAMETERS which holds wavelength
                val = getattr(self.ai, key, None)
                if val is not None:
                    defaults[key] = val
        if self.wavelength:
            defaults["wavelength"] = self.wavelength
        if self.calibrant:
            defaults["calibrant"] = self.calibrant
        if len(self.data):
            defaults["data"] = self.data
        georef = GeometryRefinement(**defaults)
        return  georef


class CliCalibration(AbstractCalibration):
    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None,
                 splineFile=None, detector=None, wavelength=None, calibrant=None):
        """Constructor of CliCalibration

        :param dataFiles: list of filenames containing data images
        :param darkFiles: list of filenames containing dark current images
        :param flatFiles: list of filenames containing flat images
        :param pixelSize: size of the pixel in meter as 2 tuple
        :param splineFile: file containing the distortion of the taper
        :param detector: Detector name or instance
        :param wavelength: radiation wavelength in meter
        :param calibrant: pyFAI.calibrant.Calibrant instance
        """
        AbstractCalibration.__init__(self, detector=detector, wavelength=wavelength, calibrant=calibrant)
        self.dataFiles = dataFiles
        self.darkFiles = darkFiles
        self.flatFiles = flatFiles

        self.detector = get_detector(detector, dataFiles)

        if splineFile and os.path.isfile(splineFile):
            self.detector.splineFile = os.path.abspath(splineFile)
        if pixelSize:
            if "__len__" in dir(pixelSize) and len(pixelSize) >= 2:
                self.detector.pixel1 = float(pixelSize[0])
                self.detector.pixel2 = float(pixelSize[1])
            else:
                self.detector.pixel1 = self.detector.pixel2 = float(pixelSize)

        self.cutBackground = None
        self.outfile = "merged.edf"
        self.saturation = 0
        self.fixed = FixedParameters(["wavelength", "rot3"])  # parameter fixed during optimization
        self.max_rings = None
        self.max_iter = 1000
        self.gui = True
        self.interactive = True
        self.filter = "mean"
        self.weighted = False
        self.polarization_factor = None
        self.parser = None
        self.nPt_1D = 1024
        self.nPt_2D_azim = 360
        self.nPt_2D_rad = 400
        self.unit = units.to_unit("2th_deg")
        self.keep = True
        self.check_calib = None
        self.fig_integrate = self.ax_xrpd_1d = self.ax_xrpd_2d = None
        self.fig_chiplot = self.ax_chiplot = None
        self.fig_center = self.ax_center = None

    def __repr__(self):
        lst = []
        if self.dataFiles:
            lst.append("data= " + ", ".join(self.dataFiles))
        else:
            lst.append("data= None")
        if self.darkFiles:
            lst.append("dark= " + ", ".join(self.darkFiles))
        else:
            lst.append("dark= None")
        if self.flatFiles:
            lst.append("flat= " + ", ".join(self.flatFiles))
        else:
            lst.append("flat= None")
        return AbstractCalibration.__repr__(self) + os.linesep + os.linesep.join(lst)

    def preprocess(self):
        """
        Common part:
        do dark, flat correction thresholding, ...
        and read missing data from keyboard if needed
        """
        # GF: self.saturation ignored if none of the other options active...
        if len(self.dataFiles) > 1 or self.cutBackground or self.darkFiles or self.flatFiles:
            self.outfile = average.average_images(self.dataFiles, self.outfile,
                                                  threshold=self.saturation,
                                                  minimum=self.cutBackground,
                                                  darks=self.darkFiles,
                                                  flats=self.flatFiles,
                                                  filter_=self.filter)
        else:
            self.outfile = self.dataFiles[0]

        url = urlparse(self.outfile)
        if (sys.platform == "win32") and (len(url.scheme) == 1):  # "c:" like path
            path = self.outfile
        else:
            if url.scheme not in self.VALID_URL:
                logger.warning("unexpected URL: %s", self.outfile)
            path = url.path
        self.basename, ext = os.path.splitext(path)
        if ext in [".gz", ".bz2"]:
            self.basename = os.path.splitext(self.basename)[0]

        if isinstance(self, Recalibration):
            self.keep = False
            self.pointfile = None
        else:
            self.pointfile = self.basename + ".npt"
        if self.wavelength is None:
            self.wavelength = self.ai.wavelength

        with fabio.open(self.outfile) as fimg:
            self.img = fimg.data

        AbstractCalibration.preprocess(self)

        # disable the callback mechanism !
        self.peakPicker.cb_refine = lambda x:None
        if not self.keep:
            self.peakPicker.points.reset()
            if not self.peakPicker.points.calibrant.wavelength:
                self.peakPicker.points.calibrant.wavelength = self.ai.wavelength
            elif self.ai.wavelength != self.peakPicker.points.calibrant.wavelength:
                self.peakPicker.points.calibrant.setWavelength_change2th(self.ai.wavelength)
        if not self.peakPicker.points.calibrant.dSpacing:
            wl = self.peakPicker.points.calibrant.wavelength
            self.read_dSpacingFile()
            if wl:
                self.peakPicker.points.calibrant.wavelength = wl
        if not self.peakPicker.points.calibrant.wavelength:
            self.read_wavelength()
            self.peakPicker.points.calibrant.wavelength = self.wavelength

################################################################################
# Calibration
################################################################################


class Calibration(CliCalibration):
    """
    class doing the calibration of frames
    """

    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None,
                 splineFile=None, detector=None, gaussianWidth=None,
                 wavelength=None, calibrant=None):
        """
        Constructor for calibration:

        :param dataFiles: list of filenames containing data images
        :param darkFiles: list of filenames containing dark current images
        :param flatFiles: list of filenames containing flat images
        :param pixelSize: size of the pixel in meter as 2 tuple
        :param splineFile: file containing the distortion of the taper
        :param detector: Detector name or instance
        :param wavelength: radiation wavelength in meter
        :param calibrant: pyFAI.calibrant.Calibrant instance

        """
        CliCalibration.__init__(self, dataFiles=dataFiles,
                                 darkFiles=darkFiles,
                                 flatFiles=flatFiles,
                                 pixelSize=pixelSize,
                                 splineFile=splineFile,
                                 detector=detector,
                                 calibrant=calibrant,
                                 wavelength=wavelength)
        self.gaussianWidth = gaussianWidth
        self.labelPattern = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    def __repr__(self):
        return CliCalibration.__repr__(self) + \
            "%sgaussian= %s" % (os.linesep, self.gaussianWidth)

    def parse(self, args=None):
        """
        parse options from command line
        """
        description = """Calibrate the diffraction setup geometry based on Debye-Sherrer rings images
without a priori knowledge of your setup.
You will need to provide a calibrant or a "d-spacing" file containing the spacing of Miller plans in
Angstrom (in decreasing order).
%s
or search in the American Mineralogist database:
http://rruff.geo.arizona.edu/AMS/amcsd.php
The --calibrant option is mandatory !""" % str(CALIBRANT_FACTORY)

        epilog = """The output of this program is a "PONI" file containing the detector description
and the 6 refined parameters (distance, center, rotation) and wavelength.
An 1D and 2D diffraction patterns are also produced. (.dat and .azim files)
        """
        usage = "pyFAI-calib [options] -w 1 -D detector -c calibrant.D imagefile.edf"
        self.configure_parser(usage=usage, description=description, epilog=epilog)  # common
        self.parser.add_argument("-r", "--reconstruct", dest="reconstruct",
                                 help="Reconstruct image where data are masked or <0  (for Pilatus "
                                 "detectors or detectors with modules)",
                                 action="store_true", default=False)

        self.parser.add_argument("-g", "--gaussian", dest="gaussian",
                                 help="""Size of the gaussian kernel.
Size of the gap (in pixels) between two consecutive rings, by default 100
Increase the value if the arc is not complete;
decrease the value if arcs are mixed together.""", default=None)
        self.parser.add_argument("--square", dest="square", action="store_true",
                                 help="Use square kernel shape for neighbor search instead of diamond shape",
                                 default=False)
        self.parser.add_argument("-p", "--pixel", dest="pixel",
                                 help="size of the pixel in micron", default=None)

        (options, _) = self.analyse_options(sysargv=args)
        # Analyse remaining aruments and options
        self.reconstruct = options.reconstruct
        self.gaussianWidth = options.gaussian
        if options.square:
            self.labelPattern = [[1] * 3] * 3
        else:
            self.labelPattern = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

        if options.pixel is not None:
            self.get_pixelSize(options.pixel)

    def preprocess(self):
        """
        do dark, flat correction thresholding, ...
        """
        CliCalibration.preprocess(self)

        if self.gaussianWidth is not None:
            self.peakPicker.massif.valley_size = self.gaussianWidth
        else:
            self.peakPicker.massif.init_valley_size()
        if self.gui:
            self.peakPicker.gui(log=True, maximize=True, pick=True,
                                widget_klass=QtMplCalibWidget)

    def gui_peakPicker(self):
        if self.peakPicker is None:
            self.preprocess()
        if os.path.isfile(self.pointfile):
            self.peakPicker.load(self.pointfile)
        if self.gui:
            self.peakPicker.widget.update()
        self.set_data(self.peakPicker.finish(self.pointfile))

    def refine(self, maxiter=1000000, fixed=None):
        """
        Contains the geometry refinement part specific to Calibration
        Sets up the initial guess when starting pyFAI-calib

        :param maxiter: number of iteration to run for in the minimizer
        :param fixed: a list of parameters for maintain fixed during the refinement. self.fixed by default.
        :return: nothing, object updated in place
        """

        fixed = self.fixed if fixed is None else fixed

        # First attempt
        self.geoRef = self.initgeoRef()
        self.geoRef.refine2(maxiter, fix=fixed)
        scor = self.geoRef.chi2()
        pars = [getattr(self.geoRef, p) for p in self.PARAMETERS]

        scores = [(scor, pars), ]

        # Second attempt
        self.geoRef = self.initgeoRef()
        self.geoRef.guess_poni()
        self.geoRef.refine2(maxiter, fix=fixed)
        scor = self.geoRef.chi2()
        pars = [getattr(self.geoRef, p) for p in self.PARAMETERS]

        scores.append((scor, pars))

        # Third attempt (can be from when a program bombed last time)
        paramfile = self.basename + ".poni"
        if os.path.isfile(paramfile):
            self.geoRef.load(paramfile)
            if self.wavelength:
                try:
                    old_wl = self.geoRef.wavelength
                except Exception as err:
                    logger.warning(err)
                else:
                    logger.warning("Overwriting wavelength from PONI file (%s) "
                                   "with the one from command line (%s)" %
                                   (old_wl, self.wavelength))
                self.geoRef.wavelength = self.wavelength
            if self.detector:
                gr_det = str(self.geoRef.detector)
                nw_det = str(self.detector)
                if gr_det != nw_det:
                    logger.warning("Overwriting detector from PONI file: %s%s "
                                   "with the one from command line %s%s" %
                                   (os.linesep, gr_det, os.linesep, nw_det))
                    self.geoRef.detector = self.detector

        # Third attempt
        self.geoRef.refine2(maxiter, fix=fixed)
        scor = self.geoRef.chi2()
        pars = [getattr(self.geoRef, p) for p in self.PARAMETERS]

        scores.append((scor, pars))

        # Choose the best scoring method: At this point we might also ask
        # a user to just type the numbers in?
        scores.sort()
        scor, pars = scores[0]
        for parval, parname in zip(pars, self.PARAMETERS):
            setattr(self.geoRef, parname, parval)

        # Now continue as before
        CliCalibration.refine(self)

################################################################################
# Recalibration
################################################################################


class Recalibration(CliCalibration):
    """
    class doing the re-calibration of frames
    """

    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None,
                 splineFile=None, detector=None, wavelength=None, calibrant=None):
        """
        Constructor for Recalibration:

        :param dataFiles: list of filenames containing data images
        :param darkFiles: list of filenames containing dark current images
        :param flatFiles: list of filenames containing flat images
        :param pixelSize: size of the pixel in meter as 2 tuple
        :param splineFile: file containing the distortion of the taper
        :param detector: Detector name or instance
        :param wavelength: radiation wavelength in meter
        :param calibrant: pyFAI.calibrant.Calibrant instance
        """
        CliCalibration.__init__(self, dataFiles=dataFiles,
                                 darkFiles=darkFiles,
                                 flatFiles=flatFiles,
                                 pixelSize=pixelSize,
                                 splineFile=splineFile,
                                 detector=detector,
                                 wavelength=wavelength,
                                 calibrant=calibrant)

    def parse(self, args=None):
        """
        parse options from command line
        """
        description = """Calibrate the diffraction setup geometry based on Debye-Sherrer rings images
with a priori knowledge of your setup (an input PONI-file).
You will need to provide a calibrant or a "d-spacing" file containing the spacing of Miller plans in
Angstrom (in decreasing order).
%s
or search in the American Mineralogist database:
http://rruff.geo.arizona.edu/AMS/amcsd.php
The --calibrant option is mandatory !
""" % str(CALIBRANT_FACTORY)

        epilog = """The main difference with pyFAI-calib is the way control-point hence Debye-Sherrer
rings are extracted. While pyFAI-calib relies on the contiguity of a region of peaks
called massif; pyFAI-recalib knows approximatly the geometry and is able to select
the region where the ring should be. From this region it selects automatically
the various peaks; making pyFAI-recalib able to run without graphical interface and
without human intervention (--no-gui and --no-interactive options).


Note that `pyFAI-recalib` program is obsolete as the same functionality is
available from within pyFAI-calib, using the `recalib` command in the
refinement process.
Two option are available for recalib: the numbe of rings to extract (similar to the -r option of this program)
and a new option which lets you choose between the original `massif` algorithm and newer ones like `blob` and `watershed` detection.
        """
        usage = "pyFAI-recalib [options] -i ponifile -w 1 -c calibrant.D imagefile.edf"
        self.configure_parser(usage=usage, description=description, epilog=epilog)

        self.parser.add_argument("-r", "--ring", dest="max_rings", type=int,
                                 help="maximum number of rings to extract. Default: all accessible", default=None)
        self.parser.add_argument("-k", "--keep", dest="keep",
                                 help="Keep existing control point and append new",
                                 default=False, action="store_true")

        options = self.parser.parse_args(args)
        args = options.args
        # Analyse aruments and options
        if (not options.poni) or (not os.path.isfile(options.poni)):
            logger.error("You should provide a PONI file as starting point !!")
        else:
            self.ai = AzimuthalIntegrator.sload(options.poni)
        if self.wavelength:
            self.ai.wavelength = self.wavelength
        self.max_rings = options.max_rings
        self.detector = self.ai.detector
        self.keep = options.keep
        self.analyse_options(options, args)

    def read_dSpacingFile(self):
        """Read the name of the file with d-spacing"""
        CliCalibration.read_dSpacingFile(self, verbose=False)

    def preprocess(self):
        """
        do dark, flat correction thresholding, ...
        """
        CliCalibration.preprocess(self)

        if self.gui:
            self.peakPicker.gui(log=True, maximize=True, pick=False,
                                widget_klass=QtMplCalibWidget)

    def refine(self, maxiter=1000000, fixed=None):
        """
        Contains the geometry refinement part specific to Recalibration

        :param maxiter: number of iteration to run for in the minimizer
        :param fixed: a list of parameters for maintain fixed during the refinement. self.fixed by default.
        :return: nothing, object updated in place
        """

        fixed = self.fixed if fixed is None else fixed
        self.geoRef = GeometryRefinement(self.data, dist=self.ai.dist, poni1=self.ai.poni1,
                                         poni2=self.ai.poni2, rot1=self.ai.rot1,
                                         rot2=self.ai.rot2, rot3=self.ai.rot3,
                                         detector=self.ai.detector, calibrant=self.calibrant,
                                         wavelength=self.wavelength)
        self.ai = self.geoRef
        self.geoRef.set_tolerance(10)
        CliCalibration.refine(self, maxiter=maxiter, fixed=fixed)


class MultiCalib(object):

    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None, splineFile=None, detector=None):
        """
        """
        self.dataFiles = dataFiles or []
        self.darkFiles = darkFiles or []
        self.flatFiles = flatFiles or []
        self.data = {}

        self.detector = get_detector(detector, dataFiles)

        if splineFile and os.path.isfile(splineFile):
            self.detector.splineFile = os.path.abspath(splineFile)
        if pixelSize:
            if "__len__" in dir(pixelSize) and len(pixelSize) >= 2:
                self.detector.pixel1 = float(pixelSize[0])
                self.detector.pixel2 = float(pixelSize[1])
            else:
                self.detector.pixel1 = self.detector.pixel2 = float(pixelSize)
        self.cutBackground = None
        self.outfile = "merged.edf"
        self.peakPicker = "blob"
        self.basename = None
        self.geoRef = None
#        self.reconstruct = False
        self.mask = None
        self.max_iter = 1000
        self.filter = "mean"
        self.saturation = 0.1
        self.calibrant = None
        self.wavelength = None
        self.weighted = False
        self.polarization_factor = 0
        self.results = {}
        self.gui = True
        self.interactive = True
        self.poni1 = None
        self.poni2 = None
        self.dist = None
        self.fixed = FixedParameters()
        self.max_rings = None
        self.rot1 = 0.0
        self.rot2 = 0.0
        self.rot3 = 0.0

    def __repr__(self):
        lst = [f"{self.__class__.__name__} object:",
               "data= " + ", ".join(self.dataFiles),
               "dark= " + ", ".join(self.darkFiles),
               "flat= " + ", ".join(self.flatFiles)]
        lst.append(self.detector.__repr__())
        return os.linesep.join(lst)

    def parse(self, exe=None, description=None, epilog=None, args=None):
        """
        parse options from command line
        :param exe: name of the program (MX-calibrate)
        :param description: Description of the program
        """
        if exe is None:
            exe = "MX-Calibrate"
            usage = "%s -w 1.54 -c CeO2 file1.cbf file2.cbf ..." % exe
            version = "%s from pyFAI version %s: %s" % (exe, PyFAI_VERSION, PyFAI_DATE)
            description = """
        Calibrate automatically a set of frames taken at various sample-detector distance.
        Return the linear regression of the fit in funtion of the sample-setector distance.

        Nota: this tool is deprecated in favor of the jupyter notebook found in the documentation
        (with the same name).
        """
            epilog = """This tool has been developed for ESRF MX-beamlines where an acceptable calibration is
        usually present is the header of the image. PyFAI reads it and does a "recalib" on
        each of them before exporting a linear regression of all parameters versus this distance.
        """
        else:
            description = description or ""
            epilog = epilog or ""
        parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
        parser.add_argument("-V", "--version", action='version', version=version)
        parser.add_argument("args", metavar="FILE", help="List of files to calibrate", nargs='+')
#        parser.add_argument("-V", "--version", dest="version", action="store_true",
#                          help="print version of the program and quit", metavar="FILE", default=False)
#        parser.add_argument("-o", "--out", dest="outfile",
#                          help="Filename where processed image is saved", metavar="FILE", default="merged.edf")
        parser.add_argument("-v", "--verbose",
                            action="store_true", dest="debug", default=False,
                            help="switch to debug/verbose mode")
#        parser.add_argument("-g", "--gaussian", dest="gaussian", help="""Size of the gaussian kernel.
# Size of the gap (in pixels) between two consecutive rings, by default 100
# Increase the value if the arc is not complete;
# decrease the value if arcs are mixed together.""", default=None)
#        parser.add_argument("-c", "--square", dest="square", action="store_true",
#                      help="Use square kernel shape for neighbor search instead of diamond shape", default=False)
        parser.add_argument("-c", "--calibrant", dest="calibrant", metavar="FILE",
                            help="file containing d-spacing of the calibrant reference sample (MANDATORY)", default=None)
        parser.add_argument("-w", "--wavelength", dest="wavelength", type=float,
                            help="wavelength of the X-Ray beam in Angstrom", default=None)
        parser.add_argument("-e", "--energy", dest="energy", type=float,
                            help="energy of the X-Ray beam in keV (hc=%skeV.A)" % hc, default=None)
        parser.add_argument("-P", "--polarization", dest="polarization_factor",
                            type=float, default=0.0,
                            help="polarization factor, from -1 (vertical) to +1 (horizontal), default is 0, synchrotrons are around 0.95")
        parser.add_argument("-b", "--background", dest="background",
                            help="Automatic background subtraction if no value are provided", default=None)
        parser.add_argument("-d", "--dark", dest="dark",
                            help="list of dark images to average and subtract", default=None)
        parser.add_argument("-f", "--flat", dest="flat",
                            help="list of flat images to average and divide", default=None)
#        parser.add_argument("-r", "--reconstruct", dest="reconstruct",
#                      help="Reconstruct image where data are masked or <0  (for Pilatus detectors or detectors with modules)",
#                      action="store_true", default=False)
        parser.add_argument("-s", "--spline", dest="spline",
                            help="spline file describing the detector distortion", default=None)
        parser.add_argument("-p", "--pixel", dest="pixel",
                            help="size of the pixel in micron", default=None)
        parser.add_argument("-D", "--detector", dest="detector_name",
                            help="Detector name (instead of pixel size+spline)", default=None)
        parser.add_argument("-m", "--mask", dest="mask",
                            help="file containing the mask (for image reconstruction)", default=None)
#        parser.add_argument("-n", "--npt", dest="npt",
#                      help="file with datapoints saved", default=None)
        parser.add_argument("--filter", dest="filter",
                            help="select the filter, either mean(default), max or median",
                            default="mean")
        parser.add_argument("--saturation", dest="saturation",
                            help="consider all pixel>max*(1-saturation) as saturated and reconstruct them",
                            default=0.1, type=float)
        parser.add_argument("-r", "--ring", dest="max_rings", type=float,
                            help="maximum number of rings to extract", default=None)

        parser.add_argument("--weighted", dest="weighted",
                            help="weight fit by intensity",
                            default=False, action="store_true")
        parser.add_argument("-l", "--distance", dest="distance", type=float,
                            help="sample-detector distance in millimeter", default=None)

        parser.add_argument("--tilt", dest="tilt",
                            help="Allow initially detector tilt to be refined (rot1, rot2, rot3). Default: Activated",
                            default=None, action="store_true")
        parser.add_argument("--no-tilt", dest="tilt",
                            help="Deactivated tilt refinement and set all rotation to 0", default=None, action="store_false")

        parser.add_argument("--dist", dest="dist", type=float,
                            help="sample-detector distance in meter", default=None)
        parser.add_argument("--poni1", dest="poni1", type=float,
                            help="poni1 coordinate in meter", default=None)
        parser.add_argument("--poni2", dest="poni2", type=float,
                            help="poni2 coordinate in meter", default=None)
        parser.add_argument("--rot1", dest="rot1", type=float,
                            help="rot1 in radians", default=None)
        parser.add_argument("--rot2", dest="rot2", type=float,
                            help="rot2 in radians", default=None)
        parser.add_argument("--rot3", dest="rot3", type=float,
                            help="rot3 in radians", default=None)

        parser.add_argument("--fix-dist", dest="fix_dist",
                            help="fix the distance parameter", default=None, action="store_true")
        parser.add_argument("--free-dist", dest="fix_dist",
                            help="free the distance parameter", default=None, action="store_false")

        parser.add_argument("--fix-poni1", dest="fix_poni1",
                            help="fix the poni1 parameter", default=None, action="store_true")
        parser.add_argument("--free-poni1", dest="fix_poni1",
                            help="free the poni1 parameter", default=None, action="store_false")

        parser.add_argument("--fix-poni2", dest="fix_poni2",
                            help="fix the poni2 parameter", default=None, action="store_true")
        parser.add_argument("--free-poni2", dest="fix_poni2",
                            help="free the poni2 parameter", default=None, action="store_false")

        parser.add_argument("--fix-rot1", dest="fix_rot1",
                            help="fix the rot1 parameter", default=None, action="store_true")
        parser.add_argument("--free-rot1", dest="fix_rot1",
                            help="free the rot1 parameter", default=None, action="store_false")

        parser.add_argument("--fix-rot2", dest="fix_rot2",
                            help="fix the rot2 parameter", default=None, action="store_true")
        parser.add_argument("--free-rot2", dest="fix_rot2",
                            help="free the rot2 parameter. Default: Activated", default=None, action="store_false")

        parser.add_argument("--fix-rot3", dest="fix_rot3",
                            help="fix the rot3 parameter", default=None, action="store_true")
        parser.add_argument("--free-rot3", dest="fix_rot3",
                            help="free the rot3 parameter. Default: Activated", default=None, action="store_false")

        parser.add_argument("--fix-wavelength", dest="fix_wavelength",
                            help="fix the wavelength parameter. Default: Activated", default=True, action="store_true")
        parser.add_argument("--free-wavelength", dest="fix_wavelength",
                            help="free the wavelength parameter. Default: Deactivated ", default=True, action="store_false")

        parser.add_argument("--no-gui", dest="gui",
                            help="force the program to run without a Graphical interface",
                            default=True, action="store_false")
        parser.add_argument("--gui", dest="gui",
                            help="force the program to run with a Graphical interface",
                            default=True, action="store_true")

        parser.add_argument("--no-interactive", dest="interactive",
                            help="force the program to run and exit without prompting for refinements",
                            default=True, action="store_false")
        parser.add_argument("--interactive", dest="interactive",
                            help="force the program to prompt for refinements",
                            default=True, action="store_true")
        parser.add_argument("--peak-picker", dest="peakPicker",
                            help="Uses the 'massif', 'blob' or 'watershed' peak-picker algorithm (default: blob)",
                            default="blob", type=str)
        options = parser.parse_args(args)

        # Analyse aruments and options
        if options.debug:
            logger.setLevel(logging.DEBUG)
        if options.background is not None:
            try:
                self.cutBackground = float(options.background)
            except Exception:
                self.cutBackground = True
        if options.dark:
            self.darkFiles = [f for f in options.dark.split(",") if os.path.isfile(f)]
        if options.flat:
            self.flatFiles = [f for f in options.flat.split(",") if os.path.isfile(f)]
        if options.mask and os.path.isfile(options.mask):
            with fabio.open(options.mask) as fimg:
                self.mask = fimg.data

        if options.detector_name:
            self.detector = get_detector(options.detector_name, options.args)
        if options.spline:
            if os.path.isfile(options.spline):
                self.detector.splineFile = os.path.abspath(options.spline)
            else:
                logger.error("Unknown spline file %s", options.spline)
        if options.pixel is not None:
            self.get_pixelSize(options.pixel)
        self.filter = options.filter
        self.saturation = options.saturation
        if options.wavelength:
            self.wavelength = 1e-10 * options.wavelength
        elif options.energy:
            self.wavelength = 1e-10 * hc / options.energy
        if not options.calibrant:
            logger.error("The calibrant is mandatory: please use the -c option")
        self.calibrant = options.calibrant
        self.polarization_factor = options.polarization_factor
        self.gui = options.gui
        self.interactive = options.interactive
        self.max_rings = options.max_rings

        self.fixed = FixedParameters()
        if options.tilt is not None:
            for key in ["rot1", "rot2", "rot3"]:
                self.fixed.add_or_discard(key, not(options.tilt))
        self.fixed.add_or_discard("dist", options.fix_dist)
        self.fixed.add_or_discard("poni1", options.fix_poni1)
        self.fixed.add_or_discard("poni2", options.fix_poni2)
        self.fixed.add_or_discard("rot1", options.fix_rot1)
        self.fixed.add_or_discard("rot2", options.fix_rot2)
        self.fixed.add_or_discard("rot3", options.fix_rot3)
        self.fixed.add_or_discard("wavelength", options.fix_wavelength)

        if options.distance:
            self.dist = 1e-3 * float(options.distance)
        if options.dist:
            self.dist = float(options.dist)
        if options.poni1:
            self.poni1 = float(options.poni1)
        if options.poni2:
            self.poni2 = float(options.poni2)
        if options.rot1:
            self.rot1 = float(options.rot1)
        if options.rot2:
            self.rot2 = float(options.rot2)
        if options.rot3:
            self.rot3 = float(options.rot3)

        self.dataFiles = [f for f in options.args if os.path.isfile(f)]
        if not self.dataFiles:
            raise RuntimeError("Please provide some calibration images ... "
                               "if you want to analyze them. Try also the --help option to see all options!")
        self.weighted = options.weighted
        if options.peakPicker.lower() in PeakPicker.VALID_METHODS:
            self.peakPicker = options.peakPicker.lower()

    def get_pixelSize(self, ans):
        """convert a comma separated sting into pixel size"""
        sp = ans.split(",")
        if len(sp) >= 2:
            try:
                pixelSizeXY = [float(i) * 1e-6 for i in sp[:2]]
            except Exception:
                logger.error("error in reading pixel size_2")
                return
        elif len(sp) == 1:
            px = sp[0]
            try:
                pixelSizeXY = [float(px) * 1e-6, float(px) * 1e-6]
            except Exception:
                logger.error("error in reading pixel size_1")
                return
        else:
            logger.error("error in reading pixel size_0")
            return
        self.detector.pixel1 = pixelSizeXY[1]
        self.detector.pixel2 = pixelSizeXY[0]

    def read_pixelsSize(self):
        """Read the pixel size from prompt if not available"""
        if (self.detector.pixel1 is None) and (self.detector.splineFile is None):
            pixelSize = [15, 15]
            ans = input("Please enter the pixel size (in micron, comma separated X, Y "
                        "i.e. %.2e,%.2e) or a spline file: " % tuple(pixelSize)).strip()
            if os.path.isfile(ans):
                self.detector.splineFile = ans
            else:
                self.get_pixelSize(ans)

    def read_dSpacingFile(self):
        """Read the name of the file with d-spacing"""
        CliCalibration.read_dSpacingFile(self, verbose=False)

    def read_wavelength(self):
        """Read the wavelength"""
        while not self.wavelength:
            ans = input("Please enter wavelength in Angstrom:\t").strip()
            try:
                self.wavelength = 1e-10 * float(ans)
            except:
                self.wavelength = None

    def process(self):
        """

        """
        self.dataFiles.sort()
        for fn in self.dataFiles:
            with fabio.open(fn) as fabimg:
                wavelength = self.wavelength
                dist = self.dist
                if self.poni2:
                    centerX = self.poni2 / self.detector.pixel2
                else:
                    centerX = None
                if self.poni1:
                    centerY = self.poni1 / self.detector.pixel1
                else:
                    centerY = None
                if "_array_data.header_contents" in fabimg.header:
                    headers = fabimg.header["_array_data.header_contents"].lower().split()
                    if "detector_distance" in headers:
                        dist = float(headers[headers.index("detector_distance") + 1])
                    if "wavelength" in headers:
                        wavelength = float(headers[headers.index("wavelength") + 1]) * 1e-10
                    if "beam_xy" in headers:
                        centerX = float(headers[headers.index("beam_xy") + 1][1:-1])
                        centerY = float(headers[headers.index("beam_xy") + 2][:-1])
                if dist is None:
                    digits = ""
                    for i in os.path.basename(fn):
                        if i.isdigit() and not digits:
                            digits += i
                        elif i.isdigit():
                            digits += i
                        elif not i.isdigit() and digits:
                            break
                    dist = int(digits) * 0.001
                if centerX is None:
                    centerX = fabimg.data.shape[1] // 2
                if centerY is None:
                    centerY = fabimg.data.shape[0] // 2
            self.results[fn] = {"wavelength": wavelength, "dist": dist}
            rec = Recalibration(dataFiles=[fn], darkFiles=self.darkFiles,
                                flatFiles=self.flatFiles, detector=self.detector,
                                calibrant=self.calibrant, wavelength=wavelength)
            rec.outfile = os.path.splitext(fn)[0] + ".proc.edf"
            rec.interactive = self.interactive
            rec.gui = self.gui
            rec.saturation = self.saturation
            rec.mask = self.mask
            rec.filter = self.filter
            rec.cutBackground = self.cutBackground
            rec.fixed = self.fixed
            rec.max_rings = self.max_rings
            rec.weighted = self.weighted
            if centerY:
                rec.ai.poni1 = centerY * self.detector.pixel1
            if centerX:
                rec.ai.poni2 = centerX * self.detector.pixel2
            if dist:
                rec.ai.dist = dist
            rec.preprocess()
            rec.extract_cpt(method=self.peakPicker)
            rec.refine()
            self.results[fn]["ai"] = rec.ai

    def regression(self):
        print(self.results)
        dist = numpy.zeros(len(self.results))
        x = dist.copy()
        poni1 = dist.copy()
        poni2 = dist.copy()
        rot1 = dist.copy()
        rot2 = dist.copy()
        rot3 = dist.copy()
        direct = dist.copy()
        tilt = dist.copy()
        trp = dist.copy()
        centerX = dist.copy()
        centerY = dist.copy()
        idx = 0
        print("")
        print("Results of linear regression for distance in mm")
        for key, dico in self.results.items():
            print(key, dico["dist"])
            print(dico["ai"])
            x[idx] = dico["dist"] * 1000
            dist[idx] = dico["ai"].dist
            poni1[idx] = dico["ai"].poni1
            poni2[idx] = dico["ai"].poni2
            rot1[idx] = dico["ai"].rot1
            rot2[idx] = dico["ai"].rot2
            rot3[idx] = dico["ai"].rot3
            f = dico["ai"].getFit2D()
            direct[idx] = f["directDist"]
            tilt[idx] = f["tilt"]
            trp[idx] = f["tiltPlanRotation"]
            centerX[idx] = f["centerX"]
            centerY[idx] = f["centerY"]
            idx += 1
        for name, elt in [("dist", dist),
                          ("poni1", poni1), ("poni2", poni2),
                          ("rot1", rot1), ("rot2", rot2), ("rot3", rot3),
                          ("direct", direct), ("tilt", tilt), ("trp", trp),
                          ("centerX", centerX), ("centerY", centerY)]:
            slope, intercept, r, _two, stderr = linregress(x, elt)

            print("%s = %s * dist_mm + %s \t R= %s\t stderr= %s" % (name, slope, intercept, r, stderr))


class CheckCalib(object):

    def __init__(self, poni=None, img=None, unit="2th_deg"):
        self.ponifile = poni
        if poni:
            self.ai = AzimuthalIntegrator.sload(poni)
        else:
            self.ai = None
        if img:
            with fabio.open(img) as fimg:
                self.img = fimg.data
        else:
            self.img = None
        self.mask = None
        self.r = None
        self.I = None
        self.wavelength = None
        self.resynth = None
        self.delta = None
        self.unit = unit
        self.masked_resynth = None
        self.masked_image = None
        self.offset = None
        self.data = None
        self.fig = None

    def __repr__(self, *args, **kwargs):
        res = [f"{self.__class__.__name__} object with:"]
        if self.ai:
            res.append("ai: " + self.ai.__repr__())
        return os.linesep.join(res)

    def parse(self, args=None):
        logger.debug("in parse")
        usage = "check_calib [options] -p param.poni image.edf"
        description = """Check_calib is a deprecated tool aiming at validating both the geometric
calibration and everything else like flat-field correction, distortion
correction, at a sub-pixel level.

Note that `check_calib` program is obsolete as the same functionality is
available from within pyFAI-calib, using the `validate` command in the
refinement process.

        :returns: True if the parsing succeed, else False
        """
        version = "check_calib from pyFAI version %s: %s" % (PyFAI_VERSION, PyFAI_DATE)
        parser = ArgumentParser(usage=usage,
                                description=description)
        parser.add_argument("-V", "--version", action='version', version=version)
        parser.add_argument("args", metavar="FILE", help="Image file to check calibration for", nargs='+')
        parser.add_argument("-v", "--verbose",
                            action="store_true", dest="verbose", default=False,
                            help="switch to debug mode")
        parser.add_argument("-d", "--dark", dest="dark", metavar="FILE", type=str,
                            help="file containing the dark images to subtract", default=None)
        parser.add_argument("-f", "--flat", dest="flat", metavar="FILE", type=str,
                            help="file containing the flat images to divide", default=None)
        parser.add_argument("-m", "--mask", dest="mask", metavar="FILE", type=str,
                            help="file containing the mask", default=None)
        parser.add_argument("-p", "--poni", dest="poni", metavar="FILE", type=str,
                            help="file containing the diffraction parameter (poni-file)",
                            default=None)
        parser.add_argument("-e", "--energy", dest="energy", type=float,
                            help="energy of the X-Ray beam in keV (hc=%skeV.A)" % hc, default=None)
        parser.add_argument("-w", "--wavelength", dest="wavelength", type=float,
                            help="wavelength of the X-Ray beam in Angstrom", default=None)

        options = parser.parse_args(args)
        if options.verbose:
            logger.setLevel(logging.DEBUG)

        if options.mask is not None:
            with fabio.open(options.mask) as fimg:
                self.mask = (fimg.data != 0)
        args = expand_args(options.args)
        if len(args) > 0:
            f = args[0]
            if os.path.isfile(f):
                with fabio.open(f) as fimg:
                    self.img = fimg.data.astype(numpy.float32)
            else:
                print("Please enter diffraction images as arguments")
                return False
            for f in args[1:]:
                with fabio.open(f) as fimg:
                    self.img += fimg.data
        if options.dark and fabio_exists(options.dark):
            with fabio.open(options.dark) as fimg:
                self.img -= fimg.data
        if options.flat and fabio_exists(options.flat):
            with fabio.open(options.flat) as fimg:
                self.img /= fimg.data
        if options.poni:
            self.ai = AzimuthalIntegrator.sload(options.poni)
        self.data = [f for f in args if os.path.isfile(f)]
        if options.poni is None:
            logger.error("PONI parameter is mandatory")
            return False
        self.ai = AzimuthalIntegrator.sload(options.poni)
        if options.wavelength:
            self.ai.wavelength = 1e-10 * options.wavelength
        elif options.energy:
            self.ai.wavelength = 1e-10 * hc / options.energy
        # else:
        #     self.read_wavelength()
        return True

    def get_1dsize(self):
        logger.debug("in get_1dsize")
        return int(numpy.sqrt(self.img.shape[0] ** 2 + self.img.shape[1] ** 2))

    size1d = property(get_1dsize)

    def integrate(self):
        logger.debug("in integrate")
        self.r, self.I = self.ai.integrate1d_ng(self.img, self.size1d, mask=self.mask,
                                                unit=self.unit, method=("full", "histo", "cython"))

    def rebuild(self):
        """
        Rebuild the diffraction image and measures the offset with the reference
        :return: offset
        """
        logger.debug("in rebuild")
        if self.r is None:
            self.integrate()

        self.resynth = self.ai.calcfrom1d(self.r, self.I, shape=self.img.shape, mask=self.mask,
                                          dim1_unit=self.unit, correctSolidAngle=True)
        if self.mask is not None:
            self.img[numpy.where(self.mask)] = 0

        self.delta = self.resynth - self.img
        if self.mask is not None:
            smooth_mask = self.smooth_mask()
        else:
            smooth_mask = 1.0
        self.masked_resynth = self.resynth * smooth_mask
        self.masked_image = self.img * smooth_mask
        self.offset = measure_offset(self.masked_resynth, self.masked_image, withLog=0)
        print("Measured offset: %s" % str(self.offset))
        return self.offset

    def smooth_mask(self, hwhm=5):
        """
        smooth out around the mask to avoid aligning on the mask
        """
        logger.debug("in smooth_mask")
        fwhm = int(round(2.0 * hwhm))
        sigma = hwhm / math.sqrt(2 * math.log(2))

        if self.mask is not None:
            if not pyFAI_morphology:
                my, mx = numpy.ogrid[-fwhm: fwhm + 1, -fwhm:fwhm + 1]
                grow = (mx * mx + my * my) <= 4.0 * hwhm * hwhm
                big_mask = morphology.binary_dilation(self.mask, grow)
            else:
                big_mask = morphology.binary_dilation(self.mask.astype(numpy.int8), fwhm)
            smooth_mask = 1.0 - gaussian_filter(big_mask.astype(numpy.float32), sigma)
            return smooth_mask

    def show(self):
        """
        Show the image with the the errors
        """
        if self.fig is None:
            self.fig = pylab.figure()
            if not gui_utils.main_loop:
                self.fig.show()
        else:
            self.fig.clf()
        ax1 = self.fig.add_subplot(2, 2, 3)
        ax1.imshow(self.delta, aspect="auto", interpolation="nearest", origin="bottom")
        ax1.set_title("Difference image")
        ax2 = self.fig.add_subplot(2, 2, 1)
        ax2.imshow(self.masked_image, aspect="auto", interpolation="nearest", origin="bottom")
        ax2.set_title("Raw image")
        ax3 = self.fig.add_subplot(2, 2, 2)
        ax3.imshow(self.masked_resynth, aspect="auto", interpolation="nearest", origin="bottom")
        ax3.set_title("Rebuild image")
        ax4 = self.fig.add_subplot(2, 2, 4)
        ax4.plot(self.r, self.I)
        ax4.set_title("powder pattern")
        ax4.set_xlabel(r"2$\theta$ ($^o$)")
        ax4.set_ylabel("Intensity")
        update_fig(self.fig)

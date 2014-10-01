#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
pyFAI-calib

A tool for determining the geometry of a detector using a reference sample.

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/10/2014"
__status__ = "production"

import os, sys, time, logging, types, math
try:
    from argparse import ArgumentParser
except ImportError:
    from .argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyFAI.calibration")
import numpy, scipy.ndimage
from scipy.stats import linregress
import fabio
from .gui_utils import pylab, update_fig, matplotlib
from .detectors import detector_factory, Detector
from .geometryRefinement import GeometryRefinement
from .peak_picker import PeakPicker
from . import units, gui_utils
from .utils import averageImages, measure_offset, expand_args, readFloatFromKeyboard
from .azimuthalIntegrator import AzimuthalIntegrator
from .units import hc
from . import version as PyFAI_VERSION
from . import date as PyFAI_DATE
from .calibrant import Calibrant, ALL_CALIBRANTS
try:
    from ._convolution import gaussian_filter
except ImportError:
    from scipy.ndimage.filters import gaussian_filter

try:
    from . import morphology
except ImportError:
    from scipy.ndimage import morphology
    pyFAI_morphology = False
else:
    pyFAI_morphology = True


def get_detector(detector, datafiles=None):
    """
    Detector factory taking into account the binning knowing the datafiles
    @param detector: string or detector or other junk
    @param datafiles: can be a list of images to be opened and their shape used.
    @return pyFAI.detector.Detector instance
    """
    res = None
    if type(detector) in types.StringTypes:
        try:
            res = detector_factory(detector)
        except RuntimeError:
            print("Not a valid detector: %s" % detector)
            sys.exit(-1)
    elif isinstance(detector, Detector):
        res = detector
    else:
        res = Detector()
    if datafiles and os.path.exists(datafiles[0]):
        shape = fabio.open(datafiles[0]).data.shape
        res.guess_binning(shape)
    return res


class AbstractCalibration(object):

    """
    Everything that is common to Calibration and Recalibration
    """

    win_error = "We are under windows, matplotlib is not able to"\
                         " display too many images without crashing, this"\
                         " is why the window showing the diffraction image"\
                         " is closed"
    HELP = {"help": "Try to get the help of a given action, like 'refine?'. Use done when finished. "
            "Most command are composed of 'action parameter value' like 'set wavelength 1e-10'.",
            "get": "print he value of a parameter",
            "set": "set the value of a parameter to the given value, i.e 'set wavelength 1e-10'",
            'fix': "fixes the value of a parameter so that its value will not be optimized, i.e. 'fix wavelength'",
            'free': "frees the parameter so that the value can be optimized, i.e. 'free wavelength'",
            'bound': "sets the upper and lower bound of a parameter: 'bound dist 0.1 0.2'",
            'bounds': "sets the upper and lower bound of all parameters",
            'refine': "performs a new cycle of refinement",
            'recalib': "extract a new set of rings and re-perform the calibration. One can specify how many rings to extract and the algorithm to use (blob or massif)",
            'done': "finishes the processing, performs an integration and quits",
            'validate': "measures the offset between the calibrated image and the back-projected image",
            'integrate': "perform the azimuthal integration and display results",
            'abort': "quit immediately, discarding any unsaved changes",
            'show': "Just print out the current parameter set",
            'reset': "Reset the geometry to the initial guess (rotation to zero, distance to 0.1m, poni at the center of the image)"
            }
    PARAMETERS = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"]
    UNITS = {"dist":"meter", "poni1":"meter", "poni2":"meter", "rot1":"radian",
             "rot2":"radian", "rot3":"radian", "wavelength":"meter"}

    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None,
                 splineFile=None, detector=None, wavelength=None, calibrant=None):
        """
        Constructor:

        @param dataFiles: list of filenames containing data images
        @param darkFiles: list of filenames containing dark current images
        @param flatFiles: list of filenames containing flat images
        @param pixelSize: size of the pixel in meter as 2 tuple
        @param splineFile: file containing the distortion of the taper
        @param detector: Detector name or instance
        @param wavelength: radiation wavelength in meter
        @param calibrant: pyFAI.calibrant.Calibrant instance
        """
        self.dataFiles = dataFiles
        self.darkFiles = darkFiles
        self.flatFiles = flatFiles
        self.pointfile = None

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
        self.peakPicker = None
        self.img = None
        self.ai = AzimuthalIntegrator(dist=1, detector=self.detector)
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
            elif calibrant in ALL_CALIBRANTS:
                self.calibrant = ALL_CALIBRANTS[calibrant]
            elif os.path.isfile(calibrant) and os.path.isfile(calibrant):
                self.calibrant = Calibrant(calibrant)
            else:
                logger.error("Unable to handle such calibrant %s" % calibrant)
                self.calibrant = None
        else:
            self.calibrant = None
        self.mask = None
        self.saturation = 0
        self.fixed = ["wavelength"]  # parameter fixed during optimization
        self.max_rings = None
        self.max_iter = 1000
        self.gui = True
        self.interactive = True
        self.filter = "mean"
        self.basename = None
        self.weighted = False
        self.polarization_factor = None
        self.parser = None
        self.nPt_1D = 1024
        self.nPt_2D_azim = 360
        self.nPt_2D_rad = 400
        self.unit = None
        self.keep = True
        self.check_calib = None
        self.fig3 = self.ax_xrpd_1d = self.ax_xrpd_2d = None

    def __repr__(self):
        lst = ["Calibration object:"]
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
                      help="polarization factor, from -1 (vertical) to +1 (horizontal),"\
                      " default is None (no correction), synchrotrons are around 0.95")
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
                      help="sample-detector distance in millimeter. Default: 0.1m", default=None)
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

        self.parser.add_argument("--saturation", dest="saturation",
                      help="consider all pixel>max*(1-saturation) as saturated and "\
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
                      help="Valid units for radial range: 2th_deg, 2th_rad, q_nm^-1,"\
                      " q_A^-1, r_mm. Default: 2th_deg", type=str, default="2th_deg")
        self.parser.add_argument("--no-gui", dest="gui",
                      help="force the program to run without a Graphical interface",
                      default=True, action="store_false")
        self.parser.add_argument("--no-interactive", dest="interactive",
                      help="force the program to run and exit without prompting"\
                      " for refinements", default=True, action="store_false")


    def analyse_options(self, options=None, args=None):
        """
        Analyse options and arguments

        @return: option,arguments
        """
        if (options is None) and  (args is None):
            options = self.parser.parse_args()
            args = options.args
        if options.debug:
            logger.setLevel(logging.DEBUG)
        self.outfile = options.outfile
        if options.dark:
            self.darkFiles = [f for f in options.dark.split(",") if os.path.isfile(f)]
            if not self.darkFiles:  #empty container !!!
                logger.error("No dark file exists !!!")
                self.darkFiles = None
        if options.flat:
            self.flatFiles = [f for f in options.flat.split(",") if os.path.isfile(f)]
            if not self.flatFiles:  #empty container !!!
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
                logger.error("Unknown spline file %s" % (options.spline))

        if options.mask and os.path.isfile(options.mask):
            self.mask = (fabio.open(options.mask).data != 0)
        else:  # Use default mask provided by detector
            self.mask = self.detector.mask


        self.pointfile = options.npt
        if options.spacing:
            if options.spacing in ALL_CALIBRANTS:
                self.calibrant = ALL_CALIBRANTS[options.spacing]
            elif os.path.isfile(options.spacing):
                self.calibrant = Calibrant(options.spacing)
            else:
                logger.error("No such Calibrant / d-Spacing file: %s" % options.spacing)

        if self.calibrant is None:
            self.read_dSpacingFile(True)

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
        self.fixed = []
        if options.fix_dist:
            self.fixed.append("dist")
        if options.fix_poni1:
            self.fixed.append("poni1")
        if options.fix_poni2:
            self.fixed.append("poni2")
        if options.fix_rot1:
            self.fixed.append("rot1")
        if options.fix_rot2:
            self.fixed.append("rot2")
        if options.fix_rot3:
            self.fixed.append("rot3")
        if options.fix_wavelength:
            self.fixed.append("wavelength")
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
            ans = raw_input("Please enter the pixel size (in micron, comma separated X,Y "\
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
                ans = raw_input("Please enter the calibrant name or the file"
                                " containing the d-spacing:\t").strip()
                if ans in ALL_CALIBRANTS:
                    self.calibrant = ALL_CALIBRANTS[ans]
                    valid = True
                elif os.path.isfile(ans):
                    self.calibrant = Calibrant(ans)
                    valid = True

    def read_wavelength(self):
        """Read the wavelength"""
        while not self.wavelength:
            ans = raw_input("Please enter wavelength in Angstrom:\t").strip()
            try:
                self.wavelength = self.ai.wavelength = 1e-10 * float(ans)
            except Exception:
                self.wavelength = None

    def preprocess(self):
        """
        Common part:
        do dark, flat correction thresholding, ...
        and read missing data from keyboard if needed
        """
        # GF: self.saturation ignored if none of the other options active...
        if len(self.dataFiles) > 1 or self.cutBackground or self.darkFiles or self.flatFiles:
            self.outfile = averageImages(self.dataFiles, self.outfile,
                                         threshold=self.saturation, minimum=self.cutBackground,
                                         darks=self.darkFiles, flats=self.flatFiles,
                                         filter_=self.filter)
        else:
            self.outfile = self.dataFiles[0]

        self.basename = os.path.splitext(self.outfile)[0]
        if isinstance(self, Recalibration):
            self.keep = False
            self.pointfile = None
        else:
            self.pointfile = self.basename + ".npt"
        if self.wavelength is None:
            self.wavelength = self.ai.wavelength

        self.peakPicker = PeakPicker(self.outfile, reconst=self.reconstruct, mask=self.mask,
                                     pointfile=self.pointfile, calibrant=self.calibrant,
                                     wavelength=self.ai.wavelength)
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

    def extract_cpt(self, method="massif"):
        """
        Performs an automatic keypoint extraction:
        Can be used in recalib or in calib after a first calibration has been performed
        """
        print("in extract_cpt with method %s" % method)
        assert self.ai
        assert self.calibrant
        assert self.peakPicker
        self.peakPicker.reset()
        self.peakPicker.init(method, False)
        if self.geoRef:
            self.ai.setPyFAI(**self.geoRef.getPyFAI())
        tth = numpy.array([ i for i in self.calibrant.get_2th() if i is not None])
        tth = numpy.unique(tth)
        dtth = numpy.zeros((tth.size, 2))
        delta = tth[1:] - tth[:-1]
        dtth[:-1, 0] = delta
        dtth[-1, 0] = delta[-1]
        dtth[1:, 1] = delta
        dtth[0, 1] = delta[0]
        dtth = dtth.min(axis= -1)
        if self.geoRef:
            ary = self.geoRef.get_ttha()
            if (ary is not None) and (ary.shape == self.peakPicker.data.shape):
                ttha = ary
            else:
                ttha = self.geoRef.twoThetaArray(self.peakPicker.data.shape)
        else:
            ttha = self.ai.twoThetaArray(self.peakPicker.data.shape)
        rings = 0
        self.peakPicker.sync_init()
        if self.max_rings is None:
            self.max_rings = tth.size
        for i in range(tth.size):
            if rings >= self.max_rings:
                break
            mask = abs(ttha - tth[i]) <= (dtth[i] / 4.0)
            if self.mask is not None:
                mask = numpy.logical_and(mask, numpy.logical_not(self.mask))
            size = mask.sum(dtype=int)
            if (size > 0):
                rings += 1
                self.peakPicker.massif_contour(mask)
                if self.gui:
                    update_fig(self.peakPicker.fig)
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
                keep = int(numpy.ceil(numpy.sqrt(size2)))

                logger.info("Extracting datapoint for ring %s (2theta = %.2f deg); "\
                            "searching for %i pts out of %i with I>%.1f" %
                            (i, numpy.degrees(tth[i]), keep, size2, upper_limit))

                res = self.peakPicker.peaks_from_area(mask2, Imin=upper_limit, keep=keep, method=method)
                self.peakPicker.points.append(res, tth[i], i)
                if self.gui:
                    # minIndex: skip redrawing of previous rings
                    self.peakPicker.display_points(minIndex=i)
                    update_fig(self.peakPicker.fig)

        self.peakPicker.points.save(self.basename + ".npt")
        if self.weighted:
            self.data = self.peakPicker.points.getWeightedList(self.peakPicker.data)
        else:
            self.data = self.peakPicker.points.getList()


    def refine(self):
        """
        Contains the common geometry refinement part
        """
        if os.name == "nt" and self.peakPicker is not None:
            logging.info(self.win_error)
            self.peakPicker.closeGUI()
        print("Before refinement, the geometry is:")
        print(self.geoRef)
        previous = sys.maxint
        finished = False
        fig2 = None
        while not finished:
            count = 0
            if "wavelength" in self.fixed:
#                print self.geoRef.calibrant
                while (previous > self.geoRef.chi2()) and (count < self.max_iter):
                    if (count == 0):
                        previous = sys.maxint
                    else:
                        previous = self.geoRef.chi2()
                    self.geoRef.refine2(1000000, fix=self.fixed)
                    print(self.geoRef)
                    count += 1
            else:
                while previous > self.geoRef.chi2_wavelength() and (count < self.max_iter):
                    if (count == 0):
                        previous = sys.maxint
                    else:
                        previous = self.geoRef.chi2()
                    self.geoRef.refine2_wavelength(1000000, fix=self.fixed)
                    print(self.geoRef)
                    count += 1
                self.peakPicker.points.setWavelength_change2th(self.geoRef.wavelength)
            self.geoRef.save(self.basename + ".poni")
            self.geoRef.del_ttha()
            self.geoRef.del_dssa()
            self.geoRef.del_chia()
            tth = self.geoRef.twoThetaArray(self.peakPicker.shape)
            dsa = self.geoRef.solidAngleArray(self.peakPicker.shape)
#            self.geoRef.chiArray(self.peakPicker.shape)
#            self.geoRef.cornerArray(self.peakPicker.shape)
            if os.name == "nt":
                logger.info(self.win_error)
            else:
                if self.gui:
                    self.peakPicker.contour(tth)
                    if self.interactive:
                        if fig2 is None:
                            fig2 = pylab.plt.figure()
                            sp = fig2.add_subplot(111)
                            im = sp.imshow(dsa, origin="lower")
                            cbar = fig2.colorbar(im)  # Add color bar
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
                previous = sys.maxint

    def prompt(self):
        """
        prompt for commands to guide the calibration process

        @return: True when the user is happy with what he has, False to request another refinement
        """

        while True:
            help = False
            print("Fixed: " + ", ".join(self.fixed))
            ans = raw_input("Modify parameters (or ? for help)?\t ").strip().lower()
            if "?" in ans:
                help = True
            if not ans:
                print("'done' to continue")
                continue
            words = ans.split()
            action = words[0]
            if action in [ "help", "?"]:
                help == True
            if help:
                for what in self.HELP.keys():
                    if action.startswith(what):
                        print("Help on %s" % what)
                        print(self.HELP[what])
                        break
                else:
                    print("Help on commands")
                    print(self.HELP["help"])
                    print("Valid actions: " + ", ".join(self.HELP.keys()))
                    print("Valid parameters: " + ", ".join(self.PARAMETERS))
            elif action == "get": #get wavelength
                if (len(words) == 2) and  words[1] in self.PARAMETERS:
                    param = words[1]
                    print("Value of parameter %s: %s %s" % (param, self.geoRef.__getattribute__(param), self.UNITS[param]))
                else:
                    print(self.HELP[action])

            elif action == "set": #set wavelength 1e-10
                if (len(words) == 3) and  words[1] in self.PARAMETERS:
                    param = words[1]
                    try:
                        value = float(words[2])
                    except:
                        logger.warning("invalid value")
                    else:
                        setattr(self.geoRef, param, value)
                else:
                    print(self.HELP[action])
            elif action == "fix": #fix wavelength
                if (len(words) == 2) and  (words[1] in self.PARAMETERS) and (words[1] not in self.fixed):
                    param = words[1]
                    print("Value of parameter %s: %s %s" % (param, self.geoRef.__getattribute__(param), self.UNITS[param]))
                    self.fixed.append(param)
                else:
                    print(self.HELP[action])
            elif action == "free": #free wavelength
                if (len(words) == 2) and  (words[1] in self.PARAMETERS) and (words[1] in self.fixed):
                    param = words[1]
                    print("Value of parameter %s: %s %s" % (param, self.geoRef.__getattribute__(param), self.UNITS[param]))
                    self.fixed.remove(param)
            elif action == "recalib":
                max_rings = None
#                 method = "blob"
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
                if len(words) == 3 and words[2] == "massif":
                    self.extract_cpt("massif")
                else:
                    self.extract_cpt("blob")
                self.geoRef.data = numpy.array(self.data, dtype=numpy.float64)
                return False
            elif action == "bound": #bound dist
                if len(words) >= 2 and  words[1] in self.PARAMETERS:
                    param = words[1]
                    if len(words) == 2:
                        readFloatFromKeyboard("Enter %s in %s " % (param, self.UNITS[param]) +
                             "(or %s_min[%.3f] %s[%.3f] %s_max[%.3f]):\t " % (
                              param, self.geoRef.__getattribute__("get_%s_min" % param)(),
                              param, self.geoRef.__getattribute__("get_%s" % param)(),
                              param, self.geoRef.__getattribute__("get_%s_max" % param)()),
                             {1:[self.geoRef.__getattribute__("set_%s" % param)],
                              2:[self.geoRef.__getattribute__("set_%s_min" % param),
                                 self.geoRef.__getattribute__("set_%s_max" % param)],
                              3:[self.geoRef.__getattribute__("set_%s_min" % param),
                                 self.geoRef.__getattribute__("set_%s" % param),
                                 self.geoRef.__getattribute__("set_%s_max" % param)]})
                    elif len(words) == 3:
                        try:
                            value = float(words[2])
                        except:
                            logger.warning("invalid value")
                        else:
                            self.geoRef.__getattribute__("set_%s" % param)(value)
                    elif len(words) == 4:
                        try:
                            value_min = float(words[2])
                            value_max = float(words[3])
                        except:
                            logger.warning("invalid value")
                        else:
                            self.geoRef.__getattribute__("set_%s_min" % param)(value_min)
                            self.geoRef.__getattribute__("set_%s_max" % param)(value_max)
                    elif len(words) == 5:
                        try:
                            value_min = float(words[2])
                            value = float(words[3])
                            value_max = float(words[4])
                        except:
                            logger.warning("invalid value")
                        else:
                            self.geoRef.__getattribute__("set_%s_min" % param)(value_min)
                            self.geoRef.__getattribute__("set_%s" % param)(value)
                            self.geoRef.__getattribute__("set_%s_max" % param)(value_max)
                    else:
                        print(self.HELP[action])
                else:
                    print(self.HELP[action])
            elif action == "bounds":
                readFloatFromKeyboard("Enter Distance in meter "
                             "(or dist_min[%.3f] dist[%.3f] dist_max[%.3f]):\t " %
                             (self.geoRef.dist_min, self.geoRef.dist, self.geoRef.dist_max),
                             {1:[self.geoRef.set_dist], 2:[ self.geoRef.set_dist_min, self.geoRef.set_dist_max],
                              3:[ self.geoRef.set_dist_min, self.geoRef.set_dist, self.geoRef.set_dist_max]})
                readFloatFromKeyboard("Enter Poni1 in meter "
                              "(or poni1_min[%.3f] poni1[%.3f] poni1_max[%.3f]):\t " %
                              (self.geoRef.poni1_min, self.geoRef.poni1, self.geoRef.poni1_max),
                               {1:[self.geoRef.set_poni1], 2:[ self.geoRef.set_poni1_min, self.geoRef.set_poni1_max],
                                3:[ self.geoRef.set_poni1_min, self.geoRef.set_poni1, self.geoRef.set_poni1_max]})
                readFloatFromKeyboard("Enter Poni2 in meter "
                              "(or poni2_min[%.3f] poni2[%.3f] poni2_max[%.3f]):\t " %
                              (self.geoRef.poni2_min, self.geoRef.poni2, self.geoRef.poni2_max),
                              {1:[self.geoRef.set_poni2], 2:[ self.geoRef.set_poni2_min, self.geoRef.set_poni2_max],
                               3:[ self.geoRef.set_poni2_min, self.geoRef.set_poni2, self.geoRef.set_poni2_max]})
                readFloatFromKeyboard("Enter Rot1 in rad "
                              "(or rot1_min[%.3f] rot1[%.3f] rot1_max[%.3f]):\t " %
                              (self.geoRef.rot1_min, self.geoRef.rot1, self.geoRef.rot1_max),
                              {1:[self.geoRef.set_rot1], 2:[ self.geoRef.set_rot1_min, self.geoRef.set_rot1_max],
                               3:[ self.geoRef.set_rot1_min, self.geoRef.set_rot1, self.geoRef.set_rot1_max]})
                readFloatFromKeyboard("Enter Rot2 in rad "
                              "(or rot2_min[%.3f] rot2[%.3f] rot2_max[%.3f]):\t " %
                              (self.geoRef.rot2_min, self.geoRef.rot2, self.geoRef.rot2_max),
                              {1:[self.geoRef.set_rot2], 2:[ self.geoRef.set_rot2_min, self.geoRef.set_rot2_max],
                               3:[ self.geoRef.set_rot2_min, self.geoRef.set_rot2, self.geoRef.set_rot2_max]})
                readFloatFromKeyboard("Enter Rot3 in rad "
                              "(or rot3_min[%.3f] rot3[%.3f] rot3_max[%.3f]):\t " %
                              (self.geoRef.rot3_min, self.geoRef.rot3, self.geoRef.rot3_max),
                              {1:[self.geoRef.set_rot3], 2:[ self.geoRef.set_rot3_min, self.geoRef.set_rot3_max],
                               3:[ self.geoRef.set_rot3_min, self.geoRef.set_rot3, self.geoRef.set_rot3_max]})
            elif action == "done":
                return True
            elif action == "quit":
                return True
            elif action == "refine":
                return False
            elif action == "fit":
                return False
            elif action == "validate":
                self.validate_calibration()
            elif action == "integrate":
                self.postProcess()
            elif action == "abort":
                sys.exit()
            elif action == "show":
                print("The current parameter set is:")
                print(self.geoRef)
            elif action == "reset":
                self.ai.dist = 0.1
                self.ai.poni1 = self.detector.pixel1 * (self.peakPicker.shape[0] / 2.)
                self.ai.poni2 = self.detector.pixel2 * (self.peakPicker.shape[1] / 2.)
                self.ai.rot1 = 0.0
                self.ai.rot2 = 0.0
                self.ai.rot3 = 0.0

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


            else:
                logger.warning("Unrecognized action: %s, type 'quit' to leave " % action)

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
        t0 = time.time()
        tth = self.geoRef.twoThetaArray(self.peakPicker.shape)
        t1 = time.time()
        dsa = self.geoRef.solidAngleArray(self.peakPicker.shape)
        t2 = time.time()
        self.geoRef.chiArray(self.peakPicker.shape)
        t2a = time.time()
        self.geoRef.cornerArray(self.peakPicker.shape)
        t2b = time.time()
        if self.gui:
            if self.fig3 is None:
                self.fig3 = pylab.plt.figure()
            else:
                self.fig3.clf()
            self.ax_xrpd_1d = self.fig3.add_subplot(1, 2, 1)
            self.ax_xrpd_2d = self.fig3.add_subplot(1, 2, 2)
        t3 = time.time()
        a, b = self.geoRef.integrate1d(self.peakPicker.data, self.nPt_1D,
                                filename=self.basename + ".xy", unit=self.unit,
                                polarization_factor=self.polarization_factor,
                                method="splitbbox")
        t4 = time.time()
        img, pos_rad, pos_azim = self.geoRef.integrate2d(self.peakPicker.data, self.nPt_2D_rad, self.nPt_2D_azim,
                                filename=self.basename + ".azim", unit=self.unit,
                                polarization_factor=self.polarization_factor,
                                method="splitbbox")
        t5 = time.time()
        logger.info(os.linesep.join(["Timings:",
                                " * two theta array generation %.3fs" % (t1 - t0),
                                " * diff Solid Angle           %.3fs" % (t2 - t1),
                                " * chi array generation       %.3fs" % (t2a - t2),
                                " * corner coordinate array    %.3fs" % (t2b - t2a),
                                " * 1D Azimuthal integration   %.3fs" % (t4 - t3),
                                " * 2D Azimuthal integration   %.3fs" % (t5 - t4)]))
        if self.gui:
            self.ax_xrpd_1d.plot(a, b)
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
                dBeamCentre = self.geoRef.getFit2D()["directDist"] # in mm!!
                xValues = dBeamCentre * numpy.tan(twoTheta)
            else:
                logger.warning('Unknown unit %s, do not plot calibration rings' % str(self.unit))
            if xValues is not None:
                for x in xValues:
                    line = matplotlib.lines.Line2D([x, x], self.ax_xrpd_1d.axis()[2:4],
                                                   color='red', linestyle='--')
                    self.ax_xrpd_1d.add_line(line)
            self.ax_xrpd_1d.set_title("1D integration")
            self.ax_xrpd_1d.set_xlabel(self.unit)
            self.ax_xrpd_1d.set_ylabel("Intensity")
            self.ax_xrpd_2d.imshow(numpy.log(img - img.min() + 1e-3), origin="lower",
                         extent=[pos_rad.min(), pos_rad.max(), pos_azim.min(), pos_azim.max()],
                         aspect="auto")
            self.ax_xrpd_2d.set_title("2D regrouping")
            self.ax_xrpd_2d.set_xlabel(self.unit)
            self.ax_xrpd_2d.set_ylabel("Azimuthal angle (deg)")
            if not gui_utils.main_loop:
                self.fig3.show()
            update_fig(self.fig3)

    def validate_calibration(self):
        """
        Validate the calivration and calculate the offset in the diffraction image
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

################################################################################
# Calibration
################################################################################

    def set_data(self, data):
        """
        call-back function for the peak-picker
        """
        self.data = data
        if not self.weighted:
            self.data = numpy.array(self.data)[:, :-1]
        self.refine()

class Calibration(AbstractCalibration):
    """
    class doing the calibration of frames
    """
    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None,
                 splineFile=None, detector=None, gaussianWidth=None,
                 wavelength=None, calibrant=None):
        """
        Constructor for calibration:

        @param dataFiles: list of filenames containing data images
        @param darkFiles: list of filenames containing dark current images
        @param flatFiles: list of filenames containing flat images
        @param pixelSize: size of the pixel in meter as 2 tuple
        @param splineFile: file containing the distortion of the taper
        @param detector: Detector name or instance
        @param wavelength: radiation wavelength in meter
        @param calibrant: pyFAI.calibrant.Calibrant instance

        """
        AbstractCalibration.__init__(self, dataFiles=dataFiles,
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
        return AbstractCalibration.__repr__(self) + \
            "%sgaussian= %s" % (os.linesep, self.gaussianWidth)

    def parse(self):
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
The --calibrant option is mandatory !""" % str(ALL_CALIBRANTS)

        epilog = """The output of this program is a "PONI" file containing the detector description
and the 6 refined parameters (distance, center, rotation) and wavelength.
An 1D and 2D diffraction patterns are also produced. (.dat and .azim files)
        """
        usage = "pyFAI-calib [options] -w 1 -D detector -c calibrant.D imagefile.edf"
        self.configure_parser(usage=usage, description=description, epilog=epilog)  # common
        self.parser.add_argument("-r", "--reconstruct", dest="reconstruct",
              help="Reconstruct image where data are masked or <0  (for Pilatus "\
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


        (options, _) = self.analyse_options()
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
        AbstractCalibration.preprocess(self)

        if self.gaussianWidth is not None:
            self.peakPicker.massif.setValleySize(self.gaussianWidth)
        else:
            self.peakPicker.massif.initValleySize()
        if self.gui:
            self.peakPicker.gui(log=True, maximize=True, pick=True)
            update_fig(self.peakPicker.fig)

    def gui_peakPicker(self):
        if self.peakPicker is None:
            self.preprocess()
#        self.peakPicker.gui(True)
        if os.path.isfile(self.pointfile):
            self.peakPicker.load(self.pointfile)
        if self.gui:
            update_fig(self.peakPicker.fig)
#        self.peakPicker.finish(self.pointfile, callback=self.set_data)
        self.set_data(self.peakPicker.finish(self.pointfile))
#        raw_input("Please press enter when you are happy with your selection" + os.linesep)
#        while self.data is None:
#            update_fig(self.peakPicker.fig)
#            time.sleep(0.1)

    def refine(self):
        """
        Contains the geometry refinement part specific to Calibration
        """
        self.geoRef = GeometryRefinement(self.data, dist=0.1, detector=self.detector,
                                         wavelength=self.wavelength,
                                         calibrant=self.calibrant)
#        print self.calibrant
        paramfile = self.basename + ".poni"
        if os.path.isfile(paramfile):
            self.geoRef.load(paramfile)
            if self.wavelength:
                try:
                    old_wl = self.geoRef.wavelength
                except:
                    pass
                else:
                    logger.warning("Overwriting wavelength from PONI file (%s) with the one from command line (%s)" % (old_wl, self.wavelength))
                self.geoRef.wavelength = self.wavelength
            if self.detector:
                gr_det = str(self.geoRef.detector)
                nw_det = str(self.detector)
                if gr_det != nw_det:
                    logger.warning("Overwriting detector from PONI file: %s%s with the one from command line %s%s" % (os.linesep, gr_det, os.linesep, nw_det))
                    self.geoRef.detector = self.detector

        AbstractCalibration.refine(self)


################################################################################
# Recalibration
################################################################################

class Recalibration(AbstractCalibration):
    """
    class doing the re-calibration of frames
    """
    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None,
                 splineFile=None, detector=None, wavelength=None, calibrant=None):
        """
        Constructor for Recalibration:

        @param dataFiles: list of filenames containing data images
        @param darkFiles: list of filenames containing dark current images
        @param flatFiles: list of filenames containing flat images
        @param pixelSize: size of the pixel in meter as 2 tuple
        @param splineFile: file containing the distortion of the taper
        @param detector: Detector name or instance
        @param wavelength: radiation wavelength in meter
        @param calibrant: pyFAI.calibrant.Calibrant instance
        """
        AbstractCalibration.__init__(self, dataFiles=dataFiles,
                                     darkFiles=darkFiles,
                                     flatFiles=flatFiles,
                                     pixelSize=pixelSize,
                                     splineFile=splineFile,
                                     detector=detector,
                                     wavelength=wavelength,
                                     calibrant=calibrant)

    def parse(self):
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
""" % str(ALL_CALIBRANTS)

        epilog = """The main difference with pyFAI-calib is the way control-point hence Debye-Sherrer
rings are extracted. While pyFAI-calib relies on the contiguity of a region of peaks
called massif; pyFAI-recalib knows approximatly the geometry and is able to select
the region where the ring should be. From this region it selects automatically
the various peaks; making pyFAI-recalib able to run without graphical interface and
without human intervention (--no-gui and --no-interactive options).


Note that `pyFAI-recalib` program is obsolete as the same functionnality is 
available from within pyFAI-calib, using the `recalib` command in the 
refinement process.  
Two option are available for recalib: the numbe of rings to extract (similar to the -r option of this program) 
and a new option which lets you choose between the original `massif` algorithm and the new `blob` detection.
        """
        usage = "pyFAI-recalib [options] -p ponifile -w 1 -c calibrant.D imagefile.edf"
        self.configure_parser(usage=usage, description=description, epilog=epilog)

        self.parser.add_argument("-r", "--ring", dest="max_rings", type=int,
                      help="maximum number of rings to extract. Default: all accessible", default=None)
        self.parser.add_argument("-p", "--poni", dest="poni", metavar="FILE",
                      help="file containing the diffraction parameter (poni-file). MANDATORY",
                      default=None)
        self.parser.add_argument("-k", "--keep", dest="keep",
                      help="Keep existing control point and append new",
                      default=False, action="store_true")

        options, args = self.parser.parse_args()
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
        AbstractCalibration.read_dSpacingFile(self, verbose=False)


    def preprocess(self):
        """
        do dark, flat correction thresholding, ...
        """
        AbstractCalibration.preprocess(self)

        if self.gui:
            self.peakPicker.gui(log=True, maximize=True, pick=False)
            update_fig(self.peakPicker.fig)



    def refine(self):
        """
        Contains the geometry refinement part specific to Recalibration
        """

        self.geoRef = GeometryRefinement(self.data, dist=self.ai.dist, poni1=self.ai.poni1,
                                         poni2=self.ai.poni2, rot1=self.ai.rot1,
                                         rot2=self.ai.rot2, rot3=self.ai.rot3,
                                         detector=self.ai.detector, calibrant=self.calibrant,
                                         wavelength=self.wavelength)
        self.ai = self.geoRef
        self.geoRef.set_tolerance(10)
        AbstractCalibration.refine(self)


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
        self.centerX = None
        self.centerY = None
        self.distance = None
        self.fixed = []
        self.max_rings = None


    def __repr__(self):
        lst = ["Multi-Calibration object:",
             "data= " + ", ".join(self.dataFiles),
             "dark= " + ", ".join(self.darkFiles),
             "flat= " + ", ".join(self.flatFiles)]
        lst.append(self.detector.__repr__())
#        lst.append("gaussian= %s" % self.gaussianWidth)
        return os.linesep.join(lst)

    def parse(self):
        """
        parse options from command line
        """
        usage = "MX-Calibrate -w 1.54 -c CeO2 file1.cbf file2.cbf ..."
        version = "MX-Calibrate from pyFAI version %s: %s" % (PyFAI_VERSION, PyFAI_DATE)
        description = """
        Calibrate automatically a set of frames taken at various sample-detector distance.
        Return the linear regression of the fit in funtion of the sample-setector distance.
        """
        epilog = """This tool has been developed for ESRF MX-beamlines where an acceptable calibration is
        usually present is the header of the image. PyFAI reads it and does a "recalib" on
        each of them before exporting a linear regression of all parameters versus this distance.
        """
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
#Size of the gap (in pixels) between two consecutive rings, by default 100
#Increase the value if the arc is not complete;
#decrease the value if arcs are mixed together.""", default=None)
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
        parser.add_argument("--no-tilt", dest="tilt",
                      help="refine the detector tilt", default=True , action="store_false")
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
                      help="free the rot2 parameter", default=None, action="store_false")

        parser.add_argument("--fix-rot3", dest="fix_rot3",
                      help="fix the rot3 parameter", default=None, action="store_true")
        parser.add_argument("--free-rot3", dest="fix_rot3",
                      help="free the rot3 parameter", default=None, action="store_false")

        parser.add_argument("--fix-wavelength", dest="fix_wavelength",
                      help="fix the wavelength parameter", default=True, action="store_true")
        parser.add_argument("--free-wavelength", dest="fix_wavelength",
                      help="free the wavelength parameter", default=True, action="store_false")


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
                      help="Uses the 'massif' or the 'blob' peak-picker algorithm (default: blob)",
                      default="blob", type=str)
        options = parser.parse_args()

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
            self.mask = fabio.open(options.mask).data

        if options.detector_name:
            self.detector = get_detector(options.detector_name, options.args)
        if options.spline:
            if os.path.isfile(options.spline):
                self.detector.splineFile = os.path.abspath(options.spline)
            else:
                logger.error("Unknown spline file %s" % (options.spline))
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
        self.fixed = []
        if not options.tilt:
            self.fixed += ["rot1", "rot2", "rot3"]
        if options.fix_dist:
            self.fixed.append("dist")
        if options.fix_poni1:
            self.fixed.append("poni1")
        if options.fix_poni2:
            self.fixed.append("poni2")
        if options.fix_rot1:
            self.fixed.append("rot1")
        if options.fix_rot2:
            self.fixed.append("rot2")
        if options.fix_rot3:
            self.fixed.append("rot3")
        if options.fix_wavelength:
            self.fixed.append("wavelength")

        self.dataFiles = [f for f in options.args if os.path.isfile(f)]
        if not self.dataFiles:
            raise RuntimeError("Please provide some calibration images ... "
                               "if you want to analyze them. Try also the --help option to see all options!")
        self.weighted = options.weighted
        if options.peakPicker.lower() in ["blob", "massif"]:
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
            ans = raw_input("Please enter the pixel size (in micron, comma separated X, Y "
                            "i.e. %.2e,%.2e) or a spline file: " % tuple(pixelSize)).strip()
            if os.path.isfile(ans):
                self.detector.splineFile = ans
            else:
                self.get_pixelSize(ans)


    def read_dSpacingFile(self):
        """Read the name of the calibrant or the file with d-spacing"""
        if self.calibrant in ALL_CALIBRANTS:
            self.calibrant = ALL_CALIBRANTS[self.calibrant]
        elif os.path.isfile(self.calibrant):
            self.calibrant = Calibrant(filename=self.calibrant)
        else:
            comments = ["MX-calibrate has changed !!!",
                        "Instead of entering the 2theta value, which was tedious,"
                        "the program takes a calibrant as in input "
                        "(either a reference one like Ceo2, either a "
                        "d-spacing file with inter planar distance in Angstrom)",
                        "and an associated wavelength", ""
                        "You will be asked to enter the ring number, "
                        "which is usually a simpler than the 2theta value."]
            print(os.linesep.join(comments))
            ans = ""
            while not self.calibrant:
                ans = raw_input("Please enter the name of the calibrant"
                                " or the file containing the d-spacing:\t").strip()
                if ans in ALL_CALIBRANTS:
                    self.calibrant = ALL_CALIBRANTS[ans]
                elif os.path.isfile(ans):
                    self.calibrant = Calibrant(filename=ans)


    def read_wavelength(self):
        """Read the wavelength"""
        while not self.wavelength:
            ans = raw_input("Please enter wavelength in Angstrom:\t").strip()
            try:
                self.wavelength = 1e-10 * float(ans)
            except:
                self.wavelength = None

    def process(self):
        """

        """
        self.dataFiles.sort()
        for fn in self.dataFiles:
            fabimg = fabio.open(fn)
            wavelength = self.wavelength
            dist = self.distance
            centerX = self.centerX
            centerY = self.centerY
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
            self.results[fn] = {"wavelength":wavelength, "dist":dist}
            rec = Recalibration(dataFiles=[fn], darkFiles=self.darkFiles, flatFiles=self.flatFiles,
                                                  detector=self.detector, calibrant=self.calibrant, wavelength=wavelength)
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
        print self.results
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
        for key, dico in  self.results.iteritems():
            print key, dico["dist"]
            print dico["ai"]
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
            slope, intercept, r, two, stderr = linregress(x, elt)

            print("%s = %s * dist_mm + %s \t R= %s\t stderr= %s" % (name, slope, intercept, r, stderr))

class CheckCalib(object):
    def __init__(self, poni=None, img=None, unit="2th_deg"):
        self.ponifile = poni
        if poni :
            self.ai = AzimuthalIntegrator.sload(poni)
        else:
            self.ai = None
        if img:
            self.img = fabio.open(img)
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
        if self.ai:
            return self.ai.__repr__()

    def parse(self):
        logger.debug("in parse")
        usage = "usage: check_calib [options] -p param.poni image.edf"
        description = """Check_calib is a research tool aiming at validating both the geometric
calibration and everything else like flat-field correction, distortion
correction, at a sub-pixel level.

Note that `check_calib` program is obsolete as the same functionnality is 
available from within pyFAI-calib, using the `validate` command in the 
refinement process.  
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

        options = parser.parse_args()
        if options.verbose:
            logger.setLevel(logging.DEBUG)

        if options.mask is not None:
            self.mask = (fabio.open(options.mask).data != 0)
        args = expand_args(options.args)
        if len(args) > 0:
            f = args[0]
            if os.path.isfile(f):
                self.img = fabio.open(f).data.astype(numpy.float32)
            else:
                print("Please enter diffraction images as arguments")
                sys.exit(1)
            for f in args[1:]:
                self.img += fabio.open(f).data
        if options.dark and os.path.exists(options.dark):
            self.img -= fabio.open(options.dark).data
        if options.flat and os.path.exists(options.flat):
            self.img /= fabio.open(options.flat).data
        if options.poni:
            self.ai = AzimuthalIntegrator.sload(options.poni)
        self.data = [f for f in args if os.path.isfile(f)]
        if options.poni is None:
            logger.error("PONI parameter is mandatory")
            sys.exit(1)
        self.ai = AzimuthalIntegrator.sload(options.poni)
        if options.wavelength:
            self.ai.wavelength = 1e-10 * options.wavelength
        elif options.energy:
            self.ai.wavelength = 1e-10 * hc / options.energy
#        else:
#            self.read_wavelength()


    def get_1dsize(self):
        logger.debug("in get_1dsize")
        return int(numpy.sqrt(self.img.shape[0] ** 2 + self.img.shape[1] ** 2))
    size1d = property(get_1dsize)

    def integrate(self):
        logger.debug("in integrate")
        self.r, self.I = self.ai.integrate1d(self.img, self.size1d, mask=self.mask,
                                             unit=self.unit, method="splitpixel")

    def rebuild(self):
        """
        Rebuild the diffraction image and measures the offset with the reference
        @return: offset
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
#        print os.linesep.join(log)
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
            smooth_mask = 1.0 - scipy.ndimage.filters.gaussian_filter(big_mask.astype(numpy.float32), sigma)
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




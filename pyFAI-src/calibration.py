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
__date__ = "15/07/2013"
__status__ = "development"

import os, sys, time, logging, types
from optparse import OptionParser
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyFAI.calibration")
import numpy, scipy.ndimage
import fabio
import matplotlib
import pylab
from .detectors import detector_factory, Detector
from .geometryRefinement import GeometryRefinement
from .peakPicker import PeakPicker
from . import units
from .utils import averageImages, measure_offset, expand_args
from .azimuthalIntegrator import AzimuthalIntegrator
from .units import hc
from . import version as PyFAI_VERSION

matplotlib.interactive(True)

class AbstractCalibration(object):

    """
    Everything that is commun to Calibration and Recalibration
    """

    win_error = "We are under windows, matplotlib is not able to"\
                         " display too many images without crashing, this"\
                         " is why the window showing the diffraction image"\
                         " is closed"
    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None,
                 splineFile=None, detector=None, spacing_file=None, wavelength=None):
        """
        Constructor:

        @param dataFiles: list of filenames containing data images
        @param darkFiles: list of filenames containing dark current images
        @param flatFiles: list of filenames containing flat images
        @param pixelSize: size of the pixel in meter as 2 tuple
        @param splineFile: file containing the distortion of the taper
        @param detector: Detector name or instance
        @param spacing_file: file containing the spacing of Miller plans (in decreasing order, in Angstrom, space separated)
        @param wavelength: radiation wavelength in meter
        """
        self.dataFiles = dataFiles
        self.darkFiles = darkFiles
        self.flatFiles = flatFiles
        self.pointfile = None

        self.detector = self.get_detector(detector)

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
        self.spacing_file = spacing_file
        self.mask = None
        self.saturation = 0.1
        self.fixed = ["wavelength"]  # parameter fixed during optimization
        self.max_rings = None
        self.max_iter = 1000
        self.gui = True
        self.interactive = True
        self.filter = "mean"
        self.basename = None
        self.saturation = 0.1
        self.weighted = False
        self.polarization_factor = None
        self.parser = None
        self.nPt_1D = 1024
        self.nPt_2D_azim = 360
        self.nPt_2D_rad = 400
        self.units = None
        self.keep = True

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

    def get_detector(self, detector):
        if type(detector) in types.StringTypes:
            try:
                return detector_factory(detector)
            except RuntimeError:
                sys.exit(-1)
        elif isinstance(detector, Detector):
            return detector
        else:
            return Detector()

    def configure_parser(self, version="%prog from pyFAI version " + PyFAI_VERSION,
                         usage="%prog [options] inputfile.edf",
                         description=None, epilog=None):
        """Common configuration for parsers
        """
        self.parser = OptionParser(usage=usage, version=version,
                              description=description, epilog=epilog)
        self.parser.add_option("-o", "--out", dest="outfile",
                          help="Filename where processed image is saved", metavar="FILE",
                          default="merged.edf")
        self.parser.add_option("-v", "--verbose",
                          action="store_true", dest="debug", default=False,
                          help="switch to debug/verbose mode")
        self.parser.add_option("-S", "--spacing", dest="spacing", metavar="FILE",
                      help="file containing d-spacing of the reference sample (MANDATORY)",
                      default=None)
        self.parser.add_option("-w", "--wavelength", dest="wavelength", type="float",
                      help="wavelength of the X-Ray beam in Angstrom", default=None)
        self.parser.add_option("-e", "--energy", dest="energy", type="float",
                      help="energy of the X-Ray beam in keV (hc=%skeV.A)" % hc, default=None)
        self.parser.add_option("-P", "--polarization", dest="polarization_factor",
                      type="float", default=None,
                      help="polarization factor, from -1 (vertical) to +1 (horizontal),"\
                      " default is None (no correction), synchrotrons are around 0.95")
        self.parser.add_option("-b", "--background", dest="background",
                      help="Automatic background subtraction if no value are provided",
                      default=None)
        self.parser.add_option("-d", "--dark", dest="dark",
                      help="list of dark images to average and subtract", default=None)
        self.parser.add_option("-f", "--flat", dest="flat",
                      help="list of flat images to average and divide", default=None)
        self.parser.add_option("-s", "--spline", dest="spline",
                      help="spline file describing the detector distortion", default=None)
        self.parser.add_option("-D", "--detector", dest="detector_name",
                      help="Detector name (instead of pixel size+spline)", default=None)
        self.parser.add_option("-m", "--mask", dest="mask",
                      help="file containing the mask (for image reconstruction)", default=None)
        self.parser.add_option("-n", "--pt", dest="npt",
                      help="file with datapoints saved. Default: basename.npt", default=None)
        self.parser.add_option("--filter", dest="filter",
                      help="select the filter, either mean(default), max or median",
                       default="mean")
        self.parser.add_option("-l", "--distance", dest="distance", type="float",
                      help="sample-detector distance in millimeter", default=None)
        self.parser.add_option("--poni1", dest="poni1", type="float",
                      help="poni1 coordinate in meter", default=None)
        self.parser.add_option("--poni2", dest="poni2", type="float",
                      help="poni2 coordinate in meter", default=None)
        self.parser.add_option("--rot1", dest="rot1", type="float",
                      help="rot1 in radians", default=None)
        self.parser.add_option("--rot2", dest="rot2", type="float",
                      help="rot2 in radians", default=None)
        self.parser.add_option("--rot3", dest="rot3", type="float",
                      help="rot3 in radians", default=None)

        self.parser.add_option("--fix-dist", dest="fix_dist",
                      help="fix the distance parameter", default=None, action="store_true")
        self.parser.add_option("--free-dist", dest="fix_dist",
                      help="free the distance parameter", default=None, action="store_false")

        self.parser.add_option("--fix-poni1", dest="fix_poni1",
                      help="fix the poni1 parameter", default=None, action="store_true")
        self.parser.add_option("--free-poni1", dest="fix_poni1",
                      help="free the poni1 parameter", default=None, action="store_false")

        self.parser.add_option("--fix-poni2", dest="fix_poni2",
                      help="fix the poni2 parameter", default=None, action="store_true")
        self.parser.add_option("--free-poni2", dest="fix_poni2",
                      help="free the poni2 parameter", default=None, action="store_false")

        self.parser.add_option("--fix-rot1", dest="fix_rot1",
                      help="fix the rot1 parameter", default=None, action="store_true")
        self.parser.add_option("--free-rot1", dest="fix_rot1",
                      help="free the rot1 parameter", default=None, action="store_false")

        self.parser.add_option("--fix-rot2", dest="fix_rot2",
                      help="fix the rot2 parameter", default=None, action="store_true")
        self.parser.add_option("--free-rot2", dest="fix_rot2",
                      help="free the rot2 parameter", default=None, action="store_false")

        self.parser.add_option("--fix-rot3", dest="fix_rot3",
                      help="fix the rot3 parameter", default=None, action="store_true")
        self.parser.add_option("--free-rot3", dest="fix_rot3",
                      help="free the rot3 parameter", default=None, action="store_false")

        self.parser.add_option("--fix-wavelength", dest="fix_wavelength",
                      help="fix the wavelength parameter", default=True, action="store_true")
        self.parser.add_option("--free-wavelength", dest="fix_wavelength",
                      help="free the wavelength parameter", default=True, action="store_false")

        self.parser.add_option("--saturation", dest="saturation",
                      help="consider all pixel>max*(1-saturation) as saturated and "\
                      "reconstruct them",
                      default=0.1, type="float")
        self.parser.add_option("--weighted", dest="weighted",
                      help="weight fit by intensity, by default not.",
                       default=False, action="store_true")
        self.parser.add_option("--npt", dest="nPt_1D",
                      help="Number of point in 1D integrated pattern, Default: 1024", type="int",
                      default=1024)
        self.parser.add_option("--npt-azim", dest="nPt_2D_azim",
                      help="Number of azimuthal sectors in 2D integrated images. Default: 360", type="int",
                      default=360)
        self.parser.add_option("--npt-rad", dest="nPt_2D_rad",
                      help="Number of radial bins in 2D integrated images. Default: 400", type="int",
                      default=400)
        self.parser.add_option("--unit", dest="unit",
                      help="Valid units for radial range: 2th_deg, 2th_rad, q_nm^-1,"\
                      " q_A^-1, r_mm. Default: 2th_deg", type="str", default="2th_deg")
        self.parser.add_option("--no-gui", dest="gui",
                      help="force the program to run without a Graphical interface",
                      default=True, action="store_false")
        self.parser.add_option("--no-interactive", dest="interactive",
                      help="force the program to run and exit without prompting"\
                      " for refinements", default=True, action="store_false")


    def analyse_options(self, options=None, args=None):
        """
        Analyse options and arguments

        @return: option,arguments
        """
        if (options is None) and  (args is None):
            options, args = self.parser.parse_args()
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
            self.detector = self.get_detector(options.detector_name)
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
        if (not options.spacing) or (not os.path.isfile(options.spacing)):
            logger.error("No such d-Spacing file: %s" % options.spacing)
            self.spacing_file = None
        else:
            self.spacing_file = options.spacing
        if self.spacing_file is None:
            self.read_dSpacingFile(True)
        else:
            self.spacing_file = os.path.abspath(self.spacing_file)
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
        """Read the name of the file with d-spacing"""
        if (self.spacing_file is None):
            comments = ["pyFAI calib has changed !!!",
                        "Instead of entering the 2theta value, which was tedious,"
                        "the program takes a d-spacing file in input "
                        "(just a serie of number representing the inter-planar "
                        "distance in Angstrom)",
                        "and an associated wavelength",
                        "You will be asked to enter the ring number,"
                        " which is usually a simpler than the 2theta value."]
            if verbose:
                print(os.linesep.join(comments))
            ans = ""
            while not os.path.isfile(ans):
                ans = raw_input("Please enter the name of the file"
                                " containing the d-spacing:\t").strip()
            self.spacing_file = os.path.abspath(ans)

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
        if len(self.dataFiles) > 1 or self.cutBackground or self.darkFiles or self.flatFiles:
            self.outfile = averageImages(self.dataFiles, self.outfile,
                                         threshold=self.saturation, minimum=self.cutBackground,
                                         darks=self.darkFiles, flats=self.flatFiles,
                                         filter_=self.filter)
        else:
            self.outfile = self.dataFiles[0]

        self.basename = os.path.splitext(self.outfile)[0]
        self.pointfile = self.basename + ".npt"
#        self.peakPicker.points.wavelength
        if self.wavelength is None:
            self.wavelength = self.ai.wavelength
        self.peakPicker = PeakPicker(self.outfile, reconst=self.reconstruct, mask=self.mask,
                                     pointfile=self.pointfile, dSpacing=self.spacing_file,
                                     wavelength=self.ai.wavelength)
        if not self.keep:
            self.peakPicker.points.reset()
            self.peakPicker.points.wavelength = self.ai.wavelength
        if not self.peakPicker.points.dSpacing:
            self.read_dSpacingFile()
            self.peakPicker.points.load_dSpacing(self.spacing_file)
        if not self.peakPicker.points.wavelength:
            self.read_wavelength()
            self.peakPicker.points.wavelength = self.wavelength

        if self.gui:
            self.peakPicker.gui(log=True, maximize=True)
            self.peakPicker.fig.canvas.draw()


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
                while (previous > self.geoRef.chi2()) and (count < self.max_iter):
                    previous = self.geoRef.chi2()
                    self.geoRef.refine2(1000000, fix=self.fixed)
                    print(self.geoRef)
                    count += 1
            else:
                while (previous > self.geoRef.chi2_wavelength()) and (count < self.max_iter):
                    previous = self.geoRef.chi2_wavelength()
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
            self.geoRef.chiArray(self.peakPicker.shape)
            self.geoRef.cornerArray(self.peakPicker.shape)
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
            if not self.interactive:
                break
            print("Fixed: " + ", ".join(self.fixed))
            change = raw_input("Modify parameters ?\t ").strip().lower()
            if (change == '') or (change[0] == "n"):
                finished = True
            elif change.startswith("help"):
                print("Type simple sentences like set wavelength 1e-10")
                print("The valid actions are: fix, set and free")
                print("The valid variables are dist, poni1, poni2, "
                      "rot1, rot2, rot3 and wavelength")
            elif change.startswith("free"):
                what = change.split()[-1]
                if what in self.fixed:
                    self.fixed.remove(what)
            elif change.startswith("fix"):
                what = change.split()[-1]
                if what not in self.fixed:
                    self.fixed.append(what)
            elif change.startswith("set"):
                words = change.split()
                if len(words) == 3:
                    what = words[1]
                    val = words[2]
                    if what in dir(self.geoRef):
                        setattr(self.geoRef, what, val)
                else:
                    print("example: set wavelength 1e-10")
            else:
                self.peakPicker.readFloatFromKeyboard("Enter Distance in meter "
                             "(or dist_min[%.3f] dist[%.3f] dist_max[%.3f]):\t " %
                             (self.geoRef.dist_min, self.geoRef.dist, self.geoRef.dist_max),
                             {1:[self.geoRef.set_dist],
                              3:[ self.geoRef.set_dist_min, self.geoRef.set_dist, self.geoRef.set_dist_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Poni1 in meter "
                              "(or poni1_min[%.3f] poni1[%.3f] poni1_max[%.3f]):\t " %
                              (self.geoRef.poni1_min, self.geoRef.poni1, self.geoRef.poni1_max),
                               {1:[self.geoRef.set_poni1],
                                3:[ self.geoRef.set_poni1_min, self.geoRef.set_poni1, self.geoRef.set_poni1_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Poni2 in meter "
                              "(or poni2_min[%.3f] poni2[%.3f] poni2_max[%.3f]):\t " %
                              (self.geoRef.poni2_min, self.geoRef.poni2, self.geoRef.poni2_max),
                              {1:[self.geoRef.set_poni2],
                               3:[ self.geoRef.set_poni2_min, self.geoRef.set_poni2, self.geoRef.set_poni2_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Rot1 in rad "
                              "(or rot1_min[%.3f] rot1[%.3f] rot1_max[%.3f]):\t " %
                              (self.geoRef.rot1_min, self.geoRef.rot1, self.geoRef.rot1_max),
                              {1:[self.geoRef.set_rot1],
                               3:[ self.geoRef.set_rot1_min, self.geoRef.set_rot1, self.geoRef.set_rot1_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Rot2 in rad "
                              "(or rot2_min[%.3f] rot2[%.3f] rot2_max[%.3f]):\t " %
                              (self.geoRef.rot2_min, self.geoRef.rot2, self.geoRef.rot2_max),
                              {1:[self.geoRef.set_rot2],
                               3:[ self.geoRef.set_rot2_min, self.geoRef.set_rot2, self.geoRef.set_rot2_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Rot3 in rad "
                              "(or rot3_min[%.3f] rot3[%.3f] rot3_max[%.3f]):\t " %
                              (self.geoRef.rot3_min, self.geoRef.rot3, self.geoRef.rot3_max),
                              {1:[self.geoRef.set_rot3],
                               3:[ self.geoRef.set_rot3_min, self.geoRef.set_rot3, self.geoRef.set_rot3_max]})
            previous = sys.maxint

    def postProcess(self):
        """
        Common part: shows the result of the azimuthal integration in 1D and 2D
        """
        if self.geoRef is None:
            self.refine()
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
            fig3 = pylab.plt.figure()
            xrpd = fig3.add_subplot(1, 2, 1)
            xrpd2 = fig3.add_subplot(1, 2, 2)
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
        print (os.linesep.join(["Timings:",
                                " * two theta array generation %.3fs" % (t1 - t0),
                                " * diff Solid Angle           %.3fs" % (t2 - t1),
                                " * chi array generation       %.3fs" % (t2a - t2),
                                " * corner coordinate array    %.3fs" % (t2b - t2a),
                                " * 1D Azimuthal integration   %.3fs" % (t4 - t3),
                                " * 2D Azimuthal integration   %.3fs" % (t5 - t4)]))
        if self.gui:
            xrpd.plot(a, b)
            xrpd.set_title("1D integration")
            xrpd.set_xlabel(self.unit)
            xrpd.set_ylabel("Intensity")
            xrpd2.imshow(numpy.log(img - img.min() + 1e-3), origin="lower",
                         extent=[pos_rad.min(), pos_rad.max(), pos_azim.min(), pos_azim.max()],
                         aspect="auto")
            xrpd2.set_title("2D regrouping")
            xrpd2.set_xlabel(self.unit)
            xrpd2.set_ylabel("Azimuthal angle (deg)")
            fig3.show()

################################################################################
# Calibration
################################################################################

class Calibration(AbstractCalibration):
    """
    class doing the calibration of frames
    """
    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None,
                 splineFile=None, detector=None, gaussianWidth=None, spacing_file=None,
                 wavelength=None):
        """
        Constructor


        """
        AbstractCalibration.__init__(self, dataFiles, darkFiles, flatFiles, pixelSize,
                                     splineFile, detector, spacing_file, wavelength)
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
You will need a "d-spacing" file containing the spacing of Miller plans in
Angstrom (in decreasing order).
If you are using a standart calibrant, look at
https://github.com/kif/pyFAI/tree/master/calibration
or search in the American Mineralogist database:
http://rruff.geo.arizona.edu/AMS/amcsd.php"""

        epilog = """The output of this program is a "PONI" file containing the detector description
and the 6 refined parameters (distance, center, rotation) and wavelength.
An 1D and 2D diffraction patterns are also produced. (.dat and .azim files)
        """
        usage = "%prog [options] -w 1 -D detector -S calibrant.D imagefile.edf"
        self.configure_parser(usage=usage, description=description, epilog=epilog)  # common
        self.parser.add_option("-r", "--reconstruct", dest="reconstruct",
              help="Reconstruct image where data are masked or <0  (for Pilatus "\
              "detectors or detectors with modules)",
              action="store_true", default=False)

        self.parser.add_option("-g", "--gaussian", dest="gaussian",
                               help="""Size of the gaussian kernel.
Size of the gap (in pixels) between two consecutive rings, by default 100
Increase the value if the arc is not complete;
decrease the value if arcs are mixed together.""", default=None)
        self.parser.add_option("-c", "--square", dest="square", action="store_true",
            help="Use square kernel shape for neighbor search instead of diamond shape",
            default=False)
        self.parser.add_option("-p", "--pixel", dest="pixel",
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


    def gui_peakPicker(self):
        if self.peakPicker is None:
            self.preprocess()
#        self.peakPicker.gui(True)
        if os.path.isfile(self.pointfile):
            self.peakPicker.load(self.pointfile)
        if self.gui:
            self.peakPicker.fig.canvas.draw()
        self.data = self.peakPicker.finish(self.pointfile)
        if not self.weighted:
            self.data = numpy.array(self.data)[:, :-1]


    def refine(self):
        """
        Contains the geometry refinement part specific to Calibration
        """
        self.geoRef = GeometryRefinement(self.data, dist=0.1, detector=self.detector,
                                         wavelength=self.wavelength,
                                         dSpacing=self.peakPicker.points.dSpacing)
        paramfile = self.basename + ".poni"
        if os.path.isfile(paramfile):
            self.geoRef.load(paramfile)
        AbstractCalibration.refine(self)


################################################################################
# Recalibration
################################################################################

class Recalibration(AbstractCalibration):
    """
    class doing the re-calibration of frames
    """
    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None,
                 splineFile=None, detector=None, spacing_file=None, wavelength=None):
        """
        """
        AbstractCalibration.__init__(self, dataFiles, darkFiles, flatFiles, pixelSize,
                                     splineFile, detector, spacing_file, wavelength)


    def parse(self):
        """
        parse options from command line
        """
        description = """Calibrate the diffraction setup geometry based on Debye-Sherrer rings images
with a priori knowledge of your setup (an input PONI-file).
You will need a "d-spacing" file containing the spacing of Miller plans in
Angstrom (in decreasing order).
If you are using a standart calibrant, look at
https://github.com/kif/pyFAI/tree/master/calibration
or search in the American Mineralogist database:
http://rruff.geo.arizona.edu/AMS/amcsd.php
"""

        epilog = """The main difference with pyFAI-calib is the way control-point hence Debye-Sherrer
rings are extracted. While pyFAI-calib relies on the contiguity of a region of peaks
called massif; pyFAI-recalib knows approximatly the geometry and is able to select
the region where the ring should be. From this region it selects automatically
the various peaks; making pyFAI-recalib able to run without graphical interface and
without human intervention (--no-gui --no-interactive options).

        """
        usage = "%prog [options] -p ponifile -w 1 -S calibrant.D imagefile.edf"
        self.configure_parser(usage=usage, description=description, epilog=epilog)  # common

        self.parser.add_option("-r", "--ring", dest="max_rings", type="int",
                      help="maximum number of rings to extract. Default: all accessible", default=None)
        self.parser.add_option("-p", "--poni", dest="poni", metavar="FILE",
                      help="file containing the diffraction parameter (poni-file). MANDATORY",
                      default=None)
        self.parser.add_option("-k", "--keep", dest="keep",
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


    def extract_cpt(self):
        d = numpy.loadtxt(self.spacing_file)
        tth = 2.0 * numpy.arcsin(self.ai.wavelength / (2.0e-10 * d))
        tth.sort()
        tth = tth[numpy.where(numpy.isnan(tth) - 1)]
        dtth = numpy.zeros((tth.size, 2))
        delta = tth[1:] - tth[:-1]
        dtth[:-1, 0] = delta
        dtth[-1, 0] = delta[-1]
        dtth[1:, 1] = delta
        dtth[0, 1] = delta[0]
        dtth = dtth.min(axis= -1)
        ttha = self.ai.twoThetaArray(self.peakPicker.data.shape)
#        self.peakPicker.points.wavelength = self.ai.wavelength
#        self.peakPicker.points.dSpacing = d
        rings = 0
        if self.max_rings is None:
            self.max_rings = tth.size
        for i in range(tth.size):
            mask = abs(ttha - tth[i]) <= (dtth[i] / 4.0)
            if self.mask is not None:
                mask = mask & (1 - self.mask)
            size = mask.sum(dtype=int)
            if (size > 0) and (rings < self.max_rings):
                rings += 1
                self.peakPicker.massif_contour(mask)
                if self.gui:
                    self.peakPicker.fig.canvas.draw()
                sub_data = self.peakPicker.data.ravel()[numpy.where(mask.ravel())]
                mean = sub_data.mean(dtype=numpy.float64)
                std = sub_data.std(dtype=numpy.float64)
                mask2 = (self.peakPicker.data > (mean + std)) & mask
                all_points = numpy.vstack(numpy.where(mask2)).T
                size2 = all_points.shape[0]
                if size2 < 1000:
                    mask2 = (self.peakPicker.data > mean) & mask
                    all_points = numpy.vstack(numpy.where(mask2)).T
                    size2 = all_points.shape[0]
                    upper_limit = mean
                else:
                    upper_limit = mean + std
                keep = int(numpy.ceil(numpy.sqrt(size2)))
                res = []
                cnt = 0
                logger.info("Extracting datapoint for ring %s (2theta = %.2f deg); "\
                            "searching for %i pts out of %i with I>%.1f" %
                            (i, numpy.degrees(tth[i]), keep, size2, upper_limit))
                numpy.random.shuffle(all_points)
                for idx in all_points:
                    out = self.peakPicker.massif.nearest_peak(idx)
                    if out is not None:
                        print("[ %3i, %3i ] -> [ %.1f, %.1f ]" %
                              (idx[1], idx[0], out[1], out[0]))
                        p0, p1 = out
                        if mask[p0, p1]:
                            if (out not in res) and\
                                (self.peakPicker.data[p0, p1] > upper_limit):
                                res.append(out)
                                cnt = 0
                    if len(res) >= keep or cnt > keep:
                        print len(res), cnt
                        break
                    else:
                        cnt += 1

                self.peakPicker.points.append(res, tth[i], i)
                if self.gui:
                    self.peakPicker.display_points()
                    self.peakPicker.fig.canvas.draw()

        self.peakPicker.points.save(self.basename + ".npt")
        if self.weighted:
            self.data = self.peakPicker.points.getWeightedList(self.peakPicker.data)
        else:
            self.data = self.peakPicker.points.getList()


    def refine(self):
        """
        Contains the geometry refinement part specific to Recalibration
        """

        self.geoRef = GeometryRefinement(self.data, dist=self.ai.dist, poni1=self.ai.poni1,
                                         poni2=self.ai.poni2, rot1=self.ai.rot1,
                                         rot2=self.ai.rot2, rot3=self.ai.rot3,
                                         detector=self.ai.detector, dSpacing=self.spacing_file,
                                         wavelength=self.ai._wavelength)
        self.ai = self.geoRef
        self.geoRef.set_tolerance(10)
        AbstractCalibration.refine(self)



class CheckCalib(object):
    def __init__(self, poni=None, img=None):
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
        self.unit = "r_mm"
        self.masked_resynth = None
        self.masked_image = None
        self.offset = None
        self.data = None

    def __repr__(self, *args, **kwargs):
        if self.ai:
            return self.ai.__repr__()

    def parse(self):
        logger.debug("in parse")
        usage = "usage: %prog [options] -p param.poni image.edf"
        description = """Check_calib is a research tool aiming at validating both the geometric
calibration and everything else like flat-field correction, distortion
correction. Maybe the future lies over there ...
        """
        parser = OptionParser(usage=usage,
                              version="%prog from pyFAI version " + PyFAI_VERSION,
                              description=description)
        parser.add_option("-v", "--verbose",
                          action="store_true", dest="verbose", default=False,
                          help="switch to debug mode")
        parser.add_option("-d", "--dark", dest="dark", metavar="FILE",
                      help="file containing the dark images to subtract", default=None)
        parser.add_option("-f", "--flat", dest="flat", metavar="FILE",
                      help="file containing the flat images to divide", default=None)
        parser.add_option("-m", "--mask", dest="mask", metavar="FILE",
                      help="file containing the mask", default=None)
        parser.add_option("-p", "--poni", dest="poni", metavar="FILE",
                      help="file containing the diffraction parameter (poni-file)",
                      default=None)
        parser.add_option("-e", "--energy", dest="energy", type="float",
                      help="energy of the X-Ray beam in keV (hc=%skeV.A)" % hc, default=None)
        parser.add_option("-w", "--wavelength", dest="wavelength", type="float",
                      help="wavelength of the X-Ray beam in Angstrom", default=None)

        (options, args) = parser.parse_args()
        if options.verbose:
            logger.setLevel(logging.DEBUG)

        if options.mask is not None:
            self.mask = (fabio.open(options.mask).data != 0)
        args = expand_args(args)
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
                                             unit=self.unit)

    def rebuild(self):
        logger.debug("in rebuild")
        if self.r is None:
            self.integrate()
        self.resynth = self.ai.calcfrom1d(self.r, self.I, mask=self.mask,
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
        self.offset, log = measure_offset(self.masked_resynth, self.masked_image, withLog=1)
        print os.linesep.join(log)

        print "offset:", self.offset

    def smooth_mask(self):
        logger.debug("in smooth_mask")
        if self.mask is not None:
            big_mask = scipy.ndimage.binary_dilation(self.mask, numpy.ones((10, 10)))
            smooth_mask = 1 - scipy.ndimage.filters.gaussian_filter(big_mask, 5)
            return smooth_mask

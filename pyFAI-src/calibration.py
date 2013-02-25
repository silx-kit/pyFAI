#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
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
__date__ = "06/02/2013"
__status__ = "development"

import os, sys, gc, threading, time, logging
from optparse import OptionParser
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyFAI.calibration")
import numpy, scipy, scipy.ndimage
from numpy import sin, cos, arccos, sqrt, floor, ceil, radians, degrees, pi
import fabio
import matplotlib
import pylab
from .detectors import detector_factory, Detector
from .geometryRefinement import GeometryRefinement
from .peakPicker import PeakPicker, Massif
from .utils import averageImages, timeit, measure_offset
from .azimuthalIntegrator import AzimuthalIntegrator
from .units import hc
from  matplotlib.path import Path
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.interactive(True)


################################################################################
# Calibration
################################################################################

class Calibration(object):
    """
    class doing the calibration of frames
    """
    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, pixelSize=None, splineFile=None, detector=None, gaussianWidth=None):
        """
        """
        self.dataFiles = dataFiles or []
        self.darkFiles = darkFiles or []
        self.flatFiles = flatFiles or []
        self.pointfile = None
        self.gaussianWidth = gaussianWidth
        self.labelPattern = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        if not detector:
            self.detector = Detector()
        if splineFile and os.path.isfile(splineFile):
            self.detector.splineFile = os.path.abspath(splineFile)
        if pixelSize:
            if __len__ in pixelSize and len(pixelSize) >= 2:
                self.detector.pixel1 = float(pixelSize[0])
                self.detector.pixel2 = float(pixelSize[1])
            else:
                self.detector.pixel1 = self.detector.pixel2 = float(pixelSize)
        self.cutBackground = None
        self.outfile = "merged.edf"
        self.peakPicker = None
        self.data = None
        self.basename = None
        self.geoRef = None
        self.reconstruct = False
        self.mask = None
        self.max_iter = 1000
        self.filter = "mean"
        self.threshold = 0.1
        self.spacing_file = None
        self.wavelength = None
        self.weighted = False
        self.polarization_factor = 0

    def __repr__(self):
        lst = ["Calibration object:",
             "data= " + ", ".join(self.dataFiles),
             "dark= " + ", ".join(self.darkFiles),
             "flat= " + ", ".join(self.flatFiles)]
        lst.append(self.detector.__repr__())
        lst.append("gaussian= %s" % self.gaussianWidth)
        return os.linesep.join(lst)

    def parse(self):
        """
        parse options from command line
        """
        parser = OptionParser()
        parser.add_option("-V", "--version", dest="version", action="store_true",
                          help="print version of the program and quit", metavar="FILE", default=False)
        parser.add_option("-o", "--out", dest="outfile",
                          help="Filename where processed image is saved", metavar="FILE", default="merged.edf")
        parser.add_option("-v", "--verbose",
                          action="store_true", dest="debug", default=False,
                          help="switch to debug/verbose mode")
        parser.add_option("-g", "--gaussian", dest="gaussian", help="""Size of the gaussian kernel.
Size of the gap (in pixels) between two consecutive rings, by default 100
Increase the value if the arc is not complete;
decrease the value if arcs are mixed together.""", default=None)
        parser.add_option("-c", "--square", dest="square", action="store_true",
                      help="Use square kernel shape for neighbor search instead of diamond shape", default=False)
        parser.add_option("--spacing", dest="spacing", metavar="FILE",
                      help="file containing d-spacing of the reference sample (MANDATORY)", default=None)
        parser.add_option("-w", "--wavelength", dest="wavelength", type="float",
                      help="wavelength of the X-Ray beam in Angstrom", default=None)
        parser.add_option("-e", "--energy", dest="energy", type="float",
                      help="energy of the X-Ray beam in keV (hc=%skeV.A)" % hc, default=None)
        parser.add_option("-P", "--polarization", dest="polarization_factor",
                      type="float", default=0.0,
                      help="polarization factor, from -1 (vertical) to +1 (horizontal), default is 0, synchrotrons are around 0.95")
        parser.add_option("-b", "--background", dest="background",
                      help="Automatic background subtraction if no value are provided", default=None)
        parser.add_option("-d", "--dark", dest="dark",
                      help="list of dark images to average and subtract", default=None)
        parser.add_option("-f", "--flat", dest="flat",
                      help="list of flat images to average and divide", default=None)
        parser.add_option("-r", "--reconstruct", dest="reconstruct",
                      help="Reconstruct image where data are masked or <0  (for Pilatus detectors or detectors with modules)",
                      action="store_true", default=False)
        parser.add_option("-s", "--spline", dest="spline",
                      help="spline file describing the detector distortion", default=None)
        parser.add_option("-p", "--pixel", dest="pixel",
                      help="size of the pixel in micron", default=None)
        parser.add_option("-D", "--detector", dest="detector_name",
                      help="Detector name (instead of pixel size+spline)", default=None)
        parser.add_option("-m", "--mask", dest="mask",
                      help="file containing the mask (for image reconstruction)", default=None)
        parser.add_option("-n", "--npt", dest="npt",
                      help="file with datapoints saved", default=None)
        parser.add_option("--filter", dest="filter",
                      help="select the filter, either mean(default), max or median",
                       default="mean")
        parser.add_option("--saturation", dest="saturation",
                      help="consider all pixel>max*(1-saturation) as saturated and reconstruct them",
                      default=0.1, type="float")
        parser.add_option("--weighted", dest="weighted",
                      help="weight fit by intensity",
                       default=False, action="store_true")


        (options, args) = parser.parse_args()

        # Analyse aruments and options
        if options.version:
            print("pyFAI-calib version %s" % pyFAI.version)
            sys.exit(0)
        if options.debug:
            logger.setLevel(logging.DEBUG)
        self.outfile = options.outfile
        self.gaussianWidth = options.gaussian
        if options.square:
            self.labelPattern = [[1] * 3] * 3
        else:
            self.labelPattern = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        if options.background is not None:
            try:
                self.cutBackground = float(options.background)
            except Exception:
                self.cutBackground = True
        if options.dark:
            print options.dark, type(options.dark)
            self.darkFiles = [f for f in options.dark.split(",") if os.path.isfile(f)]
        if options.flat:
            print options.flat, type(options.flat)
            self.flatFiles = [f for f in options.flat.split(",") if os.path.isfile(f)]
        self.reconstruct = options.reconstruct
        if options.mask and os.path.isfile(options.mask):
            self.mask = fabio.open(options.mask).data


        self.pointfile = options.npt
        if options.detector_name:
            self.detector = detector_factory(options.detector_name)
        if options.spline:
            if os.path.isfile(options.spline):
                self.detector.splineFile = os.path.abspath(options.spline)
            else:
                logger.error("Unknown spline file %s" % (options.spline))
        if options.pixel is not None:
            self.get_pixelSize(options.pixel)
        self.filter = options.filter
        self.threshold = options.saturation
        if options.wavelength:
            self.wavelength = 1e-10 * options.wavelength
        elif options.energy:
            self.wavelength = 1e-10 * hc / options.energy
        self.spacing_file = options.spacing
        self.polarization_factor = options.polarization_factor
#        if not self.spacing_file or not os.path.isfile(self.spacing_file):
#            raise RuntimeError("you must specify the d-spacing file")
        self.dataFiles = [f for f in args if os.path.isfile(f)]
        if not self.dataFiles:
            raise RuntimeError("Please provide some calibration images ... if you want to analyze them. Try also the --help option to see all options!")
        self.weighted = options.weighted

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
            ans = raw_input("Please enter the pixel size (in micron, comma separated X, Y , i.e. %.2e,%.2e) or a spline file: " % tuple(pixelSize)).strip()
            if os.path.isfile(ans):
                self.detector.splineFile = ans
            else:
                self.get_pixelSize(ans)

    def read_dSpacingFile(self):
        """Read the name of the file with d-spacing"""
        if (self.spacing_file is None):
            comments = ["pyFAI calib has changed !!!",
                        "Instead of entering the 2theta value, which was tedious,"
                        "the program takes a d-spacing file in input (just a serie of number representing the inter planar distance in Angstrom)",
                        "and an associated wavelength", ""
                        "You will be asked to enter the ring number, which is usually a simpler than the 2theta value."]
            print(os.linesep.join(comments))
            ans = ""
            while not os.path.isfile(ans):
                ans = raw_input("Please enter the name of the file containing the d-spacing:\t").strip()
            self.spacing_file = ans

    def read_wavelength(self):
        """Read the wavelength"""
        while not self.wavelength:
            ans = raw_input("Please enter wavelength in Angstrom:\t").strip()
            try:
                self.wavelength = 1e-10 * float(ans)
            except:
                self.wavelength = None

    def preprocess(self):
        """
        do dark, flat correction thresholding, ...
        """
        if len(self.dataFiles) > 1 or self.cutBackground or self.darkFiles or self.flatFiles:
            self.outfile = averageImages(self.dataFiles, self.outfile, threshold=self.threshold, minimum=self.cutBackground,
                                      darks=self.darkFiles, flats=self.flatFiles, filter_=self.filter)
        else:
            self.outfile = self.dataFiles[0]
        self.basename = os.path.splitext(self.outfile)[0]
        self.pointfile = self.basename + ".npt"
        self.peakPicker = PeakPicker(self.outfile, reconst=self.reconstruct, mask=self.mask,
                                     pointfile=self.pointfile, dSpacing=self.spacing_file, wavelength=self.wavelength)
        if self.gaussianWidth is not None:
            self.peakPicker.massif.setValleySize(self.gaussianWidth)
        else:
            self.peakPicker.massif.initValleySize()
        if not self.peakPicker.points.dSpacing:
            self.read_dSpacingFile()
            self.peakPicker.points.load_dSpacing(self.dSpacing)
        if not self.peakPicker.points.wavelength:
            self.read_wavelength()
            self.peakPicker.points.wavelength = self.wavelength

    def gui_peakPicker(self):
        if self.peakPicker is None:
            self.preprocess()
        self.peakPicker.gui(True)
        if os.path.isfile(self.pointfile):
            self.peakPicker.load(self.pointfile)
        self.data = self.peakPicker.finish(self.pointfile)
        if not self.weighted:
            self.data = numpy.array(self.data)[:, :-1]

    def refine(self):
        if os.name == "nt" and self.peakPicker is not None:
            logging.info("We are under windows, matplotlib is not able to display too many images without crashing, this is why the window showing the diffraction image is closed")
            self.peakPicker.closeGUI()
        self.geoRef = GeometryRefinement(self.data, dist=0.1, detector=self.detector, wavelength=self.wavelength, dSpacing=self.peakPicker.points.dSpacing)
        paramfile = self.basename + ".poni"
        if os.path.isfile(paramfile):
            self.geoRef.load(paramfile)
        print self.geoRef
        previous = sys.maxint
        finished = False
        fig2 = None
        while not finished:
            count = 0
            while (previous > self.geoRef.chi2()) and (count < self.max_iter):
                previous = self.geoRef.chi2()
                self.geoRef.refine2(1000000)
                print(self.geoRef)
                count += 1
            print(self.geoRef)
            self.geoRef.save(self.basename + ".poni")
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
            if os.name == "nt":
                logger.info("We are under windows, matplotlib is not able to display too many images without crashing, this is why little information is displayed")
            else:
                self.peakPicker.contour(tth)
                if fig2 is None:
                    fig2 = pylab.plt.figure()
                    sp = fig2.add_subplot(111)
                else:
                    sp.images.pop()
                sp.imshow(dsa, origin="lower")
                # self.fig.canvas.draw()
                fig2.show()

            change = raw_input("Modify parameters ?\t ").strip()
            if (change == '') or (change.lower()[0] == "n"):
                finished = True
            else:
                self.peakPicker.readFloatFromKeyboard("Enter Distance in meter (or dist_min[%.3f] dist[%.3f] dist_max[%.3f]):\t " % (self.geoRef.dist_min, self.geoRef.dist, self.geoRef.dist_max), {1:[self.geoRef.set_dist], 3:[ self.geoRef.set_dist_min, self.geoRef.set_dist, self.geoRef.set_dist_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Poni1 in meter (or poni1_min[%.3f] poni1[%.3f] poni1_max[%.3f]):\t " % (self.geoRef.poni1_min, self.geoRef.poni1, self.geoRef.poni1_max), {1:[self.geoRef.set_poni1], 3:[ self.geoRef.set_poni1_min, self.geoRef.set_poni1, self.geoRef.set_poni1_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Poni2 in meter (or poni2_min[%.3f] poni2[%.3f] poni2_max[%.3f]):\t " % (self.geoRef.poni2_min, self.geoRef.poni2, self.geoRef.poni2_max), {1:[self.geoRef.set_poni2], 3:[ self.geoRef.set_poni2_min, self.geoRef.set_poni2, self.geoRef.set_poni2_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Rot1 in rad (or rot1_min[%.3f] rot1[%.3f] rot1_max[%.3f]):\t " % (self.geoRef.rot1_min, self.geoRef.rot1, self.geoRef.rot1_max), {1:[self.geoRef.set_rot1], 3:[ self.geoRef.set_rot1_min, self.geoRef.set_rot1, self.geoRef.set_rot1_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Rot2 in rad (or rot2_min[%.3f] rot2[%.3f] rot2_max[%.3f]):\t " % (self.geoRef.rot2_min, self.geoRef.rot2, self.geoRef.rot2_max), {1:[self.geoRef.set_rot2], 3:[ self.geoRef.set_rot2_min, self.geoRef.set_rot2, self.geoRef.set_rot2_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Rot3 in rad (or rot3_min[%.3f] rot3[%.3f] rot3_max[%.3f]):\t " % (self.geoRef.rot3_min, self.geoRef.rot3, self.geoRef.rot3_max), {1:[self.geoRef.set_rot3], 3:[ self.geoRef.set_rot3_min, self.geoRef.set_rot3, self.geoRef.set_rot3_max]})
                previous = sys.maxint


    def postProcess(self):
        if self.geoRef is None:
            self.refine()
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

        fig3 = pylab.plt.figure()
        xrpd = fig3.add_subplot(111)
        fig4 = pylab.plt.figure()
        xrpd2 = fig4.add_subplot(111)
        t3 = time.time()
        a, b = self.geoRef.xrpd(self.peakPicker.data, 1024, self.basename + ".xy",
                                polarization_factor=self.polarization_factor)
        t4 = time.time()
        img = self.geoRef.xrpd2(self.peakPicker.data, 400, 360, self.basename + ".azim",
                                polarization_factor=self.polarization_factor)[0]
        t5 = time.time()
        print ("Timings:\n two theta array generation %.3fs\n diff Solid Angle  %.3fs\n\
     chi array generation %.3fs\n\
     corner coordinate array %.3fs\n\
     1D Azimuthal integration: %.3fs\n\
     2D Azimuthal integration: %.3fs" % (t1 - t0, t2 - t1, t2a - t2, t2b - t2a, t4 - t3, t5 - t4))
        xrpd.plot(a, b)
        fig3.show()
        xrpd2.imshow(numpy.log(img - img.min() + 1e-3), origin="lower")
        fig4.show()

################################################################################
# Recalibration
################################################################################

class Recalibration(object):
    """
    class doing the re-calibration of frames
    """
    def __init__(self, dataFiles=None, darkFiles=None, flatFiles=None, splineFile=None, gaussianWidth=None):
        """
        """
        self.dataFiles = dataFiles or []
        self.darkFiles = darkFiles or []
        self.flatFiles = flatFiles or []
        self.pointfile = None
        self.gaussianWidth = gaussianWidth
        self.labelPattern = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        self.splineFile = splineFile
        self.cutBackground = None
        self.outfile = "merged.edf"
        self.peakPicker = None
        self.img = None
        self.ai = None
        self.data = None
        self.basename = None
        self.geoRef = None
        self.reconstruct = False
        self.spacing_file = None
        self.mask = None
        self.saturation = 0.1
        self.fixed = ["wavelength"]  # parameter fixed during optimization
        self.max_rings = None
        self.max_iter = 1000
        self.gui = True
        self.interactive = True
        self.filter = "mean"
        self.weighted = False
        self.polarization_factor = 0

    def __repr__(self):
        lst = ["Calibration object:",
             "data= " + ", ".join(self.dataFiles),
             "dark= " + ", ".join(self.darkFiles),
             "flat= " + ", ".join(self.flatFiles)]
        lst += ["spline= %s" % self.splineFile,
             "gaussian= %s" % self.gaussianWidth]
        return os.linesep.join(lst)

    def parse(self):
        """
        parse options from command line
        """
        parser = OptionParser()
        parser.add_option("-V", "--version", dest="version", action="store_true",
                          help="print version of the program and quit", metavar="FILE", default=False)
        parser.add_option("-o", "--out", dest="outfile",
                          help="Filename where processed image is saved", metavar="FILE", default="merged.edf")
        parser.add_option("-v", "--verbose",
                          action="store_true", dest="verbose", default=False,
                          help="switch to debug mode")
        parser.add_option("-s", "--spacing", dest="spacing", metavar="FILE",
                      help="file containing d-spacing of the reference sample (MANDATORY)", default=None)
        parser.add_option("-r", "--ring", dest="max_rings", type="float",
                      help="maximum number of rings to extract", default=None)
        parser.add_option("-d", "--dark", dest="dark", metavar="FILE",
                      help="list of dark images to average and subtract", default=None)
        parser.add_option("-f", "--flat", dest="flat", metavar="FILE",
                      help="list of flat images to average and divide", default=None)
        parser.add_option("-m", "--mask", dest="mask", metavar="FILE",
                      help="file containing the mask", default=None)
        parser.add_option("-p", "--poni", dest="poni", metavar="FILE",
                      help="file containing the diffraction parameter (poni-file)", default=None)
        parser.add_option("-n", "--npt", dest="npt", metavar="FILE",
                      help="file with datapoints saved", default=None)
        parser.add_option("-e", "--energy", dest="energy", type="float",
                      help="energy of the X-Ray beam in keV (hc=%skeV.A)" % hc, default=None)
        parser.add_option("-w", "--wavelength", dest="wavelength", type="float",
                      help="wavelength of the X-Ray beam in Angstrom", default=None)
        parser.add_option("-P", "--polarization", dest="polarization_factor",
                      type="float", default=0.0,
                      help="polarization factor, from -1 (vertical) to +1 (horizontal), default is 0, synchrotrons are around 0.95")
        parser.add_option("-l", "--distance", dest="distance", type="float",
                      help="sample-detector distance in millimeter", default=None)
        parser.add_option("--poni1", dest="poni1", type="float",
                      help="poni1 coordinate in meter", default=None)
        parser.add_option("--poni2", dest="poni2", type="float",
                      help="poni2 coordinate in meter", default=None)
        parser.add_option("--rot1", dest="rot1", type="float",
                      help="rot1 in radians", default=None)
        parser.add_option("--rot2", dest="rot2", type="float",
                      help="rot2 in radians", default=None)
        parser.add_option("--rot3", dest="rot3", type="float",
                      help="rot3 in radians", default=None)

        parser.add_option("--fix-dist", dest="fix_dist",
                      help="fix the distance parameter", default=None, action="store_true")
        parser.add_option("--free-dist", dest="fix_dist",
                      help="free the distance parameter", default=None, action="store_false")

        parser.add_option("--fix-poni1", dest="fix_poni1",
                      help="fix the poni1 parameter", default=None, action="store_true")
        parser.add_option("--free-poni1", dest="fix_poni1",
                      help="free the poni1 parameter", default=None, action="store_false")

        parser.add_option("--fix-poni2", dest="fix_poni2",
                      help="fix the poni2 parameter", default=None, action="store_true")
        parser.add_option("--free-poni2", dest="fix_poni2",
                      help="free the poni2 parameter", default=None, action="store_false")

        parser.add_option("--fix-rot1", dest="fix_rot1",
                      help="fix the rot1 parameter", default=None, action="store_true")
        parser.add_option("--free-rot1", dest="fix_rot1",
                      help="free the rot1 parameter", default=None, action="store_false")

        parser.add_option("--fix-rot2", dest="fix_rot2",
                      help="fix the rot2 parameter", default=None, action="store_true")
        parser.add_option("--free-rot2", dest="fix_rot2",
                      help="free the rot2 parameter", default=None, action="store_false")

        parser.add_option("--fix-rot3", dest="fix_rot3",
                      help="fix the rot3 parameter", default=None, action="store_true")
        parser.add_option("--free-rot3", dest="fix_rot3",
                      help="free the rot3 parameter", default=None, action="store_false")

        parser.add_option("--fix-wavelength", dest="fix_wavelength",
                      help="fix the wavelength parameter", default=True, action="store_true")
        parser.add_option("--free-wavelength", dest="fix_wavelength",
                      help="free the wavelength parameter", default=True, action="store_false")
        parser.add_option("--saturation", dest="saturation",
                      help="consider all pixel>max*(1-saturation) as saturated and reconstruct them",
                      default=0.1, type="float")
        parser.add_option("--no-gui", dest="gui",
                      help="force the program to run without a Graphical interface",
                      default=True, action="store_false")
        parser.add_option("--no-interactive", dest="interactive",
                      help="force the program to run and exit without prompting for refinements",
                      default=True, action="store_false")
        parser.add_option("--filter", dest="filter",
                      help="select the filter, either mean(default), max or median",
                       default="mean")
        parser.add_option("--weighted", dest="weighted",
                      help="weight fit by intensity",
                       default=False, action="store_true")
        

        (options, args) = parser.parse_args()

        # Analyse aruments and options
        if options.version:
            print("pyFAI-recalib version %s" % pyFAI.version)
            sys.exit(0)
        if options.verbose:
            logger.setLevel(logging.DEBUG)
        self.outfile = options.outfile
        if options.dark:
            print options.dark, type(options.dark)
            self.darkFiles = [f for f in options.dark.split(",") if os.path.isfile(f)]
        if options.flat:
            print options.flat, type(options.flat)
            self.flatFiles = [f for f in options.flat.split(",") if os.path.isfile(f)]
        self.pointfile = options.npt
        self.spacing_file = options.spacing
        if not self.spacing_file or not os.path.isfile(self.spacing_file):
            self.read_dSpacingFile()
        self.ai = AzimuthalIntegrator.sload(options.poni)
        if options.wavelength:
            self.ai.wavelength = 1e-10 * options.wavelength
        elif options.energy:
            self.ai.wavelength = 1e-10 * hc / options.energy
        else:
            self.read_wavelength()
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
        if options.mask is not None:
            self.mask = fabio.open(options.mask).data
        self.dataFiles = [f for f in args if os.path.isfile(f)]
        if not self.dataFiles:
            raise RuntimeError("Please provide some calibration images ... if you want to analyze them. Try also the --help option to see all options!")
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
        self.max_rings = options.max_rings
        self.gui = options.gui
        self.interactive = options.interactive
        self.filter = options.filter
        self.weighted = options.weighted
        self.polarization_factor = options.polarization_factor
        print self.ai
        print "fixed:", self.fixed

    def read_dSpacingFile(self):
        """Read the name of the file with d-spacing"""
        if (self.spacing_file is None):
#            comments = ["pyFAI calib has changed !!!",
#                        "Instead of entering the 2theta value, which was tedious,"
#                        "the program takes a d-spacing file in input (just a serie of number representing the inter planar distance in Angstrom)",
#                        "and an associated wavelength", ""
#                        "You will be asked to enter the ring number, which is usually a simpler than the 2theta value."]
#            print(os.linesep.join(comments))
            ans = ""
            while not os.path.isfile(ans):
                ans = raw_input("Please enter the name of the file containing the d-spacing:\t").strip()
            self.spacing_file = ans

    def read_wavelength(self):
        """Read the wavelength"""
        while not self.ai.wavelength:
            ans = raw_input("Please enter wavelength in Angstrom:\t").strip()
            try:
                self.ai.wavelength = 1e-10 * float(ans)
            except:
                self.ai.wavelength = None

    def preprocess(self):
        """
        do dark, flat correction thresholding, ...
        """
        if len(self.dataFiles) > 1 or self.cutBackground or self.darkFiles or self.flatFiles:
            self.outfile = averageImages(self.dataFiles, self.outfile,
                                         threshold=self.saturation, minimum=self.cutBackground,
                                         darks=self.darkFiles, flats=self.flatFiles,
                                         filter_=self.filter)
        else:
            self.outfile = self.dataFiles[0]

        self.peakPicker = PeakPicker(self.outfile, mask=self.mask, dSpacing=self.spacing_file, wavelength=self.ai.wavelength)
        self.basename = os.path.splitext(self.outfile)[0]
        if self.gui:
            self.peakPicker.gui(log=True, maximize=True)
            self.peakPicker.fig.canvas.draw()

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
                logger.info("Extracting datapoint for ring %s (2theta = %.2f deg); searching for %i pts out of %i with I>%.1f" % (i, numpy.degrees(tth[i]), keep, size2, upper_limit))
                numpy.random.shuffle(all_points)
                for idx in all_points:
                    out = self.peakPicker.massif.nearest_peak(idx)
                    print "[ %3i, %3i ] -> [ %.1f, %.1f ]" % (idx[1], idx[0], out[1], out[0])
                    if out is not None:
                        p0, p1 = out
                        if mask[p0, p1]:
                            if (out not in res) and (self.peakPicker.data[p0, p1] > upper_limit):
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
        if os.name == "nt" and self.peakPicker is not None:
            logging.info("We are under windows, matplotlib is not able to display too many images without crashing, this is why the window showing the diffraction image is closed")
            self.peakPicker.closeGUI()
        self.geoRef = GeometryRefinement(self.data, dist=self.ai.dist, poni1=self.ai.poni1, poni2=self.ai.poni2,
                                             rot1=self.ai.rot1, rot2=self.ai.rot2, rot3=self.ai.rot3,
                                             detector=self.ai.detector, dSpacing=self.spacing_file, wavelength=self.ai._wavelength)
        self.geoRef.set_tolerance(10)
        print self.geoRef
        previous = sys.maxint
        finished = False
        fig2 = None
        while not finished:
            count = 0
            print("fixed: " + ", ".join(self.fixed))
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
            self.geoRef.save(self.basename + ".poni")
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
            if os.name == "nt":
                logger.info("We are under windows, matplotlib is not able to display too many images without crashing, this is why little information is displayed")
            else:
                if self.gui:
                    self.peakPicker.contour(tth)
                    if fig2 is None:
                        fig2 = pylab.plt.figure()
                        sp = fig2.add_subplot(111)
                    else:
                        sp.images.pop()
                    sp.imshow(dsa, origin="lower")
                    fig2.show()
            if not self.interactive:
                break
            change = raw_input("Modify parameters ?\t ").strip().lower()
            if (change == '') or (change[0] == "n"):
                finished = True
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
                        setattr(self.geoRef, what, value)
                else:
                    print("example: set wavelength 1e-10")
            else:
                self.peakPicker.readFloatFromKeyboard("Enter Distance in meter (or dist_min[%.3f] dist[%.3f] dist_max[%.3f]):\t " % (self.geoRef.dist_min, self.geoRef.dist, self.geoRef.dist_max), {1:[self.geoRef.set_dist], 3:[ self.geoRef.set_dist_min, self.geoRef.set_dist, self.geoRef.set_dist_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Poni1 in meter (or poni1_min[%.3f] poni1[%.3f] poni1_max[%.3f]):\t " % (self.geoRef.poni1_min, self.geoRef.poni1, self.geoRef.poni1_max), {1:[self.geoRef.set_poni1], 3:[ self.geoRef.set_poni1_min, self.geoRef.set_poni1, self.geoRef.set_poni1_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Poni2 in meter (or poni2_min[%.3f] poni2[%.3f] poni2_max[%.3f]):\t " % (self.geoRef.poni2_min, self.geoRef.poni2, self.geoRef.poni2_max), {1:[self.geoRef.set_poni2], 3:[ self.geoRef.set_poni2_min, self.geoRef.set_poni2, self.geoRef.set_poni2_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Rot1 in rad (or rot1_min[%.3f] rot1[%.3f] rot1_max[%.3f]):\t " % (self.geoRef.rot1_min, self.geoRef.rot1, self.geoRef.rot1_max), {1:[self.geoRef.set_rot1], 3:[ self.geoRef.set_rot1_min, self.geoRef.set_rot1, self.geoRef.set_rot1_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Rot2 in rad (or rot2_min[%.3f] rot2[%.3f] rot2_max[%.3f]):\t " % (self.geoRef.rot2_min, self.geoRef.rot2, self.geoRef.rot2_max), {1:[self.geoRef.set_rot2], 3:[ self.geoRef.set_rot2_min, self.geoRef.set_rot2, self.geoRef.set_rot2_max]})
                self.peakPicker.readFloatFromKeyboard("Enter Rot3 in rad (or rot3_min[%.3f] rot3[%.3f] rot3_max[%.3f]):\t " % (self.geoRef.rot3_min, self.geoRef.rot3, self.geoRef.rot3_max), {1:[self.geoRef.set_rot3], 3:[ self.geoRef.set_rot3_min, self.geoRef.set_rot3, self.geoRef.set_rot3_max]})
            previous = sys.maxint


    def postProcess(self):
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
            xrpd = fig3.add_subplot(111)
            fig4 = pylab.plt.figure()
            xrpd2 = fig4.add_subplot(111)
        t3 = time.time()
        a, b = self.geoRef.xrpd(self.peakPicker.data, 1024, self.basename + ".xy",
                                polarization_factor=self.polarization_factor)
        t4 = time.time()
        img = self.geoRef.xrpd2(self.peakPicker.data, 400, 360, self.basename + ".azim",
                                polarization_factor=self.polarization_factor)[0]
        t5 = time.time()
        print ("Timings:\n two theta array generation %.3fs\n diff Solid Angle  %.3fs\n\
     chi array generation %.3fs\n\
     corner coordinate array %.3fs\n\
     1D Azimuthal integration: %.3fs\n\
     2D Azimuthal integration: %.3fs" % (t1 - t0, t2 - t1, t2a - t2, t2b - t2a, t4 - t3, t5 - t4))
        if self.gui:
            xrpd.plot(a, b)
            fig3.show()
            xrpd2.imshow(numpy.log(img - img.min() + 1e-3), origin="lower")
            fig4.show()



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

    def __repr__(self, *args, **kwargs):
        if self.ai:
            return self.ai.__repr__()

    def parse(self):
        logger.debug("in parse")
        parser = OptionParser()
        parser.add_option("-V", "--version", dest="version", action="store_true",
                          help="print version of the program and quit", metavar="FILE", default=False)
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
                      help="file containing the diffraction parameter (poni-file)", default=None)
        parser.add_option("-e", "--energy", dest="energy", type="float",
                      help="energy of the X-Ray beam in keV (hc=%skeV.A)" % hc, default=None)
        parser.add_option("-w", "--wavelength", dest="wavelength", type="float",
                      help="wavelength of the X-Ray beam in Angstrom", default=None)

        (options, args) = parser.parse_args()
        if options.verbose:
            logger.setLevel(logging.DEBUG)

        if options.version:
            print("Check calibrarion: version %s" % pyFAI.version)
            sys.exit(0)
        if options.mask is not None:
            self.mask = fabio.open(options.mask).data
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
        self.r, self.I = self.ai.integrate1d(self.img, self.size1d, mask=self.mask, unit=self.unit)

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

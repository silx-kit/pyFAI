#!/usr/bin/env python
# -*- coding: utf8 -*-
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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "23/12/2011"
__status__ = "development"

import os, sys, threading, logging, gc
from math                   import ceil, sqrt, pi
import numpy
from scipy.optimize         import fmin
from scipy.ndimage.filters  import median_filter
from scipy.ndimage          import label#, binary_closing, binary_opening, binary_erosion #,binary_propagation
#import matplotlib
import pylab
import fabio
from utils                  import relabel, gaussian_filter, binning, unBinning
from bilinear               import bilinear
logger = logging.getLogger("pyFAI.peakPicker")
if os.name != "nt":
    WindowsError = RuntimeError
TARGET_SIZE = 1024

################################################################################
# PeakPicker
################################################################################
class PeakPicker(object):
    def __init__(self, strFilename):
        """
        @param: input image filename
        """
        self.strFilename = strFilename
        self.data = fabio.open(strFilename).data.astype("float32")
        self.shape = self.data.shape
        self.points = ControlPoints()
        self.lstPoints = []
        self.fig = None
        self.fig2 = None
        self.fig2sp = None
        self.ax = None
        self.ct = None
        self.msp = None
        self.massif = Massif(self.data)
        self._sem = threading.Semaphore()
        self._semGui = threading.Semaphore()
        self.defaultNbPoints = 100

    def gui(self, log=False):
        """
        @param log: show z in log scale
        """
        if self.fig is None:
            self.fig = pylab.plt.figure()
        self.ax = self.fig.add_subplot(111);
        if log:
            self.ax.imshow(numpy.log(1.0 + self.data - self.data.min()), origin="lower")
        else:
            self.ax.imshow(self.data, origin="lower")
        self.fig.show()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def load(self, filename):
        """
        load a filename and plot data on the screen (if GUI)
        """
        self.points.load(filename)
        if self.ax is not None:
            for idx, points in enumerate(self.points._points):
                if len(points) > 0:
                    pt0x = points[0][1]
                    pt0y = points[0][0]
                    self.ax.annotate("%i" % (idx), xy=(pt0x, pt0y), xytext=(pt0x + 10, pt0y + 10),
                                     color="white", arrowprops=dict(facecolor='white', edgecolor='white'))

                    npl = numpy.array(points)
                    self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)

    def onclick(self, event):
        def annontate(x, x0=None, idx=None):
            """
            Call back method to annotate the figure while calculation are going on ...
            @param x: coordinates 
            @param x0: coordinates of the starting point
            """
            if x0 is None:
                self.ax.annotate(".", xy=(x[1], x[0]), color="black")
            else:
                self.ax.annotate("%i" % (len(self.points)), xy=(x[1], x[0]), xytext=(x0[1], x0[0]), color="white",
                     arrowprops=dict(facecolor='white', edgecolor='white'),)
                self.fig.canvas.draw()

        self._sem.acquire()
        if event.button == 3: #right click
            x0 = event.xdata
            y0 = event.ydata
            listpeak = self.massif.find_peaks([y0, x0], self.defaultNbPoints, annontate, self.massif_contour)
            if len(listpeak) == 0:
                logging.warning("No peak found !!!")
                self._sem.release()
                return
            npl = numpy.array(listpeak)
            self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)

            logging.info("Added %3i points to group #%i" % (len(listpeak), len(self.points)))
            self.points.append(listpeak)
            self.fig.show()
            sys.stdout.flush()
        elif event.button == 2: #center click
            a = self.points.pop()
#            for i in a:
            if len(self.ax.texts) > 0:
                self.ax.texts.pop()
            if len(self.ax.lines) > 0:
                self.ax.lines.pop()
            self.fig.show()
            logging.info("Removing point group #%i (%5.1f %5.1f) containing %i subpoints" % (len(self.points), a[0][0], a[0][1], len(a)))
            sys.stdout.flush()
        self._sem.release()

    def readFloatFromKeyboard(self, text, dictVar):
        """
        Read float from the keyboard ....
        @param text: string to be displayed
        @param dictVar: dict of this type: {1: [set_dist_min],3: [set_dist_min, set_dist_guess, set_dist_max]}
        """
        fromkb = raw_input(text).strip()
        try:
            vals = [float(i) for i in fromkb.split()]
        except:
            logging.error("Error in parsing values")
        else:
            found = False
            for i in dictVar:
                if len(vals) == i:
                    found = True
                    for j in range(i):
                        dictVar[i][j](vals[j])
            if not found:
                logging.error("You should provide the good number of floats")


    def finish(self, filename=None):
        """
        Ask the 2theta values for the given points
        """
        logging.info("Please use the GUI and Right-click on the peaks to mark them")

        raw_input("Please press enter when you are happy; to fill in 2theta values" + os.linesep)
        self.points.readAngleFromKeyboard()
        if filename is not None:
            self.points.save(filename)
        self.lstPoints = self.points.getList()
        return self.lstPoints


    def contour(self, data):
        if self.fig is None:
            logging.warning("No diffraction image available => not showing the contour")
        else:
            if self.msp is not None:
                if len(self.msp.images) > 1:
                    self.msp.images.pop()
                    self.msp = None
            if self.ct is None:
                self.ct = self.fig.add_subplot(111)
            else:
                while len(self.ct.images) > 1:
                    self.ct.images.pop()
                while len(self.ct.collections) > 0:
                    self.ct.collections.pop()

            try:
                self.ct.contour(data)
            except MemoryError:
                logging.error("Sorry but your computer does NOT have enough memory to display the 2-theta contour plot")
            self.fig.show()

    def massif_contour(self, data):
        if self.fig is None:
            logging.error("No diffraction image available => not showing the contour")
        else:
            tmp = 100 * (1 - data.astype("uint8"))
            mask = numpy.zeros((data.shape[0], data.shape[1], 4), dtype="uint8")

            mask[:, :, 0] = tmp
            mask[:, :, 1] = tmp
            mask[:, :, 2] = tmp
            mask[:, :, 3] = tmp
            if self.msp is None:
                self.msp = self.fig.add_subplot(111)
            else:
                if len(self.msp.images) > 1:
                    self.msp.images.pop()
            try:
                self.msp.imshow(mask, cmap="gray", origin="lower")
            except MemoryError:
                logging.error("Sorry but your computer does NOT have enough memory to display the massif plot")
            #self.fig.show()
            self.fig.canvas.draw()

    def closeGUI(self):
        if self.fig is not None:
            self.fig.clear()
            self.fig = None
            gc.collect()

################################################################################
# ControlPoints
################################################################################
class ControlPoints(object):
    """
    This class contains a set of control points with (optionaly) their diffrection 2Theta angle  
    """
    def __init__(self, filename=None):
        if filename is not None:
            self.load(filename)
        self._angles = [] #angles are enforced in radians, conversion from degrees or q-space nm-1 are done on the fly
        self._points = []
        self._sem = threading.Semaphore()
        self._wavelength = None

    def __repr__(self):
        self.check()
        lstOut = ["ControlPoints instance containing %i group of point:" % len(self)]
        if self._wavelength is not None:
            lstOut = "wavelength: %s" % self._wavelength
        for angle, points in zip(self._angles, self._points):
            lstOut.append("%s: %s" % (angle, points))
        return os.linesep.join(lstOut)

    def __len__(self):
        return len(self._angles)

    def check(self):
        """
        check internal consistency of the class 
        """
        if len(self._angles) != len(self._points):
            logger.error("in ControlPoints: length of the two arrays are not consistent!!! angle: %i points: %s ",
                           len(self._angles), len(self._points))
    def reset(self):
        """
        remove all stored values and resets them to default 
        """
        with self._sem:
            self._wavelength = None
            self._angles = [] #angles are enforced in radians, conversion from degrees or q-space nm-1 are done on the fly
            self._points = []

    def append(self, points, angle=None):
        """
        @param point: list of points
        @param angle: 2-theta angle in radians 
        """
        with self._sem:
            self._angles.append(angle)
            self._points.append(points)
#    append_2theta_deg = append

    def append_2theta_deg(self, points, angle=None):
        """
        @param point: list of points
        @param angle: 2-theta angle in degrees 
        """
        with self._sem:
            self._angles.append(pi * angle / 180.)
            self._points.append(points)

    def pop(self, idx=None):
        """
        Remove the set of points at given index (by default the last)
        @param idx: poistion of the point to remove
        """
        out = None
        if idx is None:
            with self._sem:
                self._angles.pop()
                out = self._points.pop()
        else:
            with self._sem:
                self._angles.pop(idx)
                out = self._points.pop(idx)
        return out

    def save(self, filename):
        """
        Save a set of control points to a file
        @param filename: name of the file
        @return: None
        """
        self.check()
        with self._sem:
            lstOut = ["# set of control point used by pyFAI to calibrate the geometry of a scattering experiment",
                      "#angles are in radians, wavelength in meter and positions in pixels"]

            if self._wavelength is not None:
                lstOut = "wavelength: %s" % self._wavelength
            for idx, angle, points in zip(range(self.__len__()), self._angles, self._points):
                lstOut.append("")
                lstOut.append("New group of points: %i" % idx)
                lstOut.append("2theta: %s" % angle)
                for point in points:
                    lstOut.append("point: x=%s y=%s" % (point[1], point[0]))
            with open(filename, "w") as f:
                f.write(os.linesep.join(lstOut))

    def load(self, filename):
        """
        load all control points from a file 
        """
        if not os.path.isfile(filename):
            logger.error("ControlPoint.load: No such file %s", filename)
            return
        self.reset()
        tth = None
        points = []
        for line in open(filename, "r"):
            if line.startswith("#"):
                continue
            elif ":" in line:
                key, value = line.split(":", 1)
                value = value.strip()
                key = key.strip().lower()
                if key == "wavelength":
                    try:
                        self._wavelength = float(value)
                    except:
                        logger.error("ControlPoints.load: unable to convert to float %s (wavelength)", value)
                elif key == "2theta":
                    if value.lower() == "none":
                        tth = None
                    else:
                        try:
                            tth = float(value)
                        except:
                            logger.error("ControlPoints.load: unable to convert to float %s (2theta)", value)
                elif key == "point":
                    vx = None
                    vy = None
                    if "x=" in value:
                        vx = value[value.index("x=") + 2:].split()[0]
                    if "y=" in value:
                        vy = value[value.index("y=") + 2:].split()[0]
                    if (vx is not None) and (vy is not None):
                        try:
                            x = float(vx)
                            y = float(vy)
                        except:
                            logger.error("ControlPoints.load: unable to convert to float %s (point)", value)
                        else:
                            points.append([y, x])
                elif key.startswith("new"):
                    if len(points) > 0:
                        with self._sem:
                            self._angles.append(tth)
                            self._points.append(points)
                            tth = None
                            points = []
                else:
                    logger.error("Unknown key: %s", key)
        if len(points) > 0:
            self._angles.append(tth)
            self._points.append(points)


    def getList(self):
        """
        Retrieve the list of control points suitable for geometry refinement
        """
        lstOut = []
        for tth, points in zip(self._angles, self._points):
            lstOut += [[pt[0], pt[1], tth] for pt in points]
        return lstOut


    def readAngleFromKeyboard(self):
        """
        Ask the 2theta values for the given points
        """
        last2Theta = None
        for idx, tth, point in zip(range(self.__len__()), self._angles, self._points):
            bOk = False
            while not bOk:
                if tth is not None:
                    last2Theta = numpy.rad2deg(tth)
                res = raw_input("Point group #%2i (%i points)\t (%6.1f,%6.1f) \t [default=%s] 2Theta= " % (idx, len(point), point[0][1], point[0][0], last2Theta)).strip()
                if res == "":
                    res = last2Theta
                try:
                    tth = float(res)
                except (ValueError, TypeError):
                    logging.error("I did not understand your 2theta value")
                else:
                    if tth > 0:
                        last2Theta = tth
                        self._angles[idx] = numpy.deg2rad(tth)
                        bOk = True


    def setWavelength(self, value=None):
        with self._sem:
            if self._wavelength is None:
                self._wavelength = value
            else:
                logger.warning("Forbidden to change the wavelength once it is fixed !!!!")
    def getWavelength(self): return self._wavelength
    wavelength = property(getWavelength, setWavelength)

################################################################################
# Massif
################################################################################
class Massif(object):
    """
    A massif is defined as an area around a peak, it is used to find neighbouring peaks
    """
    def __init__(self, data=None):
        """
        
        """
        if isinstance(data, (str, unicode)) and os.path.isfile(data):
            self.data = fabio.open(data).data.astype("float32")
        elif  isinstance(data, fabio.fabioimage.fabioimage):
            self.data = data.data.astype("float32")
        else:
            try:
                self.data = data.astype("float32")
            except Exception as error:
                logger.error("Unable to understand this type of data %s: %s", data, error)
        self._bilin = bilinear(self.data)
        self._blured_data = None
        self._median_data = None
        self._labeled_massif = None
        self._number_massif = None
        self._valley_size = None
        self._binned_data = None
        self.binning = None #Binning is 2-list usually
        self._sem = threading.Semaphore()
        self._sem_label = threading.Semaphore()
        self._sem_binning = threading.Semaphore()
        self._sem_median = threading.Semaphore()


    def nearest_peak(self, x):
        """
        @returns the coordinates of the nearest peak       
        """
        x = numpy.array(x, dtype="float32")
        out = fmin(self._bilin.f_cy, x, disp=0).round().astype(numpy.int)
        if isinstance(out, numpy.ndarray):
            res = [int(i) for idx, i in enumerate(out) if 0 <= i < self.data.shape[idx] ]
        else:
            print out
            res = [int(i) for idx, i in enumerate(out) if 0 <= i < self.data.shape[idx] ]
        if len(res) == 2:
            return res


    def calculate_massif(self, x):
        """
        defines a map of the massif around x and returns the mask
        """
        labeled = self.getLabeledMassif()
        if labeled[x[0], x[1]] != labeled.max():
            return (labeled == labeled[x[0], x[1]])


    def find_peaks(self, x, nmax=200, annotate=None, massif_contour=None, stdout=sys.stdout):
        """
        All in one function that finds a maximum from the given seed (x) 
        then calculates the region extension and extract position of the neighboring peaks.  
        @param x: seed for the calculation, input coordinates
        @param nmax: maximum number of peak per region
        @param annotate: call back method taking number of points + coordinate as input. 
        @param massif_contour: callback to show the contour of a massif with the given index. 
        @param stdout: this is the file where output is written by default.
        @return: list of peaks
        """
        listpeaks = []
        region = self.calculate_massif(x)
        if region is None:
            logger.error("You picked a background point at %s", x)
            return listpeaks
        xinit = self.nearest_peak(x)
        if xinit is None:
            logger.error("Unable to find peak in the vinicy of %s", x)
            return listpeaks
        else:
            if not region[xinit[0], xinit[1]]:
                logger.error("Nearest peak %s is not in the same region  %s", xinit, x)
                return listpeaks

            if annotate is not None:
                try:
                    annotate(xinit, x)
                except Exception as error:
                    logger.error("Error in annotate %i: %i %i. %s" , len(listpeaks), xinit[0], xinit[1], error)

        listpeaks.append(xinit)
        idx = numpy.arange(region.size)
        idx.shape = region.shape
        regionIdx = idx[region]
        numpy.random.shuffle(regionIdx)
        nmax = min(nmax, int(ceil(sqrt(region.sum()))))
        if massif_contour is not None:
            try:
                massif_contour(region)
            except (WindowsError, MemoryError) as error:
                logger.error("Error in plotting region: %s", error)
        nbFailure = 0
        dim1 = region.shape[1]
        for idx in  regionIdx:
            x0 = idx // dim1
            x1 = idx % dim1
            if not region[x0, x1]:
                logger.warning("Input point (%s,%s) not in region !!!! " % (x0, x1))
            xopt = self.nearest_peak([x0, x1])
            if xopt is None:
                nbFailure += 1
                continue
            if (region[xopt[0], xopt[1]]) and not (xopt in listpeaks):
                stdout.write("[ %4i, %4i ] --> [ %4i, %4i ] after %3i iterations %s" % (x0, x1, xopt[0], xopt[1], nbFailure, os.linesep))
                listpeaks.append(xopt)

                nbFailure = 0
            else:
                nbFailure += 1
            if (len(listpeaks) > nmax) or (nbFailure > 2 * nmax):
                break
        return listpeaks


    def initValleySize(self):
        if self._valley_size is None:
            self.valley_size = max(5., max(self.data.shape) / 50.)


    def getValleySize(self):
        if self._valley_size is None:
            self.initValleySize()
        return self._valley_size
    def setValleySize(self, size):
        self._valley_size = size
        t = threading.Thread(target=self.getLabeledMassif)
        t.start()
    def delValleySize(self):
        self._valley_size = None
        self._blured_data = None
    valley_size = property(getValleySize, setValleySize, delValleySize, "Defines the minimum distance between two massifs")

    def getBinnedData(self):
        """
        @return binned data 
        """
        if self._binned_data is None:
            with self._sem_binning:
                if self._binned_data is None:
                    logger.info("Image size is %s", self.data.shape)
                    self.binning = []
                    for i in self.data.shape:
                        if i % TARGET_SIZE == 0:
                            self.binning.append(max(1, i // TARGET_SIZE))
                        else:
                            for j in range(i // TARGET_SIZE - 1, 0, -1):
                                if i % j == 0:
                                    self.binning.append(max(1, j))
                                    break
                            else:
                                self.binning.append(1)
#                    self.binning = max([max(1, i // TARGET_SIZE) for i in self.data.shape])
                    logger.info("Binning size is %s", self.binning)
                    self._binned_data = binning(self.data, self.binning)
        return self._binned_data

    def getMedianData(self):
        if self._median_data is None:
            with self._sem_median:
                if self._median_data is None:
                    self._median_data = median_filter(self.data, 3)
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        fabio.edfimage.edfimage(data=self._median_data).write("median_data.edf")
        return self._median_data

    def getBluredData(self):
        if self._blured_data is None:
            with self._sem:
                if self._blured_data is None:
                    logger.debug("Blurring image with kernel size: %s" , self.valley_size)
                    self._blured_data = gaussian_filter(self.getBinnedData(), [self.valley_size / i for i in  self.binning], mode="reflect")
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        fabio.edfimage.edfimage(data=self._blured_data).write("blured_data.edf")
        return self._blured_data

    def getLabeledMassif(self, pattern=None):
        if self._labeled_massif is None:
            with self._sem_label:
                if self._labeled_massif is None:
                    if pattern is None:
                        pattern = [[1] * 3] * 3#[[0, 1, 0], [1, 1, 1], [0, 1, 0]]#[[1] * 3] * 3
                    logger.debug("Labeling all massifs. This takes some time !!!")
                    labeled_massif, self._number_massif = label((self.getBinnedData() > self.getBluredData()), pattern)
                    logger.info("Labeling found %s massifs." % self._number_massif)
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        fabio.edfimage.edfimage(data=labeled_massif).write("labeled_massif_small.edf")
                    relabeled = relabel(labeled_massif, self.getBinnedData(), self.getBluredData())
                    if logger.getEffectiveLevel() == logging.DEBUG:
                            fabio.edfimage.edfimage(data=relabeled).write("relabeled_massif_small.edf")
                    self._labeled_massif = unBinning(relabeled, self.binning)
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        fabio.edfimage.edfimage(data=self._labeled_massif).write("labeled_massif.edf")
                    logger.info("Labeling found %s massifs." % self._number_massif)
        return self._labeled_massif

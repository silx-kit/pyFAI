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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/03/2013"
__status__ = "development"

import os, sys, threading, logging, gc, types
from math                   import ceil, sqrt, pi
import numpy
from scipy.optimize         import fmin
from scipy.ndimage.filters  import median_filter
from scipy.ndimage          import label
import pylab
import fabio
import utils
from .utils import gaussian_filter, binning, unBinning
from .bilinear import Bilinear
from .reconstruct import reconstruct
logger = logging.getLogger("pyFAI.peakPicker")
if os.name != "nt":
    WindowsError = RuntimeError
TARGET_SIZE = 1024


################################################################################
# PeakPicker
################################################################################
class PeakPicker(object):
    def __init__(self, strFilename, reconst=False, mask=None, pointfile=None, dSpacing=None, wavelength=None):
        """
        @param: input image filename
        @param reconst: shall mased part or negative values be reconstructed (wipe out problems with pilatus gaps)
        """
        self.strFilename = strFilename
        self.data = fabio.open(strFilename).data.astype("float32")
        if mask is not None:
            mask = mask.astype(bool)
            view = self.data.ravel()
            flat_mask = mask.ravel()
            min_valid = view[numpy.where(flat_mask == False)].min()
            view[numpy.where(flat_mask)] = min_valid

        self.shape = self.data.shape
        self.points = ControlPoints(pointfile, dSpacing=dSpacing, wavelength=wavelength)
#        self.lstPoints = []
        self.fig = None
        self.fig2 = None
        self.fig2sp = None
        self.ax = None
        self.ct = None
        self.msp = None
        if reconstruct and (reconst is not False):
            if mask is None:
                mask = self.data < 0
            self.massif = Massif(reconstruct(self.data, mask))
        else:
            self.massif = Massif(self.data)
        self._sem = threading.Semaphore()
        self._semGui = threading.Semaphore()
        self.defaultNbPoints = 100

    def gui(self, log=False, maximize=False):
        """
        @param log: show z in log scale
        """
        if self.fig is None:
            self.fig = pylab.plt.figure()
            # add 3 subplots at the same position for debye-sherrer image, contour-plot and massif contour
            self.ax = self.fig.add_subplot(111)
            self.ct = self.fig.add_subplot(111)
            self.msp = self.fig.add_subplot(111)
        if log:
            self.ax.imshow(numpy.log(1.0 + self.data - self.data.min()), origin="lower", interpolation="nearest")
        else:
            self.ax.imshow(self.data, origin="lower", interpolation="nearest")
        self.ax.autoscale_view(False, False, False)
        self.fig.show()
        if maximize:
            mng = pylab.get_current_fig_manager()
#            print mng.window.maxsize()
            # *mng.window.maxsize())
            win_shape = (1920, 1080)
            event = Event(*win_shape)
            try:
                mng.resize(event)
            except TypeError:
                 mng.resize(*win_shape)
            self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def load(self, filename):
        """
        load a filename and plot data on the screen (if GUI)
        """
        self.points.load(filename)
        self.display_points()

    def display_points(self):
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

        with self._sem:
            if event.button == 3:  # right click
                x0 = event.xdata
                y0 = event.ydata
                logger.debug("Key modifier: %s" % event.key)
                if event.key == 'shift':  # if 'shift' pressed add nearest maximum to the current group
                    points = self.points.pop() or []
                    if len(self.ax.texts) > 0:
                        self.ax.texts.pop()
                    if len(self.ax.lines) > 0:
                        self.ax.lines.pop()

                    self.fig.show()
                    newpeak = self.massif.nearest_peak([y0, x0])
                    if newpeak:
                        points.append(newpeak)
                        annontate(newpeak, [y0, x0])
                    else:
                        logging.warning("No peak found !!!")

                elif event.key == 'control':  # if 'control' pressed add nearest maximum to a new group
                    points = []
                    newpeak = self.massif.nearest_peak([y0, x0])
                    if newpeak:
                        points.append(newpeak)
                        annontate(newpeak, [y0, x0])
                    else:
                        logging.warning("No peak found !!!")
                elif event.key == 'm':  # if 'm' pressed add new group to current  group ...  ?
                    points = self.points.pop() or []
                    if len(self.ax.texts) > 0:
                        self.ax.texts.pop()
                    if len(self.ax.lines) > 0:
                        self.ax.lines.pop()
                    self.fig.show()
                    listpeak = self.massif.find_peaks([y0, x0], self.defaultNbPoints, annontate, self.massif_contour)
                    if len(listpeak) == 0:
                        logging.warning("No peak found !!!")
                    else:
                        points += listpeak
                else:  # create new group
                    points = self.massif.find_peaks([y0, x0], self.defaultNbPoints, annontate, self.massif_contour)
                    if not points:
                        logging.warning("No peak found !!!")
                if not points:
                    return
                self.points.append(points)
                npl = numpy.array(points)
                logging.info("x=%f, y=%f added to group #%i" % (x0, y0, len(self.points)))
                self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)
                self.fig.show()
                sys.stdout.flush()
            elif event.button == 2:  # center click
                poped_points = self.points.pop()
                if len(self.ax.texts) > 0:
                    self.ax.texts.pop()
                if len(self.ax.lines) > 0:
                    self.ax.lines.pop()
                self.fig.show()
                if poped_points is None:
                    logging.info("Removing No group point (non existing?)")
                else:
                    logging.info("Removing point group #%i (%5.1f %5.1f) containing %i subpoints" % (len(self.points), poped_points[0][0], poped_points[0][1], len(poped_points)))
                self.fig.canvas.draw()
                sys.stdout.flush()

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


    def finish(self, filename=None,):
        """
        Ask the ring number for the given points

        @param filename: file with the point coordinates saved
        """
        logging.info(os.linesep.join(["Please use the GUI and:",
                                      " 1) Right-click: try an auto find for a ring",
                                      " 2) Shift + Right-click: add one point to the current group",
                                      " 3) Control + Right-click : add a point to a new group",
                                      " 4) Center-click: erase the current group"]))

        raw_input("Please press enter when you are happy with your selection" + os.linesep)
        print("Now fill in the ring number. Ring number starts at 0, like point-groups.")
        self.points.readRingNrFromKeyboard()  # readAngleFromKeyboard()
        if filename is not None:
            self.points.save(filename)
#        self.lstPoints = self.points.getList()
        return self.points.getWeightedList(self.data)


    def contour(self, data):
        """
        Overlay a contour-plot

        @param data: 2darray with the 2theta values in radians...
        """
        if self.fig is None:
            logging.warning("No diffraction image available => not showing the contour")
        else:
            while len(self.msp.images) > 1:
                self.msp.images.pop()
            while len(self.ct.images) > 1:
                self.ct.images.pop()
            while len(self.ct.collections) > 0:
                self.ct.collections.pop()

            if self.points.dSpacing and  self.points._wavelength:
                angles = list(2.0 * numpy.arcsin(5e9 * self.points._wavelength / numpy.array(self.points.dSpacing)))
            else:
                angles = None
            try:
                xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
                self.ct.contour(data, levels=angles)
                self.ax.set_xlim(xlim);self.ax.set_ylim(ylim);
                print("Visually check that the curve overlays with the Debye-Sherrer rings of the image")
                print("Check also for correct indexing of rings")
            except MemoryError:
                logging.error("Sorry but your computer does NOT have enough memory to display the 2-theta contour plot")
            self.fig.show()

    def massif_contour(self, data):
        """
        @param data:
        """

        if self.fig is None:
            logging.error("No diffraction image available => not showing the contour")
        else:
            tmp = 100 * (1 - data.astype("uint8"))
            mask = numpy.zeros((data.shape[0], data.shape[1], 4), dtype="uint8")

            mask[:, :, 0] = tmp
            mask[:, :, 1] = tmp
            mask[:, :, 2] = tmp
            mask[:, :, 3] = tmp
            while len(self.msp.images) > 1:
                self.msp.images.pop()
            try:
                xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
                self.msp.imshow(mask, cmap="gray", origin="lower", interpolation="nearest")
                self.ax.set_xlim(xlim);self.ax.set_ylim(ylim);
            except MemoryError:
                logging.error("Sorry but your computer does NOT have enough memory to display the massif plot")
            # self.fig.show()
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
    This class contains a set of control points with (optionally) their ring number hence d-spacing and diffraction  2Theta angle ...
    """
    def __init__(self, filename=None, dSpacing=None, wavelength=None):
        self.dSpacing = []
        self._sem = threading.Semaphore()
        self._angles = []  # angles are enforced in radians, conversion from degrees or q-space nm-1 are done on the fly
        self._points = []
        self._ring = []  # ring number ...
        self._wavelength = wavelength

        if filename is not None:
            self.load(filename)
        have_spacing = False
        for i in self.dSpacing :
            have_spacing = have_spacing or i
        if (not have_spacing) and (dSpacing is not None):
            if type(dSpacing) in types.StringTypes:
                self.dSpacing = self.load_dSpacing(dSpacing)
            else:
                self.dSpacing = list(dSpacing)

    def __repr__(self):
        self.check()
        lstOut = ["ControlPoints instance containing %i group of point:" % len(self)]
        if self._wavelength is not None:
            lstOut.append("wavelength: %s" % self._wavelength)
        lstOut.append("dSpacing (A): " + ", ".join(["%.3f" % i for i in self.dSpacing]))
        for ring, angle, points in zip(self._ring, self._angles, self._points):
            lstOut.append("%s %s: %s" % (ring, angle, points))
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
            self._angles = []  # angles are enforced in radians, conversion from degrees or q-space nm-1 are done on the fly
            self._points = []
            self._ring = []


    def append(self, points, angle=None, ring=None):
        """
        @param point: list of points
        @param angle: 2-theta angle in radians
        """
        with self._sem:
            self._angles.append(angle)
            self._points.append(points)
            if ring is None:
                if angle in self.dSpacing:
                    self._ring.append(self.dSpacing.index(angle))
                else:
                    if angle and (angle not in self.dSpacing):
                        self.dSpacing.append(angle)
                    if angle in self.dSpacing:
                        idx = self.dSpacing.index(angle)
                    else:
                        idx = None
                    self._ring.append(idx)
            else:
                self._ring.append(ring)

    def append_2theta_deg(self, points, angle=None, ring=None):
        """
        @param point: list of points
        @param angle: 2-theta angle in degrees
        """
        with self._sem:
            self._angles.append(pi * angle / 180.)
            self._points.append(points)
            if ring is None:
                if angle in self.dSpacing:
                    self._ring.append(self.dSpacing.index(angle))
                else:
                    self.dSpacing.append(angle)
                    self._ring.append(self.dSpacing.index(angle))
            else:
                self._ring.append(ring)

    def pop(self, idx=None):
        """
        Remove the set of points at given index (by default the last)
        @param idx: position of the point to remove
        """
        out = None
        if idx is None:
            with self._sem:
                if self._angles:
                    self._angles.pop()
                    self._ring.pop()
                    out = self._points.pop()
        else:
            with self._sem:
                if idx <= len(self._angles):
                    self._angles.pop(idx)
                    self._ring.pop()
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
                lstOut.append("wavelength: %s" % self._wavelength)
            lstOut.append("dspacing:" + " ".join([str(i) for i in self.dSpacing]))
            for idx, angle, points, ring in zip(range(self.__len__()), self._angles, self._points, self._ring):
                lstOut.append("")
                lstOut.append("New group of points: %i" % idx)
                lstOut.append("2theta: %s" % angle)
                lstOut.append("ring: %s" % ring)
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
        ring = None
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
                elif key == "dspacing":
                    self.dSpacing = []
                    for val in value.split():
                        try:
                            fval = float(val)
                        except Exception:
                            fval = None
                        self.dSpacing.append(fval)
                elif key == "ring":
                    if value.lower() == "none":
                        ring = None
                    else:
                        try:
                            ring = int(value)
                        except:
                            logger.error("ControlPoints.load: unable to convert to int %s (ring)", value)
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
                            self._ring.append(ring)
                            self._points.append(points)
                            tth = None
                            points = []
                else:
                    logger.error("Unknown key: %s", key)
        if len(points) > 0:
            self._angles.append(tth)
            self._points.append(points)
            self._ring.append(ring)

    def load_dSpacing(self, filename):
        """
        Load a d-spacing file containing the inter-reticular plan distance in Angstrom
        """
        if not os.path.isfile(filename):
            logger.error("ControlPoint.load_dSpacing: No such file %s", filename)
            return
        self.dSpacing = list(numpy.loadtxt(filename))
        return self.dSpacing

    def save_dSpacing(self, filename):
        """
        save the d-spacing to a file
        """
        with open(filename) as f:
            for i in self.dSpacing:
                f.write("%s%s" % (i, os.linesep))

    def getList2theta(self):
        """
        Retrieve the list of control points suitable for geometry refinement
        """
        lstOut = []
        for tth, points in zip(self._angles, self._points):
            lstOut += [[pt[0], pt[1], tth] for pt in points]
        return lstOut

    def getListRing(self):
        """
        Retrieve the list of control points suitable for geometry refinement with ring number
        """
        lstOut = []
        for ring, points in zip(self._ring, self._points):
            lstOut += [[pt[0], pt[1], ring] for pt in points]
        return lstOut
    getList = getListRing

    def getWeightedList(self, image):
        """
        Retrieve the list of control points suitable for geometry refinement with ring number and intensities
        @param image:
        @return: a (x,4) array with pos0, pos1, ring nr and intensity
        """
        lstOut = []
        for ring, points in zip(self._ring, self._points):
            lstOut += [[pt[0], pt[1], ring, image[int(pt[0] + 0.5), int(pt[1] + 0.5)]] for pt in points]
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

    def readRingNrFromKeyboard(self):
        """
        Ask the ring number values for the given points
        """
        lastRing = None
        for idx, ring, point in zip(range(self.__len__()), self._ring, self._points):
            bOk = False
            while not bOk:
                if ring is not None:
                    lastRing = ring
                res = raw_input("Point group #%2i (%i points)\t (%6.1f,%6.1f) \t [default=%s] Ring# " % (idx, len(point), point[0][1], point[0][0], lastRing)).strip()
                if res == "":
                    res = lastRing
                try:
                    ring = int(res)
                except (ValueError, TypeError):
                    logging.error("I did not understand the ring number you entered")
                else:
                    if ring >= 0 and ring < len(self.dSpacing):
                        lastRing = ring
                        self._ring[idx] = ring
#                        print ring, self.dSpacing[ring]
                        self._angles[idx] = 2.0 * numpy.arcsin(5e9 * self.wavelength / self.dSpacing[ring])
                        bOk = True


    def setWavelength_change2th(self, value=None):
        with self._sem:
            if value:
                self._wavelength = float(value)
                if self._wavelength < 0 or self._wavelength > 1e-6:
                    logger.warning("This is an unlikely wavelength (in meter): %s" % self._wavelength)
                self._angles = list(2.0 * numpy.arcsin(5e9 * self._wavelength / numpy.array(self.dSpacing)[self._ring]))

    def setWavelength_changeDs(self, value=None):
        """
        This is probably not a good idea, but who knows !
        """
        with self._sem:
            if value :
                self._wavelength = float(value)
                if self._wavelength < 0 or self._wavelength > 1e-6:
                    logger.warning("This is an unlikely wavelength (in meter): %s" % self._wavelength)

                ds = []
                d = 5e9 * self.wavelength / numpy.sin(self.angles / 2.0)
                for i in d:
                    if i not in ds:
                        ds.append(i)
                ds.sort()
                self.dSpacing = ds
                self._ring = [self.dSpacing.index(i) for i in d]




    def setWavelength(self, value=None):
        with self._sem:
            if self._wavelength is None:
                if value:
                    self._wavelength = float(value)
                    if self._wavelength < 0 or self._wavelength > 1e-6:
                        logger.warning("This is an unlikely wavelength (in meter): %s" % self._wavelength)
            else:
                logger.warning("Forbidden to change the wavelength once it is fixed !!!!")
    def getWavelength(self):
        return self._wavelength
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
        self._bilin = Bilinear(self.data)
        self._blured_data = None
        self._median_data = None
        self._labeled_massif = None
        self._number_massif = None
        self._valley_size = None
        self._binned_data = None
        self.binning = None  # Binning is 2-list usually
        self._sem = threading.Semaphore()
        self._sem_label = threading.Semaphore()
        self._sem_binning = threading.Semaphore()
        self._sem_median = threading.Semaphore()


    def nearest_peak(self, x):
        """
        @returns the coordinates of the nearest peak
        """
#        x = numpy.array(x, dtype="float32")
#        out = fmin(self._bilin.f_cy, x, disp=0).round().astype(numpy.int)
        out = self._bilin.local_maxi(x)
        if isinstance(out, tuple):
            res = out
        elif isinstance(out, numpy.ndarray):
            res = tuple(out)
        else:
            res = [int(i) for idx, i in enumerate(out) if 0 <= i < self.data.shape[idx] ]
        if (len(res) != 2) or not((0 <= out[0] < self.data.shape[0]) and (0 <= res[1] < self.data.shape[1])):
            logger.error("in nearest_peak %s -> %s" % (x, out))
            return
        else:
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
            if not region[int(xinit[0] + 0.5), int(xinit[1] + 0.5)]:
                logger.error("Nearest peak %s is not in the same region  %s", xinit, x)
                return listpeaks

            if annotate is not None:
                try:
                    annotate(xinit, x)
                except Exception as error:
                    logger.error("Error in annotate %i: %i %i. %s" , len(listpeaks), xinit[0], xinit[1], error)

        listpeaks.append(xinit)
        mean = self.data[region].mean(dtype=numpy.float64)
        region2 = region * (self.data > mean)
        idx = numpy.vstack(numpy.where(region2)).T
        numpy.random.shuffle(idx)
        nmax = min(nmax, int(ceil(sqrt(idx.shape[0]))))
        if massif_contour is not None:
            try:
                massif_contour(region)
            except (WindowsError, MemoryError) as error:
                logger.error("Error in plotting region: %s", error)
        nbFailure = 0
        for j in idx:
            xopt = self.nearest_peak(j)
            if xopt is None:
                nbFailure += 1
                continue
            if (region2[xopt[0], xopt[1]]) and not (xopt in listpeaks):
                stdout.write("[ %4i, %4i ] --> [ %5.1f, %5.1f ] after %3i iterations %s" % (tuple(j) + tuple(xopt) + (nbFailure, os.linesep)))
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
        new_size = float(size)
        if self._valley_size != new_size:
            self._valley_size = new_size
#            self.getLabeledMassif()
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
                    self._binned_data = utils.binning(self.data, self.binning)
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
                        pattern = [[1] * 3] * 3  # [[0, 1, 0], [1, 1, 1], [0, 1, 0]]#[[1] * 3] * 3
                    logger.debug("Labeling all massifs. This takes some time !!!")
                    labeled_massif, self._number_massif = label((self.getBinnedData() > self.getBluredData()), pattern)
                    logger.info("Labeling found %s massifs." % self._number_massif)
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        fabio.edfimage.edfimage(data=labeled_massif).write("labeled_massif_small.edf")
                    relabeled = utils.relabel(labeled_massif, self.getBinnedData(), self.getBluredData())
                    if logger.getEffectiveLevel() == logging.DEBUG:
                            fabio.edfimage.edfimage(data=relabeled).write("relabeled_massif_small.edf")
                    self._labeled_massif = unBinning(relabeled, self.binning, False)
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        fabio.edfimage.edfimage(data=self._labeled_massif).write("labeled_massif.edf")
                    logger.info("Labeling found %s massifs." % self._number_massif)
        return self._labeled_massif

class Event(object):
    "Dummy class for dumm things"
    def __init__(self, width, height):
        self.width = width
        self.height = height

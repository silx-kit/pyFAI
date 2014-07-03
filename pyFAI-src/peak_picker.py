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
__date__ = "03/07/2014"
__status__ = "production"

import os, sys, threading, logging, gc, types
import operator
from math import sqrt
import numpy
from .gui_utils import pylab, update_fig, maximize_fig, QtGui, backend
import fabio
from .utils import deprecated, percentile
from .reconstruct import reconstruct
from .calibrant import Calibrant, ALL_CALIBRANTS
from .blob_detection import BlobDetection
from .massif import Massif
logger = logging.getLogger("pyFAI.peak_picker")
if os.name != "nt":
    WindowsError = RuntimeError

################################################################################
# Toolbar widget
################################################################################
#class PeakPickerToolbar(backend.NavigationToolbar2QT):
#    def __init__(self, canvas, parent, coordinates=True):
#        backend.NavigationToolbar2QT.__init__(self, canvas, parent, False)
#        self.append_mode = None
#
#
##    def _init_toolbar(self):
##        backend.NavigationToolbar2QT._init_toolbar(self)
##        self.addSeparator()
##        a = self.addAction('+pts', self.on_plus_pts_clicked)
##        a.setToolTip('Add more points to group')
##        a = self.addAction('-pts', self.on_minus_pts_clicked)
##        a.setToolTip('Remove points from group')
#
#    def pan(self):
#        self.append_mode = None
#        backend.NavigationToolbar2QT.pan(self)
#
#    def zoom(self):
#        self.append_mode = None
#        backend.NavigationToolbar2QT.zoom(self)
#
#
#    def on_plus_pts_clicked(self, *args):
#        """
#        callback function
#        """
#        self.append_mode = True
#        print(self.append_mode)
#
#    def on_minus_pts_clicked(self, *args):
#        """
#        callback function
#        """
#        self.append_mode = False
#        print(self.append_mode)


################################################################################
# PeakPicker
################################################################################
class PeakPicker(object):
    """

    This class is in charge of peak picking, i.e. find bragg spots in the image
    Two methods can be used : massif or blob

    """
    VALID_METHODS = ["massif", "blob"]

    def __init__(self, strFilename, reconst=False, mask=None,
                 pointfile=None, calibrant=None, wavelength=None, method="massif"):
        """
        @param strFilename: input image filename
        @param reconst: shall masked part or negative values be reconstructed (wipe out problems with pilatus gaps)
        @param mask: area in which keypoints will not be considered as valid
        @param pointfile:
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
        self.points = ControlPoints(pointfile, calibrant=calibrant, wavelength=wavelength)
        self.fig = None
        self.fig2 = None
        self.fig2sp = None
        self.ax = None
        self.ct = None
        self.msp = None
        self.append_mode = None
        self.spinbox = None
        self.reconstruct = reconst
        self.mask = mask
        self.massif = None  #used for massif detection
        self.blob = None    #used for blob   detection
        self._sem = threading.Semaphore()
#        self._semGui = threading.Semaphore()
        self.mpl_connectId = None
        self.defaultNbPoints = 100
        self._init_thread = None
        if method in self.VALID_METHODS:
            self.method = method
        else:
            logger.error("Not a valid peak-picker method: %s should be part of %s" % (method, self.VALID_METHODS))
            self.method = self.VALID_METHODS[0]

        if self.method == "massif":
            self.init_massif(False)
        elif self.method == "blob":
            self.init_blob(False)

    def init(self, method, sync=True):
        """
        Unified initializer
        """
        assert method in ["blob", "massif"]
        if method != self.method:
            self.__getattribute__("init_" + method)(sync)

    def sync_init(self):
        if self._init_thread:
            self._init_thread.join()



    def init_massif(self, sync=True):
        """
        Initialize PeakPicker for massif based detection
        """
        if self.reconstruct:
            if self.mask is None:
                self.mask = self.data < 0
            data = reconstruct(self.data, self.mask)
        else:
            data = self.data
        self.massif = Massif(data)
        self._init_thread = threading.Thread(target=self.massif.getLabeledMassif, name="massif_process")
        self._init_thread.start()
        self.method = "massif"
        if sync:
            self._init_thread.join()


    def init_blob(self, sync=True):
        """
        Initialize PeakPicker for blob based detection
        """
        if self.mask is not None:
            self.blob = BlobDetection(self.data, mask=self.mask)
        else:
            self.blob = BlobDetection(self.data, mask=(self.data < 0))
        self.method = "blob"
        self._init_thread = threading.Thread(target=self.blob.process, name="blob_process")
        self._init_thread.start()
        if sync:
            self._init_thread.join()

    def peaks_from_area(self, mask, Imin, keep=1000, refine=True, method=None):
        """
        Return the list of peaks within an area

        @param mask: 2d array with mask.
        @param Imin: minimum of intensity above the background to keep the point
        @param keep: maximum number of points to keep
        @param method: enforce the use of detection using "massif" or "blob"
        @return: list of peaks [y,x], [y,x], ...]
        """
        if not method:
            method = self.method
        else:
            self.init(method, True)

        obj = self.__getattribute__(method)

        return obj.peaks_from_area(mask, Imin=Imin, keep=keep, refine=refine)

    def reset(self):
        """
        Reset control point and graph (if needed)
        """
        self.points.reset()
        if self.fig and self.ax:
            #empty annotation and plots
            if len(self.ax.texts) > 0:
                self.ax.texts = []
            if len(self.ax.lines) > 0:
                self.ax.lines = []
            #Redraw the image
            self.fig.show()
            update_fig(self.fig)

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
#            self.ax.format_coord = self.format_coord
            #Add widget to the toolbar
            toolbar = self.fig.canvas.toolbar
            toolbar.addSeparator()
#            a = toolbar.addAction('+pts', self.on_plus_pts_clicked)
#            a.setToolTip('Add more points to group')
#            a = toolbar.addAction('-pts', self.on_minus_pts_clicked)
#            a.setToolTip('Remove points from group')
            label = QtGui.QLabel("ring ", toolbar)
#            self.pix_coords_label = QLabel("Pixel coordinates and intensity : x = None , y = None , i = None ", self)
            toolbar.addWidget(label)
            self.spinbox = QtGui.QSpinBox(toolbar)
            self.spinbox.setMinimum(0)
            toolbar.addWidget(self.spinbox)


            self.fig.show()
#            self.fig.canvas.toolbar.show()

        if log:
            showData = numpy.log1p(self.data - self.data.min())
            self.ax.set_title('Log colour scale (skipping lowest/highest per mille)')
        else:
            showData = self.data
            self.ax.set_title('Linear colour scale (skipping lowest/highest per mille)')

        # skip lowest and highest per mille of image values via vmin/vmax
        showMin = percentile(showData, .1)
        showMax = percentile(showData, 99.9)
        im = self.ax.imshow(showData, vmin=showMin, vmax=showMax, origin="lower", interpolation="nearest")

        self.ax.autoscale_view(False, False, False)
        self.fig.colorbar(im)
        update_fig(self.fig)
        if maximize:
            maximize_fig(self.fig)
        self.mpl_connectId = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def load(self, filename):
        """
        load a filename and plot data on the screen (if GUI)
        """
        self.points.load(filename)
        self.display_points()

    def display_points(self, minIndex=0):
        """
        display all points and their ring annotations
        @param minIndex: ring index to start with
        """
        if self.ax is not None:
            for idx, points in enumerate(self.points._points):
                if idx < minIndex:
                    continue
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
                update_fig(self.fig)

        with self._sem:
            x0 = event.xdata
            y0 = event.ydata
            if event.button == 3:  # right click: add points (1 or many) to new or existing group
                logger.debug("Button: %i, Key modifier: %s" % (event.button, event.key))
                if event.key == 'shift':  # if 'shift' pressed add nearest maximum to the current group
                    points = self.points.pop() or []
                    # no, keep annotation! if len(self.ax.texts) > 0: self.ax.texts.pop()
                    if len(self.ax.lines) > 0:
                        self.ax.lines.pop()

                    update_fig(self.fig)
                    newpeak = self.massif.nearest_peak([y0, x0])
                    if newpeak:
                        if not points:
                            # if new group, need annotation (before points.append!)
                            annontate(newpeak, [y0, x0])
                        points.append(newpeak)
                        logger.info("x=%5.1f, y=%5.1f added to group #%i" % (newpeak[1], newpeak[0], len(self.points)))
                    else:
                        logger.warning("No peak found !!!")

                elif event.key == 'control':  # if 'control' pressed add nearest maximum to a new group
                    points = []
                    newpeak = self.massif.nearest_peak([y0, x0])
                    if newpeak:
                        points.append(newpeak)
                        annontate(newpeak, [y0, x0])
                        logger.info("Create group #%i with single point x=%5.1f, y=%5.1f" % (len(self.points), newpeak[1], newpeak[0]))
                    else:
                        logger.warning("No peak found !!!")
                elif event.key == 'm':  # if 'm' pressed add new group to current  group ...  ?
                    points = self.points.pop() or []
                    # no, keep annotation! if len(self.ax.texts) > 0: self.ax.texts.pop()
                    if len(self.ax.lines) > 0:
                        self.ax.lines.pop()
                    update_fig(self.fig)
                    # need to annotate only if a new group:
                    localAnn = None if points else annontate
                    listpeak = self.massif.find_peaks([y0, x0], self.defaultNbPoints, localAnn, self.massif_contour)
                    if len(listpeak) == 0:
                        logger.warning("No peak found !!!")
                    else:
                        points += listpeak
                        logger.info("Added %i points to group #%i (now %i points)" % (len(listpeak), len(self.points), len(points)))
                else:  # create new group
                    points = self.massif.find_peaks([y0, x0], self.defaultNbPoints, annontate, self.massif_contour)
                    if not points:
                        logger.warning("No peak found !!!")
                    else:
                        logger.info("Created group #%i with %i points" % (len(self.points), len(points)))
                if not points:
                    return
                self.points.append(points)
                npl = numpy.array(points)
                self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)
                update_fig(self.fig)
                sys.stdout.flush()
            elif event.button == 2:  # center click: remove 1 or all points from current group
                logger.debug("Button: %i, Key modifier: %s" % (event.button, event.key))
                poped_points = self.points.pop() or []
                # in case not the full group is removed, would like to keep annotation
                # _except_ if the annotation is close to the removed point... too complicated!
                if len(self.ax.texts) > 0:
                    self.ax.texts.pop()
                if len(self.ax.lines) > 0:
                    self.ax.lines.pop()
                if event.key == '1' and len(poped_points) > 1:  # if '1' pressed AND > 1 point left:
                    # delete single closest point from current group
                    dists = [sqrt((p[1] - x0) ** 2 + (p[0] - y0) ** 2) for p in poped_points]  # p[1],p[0]!
                    # index and distance of smallest distance:
                    indexMin = min(enumerate(dists), key=operator.itemgetter(1))
                    removedPt = poped_points.pop(indexMin[0])
                    logger.info("x=%5.1f, y=%5.1f removed from group #%i (%i points left)" % (removedPt[1], removedPt[0], len(self.points), len(poped_points)))
                    # annotate (new?) 1st point and add remaining points back
                    pt = (poped_points[0][0], poped_points[0][1])
                    annontate(pt, (pt[0] + 10, pt[1] + 10))
                    self.points.append(poped_points)
                    npl = numpy.array(poped_points)
                    self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)
                elif len(poped_points) > 0:  # not '1' pressed or only 1 point left: remove complete group
                    logger.info("Removing group #%i containing %i points" % (len(self.points), len(poped_points)))
                else:
                    logger.info("No groups to remove")

#                self.fig.show()
                update_fig(self.fig)
                sys.stdout.flush()

    def finish(self, filename=None,):
        """
        Ask the ring number for the given points

        @param filename: file with the point coordinates saved
        """
        logging.info(os.linesep.join(["Please use the GUI and:",
                                      " 1) Right-click:         try an auto find for a ring",
                                      " 2) Right-click + Ctrl:  create new group with one point",
                                      " 3) Right-click + Shift: add one point to current group",
                                      " 4) Right-click + m:     find more points for current group",
                                      " 5) Center-click:     erase current group",
                                      " 6) Center-click + 1: erase closest point from current group"]))

        raw_input("Please press enter when you are happy with your selection" + os.linesep)
        # need to disconnect 'button_press_event':
        self.fig.canvas.mpl_disconnect(self.mpl_connectId)
        self.mpl_connectId = None
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

            if self.points.calibrant:
                angles = [ i for i in self.points.calibrant.get_2th()
                          if i is not None]
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
            update_fig(self.fig)

    def massif_contour(self, data):
        """
        Overlays a mask over a diffraction image

        @param data: mask to be overlaid
        """

        if self.fig is None:
            logging.error("No diffraction image available => not showing the contour")
        else:
            tmp = 100 * numpy.logical_not(data)
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
            update_fig(self.fig)

    def closeGUI(self):
        if self.fig is not None:
            self.fig.clear()
            self.fig = None
            gc.collect()


#    def format_coord(self, x, y):
#        """
#        Print coordinated in matplotlib toolbar
#        """
#        col = int(x + 0.5)
#        row = int(y + 0.5)
#        if col >= 0 and col < self.shape[1] and row >= 0 and row < self.shape[0]:
#            z = self.data[row, col]
#            return 'x=%.2f \t y=%.2f \t I=%1.4f' % (x, y, z)
#        else:
#            return 'x=%.2f \t y=%.2f \t I=None' % (x, y)

    def on_plus_pts_clicked(self, *args):
        """
        callback function
        """
        self.append_mode = True
        print(self.append_mode)

    def on_minus_pts_clicked(self, *args):
        """
        callback function
        """
        self.append_mode = False
        print(self.append_mode)

################################################################################
# ControlPoints
################################################################################
class ControlPoints(object):
    """
    This class contains a set of control points with (optionally) their ring number hence d-spacing and diffraction  2Theta angle ...
    """
    def __init__(self, filename=None, calibrant=None, wavelength=None):
#        self.dSpacing = []
        self._sem = threading.Semaphore()
        self._angles = []  # angles are enforced in radians, conversion from degrees or q-space nm-1 are done on the fly
        self._points = []
        self._ring = []  # ring number ...
        self.calibrant = Calibrant(wavelength=wavelength)
        if filename is not None:
            self.load(filename)
        have_spacing = False
        for i in self.dSpacing :
            have_spacing = have_spacing or i
        if (not have_spacing) and (calibrant is not None):
            if isinstance(calibrant, Calibrant):
                self.calibrant = calibrant
            elif type(calibrant) in types.StringTypes:
                if calibrant in ALL_CALIBRANTS:
                    self.calibrant = ALL_CALIBRANTS[calibrant]
                elif os.path.isfile(calibrant):
                    self.calibrant = Calibrant(calibrant)
                else:
                    logger.error("Unable to handle such calibrant: %s" % calibrant)
            elif isinstance(dSpacing, (numpy.ndarray, list, tuple, array)):
                self.calibrant = Calibrant(dSpacing=list(calibrant))
            else:
                logger.error("Unable to handle such calibrant: %s" % calibrant)
        if not self.calibrant.wavelength:
            self.calibrant.set_wavelength(wavelength)


    def __repr__(self):
        self.check()
        lstOut = ["ControlPoints instance containing %i group of point:" % len(self)]
        if self.calibrant:
            lstOut.append(self.calibrant.__repr__())
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
#            self.calibrant = Calibrant()
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
                if angle in self.calibrant.get_2th():
                    self._ring.append(self.calibrant.get_2th().index(angle))
                else:
                    if angle and (angle not in self.calibrant.get_2th()):
                        self.calibrant.append_2th(angle)
                        self.rings = [self.calibrant.get_2th_index(a) for a in self._angles]
                    else:
                        self._ring.append(None)
            else:
                self._ring.append(ring)

    def append_2theta_deg(self, points, angle=None, ring=None):
        """
        @param point: list of points
        @param angle: 2-theta angle in degrees
        """
        if angle:
            self.append(points, numpy.deg2rad(angle), ring)
        else:
            self.append(points, None, ring)

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

            if self.calibrant.wavelength is not None:
                lstOut.append("wavelength: %s" % self.calibrant.wavelength)
            lstOut.append("dspacing:" + " ".join([str(i) for i in self.calibrant.dSpacing]))
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
                        self.calibrant.set_wavelength(float(value))
                    except Exception as error:
                        logger.error("ControlPoints.load: unable to convert to float %s (wavelength): %s", value, error)
                elif key == "2theta":
                    if value.lower() == "none":
                        tth = None
                    else:
                        try:
                            tth = float(value)
                        except Exception as error:
                            logger.error("ControlPoints.load: unable to convert to float %s (2theta): %s", value, error)
                elif key == "dspacing":
                    self.dSpacing = []
                    for val in value.split():
                        try:
                            fval = float(val)
                        except Exception:
                            fval = None
                        self.calibrant.append_dSpacing(fval)
                elif key == "ring":
                    if value.lower() == "none":
                        ring = None
                    else:
                        try:
                            ring = int(value)
                        except Exception as error:
                            logger.error("ControlPoints.load: unable to convert to int %s (ring): %s", value, error)
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
                        except Exception as error:
                            logger.error("ControlPoints.load: unable to convert to float %s (point)", value, error)
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

    @deprecated
    def load_dSpacing(self, filename):
        """
        Load a d-spacing file containing the inter-reticular plan distance in Angstrom

        DEPRECATED: use a calibrant object
        """
        self.calibrant.load_file(filename)
        return self.calibrant.dSpacing
    @deprecated
    def save_dSpacing(self, filename):
        """
        save the d-spacing to a file

        DEPRECATED: use a calibrant object
        """
        self.calibrant.save_dSpacing(filename)

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
                defaultRing = 0
                if ring is not None:
                    defaultRing = ring
                elif lastRing is not None:
                    defaultRing = lastRing + 1
                res = raw_input("Point group #%2i (%i points)\t (%6.1f,%6.1f) \t [default=%s] Ring# " % (idx, len(point), point[0][1], point[0][0], defaultRing)).strip()
                if res == "":
                    res = defaultRing
                try:
                    inputRing = int(res)
                except (ValueError, TypeError):
                    logging.error("I did not understand the ring number you entered")
                else:
                    if inputRing >= 0 and inputRing < len(self.calibrant.dSpacing):
                        lastRing = ring
                        self._ring[idx] = inputRing
                        self._angles[idx] = self.calibrant.get_2th()[inputRing]
                        bOk = True
                    else:
                        logging.error("Invalid ring number %i (range 0 -> %2i)" % (inputRing, len(self.calibrant.dSpacing) - 1))



    def setWavelength_change2th(self, value=None):
        with self._sem:
            if self.calibrant is None:
                self.calibrant = Calibrant()
            self.calibrant.setWavelength_change2th(value)
            self._angles = [self.calibrant.get_2th()[i] for i in self._ring]

    def setWavelength_changeDs(self, value=None):
        """
        This is probably not a good idea, but who knows !
        """
        with self._sem:
            if value :
                if self.calibrant is None:
                    self.calibrant = Calibrant()
                self.calibrant.setWavelength_changeDs(value)

    def set_wavelength(self, value=None):
        with self._sem:
            if value:
                self.calibrant.set_wavelength(value)

    def get_wavelength(self):
        return self.calibrant._wavelength
    wavelength = property(get_wavelength, set_wavelength)

    def get_dSpacing(self):
        if self.calibrant:
            return self.calibrant.dSpacing
        else:
            return []

    def set_dSpacing(self, lst):
        if not self.calibrant:
            self.calibrant = Calibrant()
        self.calibrant.dSpacing = lst
    dSpacing = property(get_dSpacing, set_dSpacing)



class Event(object):
    "Dummy class for dummy things"
    def __init__(self, width, height):
        self.width = width
        self.height = height

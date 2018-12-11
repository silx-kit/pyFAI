#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

"""Semi-graphical tool for peak-picking and extracting visually control points
from an image with Debye-Scherer rings"""

from __future__ import print_function, absolute_import

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/10/2018"
__status__ = "production"

import os
import sys
import threading
import logging
import gc
import operator
import numpy
import fabio

logger = logging.getLogger(__name__)

try:
    from silx.gui import qt
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    qt = None

if qt is not None:
    from .utils import update_fig, maximize_fig
    from .matplotlib import matplotlib, pyplot, pylab
    from . import utils as gui_utils

from ..control_points import ControlPoints
from ..calibrant import CALIBRANT_FACTORY
from ..blob_detection import BlobDetection
from ..massif import Massif
from ..ext.reconstruct import reconstruct
from ..ext.watershed import InverseWatershed
from ..third_party import six


class PeakPicker(object):
    """
    This class is in charge of peak picking, i.e. find bragg spots in the image
    Two methods can be used : massif or blob
    """

    VALID_METHODS = ["massif", "blob", "watershed"]

    help = ["Please select rings on the diffraction image. In parenthesis, some modified shortcuts for single button mouse (Apple):",
            " * Right-click (click+n):         try an auto find for a ring",
            " * Right-click + Ctrl (click+b):  create new group with one point",
            " * Right-click + Shift (click+v): add one point to current group",
            " * Right-click + m (click+m):     find more points for current group",
            " * Center-click or (click+d):     erase current group",
            " * Center-click + 1 or (click+1): erase closest point from current group"]

    def __init__(self, data, reconst=False, mask=None,
                 pointfile=None, calibrant=None, wavelength=None, detector=None,
                 method="massif"):
        """
        :param data: input image as numpy array
        :param reconst: shall masked part or negative values be reconstructed (wipe out problems with pilatus gaps)
        :param mask: area in which keypoints will not be considered as valid
        :param pointfile:
        """
        if isinstance(data, six.string_types):
            self.data = fabio.open(data).data.astype("float32")
        else:
            self.data = numpy.ascontiguousarray(data, numpy.float32)
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
        self.refine_btn = None
        self.ref_action = None
        self.sb_action = None
        self.reconstruct = reconst
        self.detector = detector
        self.mask = mask
        self.massif = None  # used for massif detection
        self.blob = None  # used for blob   detection
        self.watershed = None  # used for inverse watershed
        self._sem = threading.Semaphore()
        self.mpl_connectId = None
        self.defaultNbPoints = 100
        self._init_thread = None
        self.point_filename = None
        self.callback = None
        self.method = None
        if method not in self.VALID_METHODS:
            logger.error("Not a valid peak-picker method: %s should be part of %s", method, self.VALID_METHODS)
            method = self.VALID_METHODS[0]
        self.init(method, False)

    def init(self, method, sync=True):
        """
        Unified initializer
        """
        assert method in self.VALID_METHODS
        if method != self.method:
            self.__getattribute__("_init_" + method)(sync)
            self.method = method

    def sync_init(self):
        if self._init_thread:
            self._init_thread.join()

    def _init_massif(self, sync=True):
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
        self._init_thread = threading.Thread(target=self.massif.get_labeled_massif, name="massif_process")
        self._init_thread.start()
        if sync:
            self._init_thread.join()

    def _init_blob(self, sync=True):
        """
        Initialize PeakPicker for blob based detection
        """
        if self.mask is not None:
            self.blob = BlobDetection(self.data, mask=self.mask)
        else:
            self.blob = BlobDetection(self.data, mask=(self.data < 0))
        self._init_thread = threading.Thread(target=self.blob.process, name="blob_process")
        self._init_thread.start()
        if sync:
            self._init_thread.join()

    def _init_watershed(self, sync=True):
        """
        Initialize PeakPicker for watershed based detection
        """
        self.watershed = InverseWatershed(self.data)
        self._init_thread = threading.Thread(target=self.watershed.init, name="iw_init")
        self._init_thread.start()
        if sync:
            self._init_thread.join()

    def peaks_from_area(self, **kwargs):
        """
        Return the list of peaks within an area

        :param mask: 2d array with mask.
        :param Imin: minimum of intensity above the background to keep the point
        :param keep: maximum number of points to keep
        :param method: enforce the use of detection using "massif" or "blob" or "watershed"
        :param ring: ring number to which assign the  points
        :param dmin: minimum distance between two peaks (in pixels)
        :param seed: good starting points.
        :return: list of peaks [y,x], [y,x], ...]
        """
        method = kwargs.get("method")
        ring = kwargs.get("ring", 0)
        if not method:
            method = self.method
        else:
            self.init(method, True)

        obj = self.__getattribute__(method)

        points = obj.peaks_from_area(**kwargs)
        if points:
            gpt = self.points.append(points, ring)
            if self.fig:
                npl = numpy.array(points)
                gpt.plot = self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)
                pt0x = gpt.points[0][1]
                pt0y = gpt.points[0][0]
                gpt.annotate = self.ax.annotate(gpt.label, xy=(pt0x, pt0y), xytext=(pt0x + 10, pt0y + 10),
                                                weight="bold", size="large", color="black",
                                                arrowprops=dict(facecolor='white', edgecolor='white'))
                update_fig(self.fig)
        return points

    def reset(self):
        """
        Reset control point and graph (if needed)
        """
        self.points.reset()
        if self.fig and self.ax:
            # empty annotate and plots
            if len(self.ax.texts) > 0:
                self.ax.texts = []
            if len(self.ax.lines) > 0:
                self.ax.lines = []
            # Redraw the image
            if not gui_utils.main_loop:
                self.fig.show()
            update_fig(self.fig)

    def gui(self, log=False, maximize=False, pick=True):
        """
        :param log: show z in log scale
        """
        if self.fig is None:
            self.fig = pyplot.figure()
            self.fig.subplots_adjust(right=0.75)
            # add 3 subplots at the same position for debye-sherrer image, contour-plot and massif contour
            self.ax = self.fig.add_subplot(111)
            self.ct = self.fig.add_subplot(111)
            self.msp = self.fig.add_subplot(111)
            toolbar = self.fig.canvas.toolbar
            toolbar.addSeparator()

            a = toolbar.addAction('Opts', self.on_option_clicked)
            a.setToolTip('open options window')
            if pick:
                label = qt.QLabel("Ring #", toolbar)
                toolbar.addWidget(label)
                self.spinbox = qt.QSpinBox(toolbar)
                self.spinbox.setMinimum(0)
                self.sb_action = toolbar.addWidget(self.spinbox)
                a = toolbar.addAction('Refine', self.on_refine_clicked)
                a.setToolTip('switch to refinement mode')
                self.ref_action = a
                self.mpl_connectId = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        if log:
            data_disp = numpy.log1p(self.data - self.data.min())
            txt = 'Log colour scale (skipping lowest/highest per mille)'
        else:
            data_disp = self.data
            txt = 'Linear colour scale (skipping lowest/highest per mille)'

        # skip lowest and highest per mille of image values via vmin/vmax
        sorted_list = data_disp.flatten()  # explicit copy
        sorted_list.sort()
        show_min = sorted_list[int(round(1e-3 * (sorted_list.size - 1)))]
        show_max = sorted_list[int(round(0.999 * (sorted_list.size - 1)))]
        im = self.ax.imshow(data_disp, vmin=show_min, vmax=show_max,
                            origin="lower", interpolation="nearest",
                            )
        self.ax.set_ylabel('y in pixels')
        self.ax.set_xlabel('x in pixels')

        if self.detector:
            s1, s2 = self.data.shape
            s1 -= 1
            s2 -= 1
            self.ax.set_xlim(0, s2)
            self.ax.set_ylim(0, s1)
            d1 = numpy.array([0, s1, s1, 0])
            d2 = numpy.array([0, 0, s2, s2])
            p1, p2, _ = self.detector.calc_cartesian_positions(d1=d1, d2=d2)
            ax = self.fig.add_subplot(1, 1, 1,
                                      xbound=False,
                                      ybound=False,
                                      xlabel=r'dim2 ($\approx m$)',
                                      ylabel=r'dim1 ($\approx m$)',
                                      xlim=(p2.min(), p2.max()),
                                      ylim=(p1.min(), p1.max()),
                                      aspect='equal',
                                      zorder=-1)
            ax.xaxis.set_label_position('top')
            ax.yaxis.set_label_position('right')
            ax.yaxis.label.set_color('blue')
            ax.xaxis.label.set_color('blue')
            ax.tick_params(colors="blue", labelbottom='off', labeltop='on',
                           labelleft='off', labelright='on')
            # ax.autoscale_view(False, False, False)

        else:
            _cbar = self.fig.colorbar(im, label=txt)
        # self.ax.autoscale_view(False, False, False)
        update_fig(self.fig)
        if maximize:
            maximize_fig(self.fig)
        if not gui_utils.main_loop:
            self.fig.show()

    def load(self, filename):
        """
        load a filename and plot data on the screen (if GUI)
        """
        self.points.load(filename)
        self.display_points()

    def display_points(self, minIndex=0, reset=False):
        """
        display all points and their ring annotations
        :param minIndex: ring index to start with
        :param reset: remove all point before re-displaying them
        """
        if self.ax is not None:
            if reset:
                self.ax.texts = []
                self.ax.lines = []

            for _lbl, gpt in self.points._groups.items():
                idx = gpt.ring
                if idx < minIndex:
                    continue
                if len(gpt) > 0:
                    pt0x = gpt.points[0][1]
                    pt0y = gpt.points[0][0]
                    gpt.annotate = self.ax.annotate(gpt.label, xy=(pt0x, pt0y), xytext=(pt0x + 10, pt0y + 10),
                                                    weight="bold", size="large", color="black",
                                                    arrowprops=dict(facecolor='white', edgecolor='white'))
                    npl = numpy.array(gpt.points)
                    gpt.plot = self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)

    def remove_grp(self, lbl):
        """
        remove a group of points

        :param lbl: label of the group of points
        """
        gpt = self.points.pop(lbl=lbl)
        if gpt and self.ax:
            print(gpt.annotate)
            if gpt.annotate in self.ax.texts:
                self.ax.texts.remove(gpt.annotate)
            for plot in gpt.plot:
                    if plot in self.ax.lines:
                        self.ax.lines.remove(plot)
            update_fig(self.fig)

    def onclick(self, event):
        """
        Called when a mouse is clicked
        """
        def annontate(x, x0=None, idx=None, gpt=None):
            """
            Call back method to annotate the figure while calculation are going on ...
            :param x: coordinates
            :param x0: coordinates of the starting point
            :param gpt: group of point, instance of PointGroup
            TODO
            """
            if x0 is None:
                annot = self.ax.annotate(".", xy=(x[1], x[0]), weight="bold", size="large", color="black")
            else:
                if gpt:
                    annot = self.ax.annotate(gpt.label, xy=(x[1], x[0]), xytext=(x0[1], x0[0]),
                                             weight="bold", size="large", color="black",
                                             arrowprops=dict(facecolor='white', edgecolor='white'),)
                    gpt.annotate = annot
                else:
                    annot = self.ax.annotate("%i" % (len(self.points)), xy=(x[1], x[0]), xytext=(x0[1], x0[0]),
                                             weight="bold", size="large", color="black",
                                             arrowprops=dict(facecolor='white', edgecolor='white'),)
                update_fig(self.fig)
            return annot

        def common_creation(points, gpt=None):
            """
            plot new set of points

            :param points: list of points
            :param gpt: : group of point, instance of PointGroup
            """
            if points:
                if not gpt:
                    gpt = self.points.append(points, ring=self.spinbox.value())
                npl = numpy.array(points)
                gpt.plot = self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)
                update_fig(self.fig)
            sys.stdout.flush()
            return gpt

        def new_grp(event):
            " * new_grp Right-click (click+n):         try an auto find for a ring"
            # ydata is a float, and matplotlib display pixels centered.
            # we use floor (int cast) instead of round to avoid use of
            # banker's rounding
            ypix, xpix = int(event.ydata + 0.5), int(event.xdata + 0.5)
            points = self.massif.find_peaks([ypix, xpix],
                                            self.defaultNbPoints,
                                            None, self.massif_contour)
            if points:
                gpt = common_creation(points)
                annontate(points[0], [ypix, xpix], gpt=gpt)
                logger.info("Created group #%2s with %i points", gpt.label, len(gpt))
            else:
                logger.warning("No peak found !!!")

        def single_point(event):
            " * Right-click + Ctrl (click+b):  create new group with one single point"
            ypix, xpix = int(event.ydata + 0.5), int(event.xdata + 0.5)
            newpeak = self.massif.nearest_peak([ypix, xpix])
            if newpeak:
                gpt = common_creation([newpeak])
                annontate(newpeak, [ypix, xpix], gpt=gpt)
                logger.info("Create group #%2s with single point x=%5.1f, y=%5.1f", gpt.label, newpeak[1], newpeak[0])
            else:
                logger.warning("No peak found !!!")

        def append_more_points(event):
            " * Right-click + m (click+m):     find more points for current group"
            gpt = self.points.get(self.spinbox.value())
            if not gpt:
                new_grp(event)
                return
            if gpt.plot:
                if gpt.plot[0] in self.ax.lines:
                    self.ax.lines.remove(gpt.plot[0])

            update_fig(self.fig)
            # matplotlib coord to pixel coord, avoinding use of banker's round
            ypix, xpix = int(event.ydata + 0.5), int(event.xdata + 0.5)
            # need to annotate only if a new group:
            listpeak = self.massif.find_peaks([ypix, xpix],
                                              self.defaultNbPoints, None,
                                              self.massif_contour)
            if listpeak:
                gpt.points += listpeak
                logger.info("Added %i points to group #%2s (now %i points)", len(listpeak), len(gpt.label), len(gpt))
            else:
                logger.warning("No peak found !!!")
            common_creation(gpt.points, gpt)

        def append_1_point(event):
            " * Right-click + Shift (click+v): add one point to current group"
            gpt = self.points.get(self.spinbox.value())
            if not gpt:
                new_grp(event)
                return
            if gpt.plot:
                if gpt.plot[0] in self.ax.lines:
                    self.ax.lines.remove(gpt.plot[0])
            update_fig(self.fig)
            # matplotlib coord to pixel coord, avoinding use of banker's round
            ypix, xpix = int(event.ydata + 0.5), int(event.xdata + 0.5)
            newpeak = self.massif.nearest_peak([ypix, xpix])
            if newpeak:
                gpt.points.append(newpeak)
                logger.info("x=%5.1f, y=%5.1f added to group #%2s", newpeak[1], newpeak[0], gpt.label)
            else:
                logger.warning("No peak found !!!")
            common_creation(gpt.points, gpt)

        def erase_grp(event):
            " * Center-click or (click+d):     erase current group"
            ring = self.spinbox.value()
            gpt = self.points.pop(ring)
            if not gpt:
                logger.warning("No group of points for ring %s", ring)
                return
            # print("Remove group from ring %s label %s" % (ring, gpt.label))
            if gpt.annotate:
                if gpt.annotate in self.ax.texts:
                    self.ax.texts.remove(gpt.annotate)
            if gpt.plot:
                if gpt.plot[0] in self.ax.lines:
                    self.ax.lines.remove(gpt.plot[0])
            if len(gpt) > 0:
                logger.info("Removing group #%2s containing %i points", gpt.label, len(gpt))
            else:
                logger.info("No groups to remove")
            update_fig(self.fig)
            sys.stdout.flush()

        def erase_1_point(event):
            " * Center-click + 1 or (click+1): erase closest point from current group"
            ring = self.spinbox.value()
            gpt = self.points.get(ring)
            if not gpt:
                logger.warning("No group of points for ring %s", ring)
                return
            # print("Remove 1 point from group from ring %s label %s" % (ring, gpt.label))
            if gpt.annotate:
                if gpt.annotate in self.ax.texts:
                    self.ax.texts.remove(gpt.annotate)
            if gpt.plot:
                if gpt.plot[0] in self.ax.lines:
                    self.ax.lines.remove(gpt.plot[0])
            if len(gpt) > 1:
                # delete single closest point from current group
                # matplotlib coord to pixel coord, avoinding use of banker's round
                y0, x0 = int(event.ydata + 0.5), int(event.xdata + 0.5)
                distsq = [((p[1] - x0) ** 2 + (p[0] - y0) ** 2) for p in gpt.points]
                # index and distance of smallest distance:
                indexMin = min(enumerate(distsq), key=operator.itemgetter(1))
                removedPt = gpt.points.pop(indexMin[0])
                logger.info("x=%5.1f, y=%5.1f removed from group #%2s (%i points left)", removedPt[1], removedPt[0], gpt.label, len(gpt))
                # annotate (new?) 1st point and add remaining points back
                pt = (gpt.points[0][0], gpt.points[0][1])
                gpt.annotate = annontate(pt, (pt[0] + 10, pt[1] + 10))
                npl = numpy.array(gpt.points)
                gpt.plot = self.ax.plot(npl[:, 1], npl[:, 0], "o", scalex=False, scaley=False)
            elif len(gpt) == 1:
                logger.info("Removing group #%2s containing 1 point", gpt.label)
                gpt = self.points.pop(ring)
            else:
                logger.info("No groups to remove")
            update_fig(self.fig)
            sys.stdout.flush()

        with self._sem:
            logger.debug("Button: %i, Key modifier: %s", event.button, event.key)

            if ((event.button == 3) and (event.key == 'shift')) or \
               ((event.button == 1) and (event.key == 'v')):
                # if 'shift' pressed add nearest maximum to the current group
                append_1_point(event)
            elif ((event.button == 3) and (event.key == 'control')) or\
                 ((event.button == 1) and (event.key == 'b')):
                # if 'control' pressed add nearest maximum to a new group
                single_point(event)
            elif (event.button in [1, 3]) and (event.key == 'm'):
                append_more_points(event)
            elif (event.button == 3) or ((event.button == 1) and (event.key == 'n')):
                # create new group
                new_grp(event)

            elif (event.key == "1") and (event.button in [1, 2]):
                erase_1_point(event)
            elif (event.button == 2) or (event.button == 1 and event.key == "d"):
                erase_grp(event)
            else:
                logger.info("Unknown combination: Button: %i, Key modifier: %s", event.button, event.key)

    def finish(self, filename=None, callback=None):
        """
        Ask the ring number for the given points

        :param filename: file with the point coordinates saved
        """
        logging.info(os.linesep.join(self.help))
        if not callback:
            if not self.points.calibrant.dSpacing:
                logger.error("Calibrant has no line ! check input parameters please, especially the '-c' option")
                print(CALIBRANT_FACTORY)
                raise RuntimeError("Invalid calibrant")
            six.moves.input("Please press enter when you are happy with your selection" + os.linesep)
            # need to disconnect 'button_press_event':
            self.fig.canvas.mpl_disconnect(self.mpl_connectId)
            self.mpl_connectId = None
            print("Now fill in the ring number. Ring number starts at 0, like point-groups.")
            self.points.readRingNrFromKeyboard()
            if filename is not None:
                self.points.save(filename)
            return self.points.getWeightedList(self.data)
        else:
            self.point_filename = filename
            self.callback = callback
            gui_utils.main_loop = True
            # MAIN LOOP
            pylab.show()

    def contour(self, data, cmap="autumn", linewidths=2, linestyles="dashed"):
        """
        Overlay a contour-plot

        :param data: 2darray with the 2theta values in radians...
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

            tth_max = data.max()
            tth_min = data.min()
            if self.points.calibrant:
                angles = [i for i in self.points.calibrant.get_2th()
                          if (i is not None) and (i >= tth_min) and (i <= tth_max)]
                if not angles:
                    angles = None
            else:
                angles = None
            try:
                xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
                if not isinstance(cmap, matplotlib.colors.Colormap):
                    cmap = matplotlib.cm.get_cmap(cmap)
                self.ct.contour(data, levels=angles, cmap=cmap, linewidths=linewidths, linestyles=linestyles)
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
                print("Visually check that the overlaid dashed curve on the Debye-Sherrer rings of the image")
                print("Check also for correct indexing of rings")
            except MemoryError:
                logging.error("Sorry but your computer does NOT have enough memory to display the 2-theta contour plot")
            except ValueError:
                logging.error("No contour-plot to display !")
            update_fig(self.fig)

    def massif_contour(self, data):
        """
        Overlays a mask over a diffraction image

        :param data: mask to be overlaid
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
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            except MemoryError:
                logging.error("Sorry but your computer does NOT have enough memory to display the massif plot")
            update_fig(self.fig)

    def closeGUI(self):
        if self.fig is not None:
            self.fig.clear()
            self.fig = None
            gc.collect()

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

    def on_option_clicked(self, *args):
        """
        callback function
        """
        print("Option!")

    def on_refine_clicked(self, *args):
        """
        callback function
        """
        print("refine, now!")
        self.sb_action.setDisabled(True)
        self.ref_action.setDisabled(True)
        self.spinbox.setEnabled(False)
        self.mpl_connectId = None
        self.fig.canvas.mpl_disconnect(self.mpl_connectId)
        pylab.ion()
        if self.point_filename:
            self.points.save(self.point_filename)
        if self.callback:
            self.callback(self.points.getWeightedList(self.data))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2021 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/01/2022"
__status__ = "production"

import os
import threading
import logging
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
    from .matplotlib import pylab
    from . import utils as gui_utils
from .mpl_calib import MplCalibWidget
from ..control_points import ControlPoints
from ..calibrant import CALIBRANT_FACTORY
from ..blob_detection import BlobDetection
from ..massif import Massif
from ..ext.reconstruct import reconstruct
from ..ext.watershed import InverseWatershed
from ..utils.callback import dangling_callback


def preprocess_image(data, log=False, clip=0.001):
    """Preforms the pre-processing of the image
    
    :param data: the input image
    :param log: set to apply logarithmic intensity scale
    :param clip: discard pixel fraction which are too weak/intense
    :return: scaled image, bounds  
    """
    if log:
        data_disp = numpy.arcsinh(data)
    else:
        data_disp = data

    # skip lowest and highest per mille of image values via vmin/vmax
    sorted_list = data_disp.flatten()  # explicit copy
    sorted_list.sort()
    show_min = sorted_list[int(round(clip * (sorted_list.size - 1)))]
    show_max = sorted_list[int(round((1.0 - clip) * (sorted_list.size - 1)))]
    bounds = (show_min, show_max)
    return  data_disp, bounds


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
        if isinstance(data, (str,)):
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
        self.widget = None
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

        self.mpl_connectId = None
        self.defaultNbPoints = 100
        self._init_thread = None
        self.point_filename = None
        self.cb_refine = dangling_callback  # Method to be called when clicked on refine
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
            if self.widget is not None:
                self.widget.add_grp(gpt.label, points)
        return points

    def reset(self):
        """
        Reset control point and graph (if needed)
        """
        self.points.reset()
        if self.widget is not None:
            try:
                self.widget.reset()
            except Exception as err:
                print(f"{type(err)}: {err} in peak_picker.reset()")

    def gui(self, log=False, maximize=False, pick=True, widget_klass=None):
        """
        Display the GUI
        
        :param log: show z in log scale
        :param maximize: set to true to maximize window
        :param pick: activate pixel picking
        :param widget_klass: provide the MplCalibWidget (either Qt or Jupyter) to instanciate the widget
        :return: None
        """

        if self.widget is None:
            if widget_klass:
                assert issubclass(widget_klass, MplCalibWidget)
            self.widget = widget_klass(new_grp_cb=self.onclick_new_grp,
                                       append_single_cb=self.onclick_append_1_point,
                                       single_point_cb=self.onclick_single_point,
                                       append_more_cb=self.onclick_append_more_points,
                                       erase_pt_cb=self.onclick_erase_1_point,
                                       erase_grp_cb=self.onclick_erase_grp,
                                       refine_cb=self.onclick_refine,
                                       option_cb=self.onclick_option,)
            self.widget.init(pick=pick)
            self.widget.show()
        data_disp, bounds = preprocess_image(self.data, False, 1e-3)
        self.widget.imshow(data_disp, bounds=bounds, log=log, update=False)
        if self.detector:
            self.widget.set_detector(self.detector, update=False)
        if maximize:
            self.widget.maximize()
        else:
            self.widget.update()

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
        if self.widget is not None:
            if reset:
                self.widget.reset(False)

            for _lbl, gpt in self.points._groups.items():
                idx = gpt.ring
                if idx < minIndex:
                    continue
                if len(gpt) > 0:
                    self.widget.add_grp(gpt.label, gpt.points, update=False)
            self.widget.update()

    def remove_grp(self, lbl):
        """
        remove a group of points

        :param lbl: label of the group of points
        """
        gpt = self.points.pop(lbl=lbl)
        if gpt and self.widget:
            print(gpt.annotate)
            self.widget.remove_grp(gpt.label, update=True)

    def _common_creation(self, points, gpt=None, ring=None):
        """
        plot new set of points on the widget

        :param points: list of points
        :param gpt: group of point, instance of PointGroup, the one to
        :param ring: ring number  
        :return: gpt
        """
        if points:
            if not gpt and self.widget:
                gpt = self.points.append(points, ring=ring)
            self.widget.add_grp(gpt.label, points)
        return gpt

    def onclick_new_grp(self, yx, ring):
        " * new_grp Right-click (click+n):         try an auto find for a ring"
        # ydata is a float, and matplotlib display pixels centered.
        # we use floor (int cast) instead of round to avoid use of
        # banker's rounding
        points = self.massif.find_peaks(yx,
                                        self.defaultNbPoints,
                                        None, self.massif_contour)
        if points:
            gpt = self._common_creation(points, ring=ring)
            logger.info("Created group #%2s with %i points", gpt.label, len(gpt))
        else:
            logger.warning("No peak found !!!")

    def onclick_single_point(self, yx, ring):
        " * Right-click + Ctrl (click+b):  create new group with one single point"
        newpeak = self.massif.nearest_peak(yx)
        if newpeak:
            gpt = self._common_creation([newpeak], ring=ring)
            logger.info("Create group #%2s with single point x=%5.1f, y=%5.1f", gpt.label, newpeak[1], newpeak[0])
        else:
            logger.warning("No peak found !!!")

    def onclick_append_more_points(self, yx, ring):
        " * Right-click + m (click+m):     find more points for current group"
        gpt = self.points.get(ring)
        if gpt:
            self.widget.remove_grp(gpt.label, update=False)
            # need to annotate only if a new group:
            listpeak = self.massif.find_peaks(yx,
                                              self.defaultNbPoints, None,
                                              self.massif_contour)
            if listpeak:
                gpt.points += listpeak
                logger.info("Added %i points to group #%2s (now %i points)", len(listpeak), len(gpt.label), len(gpt))
            else:
                logger.warning("No peak found !!!")
            self._common_creation(gpt.points, gpt, ring=ring)
        else:
            self.onclick_new_grp(yx, ring)

    def onclick_append_1_point(self, yx, ring=None):
        """ * Right-click + Shift (click+v): add one point to current group
        :param xy: 2tuple of coordinates
        """
        gpt = self.points.get(ring)
        if gpt:
            self.widget.remove_grp(gpt.label, update=False)
            # matplotlib coord to pixel coord, avoinding use of banker's round
            newpeak = self.massif.nearest_peak(yx)
            if newpeak:
                gpt.points.append(newpeak)
                logger.info("x=%5.1f, y=%5.1f added to group #%2s", newpeak[1], newpeak[0], gpt.label)
            else:
                logger.warning("No peak found !!!")
            self._common_creation(gpt.points, gpt)
        else:
            self.onclick_new_grp(yx, ring)

    def onclick_erase_grp(self, yx, ring):
        " * Center-click or (click+d):     erase current group"
        gpt = self.points.pop(ring)
        if gpt:
            self.widget.remove_grp(gpt.label, update=True)
            if len(gpt) > 0:
                logger.info("Removing group #%2s containing %i points", gpt.label, len(gpt))
            else:
                logger.info("No groups to remove")
        else:
            logger.warning("No group of points for ring %s", ring)

    def onclick_erase_1_point(self, yx, ring):
        " * Center-click + 1 or (click+1): erase closest point from current group"
        gpt = self.points.get(ring)
        if not gpt:
            self.widget.remove_grp(gpt.label, update=True)
            if len(gpt) > 1:
                # delete single closest point from current group
                # matplotlib coord to pixel coord, avoinding use of banker's round
                y0, x0 = yx
                distsq = [((p[1] - x0) ** 2 + (p[0] - y0) ** 2) for p in gpt.points]
                # index and distance of smallest distance:
                indexMin = min(enumerate(distsq), key=operator.itemgetter(1))
                removedPt = gpt.points.pop(indexMin[0])
                logger.info("x=%5.1f, y=%5.1f removed from group #%2s (%i points left)", removedPt[1], removedPt[0], gpt.label, len(gpt))
                # annotate (new?) 1st point and add remaining points back
                self._common_creation(gpt.points, gpt, ring=ring)

            elif len(gpt) == 1:
                logger.info("Removing group #%2s containing 1 point", gpt.label)
                gpt = self.points.pop(ring)
            else:
                logger.info("No groups to remove")
        else:
            logger.warning("No group of points for ring %s", ring)

    def finish(self, filename=None, callback=None):
        """
        Ask the ring number for the given points

        :param filename: file with the point coordinates saved
        :param callback:
        :return: list of control points
        """
        logging.info(os.linesep.join(self.help))
        if callback:
            self.point_filename = filename
            # self.cb_refine = callback
            gui_utils.main_loop = True
            # MAIN LOOP
            pylab.show()
            cpt = self.points.getWeightedList(self.data)
            if callable(callback):
                callback(cpt)
        else:
            if not self.points.calibrant.dSpacing:
                logger.error("Calibrant has no line ! check input parameters please, especially the '-c' option")
                print(CALIBRANT_FACTORY)
                raise RuntimeError("Invalid calibrant")
            input("Please press enter when you are happy with your selection" + os.linesep)

            # need to disconnect 'button_press_event':
            if self.widget:
                self.widget.onclick_refine()

            print("Now fill in the ring number. Ring number starts at 0, like point-groups.")
            self.points.readRingNrFromKeyboard()
            if filename is not None:
                self.points.save(filename)
            cpt = self.points.getWeightedList(self.data)
            self.cb_refine(cpt)
        return cpt

    def contour(self, data, cmap="autumn", linewidths=2, linestyles="dashed"):
        """
        Overlay a contour-plot

        :param data: 2darray with the 2theta values in radians...
        """
        if self.widget is None:
            logging.warning("No diffraction image available => not showing the contour")
        else:
            tth_max = data.max()
            tth_min = data.min()
            if self.points.calibrant:
                angles = [i for i in self.points.calibrant.get_2th()
                          if (i is not None) and (i >= tth_min) and (i <= tth_max)]
                if not angles:
                    angles = None
            else:
                angles = None
            self.widget.contour(data, angles, cmap, linewidths, linestyles, update=True)

    def massif_contour(self, data):
        """
        Overlays a mask over a diffraction image

        :param data: mask to be overlaid
        """

        if self.widget is None:
            logging.error("No diffraction image available => not showing the contour")
        else:
            self.widget.shadow(data)

    def closeGUI(self):
        if self.widget is not None:
            self.widget.close()

    @property
    def append_mode(self):
        return self.widget.append_mode

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

    def onclick_option(self, *args):
        """
        callback function
        """
        print("Option!")

    def onclick_refine(self, *args):
        """
        callback function
        """
        logger.info("refine, now!")
        if self.point_filename:
            self.points.save(self.point_filename)
        # remove the shadow of the plot, if any
        self.widget.shadow(numpy.ones(self.data.shape, dtype=numpy.int8))
        if self.cb_refine:
            data = self.points.getWeightedList(self.data)
            self.cb_refine(data)

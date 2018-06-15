#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""
ControlPoints: a set of control points associated with a calibration image

PointGroup: a group of points
"""

from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/03/2018"
__status__ = "development"
__docformat__ = 'restructuredtext'

import os

import threading
from collections import OrderedDict
import logging

import numpy
import array

from pyFAI.third_party import six
from .calibrant import Calibrant, get_calibrant, names as calibrant_names

logger = logging.getLogger(__name__)


class ControlPoints(object):
    """
    This class contains a set of control points with (optionally) their
    ring number hence d-spacing and diffraction  2Theta angle...
    """
    def __init__(self, filename=None, calibrant=None, wavelength=None):
        self._sem = threading.Semaphore()
        self._groups = OrderedDict()
        self.calibrant = Calibrant(wavelength=wavelength)
        if filename is not None:
            self.load(filename)
        have_spacing = False
        for i in self.dSpacing:
            have_spacing = have_spacing or i
        if (not have_spacing) and (calibrant is not None):
            if isinstance(calibrant, Calibrant):
                self.calibrant = calibrant
            elif type(calibrant) in six.string_types:
                if calibrant in calibrant_names():
                    self.calibrant = get_calibrant(calibrant)
                elif os.path.isfile(calibrant):
                    self.calibrant = Calibrant(calibrant)
                else:
                    logger.error("Unable to handle such calibrant: %s", calibrant)
            elif isinstance(self.dSpacing, (numpy.ndarray, list, tuple, array)):
                self.calibrant = Calibrant(dSpacing=list(calibrant))
            else:
                logger.error("Unable to handle such calibrant: %s", calibrant)
        if not self.calibrant.wavelength:
            self.calibrant.set_wavelength(wavelength)

    def __repr__(self):
        self.check()
        lstout = ["ControlPoints instance containing %i group of point:" % len(self)]
        if self.calibrant:
            lstout.append(self.calibrant.__repr__())
        labels = self.get_labels()
        lstout.append("Containing %s groups of points:" % len(labels))
        for lbl in labels:
            lstout.append(str(self._groups[lbl]))
        return os.linesep.join(lstout)

    def __len__(self):
        return len(self._groups)

    def check(self):
        """check internal consistency of the class, disabled for now
        """
        pass

    def reset(self):
        """remove all stored values and resets them to default
        """
        with self._sem:
            self._groups = OrderedDict()
            PointGroup.reset_label()

    def append(self, points, ring=None, annotate=None, plot=None):
        """Append a group of points to a given ring

        :param point: list of points
        :param ring: ring number
        :param annotate: matplotlib.annotate reference
        :param plot: matplotlib.plot reference
        :return: PointGroup instance
        """
        with self._sem:
            gpt = PointGroup(points, ring, annotate, plot)
            self._groups[gpt.label] = gpt
        return gpt

    def append_2theta_deg(self, points, angle=None, ring=None):
        """Append a group of points to a given ring

        :param point: list of points
        :param angle: 2-theta angle in degrees
        :param: ring: ring number
        """
        if angle:
            self.append(points, numpy.deg2rad(angle), ring)
        else:
            self.append(points, None, ring)

    def get(self, ring=None, lbl=None):
        """Retireves the last group of points for a given ring (by default the last)

        :param ring: index of ring to search for
        :param lbl: label of the group to retrieve
        """
        out = None
        with self._sem:
            if lbl is None:
                if (ring is None):
                    lst = self.get_labels()
                    if not lst:
                        logger.warning("No group in ControlPoints.get")
                        return
                    lbl = lst[-1]
                else:
                    lst = [l for l, gpt in self._groups.items() if gpt.ring == ring]
                    lst.sort(key=lambda item: self._groups[item].code)
                    if not lst:
                        logger.warning("No group for ring %s in ControlPoints.get", ring)
                        return
                    lbl = lst[-1]
            if lbl in self._groups:
                out = self._groups.get(lbl)
            else:
                logger.warning("No such group %s in ControlPoints.pop", lbl)
        return out

    def pop(self, ring=None, lbl=None):
        """
        Remove the set of points, either from its code or from a given ring (by default the last)

        :param ring: index of ring of which remove the last group
        :param lbl: code of the ring to remove
        """
        out = None
        with self._sem:
            if lbl is None:
                if (ring is None):
                    lst = list(self._groups.keys())
                    lst.sort(key=lambda item: self._groups[item].code)
                    if not lst:
                        logger.warning("No group in ControlPoints.pop")
                        return
                    lbl = lst[-1]
                else:
                    lst = [l for l, gpt in self._groups.items() if gpt.ring == ring]
                    lst.sort(key=lambda item: self._groups[item].code)
                    if not lst:
                        logger.warning("No group for ring %s in ControlPoints.pop", ring)
                        return
                    lbl = lst[-1]
            if lbl in self._groups:
                out = self._groups.pop(lbl)
            else:
                logger.warning("No such group %s in ControlPoints.pop", lbl)
        return out

    def save(self, filename):
        """
        Save a set of control points to a file
        :param filename: name of the file
        :return: None
        """
        self.check()
        with self._sem:
            lstout = ["# set of control point used by pyFAI to calibrate the geometry of a scattering experiment",
                      "#angles are in radians, wavelength in meter and positions in pixels"]
            if self.calibrant:
                lstout.append("calibrant: %s" % self.calibrant)
            if self.calibrant.wavelength is not None:
                lstout.append("wavelength: %s" % self.calibrant.wavelength)
            lstout.append("dspacing:" + " ".join([str(i) for i in self.calibrant.dSpacing]))
            lst = self.get_labels()
            tth = self.calibrant.get_2th()
            for idx, lbl in enumerate(lst):
                gpt = self._groups[lbl]
                ring = gpt.ring
                lstout.append("")
                lstout.append("New group of points: %i" % idx)
                if ring < len(tth):
                    lstout.append("2theta: %s" % tth[ring])
                lstout.append("ring: %s" % ring)
                for point in gpt.points:
                    lstout.append("point: x=%s y=%s" % (point[1], point[0]))
            with open(filename, "w") as f:
                f.write("\n".join(lstout))

    def load(self, filename):
        """
        load all control points from a file
        """
        if not os.path.isfile(filename):
            logger.error("ControlPoint.load: No such file %s", filename)
            return
        self.reset()
        ring = None
        points = []
        calibrant = None
        wavelength = None
        dspacing = []

        for line in open(filename, "r"):
            if line.startswith("#"):
                continue
            elif ":" in line:
                key, value = line.split(":", 1)
                value = value.strip()
                key = key.strip().lower()
                if key == "calibrant":
                    words = value.split()
                    if words[0] in calibrant_names():
                        calibrant = get_calibrant(words[0])
                    try:
                        wavelength = float(words[-1])
                        calibrant.set_wavelength(wavelength)
                    except Exception as error:
                        logger.error("ControlPoints.load: unable to convert to float %s (wavelength): %s", value, error)
                elif key == "wavelength":
                    try:
                        wavelength = float(value)
                    except Exception as error:
                        logger.error("ControlPoints.load: unable to convert to float %s (wavelength): %s", value, error)
                elif key == "dspacing":
                    for val in value.split():
                        try:
                            fval = float(val)
                        except Exception:
                            fval = None
                        dspacing.append(fval)
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
                            logger.error("ControlPoints.load: unable to convert to float %s (point): %s", value, error)
                        else:
                            points.append([y, x])
                elif key.startswith("new"):
                    if len(points) > 0:
                        with self._sem:
                            gpt = PointGroup(points, ring)
                            self._groups[gpt.label] = gpt
                            points = []
                            ring = None
                elif key in ["2theta"]:
                    # Deprecated keys
                    pass
                else:
                    logger.error("Unknown key: %s", key)
        if len(points) > 0:
            with self._sem:
                gpt = PointGroup(points, ring)
                self._groups[gpt.label] = gpt
        # Update calibrant if needed.
        if not calibrant and dspacing:
            calibrant = Calibrant()
            calibrant.dSpacing = dspacing
        if calibrant and calibrant.wavelength is None and wavelength:
            calibrant.wavelength = wavelength
        if calibrant:
            self.calibrant = calibrant

    def getList2theta(self):
        """
        Retrieve the list of control points suitable for geometry refinement
        """
        lstout = []
        tth = self.calibrant.get_2th()
        for gpt in self._groups:
            if gpt.ring < len(tth):
                tthi = tth[gpt.ring]
                lstout += [[pt[0], pt[1], tthi] for pt in gpt.points]
        return lstout

    def getListRing(self):
        """
        Retrieve the list of control points suitable for geometry refinement with ring number
        """
        lstout = []
        for gpt in self._groups.values():
            lstout += [[pt[0], pt[1], gpt.ring] for pt in gpt.points]
        return lstout

    getList = getListRing

    def getWeightedList(self, image):
        """
        Retrieve the list of control points suitable for geometry refinement with ring number and intensities
        :param image:
        :return: a (x,4) array with pos0, pos1, ring nr and intensity

        #TODO: refine the value of the intensity using 2nd order polynomia
        """
        lstout = []
        for gpt in self._groups.values():
            lstout += [[pt[0], pt[1], gpt.ring, image[int(pt[0] + 0.5), int(pt[1] + 0.5)]] for pt in gpt.points]
        return lstout

    def readRingNrFromKeyboard(self):
        """
        Ask the ring number values for the given points
        """
        lastRing = None
        lst = list(self._groups.keys())
        lst.sort(key=lambda item: self._groups[item].code)
        for lbl in lst:
            bOk = False
            gpt = self._groups[lbl]
            while not bOk:
                defaultRing = 0
                ring = gpt.ring
                if ring is not None:
                    defaultRing = ring
                elif lastRing is not None:
                    defaultRing = lastRing + 1
                msg = "Point group #%2s (%i points)\t (%6.1f,%6.1f) \t [default=%s] Ring# "
                res = six.moves.input(msg % (lbl, len(gpt), gpt.points[0][1], gpt.points[0][0], defaultRing))
                res = res.strip()
                if res == "":
                    res = defaultRing
                try:
                    input_ring = int(res)
                except (ValueError, TypeError):
                    logging.error("I did not understand the ring number you entered")
                else:
                    if input_ring >= 0 and input_ring < len(self.calibrant.dSpacing):
                        lastRing = ring
                        gpt.ring = input_ring
                        bOk = True
                    else:
                        logging.error("Invalid ring number %i (range 0 -> %2i)",
                                      input_ring, len(self.calibrant.dSpacing) - 1)

    def setWavelength_change2th(self, value=None):
        with self._sem:
            if self.calibrant is None:
                self.calibrant = Calibrant()
            self.calibrant.setWavelength_change2th(value)

    def setWavelength_changeDs(self, value=None):
        """
        This is probably not a good idea, but who knows !
        """
        with self._sem:
            if value:
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

    def get_labels(self):
        """Retieve the list of labels

        :return: list of labels as string
        """
        labels = list(self._groups.keys())
        labels.sort(key=lambda item: self._groups[item].code)
        return labels


class PointGroup(object):
    """
    Class contains a group of points ...
    They all belong to the same Debye-Scherrer ring
    """
    last_label = 0

    @classmethod
    def get_label(cls):
        """
        return the next label
        """
        code = cls.last_label
        cls.last_label += 1
        if code < 26:
            label = chr(97 + code)
        elif code < 26 * 26:
            label = chr(96 + code // 26) + chr(97 + code % 26)
        else:
            a = code % 26
            b = code // 26
            label = chr(96 + b // 26) + chr(97 + b % 26) + chr(97 + a)
        return label, code

    @classmethod
    def set_label(cls, label):
        """
        update the internal counter if needed
        """
        if len(label) == 1:
            code = ord(label) - 97
        elif len(label) == 2:
            code = (ord(label[0]) - 96) * 26 + (ord(label[1]) - 97)
        else:
            code = (ord(label[0]) - 96) * 26 * 26 + \
                   (ord(label[1]) - 97) * 26 + \
                   (ord(label) - 97)
        if cls.last_label <= code:
            cls.last_label = code + 1
        return code

    @classmethod
    def reset_label(cls):
        """
        reset intenal counter
        """
        cls.last_label = 0

    def __init__(self, points=None, ring=None, annotate=None, plot=None, force_label=None):
        """
        Constructor

        :param points: list of points
        :param ring: ring number
        :param annotate: reference to the matplotlib annotate output
        :param plot: reference to the matplotlib plot
        :param force_label: allows to enforce the label
        """
        if points:
            self.points = points
        else:
            self.points = []
        if force_label:
            self.__label = force_label
            self.__code = self.set_label(force_label)
        else:
            self.__label, self.__code = self.get_label()
        if ring is not None:
            self._ring = int(ring)
        else:
            self._ring = None
        # placeholder of matplotlib references...
        self.annotate = annotate
        self.plot = plot

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return "#%2s ring %s: %s points" % (self.label, self.ring, len(self.points))

    def get_ring(self):
        return self._ring

    def set_ring(self, value):
        if type(value) != int:
            logger.error("Ring: %s", value)
            import traceback
            traceback.print_stack()
            self._ring = int(value)
        self._ring = value

    ring = property(get_ring, set_ring)

    @property
    def code(self):
        """
        Numerical value for the label: mainly for sorting
        """
        return self.__code

    @property
    def label(self):
        return self.__label

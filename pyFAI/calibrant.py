#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2018 European Synchrotron Radiation Facility, Grenoble, France
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


"""Calibrant

A module containing classical calibrant and also tools to generate d-spacing.

Interesting formula:
http://geoweb3.princeton.edu/research/MineralPhy/xtalgeometry.pdf
"""

from __future__ import absolute_import, print_function, with_statement

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/05/2019"
__status__ = "production"


import os
import logging
import numpy
import itertools
from math import sin, asin, cos, sqrt, pi, ceil
import threading
from .utils import get_calibration_dir
from .utils.decorators import deprecated
from . import units

logger = logging.getLogger(__name__)
epsilon = 1.0e-6  # for floating point comparison


class Cell(object):
    """
    This is a cell object, able to calculate the volume and d-spacing according to formula from:

    http://geoweb3.princeton.edu/research/MineralPhy/xtalgeometry.pdf
    """
    lattices = ["cubic", "tetragonal", "hexagonal", "rhombohedral", "orthorhombic", "monoclinic", "triclinic"]
    types = {"P": "Primitive",
             "I": "Body centered",
             "F": "Face centered",
             "C": "Side centered",
             "R": "Rhombohedral"}

    def __init__(self, a=1, b=1, c=1, alpha=90, beta=90, gamma=90, lattice="triclinic", lattice_type="P"):
        """Constructor of the Cell class:

        Crystalographic units are Angstrom for distances and degrees for angles !

        :param a,b,c: unit cell length in Angstrom
        :param alpha, beta, gamma: unit cell angle in degrees
        :param lattice: "cubic", "tetragonal", "hexagonal", "rhombohedral", "orthorhombic", "monoclinic", "triclinic"
        :param lattice_type: P, I, F, C or R
        """
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lattice = lattice if lattice in self.lattices else "triclinic"

        self._volume = None
        self.S11 = None
        self.S12 = None
        self.S13 = None
        self.S22 = None
        self.S23 = None
        self.selection_rules = []
        "contains a list of functions returning True(allowed)/False(forbiden)/None(unknown)"
        self._type = "P"
        self.set_type(lattice_type)

    def __repr__(self, *args, **kwargs):
        return "%s %s cell a=%.4f b=%.4f c=%.4f alpha=%.3f beta=%.3f gamma=%.3f" % \
            (self.types[self.type], self.lattice, self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    @classmethod
    def cubic(cls, a, lattice_type="P"):
        """Factory for cubic lattices

        :param a: unit cell length
        """
        a = float(a)
        self = cls(a, a, a, 90, 90, 90,
                   lattice="cubic", lattice_type=lattice_type)
        return self

    @classmethod
    def tetragonal(cls, a, c, lattice_type="P"):
        """Factory for tetragonal lattices

        :param a: unit cell length
        :param c: unit cell length
        """
        a = float(a)
        self = cls(a, a, float(c), 90, 90, 90,
                   lattice="tetragonal", lattice_type=lattice_type)
        return self

    @classmethod
    def orthorhombic(cls, a, b, c, lattice_type="P"):
        """Factory for orthorhombic lattices

        :param a: unit cell length
        :param b: unit cell length
        :param c: unit cell length
        """
        self = cls(float(a), float(b), float(c), 90, 90, 90,
                   lattice="orthorhombic", lattice_type=lattice_type)
        return self

    @classmethod
    def hexagonal(cls, a, c, lattice_type="P"):
        """Factory for hexagonal lattices

        :param a: unit cell length
        :param c: unit cell length
        """
        a = float(a)
        self = cls(a, a, float(c), 90, 90, 120,
                   lattice="hexagonal", lattice_type=lattice_type)
        return self

    @classmethod
    def monoclinic(cls, a, b, c, beta, lattice_type="P"):
        """Factory for hexagonal lattices

        :param a: unit cell length
        :param b: unit cell length
        :param c: unit cell length
        :param beta: unit cell angle
        """
        self = cls(float(a), float(b), float(c), 90, float(beta), 90,
                   lattice_type=lattice_type, lattice="monoclinic")
        return self

    @classmethod
    def rhombohedral(cls, a, alpha, lattice_type="P"):
        """Factory for hexagonal lattices

        :param a: unit cell length
        :param alpha: unit cell angle
        """
        a = float(a)
        alpha = float(a)
        self = cls(a, a, a, alpha, alpha, alpha,
                   lattice="rhombohedral", lattice_type=lattice_type)
        return self

    @classmethod
    def diamond(cls, a):
        """Factory for Diamond type FCC like Si and Ge

        :param a: unit cell length
        """
        self = cls.cubic(a, lattice_type="F")
        self.selection_rules.append(lambda h, k, l: not((h % 2 == 0) and (k % 2 == 0) and (l % 2 == 0) and ((h + k + l) % 4 != 0)))
        return self

    @property
    def volume(self):
        if self._volume is None:
            self._volume = self.a * self.b * self.c
            if self.lattice not in ["cubic", "tetragonal", "orthorhombic"]:
                cosa = cos(self.alpha * pi / 180.)
                cosb = cos(self.beta * pi / 180.)
                cosg = cos(self.gamma * pi / 180.)
                self._volume *= sqrt(1 - cosa ** 2 - cosb ** 2 - cosg ** 2 + 2 * cosa * cosb * cosg)
        return self._volume

    def get_type(self):
        return self._type

    def set_type(self, lattice_type):
        self._type = lattice_type if lattice_type in self.types else "P"
        self.selection_rules = [lambda h, k, l: not(h == 0 and k == 0 and l == 0)]
        if self._type == "I":
            self.selection_rules.append(lambda h, k, l: (h + k + l) % 2 == 0)
        if self._type == "F":
            self.selection_rules.append(lambda h, k, l: (h % 2 + k % 2 + l % 2) in (0, 3))
        if self._type == "R":
            self.selection_rules.append(lambda h, k, l: ((h - k + l) % 3 == 0))
    type = property(get_type, set_type)

    def d(self, hkl):
        """
        Calculate the actual d-spacing for a 3-tuple of integer representing a
        family of Miller plans

        :param hkl: 3-tuple of integers
        :return: the inter-planar distance
        """
        h, k, l = hkl
        if self.lattice in ["cubic", "tetragonal", "orthorhombic"]:
            invd2 = (h / self.a) ** 2 + (k / self.b) ** 2 + (l / self.c) ** 2
        else:
            if self.S11 is None:
                alpha = self.alpha * pi / 180.
                cosa = cos(alpha)
                sina = sin(alpha)
                beta = self.beta * pi / 180.
                cosb = cos(beta)
                sinb = sin(beta)
                gamma = self.gamma * pi / 180.
                cosg = cos(gamma)
                sing = sin(gamma)

                self.S11 = (self.b * self.c * sina) ** 2
                self.S22 = (self.a * self.c * sinb) ** 2
                self.S33 = (self.a * self.b * sing) ** 2
                self.S12 = self.a * self.b * self.c * self.c * (cosa * cosb - cosg)
                self.S23 = self.a * self.a * self.b * self.c * (cosb * cosg - cosa)
                self.S13 = self.a * self.b * self.b * self.c * (cosg * cosa - cosb)

            invd2 = (self.S11 * h * h +
                     self.S22 * k * k +
                     self.S33 * l * l +
                     2 * self.S12 * h * k +
                     2 * self.S23 * k * l +
                     2 * self.S13 * h * l)
            invd2 /= (self.volume) ** 2
        return sqrt(1 / invd2)

    def d_spacing(self, dmin=1.0):
        """Calculate all d-spacing down to dmin

        applies selection rules

        :param dmin: minimum value of spacing requested
        :return: dict d-spacing as string, list of tuple with Miller indices
                preceded with the numerical value
        """
        hmax = int(ceil(self.a / dmin))
        kmax = int(ceil(self.b / dmin))
        lmax = int(ceil(self.c / dmin))
        res = {}
        for hkl in itertools.product(range(-hmax, hmax + 1),
                                     range(-kmax, kmax + 1),
                                     range(-lmax, lmax + 1)):
            # Apply selection rule
            valid = True
            for rule in self.selection_rules:
                valid = rule(*hkl)
                if not valid:
                    break
            if not valid:
                continue

            d = self.d(hkl)
            strd = "%.8e" % d
            if d < dmin:
                continue
            if strd in res:
                res[strd].append(hkl)
            else:
                res[strd] = [d, hkl]
        return res

    def save(self, name, long_name=None, doi=None, dmin=1.0, dest_dir=None):
        """Save informations about the cell in a d-spacing file, usable as Calibrant

        :param name: name of the calibrant
        :param doi: reference of the publication used to parametrize the cell
        :param dmin: minimal d-spacing
        :param dest_dir: name of the directory where to save the result
        """
        fname = name + ".D"
        if dest_dir:
            fname = os.path.join(dest_dir, fname)
        with open(fname, "w") as f:
            if long_name:
                f.write("# Calibrant: %s (%s)%s" % (long_name, name, os.linesep))
            else:
                f.write("# Calibrant: %s%s" % (name, os.linesep))
            f.write("# %s%s" % (self, os.linesep))
            if doi:
                f.write("# Ref: %s%s" % (doi, os.linesep))
            d = self.d_spacing(dmin)
            ds = [i[0] for i in d.values()]
            ds.sort(reverse=True)
            for k in ds:
                strk = "%.8e" % k
                f.write("%.8f # %s %s%s" % (k, d[strk][-1], len(d[strk]) - 1, os.linesep))


class Calibrant(object):
    """
    A calibrant is a reference compound where the d-spacing (interplanar distances)
    are known. They are expressed in Angstrom (in the file)
    """
    def __init__(self, filename=None, dSpacing=None, wavelength=None):
        object.__init__(self)
        self._filename = filename
        self._wavelength = wavelength
        self._sem = threading.Semaphore()
        self._2th = []
        if filename is not None:
            self._dSpacing = None
        elif dSpacing is None:
            self._dSpacing = []
        else:
            self._dSpacing = list(dSpacing)
        self._out_dSpacing = []
        if self._dSpacing and self._wavelength:
            self._calc_2th()

    def __eq__(self, other):
        """
        Test the equality with another object

        It only takes into acount the wavelength and dSpacing, not the
        filename.

        :param object other: Another object
        :rtype: bool
        """
        if other is None:
            return False
        if not isinstance(other, Calibrant):
            return False
        if self._wavelength != other._wavelength:
            return False
        if self.dSpacing != other.dSpacing:
            return False
        return True

    def __ne__(self, other):
        """
        Test the non-equality with another object

        It only takes into acount the wavelength and dSpacing, not the
        filename.

        :param object other: Another object
        :rtype: bool
        """
        return not (self == other)

    def __hash__(self):
        """
        Returns the hash of the object.

        It only takes into acount the wavelength and dSpacing, not the
        filename.

        :rtype: int
        """
        h = hash(self._wavelength)
        for d in self.dSpacing:
                h = h ^ hash(d)
        return h

    def __copy__(self):
        """
        Copy a calibrant

        :rtype: Calibrant
        """
        self._initialize()
        return Calibrant(filename=self._filename,
                         dSpacing=self._dSpacing + self._out_dSpacing,
                         wavelength=self._wavelength)

    def __repr__(self):
        name = "undefined"
        if self._filename:
            name = os.path.splitext(os.path.basename(self._filename))[0]
        name += " Calibrant "
        if len(self.dSpacing):
            name += "with %i reflections " % len(self._dSpacing)
        if self._wavelength:
            name += "at wavelength %s" % self._wavelength
        return name

    def get_filename(self):
        return self._filename

    filename = property(get_filename)

    def load_file(self, filename=None):
        with self._sem:
            self._load_file(filename)

    def _load_file(self, filename=None):
        if filename:
            self._filename = filename
        if not os.path.isfile(self._filename):
            logger.error("No such calibrant file: %s", self._filename)
            return
        self._filename = os.path.abspath(self._filename)
        self._dSpacing = numpy.unique(numpy.loadtxt(self._filename))
        self._dSpacing = list(self._dSpacing[-1::-1])  # reverse order
        # self._dSpacing.sort(reverse=True)
        if self._wavelength:
            self._calc_2th()

    def _initialize(self):
        """Initialize the object if expected."""
        if self._dSpacing is None:
            if self._filename:
                self._load_file()
            else:
                self._dSpacing = []

    def count_registered_dSpacing(self):
        """Count of registered dSpacing positons."""
        self._initialize()
        return len(self._dSpacing) + len(self._out_dSpacing)

    def save_dSpacing(self, filename=None):
        """
        save the d-spacing to a file

        """
        self._initialize()
        if (filename is None) and (self._filename is not None):
            filename = self._filename
        else:
            return
        with open(filename) as f:
            f.write("# %s Calibrant" % filename)
            for i in self.dSpacing:
                f.write("%s\n" % i)

    def get_dSpacing(self):
        self._initialize()
        return self._dSpacing

    def set_dSpacing(self, lst):
        self._dSpacing = list(lst)
        self._out_dSpacing = []
        self._filename = "Modified"
        if self._wavelength:
            self._calc_2th()

    dSpacing = property(get_dSpacing, set_dSpacing)

    def append_dSpacing(self, value):
        self._initialize()
        with self._sem:
            delta = [abs(value - v) / v for v in self._dSpacing if v is not None]
            if not delta or min(delta) > epsilon:
                self._dSpacing.append(value)
                self._dSpacing.sort(reverse=True)
                self._calc_2th()

    def append_2th(self, value):
        with self._sem:
            self._initialize()
            if value not in self._2th:
                self._2th.append(value)
                self._2th.sort()
                self._calc_dSpacing()

    def setWavelength_change2th(self, value=None):
        with self._sem:
            if value:
                self._wavelength = float(value)
                if self._wavelength < 1e-15 or self._wavelength > 1e-6:
                    logger.warning("This is an unlikely wavelength (in meter): %s", self._wavelength)
                self._calc_2th()

    def setWavelength_changeDs(self, value=None):
        """
        This is probably not a good idea, but who knows !
        """
        with self._sem:
            if value:
                self._wavelength = float(value)
                if self._wavelength < 1e-15 or self._wavelength > 1e-6:
                    logger.warning("This is an unlikely wavelength (in meter): %s", self._wavelength)
                self._calc_dSpacing()

    def set_wavelength(self, value=None):
        updated = False
        with self._sem:
            if self._wavelength is None:
                if value:
                    self._wavelength = float(value)
                    if (self._wavelength < 1e-15) or (self._wavelength > 1e-6):
                        logger.warning("This is an unlikely wavelength (in meter): %s", self._wavelength)
                    updated = True
            elif abs(self._wavelength - value) / self._wavelength > epsilon:
                logger.warning("Forbidden to change the wavelength once it is fixed !!!!")
                logger.warning("%s != %s, delta= %s", self._wavelength, value, self._wavelength - value)
        if updated:
            self._calc_2th()

    def get_wavelength(self):
        return self._wavelength

    wavelength = property(get_wavelength, set_wavelength)

    def _calc_2th(self):
        """Calculate the 2theta positions for all peaks"""
        self._initialize()
        if self._wavelength is None:
            logger.error("Cannot calculate 2theta angle without knowing wavelength")
            return
        tths = []
        dSpacing = self._dSpacing[:] + self._out_dSpacing  # explicit copy
        try:
            for ds in dSpacing:
                tth = 2.0 * asin(5.0e9 * self._wavelength / ds)
                tths.append(tth)
        except ValueError:
            size = len(tths)
            # remove dSpacing outside of 0..180
            self._dSpacing = dSpacing[:size]
            self._out_dSpacing = dSpacing[size:]
        else:
            self._dSpacing = dSpacing
            self._out_dSpacing = []
        self._2th = tths

    def _calc_dSpacing(self):
        if self._wavelength is None:
            logger.error("Cannot calculate 2theta angle without knowing wavelength")
            return
        self._dSpacing = [5.0e9 * self._wavelength / sin(tth / 2.0) for tth in self._2th]

    def get_2th(self):
        """Returns the 2theta positions for all peaks (cached)"""
        if not self._2th:
            self._initialize()
            if not self._dSpacing:
                logger.error("Not d-spacing for calibrant: %s", self)
            with self._sem:
                if not self._2th:
                    self._calc_2th()
        return self._2th

    def get_2th_index(self, angle, delta=None):
        """Returns the index in the 2theta angle index

        :param angle: expected angle in radians
        :param delta: precision on angle
        :return: 0-based index or None
        """
        if angle and angle in self._2th:
            return self._2th.index(angle)
        if delta:
            d2th = abs(numpy.array(self._2th) - angle)
            if d2th.min() < delta:
                return d2th.argmin()

    def get_max_wavelength(self, index=None):
        """Calculate the maximum wavelength assuming the ring at index is visible
        :param index: Ring number, otherwise assumes all rings are visible
        :return: the maximum visible wavelength
        """
        dSpacing = self._dSpacing[:] + self._out_dSpacing  # get all rings
        if index is None:
            index = len(dSpacing) - 1
        if index >= len(dSpacing):
            raise IndexError("There are not than many (%s) rings indices in this calibrant" % (index))
        return dSpacing[index] * 1e-10

    def get_peaks(self, unit="2th_deg"):
        """Calculate the peak position as
        :return: numpy array (unlike other methods which return lists)
        """
        unit = units.to_unit(unit)
        scale = unit.scale
        name = unit.name
        size = len(self.get_2th())
        if name.startswith("2th"):
            values = numpy.array(self.get_2th())
        elif name.startswith("q"):
            values = 20.0 * pi / numpy.array(self.get_dSpacing()[:size])
        else:
            raise ValueError("Only 2\theta and *q* units are supported for now")

        return values * scale

    def fake_calibration_image(self, ai, shape=None, Imax=1.0,
                               U=0, V=0, W=0.0001):
        """
        Generates a fake calibration image from an azimuthal integrator

        :param ai: azimuthal integrator
        :param Imax: maximum intensity of rings
        :param U, V, W: width of the peak from Caglioti's law (FWHM^2 = Utan(th)^2 + Vtan(th) + W)
        """
        if shape is None:
            if ai.detector.shape:
                shape = ai.detector.shape
            elif ai.detector.max_shape:
                shape = ai.detector.max_shape
        if shape is None:
            raise RuntimeError("No shape available")
        if (self.wavelength is None) and (ai._wavelength is not None):
            self.wavelength = ai.wavelength
        elif (self.wavelength is None) and (ai._wavelength is None):
            raise RuntimeError("Wavelength needed to calculate 2theta position")
        elif (self.wavelength is not None) and (ai._wavelength is not None) and\
                abs(self.wavelength - ai.wavelength) > 1e-15:
            logger.warning("Mismatch between wavelength for calibrant (%s) and azimutal integrator (%s)",
                           self.wavelength, ai.wavelength)
        tth = ai.twoThetaArray(shape)
        tth_min = tth.min()
        tth_max = tth.max()
        dim = int(numpy.sqrt(shape[0] * shape[0] + shape[1] * shape[1]))
        tth_1d = numpy.linspace(tth_min, tth_max, dim)
        tanth = numpy.tan(tth_1d / 2.0)
        fwhm2 = U * tanth ** 2 + V * tanth + W
        sigma2 = fwhm2 / (8.0 * numpy.log(2.0))
        signal = numpy.zeros_like(sigma2)
        for t in self.get_2th():
            if t >= tth_max:
                break
            else:
                signal += Imax * numpy.exp(-(tth_1d - t) ** 2 / (2.0 * sigma2))
        res = ai.calcfrom1d(tth_1d, signal, shape=shape, mask=ai.mask,
                            dim1_unit='2th_rad', correctSolidAngle=True)
        return res

    def __getnewargs_ex__(self):
        return (self._filename, self._dSpacing, self._wavelength), {}

    def __getstate__(self):
        state_blacklist = ('_sem',)
        state = self.__dict__.copy()
        for key in state_blacklist:
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        for statekey, statevalue in state.items():
            setattr(self, statekey, statevalue)
        self._sem = threading.Semaphore()


class CalibrantFactory(object):
    """Behaves like a dict but is actually a factory:

    Each time one retrieves an object it is a new geniune new calibrant (unmodified)
    """
    def __init__(self, basedir=None):
        """
        Constructor

        :param basedir: directory name where to search for the calibrants
        """
        if basedir is None:
            basedir = get_calibration_dir()
        self.directory = basedir
        if not os.path.isdir(self.directory):
            logger.warning("No calibrant directory: %s", self.directory)
            self.all = {}
        else:
            self.all = dict([(os.path.splitext(i)[0], os.path.join(self.directory, i))
                             for i in os.listdir(self.directory)
                             if i.endswith(".D")])

    def __call__(self, calibrant_name):
        """Returns a new instance of a calibrant by it's name."""
        return Calibrant(self.all[calibrant_name])

    def get(self, what, notfound=None):
        if what in self.all:
            return Calibrant(self.all[what])
        else:
            return notfound

    def __contains__(self, k):
        return k in self.all

    def __repr__(self):
        return "Calibrants available: %s" % (", ".join(list(self.all.keys())))

    def __len__(self):
        return len(self.all)

    def keys(self):
        return list(self.all.keys())

    def values(self):
        return [Calibrant(i) for i in self.all.values()]

    def items(self):
        return [(i, Calibrant(j)) for i, j in self.all.items()]

    @deprecated  # added on 2017-03-06
    def __getitem__(self, calibration_name):
        return self(calibration_name)

    has_key = __contains__


CALIBRANT_FACTORY = CalibrantFactory()
"""Default calibration factory provided by the library."""


ALL_CALIBRANTS = CALIBRANT_FACTORY


@deprecated  # added on 2017-03-06
class calibrant_factory(CalibrantFactory):
    pass


def get_calibrant(calibrant_name):
    """Returns a new instance of the calibrant by it's name.

    :param str calibrant_name: Name of the calibrant
    """
    return CALIBRANT_FACTORY(calibrant_name)


def names():
    """Returns the list of registred calibrant names.

    :rtype: str
    """
    return CALIBRANT_FACTORY.keys()

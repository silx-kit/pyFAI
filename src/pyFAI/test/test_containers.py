#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2025-2025 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Test suite for container module"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/09/2026"

import copy
import logging
import unittest

import fabio
import numpy

from .. import containers
from .. import load as pyFAI_load
from ..utils.decorators import depreclog
from .utilstest import TestLogging, UtilsTest

logger = logging.getLogger(__name__)


class TestContainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = fabio.open(UtilsTest.getimage("moke.tif")).data
        cls.ai = pyFAI_load(
            {
                "poni_version": 2.1,
                "detector": "Detector",
                "detector_config": {
                    "pixel1": 1e-4,
                    "pixel2": 1e-4,
                    "max_shape": [500, 600],
                    "orientation": 3,
                },
                "dist": 0.1,
                "poni1": 0.03,
                "poni2": 0.03,
                "rot1": 0.0,
                "rot2": 0.0,
                "rot3": 0.0,
                "wavelength": 1.0178021533473887e-10,
            }
        )

    @classmethod
    def tearDownClass(cls):
        cls.img = cls.ai = None

    def test_recalculate_means(self):
        "Ensure means remains the same after recalculation ..."
        method = ("bbox", "csr", "cython")
        res = self.ai.integrate1d(
            self.img, 50, method=method, error_model="poisson"
        )
        ref = copy.deepcopy(res)
        res.__recalculate_means__()

        self.assertTrue(numpy.allclose(ref.radial, res.radial), "radial matches")
        self.assertTrue(numpy.allclose(ref.intensity, res.intensity), "intensity matches")
        self.assertTrue(numpy.allclose(ref.sem, res.sem), "sem matches")
        self.assertTrue(numpy.allclose(ref.std, res.std), "std matches")


    def test_rebin1d(self):
        method = ("no", "histogram", "cython")
        res2d = self.ai.integrate2d(
            self.img, 500, 360, method=method, error_model="poisson"
        )
        ref1d = self.ai.integrate1d(self.img, 500, method=method, error_model="poisson")
        with TestLogging(logger=depreclog, warning=1):
            res1d = containers.rebin1d(res2d)
        self.assertTrue(numpy.allclose(res1d[0], ref1d[0]), "radial matches")
        self.assertTrue(numpy.allclose(res1d[1], ref1d[1]), "intensity matches")
        self.assertTrue(numpy.allclose(res1d[2], ref1d[2]), "sem matches")

    def test_copy(self):
        "Every attribute survives copy and deepcopy, the ranges included"
        kwargs = {"method": ("no", "histogram", "cython"),
                  "error_model": "poisson",
                  "radial_range": (0.0, 10.0),
                  "azimuth_range": (-90.0, 90.0)}
        for res in (self.ai.integrate1d(self.img, 50, **kwargs),
                    self.ai.integrate2d(self.img, 50, 36, **kwargs)):
            name = type(res).__name__
            # an attribute missing from COPYABLE_ATTR is silently dropped by copy
            self.assertEqual(set(vars(res)) - set(res.COPYABLE_ATTR), set(),
                             f"{name}: every attribute is declared in COPYABLE_ATTR")
            self.assertIsNotNone(res.radial_range, f"{name}: radial_range was recorded")
            self.assertIsNotNone(res.azimuth_range, f"{name}: azimuth_range was recorded")
            for copied in (copy.copy(res), copy.deepcopy(res)):
                self.assertEqual(copied.radial_range, res.radial_range,
                                 f"{name}: radial_range is copied")
                self.assertEqual(copied.azimuth_range, res.azimuth_range,
                                 f"{name}: azimuth_range is copied")
                self.assertTrue(numpy.allclose(copied.sum_signal, res.sum_signal),
                                f"{name}: sum_signal is copied")

    def test_symmetrize(self):
        res2d = self.ai.integrate2d(
            self.img,
            500,
            360,
            error_model="poisson",
            radial_range=(0, 12),
            unit="2th_deg",
        )
        sym = containers.symmetrize(res2d)
        self.assertAlmostEqual(res2d.intensity.mean(), sym.intensity.mean(), places=0)

    def test_spottiness(self):
        python = self.ai.integrate1d(self.img, 100, method=("no","csr", "python"), error_model="azimuth").calc_spottiness()
        cython = self.ai.integrate1d(self.img, 100, method=("no","csr", "cython"), error_model="azimuth").calc_spottiness()
        self.assertAlmostEqual(python, 0.06396, msg="python", places=4)
        self.assertAlmostEqual(cython, 0.06396, msg="cython", places=4)

    def test_maths(self):
        method = ("no", "histogram", "cython")
        a1d = self.ai.integrate1d(self.img, 10, method=method, error_model="poisson")
        b1d = self.ai.integrate1d(
            numpy.ones_like(self.img), 10, method=method, error_model="poisson"
        )

        c1d = a1d + b1d
        self.assertTrue(numpy.allclose(c1d.sum_signal, a1d.sum_signal + b1d.sum_signal))
        self.assertTrue(
            numpy.allclose(c1d.sum_variance, a1d.sum_variance + b1d.sum_variance)
        )
        self.assertTrue(numpy.allclose(c1d.sum_normalization, b1d.sum_normalization))
        self.assertTrue(numpy.allclose(c1d.sum_normalization2, b1d.sum_normalization2))
        self.assertTrue(numpy.allclose(c1d.count, b1d.count))
        self.assertTrue(numpy.allclose(c1d.radial, b1d.radial))
        self.assertTrue(numpy.all(c1d.intensity > a1d.intensity))
        self.assertTrue(numpy.all(c1d.intensity >= b1d.intensity))
        self.assertTrue(numpy.all(c1d.std > a1d.std))
        self.assertTrue(numpy.all(c1d.std > b1d.std))
        self.assertTrue(numpy.all(c1d.sem > a1d.sem))
        self.assertTrue(numpy.all(c1d.sem > b1d.sem))
        self.assertTrue(numpy.all(c1d.sigma > a1d.sigma))
        self.assertTrue(numpy.all(c1d.sigma > b1d.sigma))

        d1d = c1d - b1d
        self.assertTrue(numpy.allclose(d1d.sum_signal, a1d.sum_signal))
        self.assertTrue(
            numpy.allclose(d1d.sum_variance, a1d.sum_variance + 2 * b1d.sum_variance)
        )
        self.assertTrue(numpy.allclose(d1d.sum_normalization, b1d.sum_normalization))
        self.assertTrue(numpy.allclose(d1d.sum_normalization2, b1d.sum_normalization2))
        self.assertTrue(numpy.allclose(d1d.count, b1d.count))
        self.assertTrue(numpy.allclose(d1d.radial, b1d.radial))
        self.assertTrue(numpy.allclose(d1d.intensity, a1d.intensity))
        self.assertTrue(numpy.all(d1d.std > a1d.std))
        self.assertTrue(numpy.all(d1d.std > b1d.std))
        self.assertTrue(numpy.all(d1d.sem > a1d.sem))
        self.assertTrue(numpy.all(d1d.sem > b1d.sem))
        self.assertTrue(numpy.all(d1d.sigma > a1d.sigma))
        self.assertTrue(numpy.all(d1d.sigma > b1d.sigma))

        e1d = copy.deepcopy(a1d)
        e1d += b1d
        self.assertTrue(numpy.allclose(e1d.sum_signal, c1d.sum_signal))
        self.assertTrue(numpy.allclose(e1d.sum_variance, c1d.sum_variance))
        self.assertTrue(numpy.allclose(e1d.sum_normalization, c1d.sum_normalization))
        self.assertTrue(numpy.allclose(e1d.sum_normalization2, c1d.sum_normalization2))
        self.assertTrue(numpy.allclose(e1d.count, c1d.count))
        self.assertTrue(numpy.allclose(e1d.radial, c1d.radial))
        self.assertTrue(numpy.allclose(e1d.intensity, c1d.intensity))
        self.assertTrue(numpy.allclose(e1d.std, c1d.std))
        self.assertTrue(numpy.allclose(e1d.sem, c1d.sem))
        self.assertTrue(numpy.allclose(e1d.sigma, c1d.sigma))

        f1d = copy.deepcopy(c1d)
        f1d -= b1d
        self.assertTrue(numpy.allclose(f1d.sum_signal, a1d.sum_signal))
        self.assertTrue(numpy.allclose(f1d.sum_variance, d1d.sum_variance))
        self.assertTrue(numpy.allclose(f1d.sum_normalization, a1d.sum_normalization))
        self.assertTrue(numpy.allclose(f1d.sum_normalization2, a1d.sum_normalization2))
        self.assertTrue(numpy.allclose(f1d.count, a1d.count))
        self.assertTrue(numpy.allclose(f1d.radial, a1d.radial))
        self.assertTrue(numpy.allclose(f1d.intensity, a1d.intensity))
        self.assertTrue(numpy.allclose(f1d.std, d1d.std))
        self.assertTrue(numpy.allclose(f1d.sem, d1d.sem))
        self.assertTrue(numpy.allclose(f1d.sigma, d1d.sigma))

        g1d = a1d.union(b1d)
        self.assertTrue(numpy.allclose(g1d.sum_signal, a1d.sum_signal+b1d.sum_signal))
        self.assertTrue(numpy.allclose(g1d.sum_variance, a1d.sum_variance+b1d.sum_variance))
        self.assertTrue(numpy.allclose(g1d.sum_normalization, a1d.sum_normalization+b1d.sum_normalization))
        self.assertTrue(numpy.allclose(g1d.sum_normalization2, a1d.sum_normalization2+b1d.sum_normalization2))
        self.assertTrue(numpy.allclose(g1d.count, a1d.count+b1d.count))
        self.assertTrue(numpy.allclose(g1d.radial, a1d.radial))
        self.assertFalse(numpy.allclose(g1d.intensity, a1d.intensity))
        self.assertFalse(numpy.allclose(g1d.std, a1d.std))
        self.assertFalse(numpy.allclose(g1d.sem, a1d.sem))
        self.assertFalse(numpy.allclose(g1d.sigma, a1d.sigma))

        # same with 2D arrays
        a2d = self.ai.integrate2d(
            self.img, 40, 36, method=method, error_model="poisson"
        )
        b2d = self.ai.integrate2d(
            numpy.ones_like(self.img), 40, 36, method=method, error_model="poisson"
        )

        c2d = a2d + b2d
        self.assertTrue(numpy.allclose(c2d.sum_signal, a2d.sum_signal + b2d.sum_signal))
        self.assertTrue(
            numpy.allclose(c2d.sum_variance, a2d.sum_variance + b2d.sum_variance)
        )
        self.assertTrue(numpy.allclose(c2d.sum_normalization, b2d.sum_normalization))
        self.assertTrue(numpy.allclose(c2d.sum_normalization2, b2d.sum_normalization2))
        self.assertTrue(numpy.allclose(c2d.count, b2d.count))
        self.assertTrue(numpy.allclose(c2d.radial, b2d.radial))
        self.assertTrue(numpy.all(c2d.intensity >= a2d.intensity))
        self.assertTrue(numpy.all(c2d.intensity >= b2d.intensity))
        self.assertTrue(numpy.all(c2d.std >= a2d.std))
        self.assertTrue(numpy.all(c2d.std >= b2d.std))
        self.assertTrue(numpy.all(c2d.sem >= a2d.sem))
        self.assertTrue(numpy.all(c2d.sem >= b2d.sem))
        self.assertTrue(numpy.all(c2d.sigma >= a2d.sigma))
        self.assertTrue(numpy.all(c2d.sigma >= b2d.sigma))

        d2d = c2d - b2d
        self.assertTrue(numpy.allclose(d2d.sum_signal, a2d.sum_signal))
        self.assertTrue(
            numpy.allclose(d2d.sum_variance, a2d.sum_variance + 2 * b2d.sum_variance)
        )
        self.assertTrue(numpy.allclose(d2d.sum_normalization, b2d.sum_normalization))
        self.assertTrue(numpy.allclose(d2d.sum_normalization2, b2d.sum_normalization2))
        self.assertTrue(numpy.allclose(d2d.count, b2d.count))
        self.assertTrue(numpy.allclose(d2d.radial, b2d.radial))
        self.assertTrue(numpy.allclose(d2d.intensity, a2d.intensity))
        self.assertTrue(numpy.all(d2d.std >= a2d.std))
        self.assertTrue(numpy.all(d2d.std >= b2d.std))
        self.assertTrue(numpy.all(d2d.sem >= a2d.sem))
        self.assertTrue(numpy.all(d2d.sem >= b2d.sem))
        self.assertTrue(numpy.all(d2d.sigma >= a2d.sigma))
        self.assertTrue(numpy.all(d2d.sigma >= b2d.sigma))

        e2d = copy.deepcopy(a2d)
        e2d += b2d
        self.assertTrue(numpy.allclose(e2d.sum_signal, c2d.sum_signal))
        self.assertTrue(numpy.allclose(e2d.sum_variance, c2d.sum_variance))
        self.assertTrue(numpy.allclose(e2d.sum_normalization, c2d.sum_normalization))
        self.assertTrue(numpy.allclose(e2d.sum_normalization2, c2d.sum_normalization2))
        self.assertTrue(numpy.allclose(e2d.count, c2d.count))
        self.assertTrue(numpy.allclose(e2d.radial, c2d.radial))
        self.assertTrue(numpy.allclose(e2d.intensity, c2d.intensity))
        self.assertTrue(numpy.allclose(e2d.std, c2d.std))
        self.assertTrue(numpy.allclose(e2d.sem, c2d.sem))
        self.assertTrue(numpy.allclose(e2d.sigma, c2d.sigma))

        f2d = copy.deepcopy(c2d)
        f2d -= b2d
        self.assertTrue(numpy.allclose(f2d.sum_signal, a2d.sum_signal))
        self.assertTrue(numpy.allclose(f2d.sum_variance, d2d.sum_variance))
        self.assertTrue(numpy.allclose(f2d.sum_normalization, a2d.sum_normalization))
        self.assertTrue(numpy.allclose(f2d.sum_normalization2, a2d.sum_normalization2))
        self.assertTrue(numpy.allclose(f2d.count, a2d.count))
        self.assertTrue(numpy.allclose(f2d.radial, a2d.radial))
        self.assertTrue(numpy.allclose(f2d.intensity, a2d.intensity))
        self.assertTrue(numpy.allclose(f2d.std, d2d.std))
        self.assertTrue(numpy.allclose(f2d.sem, d2d.sem))
        self.assertTrue(numpy.allclose(f2d.sigma, d2d.sigma))

        g2d = a2d.union(b2d)
        self.assertTrue(numpy.allclose(g2d.sum_signal, a2d.sum_signal+b2d.sum_signal))
        self.assertTrue(numpy.allclose(g2d.sum_variance, a2d.sum_variance+b2d.sum_variance))
        self.assertTrue(numpy.allclose(g2d.sum_normalization, a2d.sum_normalization+b2d.sum_normalization))
        self.assertTrue(numpy.allclose(g2d.sum_normalization2, a2d.sum_normalization2+b2d.sum_normalization2))
        self.assertTrue(numpy.allclose(g2d.count, a2d.count+b2d.count))
        self.assertTrue(numpy.allclose(g2d.radial, a2d.radial))
        self.assertFalse(numpy.allclose(g2d.intensity, a2d.intensity))
        self.assertFalse(numpy.allclose(g2d.std, a2d.std))
        self.assertFalse(numpy.allclose(g2d.sem, a2d.sem))
        self.assertFalse(numpy.allclose(g2d.sigma, a2d.sigma))

        # azimuthal propagation:
        method = ("no", "csr", "cython")

        a1 = self.ai.integrate1d(self.img, 10, method=method, error_model="azimuthal")
        noise = numpy.random.random(self.img.shape)
        a2 = self.ai.integrate1d(noise, 10, method=method, error_model="azimuthal")
        b = a1.union(a2)
        c =  self.ai.integrate1d((self.img + noise), 10, method=method, error_model="azimuthal")
        self.assertTrue(numpy.allclose(c.sum_signal, b.sum_signal))

        self.assertTrue(numpy.allclose(c.sum_normalization, a1.sum_normalization))
        self.assertTrue(numpy.allclose(c.sum_normalization2, a1.sum_normalization2))
        self.assertTrue(numpy.allclose(b.sum_normalization, 2*a1.sum_normalization))
        self.assertTrue(numpy.allclose(b.sum_normalization2, 2*a1.sum_normalization2))
        self.assertTrue(numpy.allclose(c.count, a1.count))
        self.assertTrue(numpy.allclose(c.radial, b.radial))
        self.assertTrue(numpy.allclose(c.intensity, 2*b.intensity, rtol=0.1))
        self.assertTrue(numpy.allclose(c.sum_variance, b.sum_variance, rtol=1))
        self.assertTrue(numpy.allclose(c.std**2, 2*b.std**2, rtol=1))
        # self.assertTrue(numpy.allclose(c.sem, b.sem, rtol=0.4))
        # self.assertTrue(numpy.allclose(c.sigma, b.sigma, rtol=0.4))

    def test_renormalize(self):
        method = ("bbox", "histogram", "cython")
        res1 = self.ai.integrate1d(
            self.img, 50, method=method, error_model="poisson"
        )
        res2 = res1.renormalize(2)
        self.assertTrue(numpy.allclose(res1[0], res2[0]), "radial matches")
        self.assertTrue(numpy.allclose(res1[1], 2*res2[1]), "intensity matches")
        self.assertTrue(numpy.allclose(res1[2], 2*res2[2]), "sem matches")
        self.assertTrue(numpy.allclose(res1.std, 2*res2.std), "std matches")

    def test_immutable_dict(self):
        d = containers.ImmutableDict({"a": 1, "b": 2})
        e = containers.ImmutableDict({"b": 2, "a": 1})
        self.assertEqual(d, e)
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2)
        for key in d:
            self.assertTrue(key in ("a", "b"))
        self.assertEqual(len(d), 2)
        self.assertEqual(list(d.keys()), ["a", "b"])
        self.assertEqual(list(e.values()), [1, 2])
        self.assertEqual(list(d.items()), [("a", 1), ("b", 2)])


class TestRebin1dSector(unittest.TestCase):
    """Validate `Integrate2dResult.rebin1d` on an azimuthally modulated image.

    The synthetic image holds a few Debye-Scherrer rings whose intensity is
    modulated by cos²(χ). Re-binning a fine 2D integration over an azimuthal
    sector has to match, bin per bin, the 1D integration performed on this very
    same sector.

    All integrations are performed with explicit `radial_range`/`azimuth_range`
    and a pixel-splitting-free method, so that the bin boundaries of the 2D
    integration fall exactly on the requested sector limits: the comparison is
    then expected to be exact, not approximate.
    """

    NPT_RAD = 500
    NPT_AZIM = 360
    RADIAL_RANGE = (0.0, 8.0)
    AZIMUTH_RANGE = (-180.0, 180.0)
    UNIT = "2th_deg"
    METHOD = ("no", "histogram", "cython")
    RINGS = (2.0, 4.0, 6.0)
    BACKGROUND = 10.0

    @classmethod
    def setUpClass(cls):
        cls.ai = pyFAI_load(
            {
                "poni_version": 2.1,
                "detector": "Detector",
                "detector_config": {
                    "pixel1": 1e-4,
                    "pixel2": 1e-4,
                    "max_shape": [200, 200],
                    "orientation": 3,
                },
                "dist": 0.1,
                # slightly off the geometrical center of the detector, on purpose:
                # a PONI centered, or shifted by a whole number of pixels, puts thousands of
                # pixels exactly on the ±45°/±135° diagonals, i.e. exactly on a bin boundary,
                # where the 1D and the 2D integration are free to disagree by one bin.
                "poni1": 0.010321,
                "poni2": 0.009717,
                "rot1": 0.0,
                "rot2": 0.0,
                "rot3": 0.0,
                "wavelength": 1e-10,
            }
        )
        tth = cls.ai.array_from_unit(unit=cls.UNIT, typ="center", scale=True)
        chi = numpy.rad2deg(cls.ai.center_array(unit="chi_rad", scale=False))
        rings = numpy.zeros(tth.shape, dtype=numpy.float64)
        for position in cls.RINGS:
            rings += 1000.0 * numpy.exp(-0.5 * ((tth - position) / 0.1) ** 2)
        # cos² modulation of the rings, on top of a flat background
        cls.img = rings * numpy.cos(numpy.deg2rad(chi)) ** 2 + cls.BACKGROUND

    @classmethod
    def tearDownClass(cls):
        cls.ai = cls.img = None

    def integrate2d(self, **kwargs):
        """Fine 2D integration used as the input of rebin1d"""
        kwargs.setdefault("radial_range", self.RADIAL_RANGE)
        kwargs.setdefault("azimuth_range", self.AZIMUTH_RANGE)
        return self.ai.integrate2d(self.img, self.NPT_RAD, self.NPT_AZIM,
                                   method=self.METHOD, unit=self.UNIT,
                                   error_model="poisson", **kwargs)

    def integrate1d(self, **kwargs):
        """Reference 1D integration, sharing the radial binning of integrate2d"""
        kwargs.setdefault("radial_range", self.RADIAL_RANGE)
        return self.ai.integrate1d(self.img, self.NPT_RAD,
                                   method=self.METHOD, unit=self.UNIT,
                                   error_model="poisson", **kwargs)

    def assertSameCurve(self, obtained, expected, radial_slice=slice(None), msg=""):
        """Compare a rebinned result with a reference 1D integration"""
        for name in ("radial", "sum_signal", "sum_normalization", "count",
                     "sum_variance", "intensity", "sem", "std"):
            ref = getattr(expected, name)[radial_slice]
            obt = getattr(obtained, name)
            self.assertEqual(obt.shape, ref.shape, f"{msg}: shape of {name}")
            self.assertTrue(numpy.allclose(obt, ref, rtol=1e-5, atol=1e-6, equal_nan=True),
                            f"{msg}: {name} differs by "
                            f"{abs(numpy.nan_to_num(obt) - numpy.nan_to_num(ref)).max()}")

    def test_azimuthal_sector(self):
        "rebin1d over a sector matches integrate1d restricted to that sector"
        res2d = self.integrate2d()
        for sector in ((30.0, 60.0), (-90.0, -30.0), (0.0, 180.0), (-45.0, 45.0)):
            res1d = res2d.rebin1d(azimuth_range=sector)
            ref1d = self.integrate1d(azimuth_range=sector)
            self.assertSameCurve(res1d, ref1d, msg=f"sector {sector}")

    def test_azimuthal_sector_is_a_subset(self):
        "The rebinned sectors sum back to the full azimuthal range"
        res2d = self.integrate2d()
        total = res2d.rebin1d()
        parts = [res2d.rebin1d(azimuth_range=sector)
                 for sector in ((-180.0, -60.0), (-60.0, 60.0), (60.0, 180.0))]
        self.assertTrue(numpy.allclose(sum(p.sum_signal for p in parts), total.sum_signal),
                        "signal is conserved")
        self.assertTrue(numpy.allclose(sum(p.count for p in parts), total.count),
                        "count is conserved")

    def test_radial_range(self):
        "rebin1d with a radial_range crops the curve, keeping the bin size"
        res2d = self.integrate2d()
        ref1d = self.integrate1d()
        # 8 degrees over 500 bins: 0.016 deg per bin, (2, 6) falls on bin edges
        res1d = res2d.rebin1d(radial_range=(2.0, 6.0))
        self.assertEqual(res1d.radial.size, 250, "number of radial bins")
        self.assertSameCurve(res1d, ref1d, slice(125, 375), msg="radial crop")

    def test_radial_and_azimuthal_range(self):
        "Both ranges can be combined"
        sector = (30.0, 60.0)
        res2d = self.integrate2d()
        res1d = res2d.rebin1d(radial_range=(2.0, 6.0), azimuth_range=sector)
        ref1d = self.integrate1d(azimuth_range=sector)
        self.assertSameCurve(res1d, ref1d, slice(125, 375), msg="both ranges")

    def test_implicit_ranges(self):
        "rebin1d works as well when integrate2d was called without any range"
        res2d = self.ai.integrate2d(self.img, self.NPT_RAD, self.NPT_AZIM,
                                    method=self.METHOD, unit=self.UNIT,
                                    error_model="poisson")
        # The bin edges are no more aligned on round values: rebuild the ones
        # actually used by the 2D integration to build the reference.
        azim_edges = containers._to_edges(res2d.azimuthal)
        radial_edges = containers._to_edges(res2d.radial)
        first, last = 210, 240
        sector = (azim_edges[first], azim_edges[last])
        res1d = res2d.rebin1d(azimuth_range=sector)
        ref1d = self.ai.integrate1d(self.img, self.NPT_RAD, method=self.METHOD,
                                    unit=self.UNIT, error_model="poisson",
                                    radial_range=(radial_edges[0], radial_edges[-1]),
                                    azimuth_range=sector)
        self.assertEqual(res1d.radial.size, self.NPT_RAD, "all radial bins are kept")
        self.assertTrue(numpy.allclose(res1d.count, ref1d.count), "count matches")
        self.assertTrue(numpy.allclose(res1d.sum_signal, ref1d.sum_signal, rtol=1e-5),
                        "signal matches")

    def mean_peak_intensity(self, res1d):
        """Weighted average intensity around the ring at 4°, background subtracted.

        The ring at 4° lies well inside the detector, so it is fully covered
        whatever the azimuthal sector: the radial weighting is the same for all
        sectors and the ratio of two sectors is the ratio of their modulation.
        """
        window = numpy.logical_and(res1d.radial > 3.5, res1d.radial < 4.5)
        signal = res1d.sum_signal[window].sum()
        normalization = res1d.sum_normalization[window].sum()
        return signal / normalization - self.BACKGROUND

    def test_azimuthal_modulation(self):
        "The cos² modulation is recovered sector per sector"
        res2d = self.integrate2d()
        reference = self.mean_peak_intensity(res2d.rebin1d())
        half_width = 15.0
        for center in (0.0, 30.0, 60.0, 90.0, 150.0):
            sector = (center - half_width, center + half_width)
            res1d = res2d.rebin1d(azimuth_range=sector)
            # analytical average of cos²(χ) over the sector, the mean over the
            # whole azimuthal range being 1/2
            low, high = (numpy.deg2rad(i) for i in sector)
            expected = 1.0 + (numpy.sin(2 * high) - numpy.sin(2 * low)) / (2 * (high - low))
            obtained = self.mean_peak_intensity(res1d) / reference
            self.assertAlmostEqual(obtained, expected, delta=0.03,
                                   msg=f"cos² modulation at χ={center}°")

    def test_other_engines(self):
        "The agreement holds for every pixel-splitting-free engine"
        sector = (30.0, 60.0)
        for algo in ("histogram", "csr", "csc", "lut"):
            method = ("no", algo, "cython")
            res2d = self.ai.integrate2d(self.img, self.NPT_RAD, self.NPT_AZIM,
                                        method=method, unit=self.UNIT,
                                        error_model="poisson",
                                        radial_range=self.RADIAL_RANGE,
                                        azimuth_range=self.AZIMUTH_RANGE)
            ref1d = self.ai.integrate1d(self.img, self.NPT_RAD, method=method,
                                        unit=self.UNIT, error_model="poisson",
                                        radial_range=self.RADIAL_RANGE,
                                        azimuth_range=sector)
            res1d = res2d.rebin1d(azimuth_range=sector)
            self.assertSameCurve(res1d, ref1d, msg=f"engine {algo}")

    def test_ranges_are_stored_in_the_public_unit(self):
        "radial_range/azimuth_range match the unit of the position arrays"
        sector = (30.0, 60.0)
        res2d = self.integrate2d()
        self.assertEqual(res2d.radial_range, self.RADIAL_RANGE, "radial_range of integrate2d")
        self.assertTrue(numpy.allclose(res2d.azimuth_range, self.AZIMUTH_RANGE),
                        f"azimuth_range of integrate2d: {res2d.azimuth_range}")
        for axis, stored in (("radial", res2d.radial_range),
                             ("azimuthal", res2d.azimuth_range)):
            edges = containers._to_edges(getattr(res2d, axis))
            self.assertTrue(numpy.allclose(stored, (edges[0], edges[-1]), atol=1e-5),
                            f"the {axis} range {stored} frames the axis {(edges[0], edges[-1])}")

        ref1d = self.integrate1d(azimuth_range=sector)
        self.assertEqual(ref1d.radial_range, self.RADIAL_RANGE, "radial_range of integrate1d")
        self.assertTrue(numpy.allclose(ref1d.azimuth_range, sector),
                        f"azimuth_range of integrate1d: {ref1d.azimuth_range}")

        # the ranges are the boundaries of the bins which were actually kept
        res1d = res2d.rebin1d(radial_range=(2.0, 6.0), azimuth_range=sector)
        self.assertTrue(numpy.allclose(res1d.radial_range, (2.0, 6.0)),
                        f"radial_range of rebin1d: {res1d.radial_range}")
        self.assertTrue(numpy.allclose(res1d.azimuth_range, sector),
                        f"azimuth_range of rebin1d: {res1d.azimuth_range}")
        # ... hence consistent with the position array they describe
        delta = 0.5 * (self.RADIAL_RANGE[1] - self.RADIAL_RANGE[0]) / self.NPT_RAD
        self.assertAlmostEqual(res1d.radial[0] - delta, res1d.radial_range[0],
                               delta=1e-5, msg="lower bound of the radial axis")
        self.assertAlmostEqual(res1d.radial[-1] + delta, res1d.radial_range[1],
                               delta=1e-5, msg="upper bound of the radial axis")

    def test_ranges_in_radian(self):
        "The scaling follows the requested unit, radians included"
        # with an azimuthal unit in radians, azimuth_range is expected in radians too
        res2d = self.ai.integrate2d(self.img, self.NPT_RAD, self.NPT_AZIM,
                                    method=self.METHOD, unit=("2th_rad", "chi_rad"),
                                    radial_range=(0.0, 0.1),
                                    azimuth_range=(-numpy.pi, numpy.pi))
        self.assertTrue(numpy.allclose(res2d.radial_range, (0.0, 0.1)),
                        f"radial_range in radians: {res2d.radial_range}")
        # the azimuthal axis is in radians: so is the range, which frames it exactly
        edges = containers._to_edges(res2d.azimuthal)
        self.assertTrue(numpy.allclose(res2d.azimuth_range, (edges[0], edges[-1]), atol=1e-9),
                        f"azimuth_range {res2d.azimuth_range} frames {(edges[0], edges[-1])}")
        # rebin1d then expects its sector in radians as well
        sector = (edges[100], edges[130])
        res1d = res2d.rebin1d(azimuth_range=sector)
        self.assertEqual(res1d.count.size, self.NPT_RAD, "all radial bins are kept")
        self.assertTrue(numpy.allclose(res1d.azimuth_range, sector, atol=1e-9),
                        f"azimuth_range of rebin1d: {res1d.azimuth_range}")
        self.assertTrue(numpy.allclose(res1d.count,
                                       res2d.count[100:130].sum(axis=0)),
                        "exactly the 30 bins of the sector were kept")

    def test_irregular_binning_warning(self):
        "A non uniform binning is signaled, since the bin boundaries are then guessed"
        self.assertTrue(containers._check_regular_binning(numpy.arange(10.0), "test"),
                        "a regular axis is accepted silently")
        res2d = copy.deepcopy(self.ai.integrate2d(self.img, self.NPT_RAD, self.NPT_AZIM,
                                                  method=self.METHOD, unit=self.UNIT))
        # the position arrays are plain writable arrays: distort the radial one
        res2d.radial[:] = numpy.sqrt(res2d.radial)
        with TestLogging(logger="pyFAI.containers", warning=1):
            res2d.rebin1d(radial_range=(1.0, 2.0))

    def test_empty_sector(self):
        "An azimuthal range holding no bin is rejected"
        res2d = self.integrate2d()
        with self.assertRaises(ValueError):
            res2d.rebin1d(azimuth_range=(30.0, 30.0))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestContainer))
    testsuite.addTest(loader(TestRebin1dSector))
    return testsuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

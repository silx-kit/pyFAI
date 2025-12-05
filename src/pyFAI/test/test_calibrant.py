#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suites for calibrants"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/12/2025"

import unittest
import itertools
import logging
import sys
import os
import copy
import numpy
import h5py
from .utilstest import UtilsTest
from ..calibrant import Calibrant, get_calibrant, Cell, CALIBRANT_FACTORY
from ..detectors import ALL_DETECTORS
from ..integrator.azimuthal import AzimuthalIntegrator
from ..crystallography.space_groups import ReflectionCondition
from ..io.calibrant_config import CalibrantConfig
logger = logging.getLogger(__name__)


class TestCalibrant(unittest.TestCase):
    """
    Test calibrant installation and loading
    """

    def test_factory(self):
        # by default we provide 11 calibrants
        count = len(CALIBRANT_FACTORY)
        self.assertTrue(count > 10, "at least 11 calibrants are available, got %s" % count)

        self.assertTrue("LaB6" in CALIBRANT_FACTORY, "LaB6 is a calibrant")

        # ensure each calibrant instance is unique
        cal1 = get_calibrant("LaB6")
        cal1.wavelength = 1e-10
        cal2 = get_calibrant("LaB6")
        self.assertTrue(cal2.wavelength is None, "calibrant is delivered without wavelength")

        # check that it is possible to instantiate all calibrant
        for _k, v in CALIBRANT_FACTORY.items():
            self.assertTrue(isinstance(v, Calibrant))

    def test_2th(self):
        lab6 = get_calibrant("LaB6")
        lab6.wavelength = 1.54e-10
        tth = lab6.get_2th()
        self.assertEqual(len(tth), 25, "We expect 25 rings for LaB6")

        lab6.setWavelength_change2th(1e-10)
        tth = lab6.get_2th()
        self.assertEqual(len(tth), 59, "We expect 59 rings for LaB6")

        lab6.setWavelength_change2th(2e-10)
        tth = lab6.get_2th()
        self.assertEqual(len(tth), 15, "We expect 15 rings for LaB6")

        self.assertEqual(lab6.get_2th_index(1.0, 0.04), 3, "right index picked")

    def test_fake(self):
        """test for fake image generation"""
        detectors = set(ALL_DETECTORS.values())
        for _idx, detector in enumerate(detectors):
            det = detector()
            # Skip generic detectors
            if "MAX_SHAPE" not in dir(det):
                continue
            # skip the big detectors for now
            if max(det.MAX_SHAPE) > 1000:
                continue
            ai = AzimuthalIntegrator(dist=0.01, poni1=0, poni2=0, detector=det)
            calibrant = get_calibrant("LaB6")
            calibrant.wavelength = 1e-10
            img = calibrant.fake_calibration_image(ai)

            logger.info("%s min: %s max: %s ", det.name, img.min(), img.max())
            self.assertTrue(img.shape == det.shape, "Image (%s) has the right size" % (det.name,))
            self.assertTrue(img.sum() > 0, "Image (%s) contains some data" % (det.name,))
            sys.stderr.write(".")

    def test_get_peaks(self):
        calibrant = get_calibrant("LaB6")
        calibrant.wavelength = 1e-10
        ref = calibrant.get_2th()

        delta = abs(calibrant.get_peaks() - numpy.rad2deg(ref))
        self.assertLess(delta.max(), 1e-10, "results are the same")

        self.assertEqual(len(calibrant.get_peaks("q_A^-1")), len(ref), "length is OK")

    def test_amount_of_calibrant(self):
        c = get_calibrant("LaB6")
        nb = c.count_registered_dSpacing()
        c.setWavelength_change2th(0.00000000002)
        c.setWavelength_change2th(0.0000000002)
        c.setWavelength_change2th(0.00000000002)
        c.setWavelength_change2th(0.0000000002)
        self.assertEqual(c.count_registered_dSpacing(), nb)

    def test_factory_create_calibrant(self):
        c1 = get_calibrant("LaB6")
        c2 = get_calibrant("LaB6")
        self.assertIsNot(c1, c2)

    def test_same(self):
        c1 = get_calibrant("LaB6")
        c2 = get_calibrant("LaB6")
        self.assertEqual(c1, c2)

    def test_same2(self):
        c1 = get_calibrant("LaB6")
        c2 = get_calibrant("LaB6")
        c1.wavelength = 1e-10 
        c2.wavelength = 1e-10
        self.assertEqual(c1, c2)

    def test_not_same_dspace(self):
        "this test checked the 2 calibrant are actually lazy-loaded"
        c1 = get_calibrant("LaB6_SRM660a")
        c2 = get_calibrant("LaB6_SRM660b")
        self.assertNotEqual(c1, c2)

    def test_not_same_wavelength(self):
        c1 = get_calibrant("LaB6")
        c1.wavelength=1e-10
        c2 = get_calibrant("LaB6")
        self.assertNotEqual(c1, c2)

    def test_copy(self):
        c1 = get_calibrant("AgBh")
        c2 = copy.copy(c1)
        self.assertIsNot(c1, c2)
        self.assertEqual(c1, c2)
        c2.wavelength=1e-10
        self.assertNotEqual(c1, c2)

    def test_hash(self):
        c1 = get_calibrant("AgBh")
        c2 = get_calibrant("AgBh")
        c3 = get_calibrant("AgBh")
        c3.wavelength=1e-10
        c4 = get_calibrant("LaB6")
        store = {}
        store[c1] = True
        self.assertTrue(c1 in store)
        self.assertTrue(c2 in store)
        self.assertTrue(c3 not in store)
        self.assertTrue(c4 not in store)

    def test_all_calibrants_idempotent(self):
        """Check that all calibrant from the factory can be:
        * instantiated
        * parsed without loss of information"""
        for c in CALIBRANT_FACTORY.all:
            print(c, end=": ")
            cal = CALIBRANT_FACTORY(c)
            print(cal)
            # check for idempotence of the parser ...
            filename = Calibrant._get_abs_path(CALIBRANT_FACTORY.all[c])
            with open(filename) as fd:
                ref = numpy.array([i.strip() for i in fd.readlines()])
            cal = str(CalibrantConfig.from_dspacing(filename=filename))
            obt = numpy.array([i.strip() for i in cal.split(os.linesep)])
            res = ref != obt
            self.assertFalse(res.any(), f"Non idempotent: `{c}` lines {numpy.where(res)[0]}: {ref[res]} vs {obt[res]}")


    def test_energy(self):
        calibrant = get_calibrant("LaB6")
        calibrant.energy = 20  # keV
        self.assertAlmostEqual(calibrant.wavelength, 6.19920e-11, places=15)
        self.assertEqual(len(calibrant.dspacing), 151)
        self.assertEqual(calibrant.energy, 20)



class TestCell(unittest.TestCase):
    """
    Test generation of a calibrant from a cell
    """

    def test_class(self):
        c = Cell()
        self.assertAlmostEqual(c.volume, 1.0, msg="Volume of triclinic 1,1,1,90,90,90 == 1.0, got %s" % c.volume)
        c = Cell(1, 2, 3)
        self.assertAlmostEqual(c.volume, 6.0, msg="Volume of triclinic 1,2,3,90,90,90 == 6.0, got %s" % c.volume)
        c = Cell(1, 2, 3, 90, 30, 90)
        self.assertAlmostEqual(c.volume, 3.0, msg="Volume of triclinic 1,2,3,90,30,90 == 3.0, got %s" % c.volume)

    def test_classmethods(self):
        c = Cell.cubic(1)
        self.assertAlmostEqual(c.volume, 1.0, msg="Volume of cubic 1 == 1.0, got %s" % c.volume)
        c = Cell.tetragonal(2, 3)
        self.assertAlmostEqual(c.volume, 12.0, msg="Volume of tetragonal 2,3 == 12.0, got %s" % c.volume)
        c = Cell.orthorhombic(1, 2, 3)
        self.assertAlmostEqual(c.volume, 6.0, msg="Volume of orthorhombic 1,2,3 == 6.0, got %s" % c.volume)

    def test_dspacing(self):
        c = Cell.cubic(1)
        cd = c.calculate_dspacing(0.1)
        cds = list(cd.keys())
        cds.sort()

        t = Cell()
        td = t.calculate_dspacing(0.1)
        tds = list(td.keys())
        tds.sort()

        self.assertEqual(cds, tds, msg="d-spacings are the same")
        for k in cds:
            self.assertEqual(cd[k], td[k], msg="plans are the same for d=%s" % k)

    def test_helium(self):
        # self.skipTest("Not working")
        a = 4.242
        href = "A.F. Schuch and R.L. Mills, Phys. Rev. Lett., 1961, 6, 596."
        he = Cell.cubic(a)
        self.assertTrue(len(he.calculate_dspacing(1)) == 15, msg="got 15 lines for He")
        he.save("He", "Helium", href, 1.0, UtilsTest.tempdir)
        calibrant = he.to_calibrant(dmin=1.0)
        self.assertTrue(isinstance(calibrant, Calibrant))
        self.assertEqual(len(calibrant.dspacing), 15)

    def test_hydrogen(self):
        # self.skipTest("Not working")
        href = "DOI: 10.1126/science.239.4844.1131"
        h = Cell.hexagonal(2.6590, 4.3340)
        self.assertAlmostEqual(h.volume, 26.537, places=3, msg="Volume for H cell is correct")
        self.assertTrue(len(h.calculate_dspacing(1)) == 14, msg="got 14 lines for H")
        h.save("H", "Hydrogen", href, 1.0, UtilsTest.tempdir)
        calibrant = Calibrant.from_cell(h)
        self.assertEqual(len(calibrant.dSpacing), 14)

class TestReflection(unittest.TestCase):
    """
    Test space group reflection allowance against xrayutilities obtained against:

    import xrayutilities.materials.spacegrouplattice
    import h5py
    hdf5 = h5py.File("reflection.h5")
    for sgn in range (1,3):  # Triclinic
         sg=xrayutilities.materials.spacegrouplattice.SGLattice(sgn, 1.1,1.2,1.3,95,96,97)
         table=numpy.zeros((size, size,size),dtype=bool)
         for i in itertools.product(range(size),range(size), range(size)): table[i]=sg.hkl_allowed(i)
         hdf5[sg.space_group] = table


    for sgn in range (3,16):  # Monoclinic
         sg=xrayutilities.materials.spacegrouplattice.SGLattice(sgn, 1.1,1.2,1.3,95)
         table=numpy.zeros((size, size,size),dtype=bool)
         for i in itertools.product(range(size),range(size), range(size)): table[i]=sg.hkl_allowed(i)
         hdf5[sg.space_group] = table


    for sgn in range (16,75):  # orthorhombic
         sg=xrayutilities.materials.spacegrouplattice.SGLattice(sgn, 1.1,1.2,1.3)
         table=numpy.zeros((size, size,size),dtype=bool)
         for i in itertools.product(range(size),range(size), range(size)): table[i]=sg.hkl_allowed(i)
         hdf5[sg.space_group] = table


    for sgn in range (75,143):  # Tetragonal
         sg=xrayutilities.materials.spacegrouplattice.SGLattice(sgn, 1.1,1.2)
         table=numpy.zeros((size, size,size),dtype=bool)
         for i in itertools.product(range(size),range(size), range(size)): table[i]=sg.hkl_allowed(i)
         hdf5[sg.space_group] = table


    for sgn in range (143,168):  # Trigonal
         sg=xrayutilities.materials.spacegrouplattice.SGLattice(sgn, 1.1,1.2)
         table=numpy.zeros((size, size,size),dtype=bool)
         for i in itertools.product(range(size),range(size), range(size)): table[i]=sg.hkl_allowed(i)
         hdf5[sg.space_group] = table


    for sgn in range (168,195):  # Hexagonal
         sg=xrayutilities.materials.spacegrouplattice.SGLattice(sgn, 1.1,1.2)
         table=numpy.zeros((size, size,size),dtype=bool)
         for i in itertools.product(range(size),range(size), range(size)): table[i]=sg.hkl_allowed(i)
         hdf5[sg.space_group] = table


    for sgn in range (195,231):  # Cubic
         sg=xrayutilities.materials.spacegrouplattice.SGLattice(sgn, 1.1)
         table=numpy.zeros((size, size,size),dtype=bool)
         for i in itertools.product(range(size),range(size), range(size)): table[i]=sg.hkl_allowed(i)
         hdf5[sg.space_group] = table
    """


    #print(UtilsTest.resources.data_home)

    @classmethod
    def setUpClass(cls):
        cls.reflection_file = UtilsTest.getimage("reflection.h5")
        #print(f"reflection_file: {cls.reflection_file!r}")

    @staticmethod
    def build_table( funct, size=10):
        """build a 10x10x10 map with True for allowed reflection"""
        table = numpy.zeros((size, size, size), dtype=bool)
        for i in itertools.product(range(size),range(size), range(size)):
            table[i]=funct(*i)
        return table



    def test_code(self):
        "Checks the class has no issue and that validated methods are actually correct !"
        with h5py.File(self.reflection_file) as reflections:
            groups = []
            for name in dir(ReflectionCondition):
                if name.startswith("group"):
                    groups.append(name)
            groups.sort(key=lambda i:int(i[5:].split("_",1)[0]))
            for name in groups:
                method = getattr(ReflectionCondition, name)
                table = self.build_table(method)
                nr = name[5:].split("_",1)[0]
                if nr in reflections:
                    ref = reflections[nr][()]
                else:
                    good = [i for i in reflections.keys() if i.startswith(nr+":")]
                    if good:
                        ref = reflections[good[0]][()]
                if "validated" in method.__doc__.lower():
                    if not numpy.all(ref==table):
                        print(name, "differ at hkl=", [(int(h), int(k), int(l)) for h,k,l in zip(*numpy.where(ref!=table))])  # noqa
                        raise AssertionError(f"Space group {name} did not validate against xrayutilities")

def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestCalibrant))
    testsuite.addTest(loader(TestCell))
    testsuite.addTest(loader(TestReflection))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

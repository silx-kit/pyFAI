#!/usr/bin/env python3
# coding: utf-8
#
#    Project: PyFAI: diffraction signal analysis
#             https://github.com/silx-kit/pyFAI
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
Simple test of peak-pickers within pyFAI
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2020-2021 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/10/2021"

import logging
import numpy

import unittest
from .. import ocl
import fabio
from ...test.utilstest import UtilsTest
from ...azimuthalIntegrator import AzimuthalIntegrator
if ocl:
    from ..peak_finder import OCL_SimplePeakFinder, OCL_PeakFinder, densify
logger = logging.getLogger(__name__)


@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestOclPeakFinder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestOclPeakFinder, cls).setUpClass()
        if ocl:
            if logger.getEffectiveLevel() <= logging.INFO:
                cls.PROFILE = True
            else:
                cls.PROFILE = False
        cls.ref = numpy.array([(88, 705), (1097, 907), (833, 930), (1520, 1083),
                               (1463, 1249), (1721, 1281), (1274, 1316), (1662, 1372),
                               (165, 1433), (304, 1423), (1058, 1449), (1260, 1839),
                               (806, 2006), (129, 2149), (1981, 2272), (1045, 2446)],
                               dtype=[('x', '<i4'), ('y', '<i4')])
        cls.img = fabio.open(UtilsTest.getimage("Pilatus6M.cbf")).data
        cls.ai = AzimuthalIntegrator.sload(UtilsTest.getimage("Pilatus6M.poni"))

    @classmethod
    def tearDownClass(cls):
        super(TestOclPeakFinder, cls).tearDownClass()
        cls.ai = None
        cls.img = None
        cls.ref = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_simple_peak_finder(self):
        """
        test for simple peak picker
        """
        msk = self.img < 0
        pf = OCL_SimplePeakFinder(mask=msk)
        res = pf(self.img, window=11)
        s1 = set((i["x"], i["y"]) for i in self.ref)
        s2 = set(zip(res.x, res.y))
        self.assertGreater(len(s2), len(s1), "Many more peaks with default settings")
        self.assertFalse(bool(s1.difference(s1.intersection(s2))), "All peaks found")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_azimuthal_peak_finder(self):
        """
        test for peak picker with background calculate from an azimuthal sigma-clipping
        """
        unit = "r_m"
        msk = self.img < 0
        engine = self.ai.setup_CSR(self.img.shape, 128, mask=msk, split="no", unit=unit)
        bin_centers = engine.bin_centers
        lut = engine.lut
        distance = self.ai._cached_array["r_center"]
        pf = OCL_PeakFinder(lut, self.img.size, unit=unit, radius=distance, bin_centers=bin_centers, mask=msk)
        res = pf(self.img, error_model="poisson", dummy=-1)
        s1 = set((i["x"], i["y"]) for i in self.ref)
        s2 = set(zip(res.x, res.y))
        self.assertGreater(len(s2), len(s1), "Many more peaks with default settings")
        self.assertFalse(bool(s1.difference(s1.intersection(s2))), "All peaks found")
        # Test densification function
        dense = densify(res)
        self.assertLess(abs(dense - self.img).max(), 20, "max difference is contained")
        self.assertLess(abs((dense - self.img).mean()), 1, "mean of difference is close to zero")
        self.assertLess((dense - self.img).std(), 3, "standard deviation of difference is contained")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_azimuthal_peak_finder_single(self):
        """
        test for peak picker with background calculate from an azimuthal sigma-clipping

        single threaded version
        """
        unit = "r_m"
        msk = self.img < 0
        engine = self.ai.setup_CSR(self.img.shape, 128, mask=msk, split="no", unit=unit)
        bin_centers = engine.bin_centers
        lut = engine.lut
        distance = self.ai._cached_array["r_center"]
        pf = OCL_PeakFinder(lut, self.img.size, unit=unit, radius=distance, bin_centers=bin_centers, mask=msk,
                            block_size=1)
        res = pf(self.img, error_model="poisson", dummy=-1)
        s1 = set((i["x"], i["y"]) for i in self.ref)
        s2 = set(zip(res.x, res.y))
        self.assertGreater(len(s2), len(s1), "Many more peaks with default settings")
        self.assertFalse(bool(s1.difference(s1.intersection(s2))), "All peaks found")
        # Test densification function
        dense = densify(res)
        self.assertLess(abs(dense - self.img).max(), 20, "max difference is contained")
        self.assertLess(abs((dense - self.img).mean()), 1, "mean of difference is close to zero")
        self.assertLess((dense - self.img).std(), 3, "standard deviation of difference is contained")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_azimuthal_peak_finder_chauvenet(self):
        """
        test for peak picker with background calculate from an azimuthal sigma-clipping
        test using cut-off value obtained from Chauvenet criterion
        """
        unit = "r_m"
        msk = self.img < 0
        engine = self.ai.setup_CSR(self.img.shape, 128, mask=msk, split="no", unit=unit)
        bin_centers = engine.bin_centers
        lut = engine.lut
        distance = self.ai._cached_array["r_center"]
        pf = OCL_PeakFinder(lut, self.img.size, unit=unit, radius=distance, bin_centers=bin_centers, mask=msk)
        res = pf(self.img, error_model="poisson", dummy=-1, cutoff_clip=0)
        s1 = set((i["x"], i["y"]) for i in self.ref)
        s2 = set(zip(res.x, res.y))
        self.assertGreater(len(s2), len(s1), "Many more peaks with default settings")
        self.assertFalse(bool(s1.difference(s1.intersection(s2))), "All peaks found")
        # Test densification function
        dense = densify(res)
        self.assertLess(abs(dense - self.img).max(), 20, "max difference is contained")
        self.assertLess(abs((dense - self.img).mean()), 1, "mean of difference is close to zero")
        self.assertLess((dense - self.img).std(), 3, "standard deviation of difference is contained")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_peakfinder8(self):
        """
        test for peakfinder8
        """
        unit = "r_m"
        msk = self.img < 0
        engine = self.ai.setup_CSR(self.img.shape, 1000, mask=msk, split="no", unit=unit)
        bin_centers = engine.bin_centers
        lut = engine.lut
        distance = self.ai._cached_array["r_center"]
        pf = OCL_PeakFinder(lut, self.img.size, unit=unit, radius=distance, bin_centers=bin_centers, mask=msk, 
                            block_size = 32) # leads to a 4x8 patch size
        
        np = pf.count(self.img, error_model="poisson")
        res = pf.peakfinder8(self.img, error_model="poisson")
        
        s1 = numpy.vstack((self.ref["x"], self.ref["y"])).T
        s2 = numpy.vstack((res["pos1"], res["pos0"])).T
        from scipy.spatial import distance_matrix
        dm = distance_matrix(s1, s2)
        self.assertLess(len(res), np, "Many more peaks with default settings")
        self.assertLess(numpy.median(dm.min(axis=1)), 1, "Most peaks are found")

def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testSuite = unittest.TestSuite()
    testSuite.addTest(loader(TestOclPeakFinder))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")

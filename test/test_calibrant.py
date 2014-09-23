#!/usr/bin/env python

"""
Test suites for calibrants 
"""


import unittest
import numpy
import sys
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

from pyFAI import calibrant


class TestCalibrant(unittest.TestCase):
    """
    Test calibrant installation and loading
    """
    def test_factory(self):
        # by default we provide 11 calibrants
        l = len(calibrant.ALL_CALIBRANTS)
        self.assert_(l > 10, "at least 11 calibrants are available, got %s" % l)

        self.assert_("LaB6" in calibrant.ALL_CALIBRANTS, "LaB6 is a calibrant")

        #ensure each calibrant instance is uniq
        cal1 = calibrant.ALL_CALIBRANTS["LaB6"]
        cal1.wavelength = 1e-10
        cal2 = calibrant.ALL_CALIBRANTS["LaB6"]
        self.assert_(cal2.wavelength is None, "calibrant is delivered without wavelength")

    def test_2th(self):
        lab6 = calibrant.ALL_CALIBRANTS["LaB6"]
        lab6.wavelength = 1.54e-10
        tth = lab6.get_2th()
        self.assert_(len(tth) == 25, "We expect 25 rings for LaB6")
        lab6.setWavelength_change2th(1e-10)
        tth = lab6.get_2th()
        self.assert_(len(tth) == 25, "We still expect 25 rings for LaB6 (some are missing lost)")
        lab6.setWavelength_change2th(2e-10)
        tth = lab6.get_2th()
        self.assert_(len(tth) == 15, "Only 15 remaining out of 25 rings for LaB6 (some additional got lost)")

    def test_fake(self):
        """test for fake image generation"""
        det = pyFAI.detectors.detector_factory("pilatus1m")
        ai = pyFAI.AzimuthalIntegrator(dist=0.1, poni1=0.1, poni2=0.1, detector=det)
        lab6 = pyFAI.calibrant.ALL_CALIBRANTS["LaB6"]
        lab6.set_wavelength(1e-10)
        img = lab6.fake_calibration_image(ai)
        self.assert_(img.max() > 0.8, "Image contains some data")
        self.assert_(img.min() == 0, "Image contains some data")


def test_suite_all_calibrant():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestCalibrant("test_factory"))
    testSuite.addTest(TestCalibrant("test_2th"))
    testSuite.addTest(TestCalibrant("test_fake"))
    return testSuite


if __name__ == '__main__':
    mysuite = test_suite_all_calibrant()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)

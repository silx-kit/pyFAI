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
#    author: Picca Frédéric-Emmanuel <picca@synchrotron-soleil.fr>
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
"test suite for masked arrays"

__author__ = "Picca Frédéric-Emmanuel"
__contact__ = "picca@synchrotron-soleil.fr"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "24/09/2013"

import sys
import unittest
import numpy
from utilstest import getLogger  # UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI.detectors import detector_factory


class TestDetector(unittest.TestCase):

    def test_detector_imxpad_s140(self):
        """
        The masked image has a masked ring around 1.5deg with value
        -10 without mask the pixels should be at -10 ; with mask they
        are at 0
        """
        imxpad = detector_factory("imxpad_s140")

        # check that the cartesian coordinates is cached
        self.assertEqual(hasattr(imxpad, 'COORDINATES'), False)
        y, x = imxpad.calc_cartesian_positions()
        self.assertEqual(hasattr(imxpad, 'COORDINATES'), True)

        # now check that the cached values are identical for each
        # method call
        y1, x1 = imxpad.calc_cartesian_positions()
        self.assertEqual(numpy.all(numpy.equal(y1, y)), True)
        self.assertEqual(numpy.all(numpy.equal(x1, x)), True)

        # check that a few pixel positiopns are ok.
        self.assertAlmostEqual(y[0], 130e-6 / 2.)
        self.assertAlmostEqual(y[1], y[0] + 130e-6)
        self.assertAlmostEqual(y[119], y[118] + 130e-6 * 3.5 / 2.)

        self.assertAlmostEqual(x[0], 130e-6 / 2.)
        self.assertAlmostEqual(x[1], x[0] + 130e-6)
        self.assertAlmostEqual(x[79], x[78] + 130e-6 * 3.5 / 2.)

def test_suite_all_detectors():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestDetector("test_detector_imxpad_s140"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_detectors()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)

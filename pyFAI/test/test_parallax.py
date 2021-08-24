#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021 European Synchrotron Radiation Facility, Grenoble, France
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

"""Test suites for parallax correction"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "24/08/2021"

import unittest
import numpy
import logging
logger = logging.getLogger(__name__)
from ..parallax import Beam, ThinSensor, BaseSensor, Parallax


class TestParallax(unittest.TestCase):
    """Test Azimuthal integration based sparse matrix multiplication methods
    Bounding box pixel splitting
    """
    def test_beam(self):
        width = 1e-3
        for profile in ("gaussian", "circle", "square"):
            beam =  Beam(width, profile)
            x,y = beam()
            self.assertGreaterEqual(x[-1]-x[0], width, "{profile} profile is large enough")
            self.assertTrue(numpy.isclose(y.sum(), 1.0), "intensity are normalized")
    
        
    
    def test_decay(self):
        t = ThinSensor(450e-6, 0.3)
        self.assertTrue(isinstance(t, BaseSensor))
        self.assertTrue(t.test(), msg="autotest OK")
    
    def test_serialize1(self):
        beam = Beam(1e-3)
        sensor=ThinSensor(1e-3, 0.3)
        p = Parallax(beam=beam, sensor=sensor); q=Parallax()
        q.set_config(p.get_config())
        self.assertEqual(str(p), str(q))

    def test_serialize2(self):
        beam = Beam(1e-3)
        sensor=BaseSensor(1e-3)
        p = Parallax(beam=beam, sensor=sensor); q=Parallax()
        q.set_config(p.get_config())
        self.assertEqual(str(p), str(q))

        
def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestParallax))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

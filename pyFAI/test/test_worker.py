#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, division, print_function

__doc__ = "test suite for worker"
__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/05/2016"


import unittest
import numpy
from .utilstest import getLogger
from .. import units
from ..worker import Worker
from ..azimuthalIntegrator import AzimuthalIntegrator

logger = getLogger(__file__)


class AzimuthalIntegratorMocked():

    def __init__(self, result=None):
        self._integrate1d_called = 0
        self._integrate2d_called = 0
        self._result = result
        self._csr_integrator = 0
        self._lut_integrator = numpy.array([0])

    def integrate1d(self, **kargs):
        self._integrate1d_called += 1
        self._integrate1d_kargs = kargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def integrate2d(self, **kargs):
        self._integrate2d_called += 1
        self._integrate2d_kargs = kargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class TestWorker(unittest.TestCase):

    def test_constructor_ai(self):
        ai = AzimuthalIntegrator()
        w = Worker(ai)
        self.assertIsNotNone(w)

    def test_constructor(self):
        w = Worker()
        self.assertIsNotNone(w)

    def test_process_1d(self):
        ai_result = numpy.array([0, 1]), numpy.array([2, 3])
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai)
        data = numpy.array([0])
        worker.unit = units.TTH_DEG
        worker.dummy = "b"
        worker.delta_dummy = "c"
        worker.method = "d"
        worker.polarization = "e"
        worker.correct_solid_angle = "f"
        worker.nbpt_rad = "g"
        worker.nbpt_azim = 1
        worker.output = "numpy"
        result = worker.process(data)

        # ai calls
        self.assertEquals(ai._integrate1d_called, 1)
        self.assertEquals(ai._integrate2d_called, 0)
        ai_args = ai._integrate1d_kargs
        self.assertEquals(ai_args["unit"], worker.unit)
        self.assertEquals(ai_args["dummy"], worker.dummy)
        self.assertEquals(ai_args["delta_dummy"], worker.delta_dummy)
        self.assertEquals(ai_args["method"], worker.method)
        self.assertEquals(ai_args["polarization_factor"], worker.polarization)
        self.assertEquals(ai_args["safe"], True)
        self.assertEquals(ai_args["data"], data)
        self.assertEquals(ai_args["correctSolidAngle"], worker.correct_solid_angle)
        self.assertEquals(ai_args["npt"], worker.nbpt_rad)

        # result
        print(worker.radial)
        self.assertEquals(result.tolist(), [[0, 2], [1, 3]])
        self.assertEquals(worker.radial.tolist(), [0, 1])
        self.assertEquals(worker.azimuthal, None)

    def test_process_2d(self):
        ai_result = numpy.array([0]), numpy.array([1]), numpy.array([2])
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai)
        data = numpy.array([0])
        worker.unit = units.TTH_RAD
        worker.dummy = "b"
        worker.delta_dummy = "c"
        worker.method = "d"
        worker.polarization = "e"
        worker.correct_solid_angle = "f"
        worker.nbpt_rad = "g"
        worker.nbpt_azim = 2
        worker.output = "numpy"
        result = worker.process(data)

        # ai calls
        self.assertEqual(ai._integrate1d_called, 0)
        self.assertEqual(ai._integrate2d_called, 1)
        ai_args = ai._integrate2d_kargs
        self.assertEquals(ai_args["unit"], worker.unit)
        self.assertEquals(ai_args["dummy"], worker.dummy)
        self.assertEquals(ai_args["delta_dummy"], worker.delta_dummy)
        self.assertEquals(ai_args["method"], worker.method)
        self.assertEquals(ai_args["polarization_factor"], worker.polarization)
        self.assertEquals(ai_args["safe"], True)
        self.assertEquals(ai_args["data"], data)
        self.assertEquals(ai_args["correctSolidAngle"], worker.correct_solid_angle)
        self.assertEquals(ai_args["npt_rad"], worker.nbpt_rad)
        self.assertEquals(ai_args["npt_azim"], worker.nbpt_azim)

        # result
        self.assertEquals(result.tolist(), [0])
        self.assertEquals(worker.radial.tolist(), [1])
        self.assertEquals(worker.azimuthal.tolist(), [2])

    def test_process_exception(self):
        ai = AzimuthalIntegratorMocked(result=Exception("Out of memory"))
        worker = Worker(ai)
        data = numpy.array([0])
        worker.nbpt_azim = 2
        try:
            worker.process(data)
        except:
            pass

    def test_process_poisson(self):
        ai_result = numpy.array([0]), numpy.array([1]), numpy.array([2])
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai)
        data = numpy.array([0])
        worker.nbpt_azim = 1
        worker.do_poisson = True
        worker.process(data)
        self.assertIn("error_model", ai._integrate1d_kargs)
        self.assertEquals(ai._integrate1d_kargs["error_model"], "poisson")

    def test_process_no_output(self):
        ai_result = numpy.array([0]), numpy.array([1]), numpy.array([2])
        ai = AzimuthalIntegratorMocked(result=ai_result)
        worker = Worker(ai)
        data = numpy.array([0])
        worker.output = None
        worker.nbpt_azim = 1
        worker.do_poisson = True
        result = worker.process(data)
        self.assertIsNone(result)


def suite():
    testsuite = unittest.TestSuite()
    test_names = unittest.getTestCaseNames(TestWorker, "test")
    for test in test_names:
        testsuite.addTest(TestWorker(test))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

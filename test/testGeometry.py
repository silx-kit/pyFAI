#!/usr/bin/env python

"""tests for Jon's geometry changes
FIXME : make some tests that the functions do what is expected

"""


import unittest, numpy, os, sys, time
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
import pyFAI.geometry
geometry = pyFAI.geometry

class ParameterisedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parameterised should
        inherit from this class.
        From Eli Bendersky's website
        http://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases/
    """
    def __init__(self, methodName='runTest', param=None):
        super(ParameterisedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parameterise(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite

class TestGeometry(ParameterisedTestCase):

    def testGeometryFunctions(self):
        func, statargs, varargs, kwds, expectedFail = self.param
        kwds["pixel1"] = 1
        kwds["pixel2"] = 1
        g = geometry.Geometry(**kwds)
        g.wavelength = 1e-10
        t0 = time.time()
        oldret = getattr(g, func)(*statargs, path=varargs[0])
        t1 = time.time()
        newret = getattr(g, func)(*statargs, path=varargs[1])
        t2 = time.time()
        logger.debug("TIMINGS\t meth: %s t=%.3fs\t meth: %s t=%.3fs" % (varargs[0], t1 - t0, varargs[1], t2 - t1))
        maxDelta = abs(oldret - newret).max()
        msg = "geo=%s%s max delta=%.3f" % (g, os.linesep, maxDelta)
        if expectedFail:
            self.assertNotAlmostEquals(maxDelta, 0, 3, msg)
        else:
            self.assertAlmostEquals(maxDelta, 0, 3, msg)
        logger.info(msg)

size = 1024
d1, d2 = numpy.mgrid[-size:size:32, -size:size:32]

TESTCASES = [
 ("tth", (d1, d2), ("cos", "tan"), {'dist':1, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-1, 'rot2':1, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-1, 'rot2':-.2, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-1.2, 'rot2':1, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'dist':1e10, 'rot1':-2, 'rot2':2, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'dist':1, 'rot1':3, 'rot2':0, 'rot3':0}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-1, 'rot2':1, 'rot3':3}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-3, 'rot2':-.2, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'rot1':-1.2, 'rot2':1.6, 'rot3':1}, False),
 ("tth", (d1, d2), ("cos", "tan"), {'dist':1e10, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'dist':1, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-1, 'rot2':1, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-1, 'rot2':-.2, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-1.2, 'rot2':1, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'dist':1e10, 'rot1':-2, 'rot2':2, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'dist':1, 'rot1':3, 'rot2':0, 'rot3':0}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-1, 'rot2':1, 'rot3':3}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-3, 'rot2':-.2, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'rot1':-1.2, 'rot2':1.6, 'rot3':1}, False),
 ("tth", (d1, d2), ("tan", "cython"), {'dist':1e10, 'rot1':0, 'rot2':0, 'rot3':0}, False),

 ("qFunction", (d1, d2), ("cython", "tan"), {'dist':1, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-1, 'rot2':1, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-1, 'rot2':-.2, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-1.2, 'rot2':1, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'dist':1e10, 'rot1':-2, 'rot2':2, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'dist':1, 'rot1':3, 'rot2':0, 'rot3':0}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-1, 'rot2':1, 'rot3':3}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-3, 'rot2':-.2, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'rot1':-1.2, 'rot2':1.6, 'rot3':1}, False),
 ("qFunction", (d1, d2), ("cython", "tan"), {'dist':1e10, 'rot1':0, 'rot2':0, 'rot3':0}, False),

 ("rFunction", (d1, d2), ("cython", "numpy"), {'dist':1, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-1, 'rot2':1, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-1, 'rot2':-.2, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-1.2, 'rot2':1, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'dist':1e10, 'rot1':-2, 'rot2':2, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'dist':1, 'rot1':3, 'rot2':0, 'rot3':0}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-1, 'rot2':1, 'rot3':3}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-3, 'rot2':-.2, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'rot1':-1.2, 'rot2':1.6, 'rot3':1}, False),
 ("rFunction", (d1, d2), ("cython", "numpy"), {'dist':1e10, 'rot1':0, 'rot2':0, 'rot3':0}, False),


 ]


def test_suite_all_Geometry():
    testSuite = unittest.TestSuite()
    for param in TESTCASES:
        testSuite.addTest(ParameterisedTestCase.parameterise(
                TestGeometry, param))
    return testSuite



if __name__ == '__main__':
    mysuite = test_suite_all_Geometry()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)

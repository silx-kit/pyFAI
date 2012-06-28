#!/usr/bin/env python

"""tests for Jon's geometry changes
FIXME : make some tests that the functions do what is expected

"""


import unittest, numpy, os
from pyFAI import geometry as geometry


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
        g = geometry.Geometry(**kwds)
        oldret = getattr(g, func)(*statargs, path=varargs[0])
        newret = getattr(g, func)(*statargs, path=varargs[1])
        maxDelta = abs(oldret - newret).max()
        msg = "geo=%s%s max delta=%.3f" % (g, os.linesep, maxDelta)
        if expectedFail:
            self.assertNotAlmostEquals(maxDelta, 0, 3, msg)
        else:
            self.assertAlmostEquals(maxDelta, 0, 3, msg)
        print msg

TESTCASES = [
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'dist':1, 'rot1':0, 'rot2':0, 'rot3':0}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'rot1':-1, 'rot2':1, 'rot3':1}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'rot1':-1, 'rot2':-.2, 'rot3':1}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'rot1':-1.2, 'rot2':1, 'rot3':1}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'dist':1e10, 'rot1':-2, 'rot2':2, 'rot3':1}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'dist':1, 'rot1':3, 'rot2':0, 'rot3':0}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'rot1':-1, 'rot2':1, 'rot3':3}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'rot1':-.2, 'rot2':1, 'rot3':-.1}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'rot1':-3, 'rot2':-.2, 'rot3':1}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'rot1':1, 'rot2':5, 'rot3':.4}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'rot1':-1.2, 'rot2':1.6, 'rot3':1}, False),
 ("tth", (numpy.arange(1000), numpy.arange(1000)), ("cos", "tan"), {'dist':1e10, 'rot1':0, 'rot2':0, 'rot3':0}, False),
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
